"""VAE encodes kept across calls: an exact cache on comfy.sd.VAE.encode.

A VAE encode is a pure function of the weights and the pixels it is handed. The
same pixels come back more often than the callers can see:

- Continuity keys its own reference cache on the *generation's* canvas (under
  `ref_size: match`), and a render compiles its pass at two canvases (the main
  pass and the 768x768 face pass). A reference clip whose own canvas is smaller
  than both is resized to the same pixels twice, misses Continuity's cache
  twice, and the MiniMax H3 video VAE spends ~21 s on each. Measured
  2026-11-07: the two latents are bit-identical (`fe049e5c...` both).
- Any workflow that re-runs an encode node after an unrelated graph change.

This keys on content, for any model and any workflow:

- **The weights**: the loader ComfyUI records on the patcher
  (`cached_patcher_init`) and its arguments, files by path, size and mtime.
- **How the VAE runs**: its model class, weight and output dtypes, device
  type, tiling settings, object patches and model options.
- **The pixels**: shape, dtype and every byte, by a fixed 8-way SHA-256 tree
  (~0.04 s for a 617 MB clip on the GB10's Grace cores).

Only single-batch encodes are kept: ComfyUI splits a batch by free memory, and
which split a call gets is not part of the key. A VAE with weight patches, or a
loader that recorded nothing, is not cached. Results are kept on the CPU and
handed out as copies, on the device and in the dtype the encode returns.
Least recently used first out, within SPARK_VAE_CACHE_MB (default 1024).
Opt-out: SPARK_VAE_CACHE=0.

The cache stays the outermost wrapper on VAE.encode: dgx-monarch wraps it too
when its mesh arms (the Worker-assisted encode split), and a hit must return
before a split opens a cluster session. `install()` runs again on every prompt
and puts the cache back on top; a cache layer left underneath steps aside.
"""
import contextvars
import functools
import hashlib
import logging
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

from .te_cache import _Opaque, _feed, _file_identity

log = logging.getLogger("spark.vae_cache")

_lock = threading.Lock()
_store = OrderedDict()      # key -> (cpu tensor, encode seconds, bytes)
_bytes = 0
STATS = {"hits": 0, "misses": 0, "skipped": 0, "saved_s": 0.0}
SKIPS = {}

TREE = 8                    # fixed: the digest must not depend on the thread count
_pool = ThreadPoolExecutor(TREE, thread_name_prefix="vae-cache-digest")
_INSIDE = contextvars.ContextVar("spark_vae_cache_inside", default=False)
_WRAPPER = None             # the outermost cache wrapper this process installed


def enabled():
    return os.environ.get("SPARK_VAE_CACHE", "1").strip().lower() not in (
        "0", "false", "no", "off")


def _budget():
    try:
        return max(0, int(os.environ.get("SPARK_VAE_CACHE_MB", "1024"))) << 20
    except ValueError:
        return 1024 << 20


def _skip(reason):
    with _lock:
        SKIPS[reason] = SKIPS.get(reason, 0) + 1
    return None


def pixels_digest(pixels):
    """Shape, dtype and every byte of a tensor -> hex digest (SHA-256 of 8 chunk digests)."""
    import torch

    t = pixels.detach()
    if t.is_sparse or t.is_quantized:
        raise _Opaque("tensor layout")
    t = t.to("cpu").contiguous()
    head = f"T{tuple(t.shape)}{t.dtype};".encode()
    flat = t.reshape(-1).view(torch.uint8).numpy() if t.numel() else b""
    n = len(flat)
    chunks = [flat[i * n // TREE:(i + 1) * n // TREE] for i in range(TREE)]
    digests = list(_pool.map(lambda c: hashlib.sha256(c).digest(), chunks))
    return hashlib.sha256(head + b"".join(digests)).hexdigest()


def _batch(vae, pixels):
    """How many items ComfyUI's encode loop would split, as VAE.encode reshapes them."""
    if getattr(vae, "latent_dim", None) == 3 and pixels.ndim < 5:
        return pixels.shape[0] if getattr(vae, "not_video", False) else 1
    return pixels.shape[0]


def _key(vae, pixels):
    patcher = getattr(vae, "patcher", None)
    origin = getattr(patcher, "cached_patcher_init", None)
    if patcher is None or not origin:
        return _skip("VAE of unknown origin")
    if getattr(patcher, "patches", None):
        return _skip("weight patches")
    if getattr(patcher, "weight_wrapper_patches", None):
        return _skip("weight-wrapper patches")
    if _batch(vae, pixels) != 1:
        return _skip("batched encode")
    fsm = getattr(vae, "first_stage_model", None)
    h = hashlib.sha256()
    stage = "loader"
    try:
        _feed(h, ["origin", origin[0], _file_identity(list(origin[1])), list(origin[2:])])
        stage = "settings"
        out_dtype = vae.vae_output_dtype() if hasattr(vae, "vae_output_dtype") else None
        _feed(h, ["vae", f"{type(fsm).__module__}.{type(fsm).__qualname__}",
                  getattr(vae, "vae_dtype", None), out_dtype,
                  str(getattr(vae, "device", "")), str(getattr(vae, "output_device", "")),
                  getattr(vae, "latent_dim", None), bool(getattr(vae, "not_video", False)),
                  bool(getattr(vae, "handles_tiling", False)),
                  getattr(fsm, "tiling", None), getattr(fsm, "tile_size", None),
                  getattr(fsm, "tile_overlap_min", None),
                  getattr(vae, "format_encoded", None)])
        stage = "object patches"
        _feed(h, ["object_patches", getattr(patcher, "object_patches", None) or {}])
        stage = "model_options"
        _feed(h, ["model_options", getattr(patcher, "model_options", None) or {}])
        stage = "pixels"
        _feed(h, ["pixels", pixels_digest(pixels)])
    except _Opaque as why:
        return _skip(f"unreadable {stage}: {why}")
    except OSError as why:
        return _skip(f"weights file: {type(why).__name__}")
    return h.hexdigest()


def _put(key, latent, took):
    global _bytes
    kept = latent.detach().to("cpu", copy=True).contiguous()
    size = kept.numel() * kept.element_size()
    budget = _budget()
    if size > budget:
        return
    with _lock:
        old = _store.pop(key, None)
        if old is not None:
            _bytes -= old[2]
        _store[key] = (kept, took, size)
        _bytes += size
        while _bytes > budget and _store:
            _, (_, _, gone) = _store.popitem(last=False)
            _bytes -= gone


def _get(key):
    with _lock:
        found = _store.get(key)
        if found is not None:
            _store.move_to_end(key)
    return found


def clear():
    global _bytes
    with _lock:
        _store.clear()
        _bytes = 0


def status():
    with _lock:
        return {"enabled": enabled(), "entries": len(_store), "mb": round(_bytes / 2**20, 1),
                "budget_mb": _budget() >> 20, **STATS, "skip_reasons": dict(SKIPS)}


def _cached(original):
    @functools.wraps(original)
    def encode(self, pixel_samples):
        if _INSIDE.get():                      # an outer cache layer already looked
            return original(self, pixel_samples)
        marker = _INSIDE.set(True)
        try:
            return _encode(self, pixel_samples)
        finally:
            _INSIDE.reset(marker)

    def _encode(self, pixel_samples):
        key = None
        if enabled():
            try:
                key = _key(self, pixel_samples)
            except Exception as why:  # noqa: BLE001 - never fail an encode over the cache
                key = _skip(f"key: {type(why).__name__}")
        if key is None:
            STATS["skipped"] += 1
            return original(self, pixel_samples)
        found = _get(key)
        if found is not None:
            latent, took, _ = found
            STATS["hits"] += 1
            STATS["saved_s"] += took
            logging.info("VAE encode: same weights and pixels as an earlier encode, "
                         "result reused (%.1f s saved)", took)
            device = getattr(self, "output_device", None) or latent.device
            return latent.to(device, copy=True)
        started = time.monotonic()
        out = original(self, pixel_samples)
        took = time.monotonic() - started
        STATS["misses"] += 1
        try:
            _put(key, out, took)
        except Exception:  # noqa: BLE001 - the encode stands, uncached
            log.debug("VAE encode cache: result not kept")
        return out

    encode._spark_vae_cache = True
    return encode


def install():
    """Put the cache on top of comfy.sd.VAE.encode. -> True when in place.

    Idempotent while it is on top. When something wrapped it since (dgx-monarch's
    encode split), a new layer goes on top and the old one steps aside.
    """
    global _WRAPPER
    try:
        import comfy.sd

        current = comfy.sd.VAE.encode
        if _WRAPPER is None or current is not _WRAPPER:
            _WRAPPER = _cached(current)
            comfy.sd.VAE.encode = _WRAPPER
    except Exception:  # noqa: BLE001 - an optimization; encodes go on without it
        log.exception("VAE encode cache not installed")
        return False
    return True
