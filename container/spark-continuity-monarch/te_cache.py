"""Text-encoder results kept across renders: an exact cache on CLIP encodes.

A text encoder's output is a pure function of its weights, its patches, how it
was loaded, and the tokens it is given (images, video frames and audio included,
for the vision-language encoders). A re-roll that changes only the seed hands it
the same tokens again, and the 32B Qwen3-VL behind MiniMax H3 spends ~50 s per
pass recomputing what it already returned. ComfyUI cannot keep it: its loaders
re-run on every prompt and hand out a new CLIP object, so nothing keyed on the
object survives.

This keys on content instead, for any model and any workflow:

- **Where the weights came from**: the loader ComfyUI records on the patcher
  (`cached_patcher_init`) and its arguments; files by path, size and mtime.
- **What is patched onto them**: the LoRA patches, by a digest of their
  tensors and strengths (once per patch set), the clip layer, and the
  tokenizer options.
- **What is encoded**: a digest of the full token structure, every tensor's
  bytes included, and the call's arguments.
- **At what precision**: the SPARK_TE_COMPUTE setting (mod gb10-te-bf16).

Whatever cannot be fingerprinted exactly is not cached and encodes as usual: a
CLIP from a loader that recorded nothing, hooks or scheduled conditioning,
weight-wrapper patches, object patches it cannot read (code, modules), and anything the digest does not know how to read. Results are
handed out as deep copies, so a caller that mutates one cannot touch the cache.
Kept in RAM, least recently used first out, within SPARK_TE_CACHE_MB
(default 4096). Opt-out: SPARK_TE_CACHE=0.
"""
import copy
import functools
import hashlib
import logging
import os
import threading
import time
from collections import OrderedDict

log = logging.getLogger("spark.te_cache")

_lock = threading.Lock()
_store = OrderedDict()      # key -> (conditioning, bytes)
_bytes = 0
_patch_digests = {}         # patches_uuid -> digest of that patch set
STATS = {"hits": 0, "misses": 0, "skipped": 0, "saved_s": 0.0}
SKIPS = {}                  # reason -> count: why encodes were not cached


def enabled():
    return os.environ.get("SPARK_TE_CACHE", "1").strip().lower() not in (
        "0", "false", "no", "off")


def _budget():
    try:
        return max(0, int(os.environ.get("SPARK_TE_CACHE_MB", "4096"))) << 20
    except ValueError:
        return 4096 << 20


class _Opaque(Exception):
    """A value this digest cannot read exactly."""


def _feed(h, value):
    import torch

    if isinstance(value, torch.Tensor):
        t = value.detach()
        if t.is_sparse or t.is_quantized:
            raise _Opaque("tensor layout")
        t = t.to("cpu").contiguous()
        h.update(f"T{tuple(t.shape)}{t.dtype};".encode())
        if t.numel():
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, (torch.dtype, torch.device)):
        h.update(f"{type(value).__name__}:{value};".encode())
    elif isinstance(value, bool) or value is None:
        h.update(f"{value!r};".encode())
    elif isinstance(value, (str, int, float, bytes)):
        h.update(f"{type(value).__name__}:{value!r};".encode())
    elif isinstance(value, (list, tuple)):
        h.update(f"{type(value).__name__}[{len(value)}:".encode())
        for item in value:
            _feed(h, item)
        h.update(b"]")
    elif isinstance(value, dict):
        h.update(f"d{{{len(value)}:".encode())
        for k in sorted(value, key=repr):
            _feed(h, k)
            _feed(h, value[k])
        h.update(b"}")
    elif hasattr(value, "value") and type(value).__module__.startswith(("comfy", "enum")):
        h.update(f"E{type(value).__name__}.{value!r};".encode())   # CLIPType and friends
    elif isinstance(value, type) or callable(value):
        name = f"{getattr(value, '__module__', '?')}.{getattr(value, '__qualname__', '?')}"
        if "<locals>" in name or "<lambda>" in name:
            raise _Opaque(name)                  # a closure: its state is not in its name
        h.update(f"F{name};".encode())
    else:
        raise _Opaque(type(value).__name__)


def _file_identity(value):
    """A path to a file stands for its contents by path, size and mtime."""
    if isinstance(value, str) and os.path.isfile(value):
        st = os.stat(value)
        return f"file:{os.path.realpath(value)}:{st.st_size}:{st.st_mtime_ns}"
    if isinstance(value, (list, tuple)):
        return type(value)(_file_identity(v) for v in value)
    return value


def _compute_mode():
    """The text encoders' compute precision setting (mod gb10-te-bf16), so a
    result computed at one precision is never handed out at the other."""
    import sys

    patched = hasattr(sys.modules.get("comfy.sd1_clip"), "_spark_te_act_dtype")
    return f"{patched}:{os.environ.get('SPARK_TE_COMPUTE', 'bf16').strip().lower()}"


def _patches_digest(patcher):
    patches = getattr(patcher, "patches", None) or {}
    if not patches:
        return "none"
    uid = getattr(patcher, "patches_uuid", None)
    with _lock:
        found = _patch_digests.get(uid)
    if found is not None:
        return found
    h = hashlib.sha256()
    _feed(h, {k: [p[:3] for p in v] for k, v in patches.items()})  # strength, patch, strength
    digest = h.hexdigest()
    with _lock:
        _patch_digests[uid] = digest
        while len(_patch_digests) > 64:
            _patch_digests.pop(next(iter(_patch_digests)))
    return digest


def _skip(reason):
    with _lock:
        SKIPS[reason] = SKIPS.get(reason, 0) + 1
    return None


def _key(clip, tokens, unprojected, add_dict):
    """-> the key of this encode, or None when it cannot be fingerprinted exactly."""
    patcher = getattr(clip, "patcher", None)
    origin = getattr(patcher, "cached_patcher_init", None)
    if patcher is None or not origin:
        return _skip("encoder of unknown origin")
    if getattr(patcher, "forced_hooks", None) is not None or getattr(clip, "apply_hooks_to_conds", None):
        return _skip("hooks")
    if getattr(patcher, "weight_wrapper_patches", None):
        return _skip("weight-wrapper patches")
    h = hashlib.sha256()
    stage = "loader"
    try:
        loader, args = origin[0], origin[1]
        stage = "loader"
        _feed(h, ["origin", loader, _file_identity(list(args)), list(origin[2:])])
        stage = "patches"
        _feed(h, ["patches", _patches_digest(patcher)])
        stage = "object patches"          # e.g. ComfyUI's own manual_cast_dtype
        _feed(h, ["object_patches", getattr(patcher, "object_patches", None) or {}])
        stage = "model_options"
        _feed(h, ["model_options", getattr(patcher, "model_options", None) or {}])
        stage = "encoder settings"
        _feed(h, ["clip", type(getattr(clip, "cond_stage_model", None)).__qualname__,
                  getattr(clip, "layer_idx", None), getattr(clip, "tokenizer_options", None) or {},
                  bool(getattr(clip, "use_clip_schedule", False))])
        stage = "call arguments"
        _feed(h, ["call", unprojected, add_dict or {}])
        _feed(h, ["compute", _compute_mode()])
        stage = "tokens"
        _feed(h, ["tokens", tokens])
    except _Opaque as why:
        return _skip(f"unreadable {stage}: {why}")
    except OSError as why:
        return _skip(f"weights file: {type(why).__name__}")
    return h.hexdigest()


def _size(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_size(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_size(v) for v in value)
    return 64


def _put(key, value):
    global _bytes
    size = _size(value)
    budget = _budget()
    if size > budget:
        return
    with _lock:
        old = _store.pop(key, None)
        if old is not None:
            _bytes -= old[1]
        _store[key] = (value, size)
        _bytes += size
        while _bytes > budget and _store:
            _, (_, gone) = _store.popitem(last=False)
            _bytes -= gone


def _get(key):
    with _lock:
        found = _store.get(key)
        if found is not None:
            _store.move_to_end(key)
    return None if found is None else found[0]


def clear():
    global _bytes
    with _lock:
        _store.clear()
        _patch_digests.clear()
        _bytes = 0


def status():
    with _lock:
        return {"enabled": enabled(), "entries": len(_store), "mb": round(_bytes / 2**20, 1),
                "budget_mb": _budget() >> 20, **STATS, "skip_reasons": dict(SKIPS)}


def _cached(original):
    @functools.wraps(original)
    def encode_from_tokens_scheduled(self, tokens, unprojected=False, add_dict={}, show_pbar=True):
        key = _key(self, tokens, unprojected, add_dict) if enabled() else None
        if key is None:
            STATS["skipped"] += 1
            return original(self, tokens, unprojected=unprojected, add_dict=add_dict,
                            show_pbar=show_pbar)
        found = _get(key)
        if found is not None:
            STATS["hits"] += 1
            STATS["saved_s"] += found[1]
            logging.info("Text encoder: same weights and inputs as an earlier encode, "
                         "result reused (%.1f s saved)", found[1])
            return copy.deepcopy(found[0])
        started = time.monotonic()
        out = original(self, tokens, unprojected=unprojected, add_dict=add_dict,
                       show_pbar=show_pbar)
        took = time.monotonic() - started
        STATS["misses"] += 1
        try:
            _put(key, (copy.deepcopy(out), took))
        except Exception:  # noqa: BLE001 - uncopyable: the encode stands, uncached
            log.debug("text encoder cache: result not copyable, not kept")
        return out

    encode_from_tokens_scheduled._spark_te_cache = True
    return encode_from_tokens_scheduled


def install():
    """Wrap comfy.sd.CLIP.encode_from_tokens_scheduled once. -> True when in place."""
    try:
        import comfy.sd

        current = comfy.sd.CLIP.encode_from_tokens_scheduled
        if not getattr(current, "_spark_te_cache", False):
            comfy.sd.CLIP.encode_from_tokens_scheduled = _cached(current)
    except Exception:  # noqa: BLE001 - an optimization; encodes go on without it
        log.exception("text encoder cache not installed")
        return False
    return True
