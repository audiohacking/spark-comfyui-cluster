"""Encode once per render: Continuity's face pass re-encodes what the pass did.

The face pass is a second segment node at the crop canvas. Its references are
`max`-sized (independent of the canvas) and its prompt is the pass's, so its
encodes repeat the pass's. Two things kept that from being free:

- **Reference latents.** The pass's segment node is told its VAE checkpoints'
  names; the face pass's is not (`H3.emit_face`), so Continuity's reference
  cache keys the face pass's latents on a different VAE fingerprint and a new
  reference (or a new length) is VAE-encoded twice. The names are added here.
- **The text encoder** (~37 s on a GB10 with a reference video) runs again on
  tokens identical to the pass's. Its output is a pure function of the tokens
  and the encoder's weights and patches, so it is memoized for the render:
  keyed on a digest of the tokens, handed out as copies, only from inside
  Continuity's H3 encode. Anything that cannot be digested passes through.

Opt-out: SPARK_ENCODE_REUSE=0.
"""
import copy
import functools
import hashlib
import importlib
import logging
import os
import threading
import time

log = logging.getLogger("spark.encode_reuse")

_KEEP = 2                 # the pass's conditioning and one other
_memo = {}                # key -> conditioning (insertion order = age)
_lock = threading.Lock()


def enabled():
    return os.environ.get("SPARK_ENCODE_REUSE", "1").strip().lower() not in (
        "0", "false", "no", "off")


class _Opaque(Exception):
    """A token holds something this digest does not know how to read."""


def _feed(h, value):
    import torch

    if isinstance(value, torch.Tensor):
        t = value.detach().to("cpu").contiguous()
        h.update(f"T{tuple(t.shape)}{t.dtype}".encode())
        h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    elif isinstance(value, (str, int, float, bool)) or value is None:
        h.update(f"{type(value).__name__}:{value!r};".encode())
    elif isinstance(value, (list, tuple)):
        h.update(f"{type(value).__name__}[{len(value)}".encode())
        for item in value:
            _feed(h, item)
        h.update(b"]")
    elif isinstance(value, dict):
        h.update(f"d{{{len(value)}".encode())
        for k in sorted(value, key=repr):
            _feed(h, k)
            _feed(h, value[k])
        h.update(b"}")
    else:
        raise _Opaque(type(value).__name__)


def _key(clip, tokens, kwargs):
    """-> a key for this encode, or None when it should not be memoized."""
    if kwargs:
        return None
    patcher = getattr(clip, "patcher", None)
    h = hashlib.sha256()
    try:
        _feed(h, tokens)
    except _Opaque:
        return None
    return (id(getattr(clip, "cond_stage_model", clip)),
            str(getattr(patcher, "patches_uuid", "")),
            h.hexdigest())


def _remember(key, cond):
    with _lock:
        _memo.pop(key, None)
        _memo[key] = cond
        while len(_memo) > _KEEP:
            _memo.pop(next(iter(_memo)))


def _memoized(clip, original):
    @functools.wraps(original)
    def encode_from_tokens_scheduled(tokens, *args, **kwargs):
        if args:
            return original(tokens, *args, **kwargs)
        key = _key(clip, tokens, kwargs)
        if key is None:
            return original(tokens, **kwargs)
        with _lock:
            found = _memo.get(key)
        if found is not None:
            try:
                out = copy.deepcopy(found)
            except Exception:  # noqa: BLE001 - uncopyable: encode as usual
                return original(tokens, **kwargs)
            log.info("Continuity: text encoder output reused (same prompt and references)")
            return out
        started = time.monotonic()
        cond = original(tokens, **kwargs)
        try:
            _remember(key, copy.deepcopy(cond))
        except Exception:  # noqa: BLE001
            pass
        log.debug("text encoder: %.1f s, kept for reuse", time.monotonic() - started)
        return cond

    return encode_from_tokens_scheduled


def _wrap_encode(encode_mod):
    orig = encode_mod.encode
    if getattr(orig, "_spark_reuse", False):
        return

    @functools.wraps(orig)
    def encode(clip, *args, **kwargs):
        if not enabled() or clip is None or not callable(
                getattr(clip, "encode_from_tokens_scheduled", None)):
            return orig(clip, *args, **kwargs)
        had = "encode_from_tokens_scheduled" in vars(clip)
        before = vars(clip).get("encode_from_tokens_scheduled")
        try:
            clip.encode_from_tokens_scheduled = _memoized(clip, clip.encode_from_tokens_scheduled)
        except Exception:  # noqa: BLE001 - not patchable: encode as usual
            return orig(clip, *args, **kwargs)
        try:
            return orig(clip, *args, **kwargs)
        finally:
            if had:
                clip.encode_from_tokens_scheduled = before
            else:
                del clip.encode_from_tokens_scheduled

    encode._spark_reuse = True
    encode_mod.encode = encode


def _wrap_face(cls, segment_node):
    orig = getattr(cls, "emit_face", None)
    if not callable(orig) or getattr(orig, "_spark_reuse", False):
        return

    @functools.wraps(orig)
    def emit_face(self, graph, links, payload, compiled, face, written, weights,
                  *args, **kwargs):
        node = getattr(graph, "node", None)
        if not enabled() or not callable(node):
            return orig(self, graph, links, payload, compiled, face, written,
                        weights, *args, **kwargs)

        def named(class_type, *a, **inputs):
            if class_type == segment_node:
                if "vae" in inputs and "vae_name" not in inputs:
                    inputs["vae_name"] = getattr(weights, "vae", None) or ""
                if "audio_vae" in inputs and "audio_vae_name" not in inputs:
                    inputs["audio_vae_name"] = getattr(weights, "audio_vae", None) or ""
            return node(class_type, *a, **inputs)

        try:
            graph.node = named
        except Exception:  # noqa: BLE001 - not patchable: emit as usual
            return orig(self, graph, links, payload, compiled, face, written,
                        weights, *args, **kwargs)
        try:
            return orig(self, graph, links, payload, compiled, face, written,
                        weights, *args, **kwargs)
        finally:
            del graph.node

    emit_face._spark_reuse = True
    cls.emit_face = emit_face


def install(render_mod):
    """Patch Continuity's H3 encode and face pass once. -> True when in place."""
    try:
        encode_mod = importlib.import_module(
            render_mod.__name__.rsplit(".", 1)[0] + ".encode")
        if not callable(getattr(encode_mod, "encode", None)):
            raise AttributeError("encode.encode")
        segment_node = render_mod.SEGMENT_NODE
        _wrap_encode(encode_mod)
        _wrap_face(render_mod.H3, segment_node)
    except Exception:  # noqa: BLE001 - an optimization; the render goes on without it
        log.exception("Continuity: encode reuse not installed; the face pass "
                      "encodes on its own")
        return False
    return True
