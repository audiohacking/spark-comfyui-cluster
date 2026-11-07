"""The face pass finds the pass's reference latents in Continuity's cache.

The face pass is a second segment node at the crop canvas. Continuity's
reference cache keys VAE latents on the VAE checkpoints' names, and the pass's
segment node is told them while the face pass's is not (`H3.emit_face`), so it
keyed the face pass's latents on a different VAE fingerprint and a new
reference (or a new length) was VAE-encoded twice. The names are added here.

The text encoder's repeats are handled for every workflow by te_cache.py.

Opt-out: SPARK_ENCODE_REUSE=0.
"""
import functools
import logging
import os

log = logging.getLogger("spark.encode_reuse")


def enabled():
    return os.environ.get("SPARK_ENCODE_REUSE", "1").strip().lower() not in (
        "0", "false", "no", "off")


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
    """Patch Continuity's H3 face pass once. -> True when in place."""
    try:
        _wrap_face(render_mod.H3, render_mod.SEGMENT_NODE)
    except Exception:  # noqa: BLE001 - an optimization; the render goes on without it
        log.exception("Continuity: encode reuse not installed; the face pass "
                      "encodes its references on its own")
        return False
    return True
