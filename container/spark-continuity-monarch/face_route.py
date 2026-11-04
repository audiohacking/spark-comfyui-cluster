"""Continuity's face pass, sampled on the cluster.

The face pass (`MiniMaxH3FacePass`, Continuity's facepass.py) detects the face,
crops it, VAE-encodes the crop and samples it with one ComfyUI call per window:

    comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler,
                        positive, negative, samples, denoise=..., noise_mask=mask,
                        seed=..., callback=..., disable_pbar=...)

where `mask` holds the per-latent-frame repair strength for the picture and
zeros for the sound. Everything else (detection, crops, paste masks, feather,
composite) stays exactly as Continuity does it on this Spark. Only that one
call goes to dgx-monarch's sampler, with the same latent, mask, seed, steps,
sampler, scheduler and denoise. dgx-monarch's workers run the same stock
`comfy.sample.sample` with the mask (actor/sampling.py), and the image's
monarch-h3-mask.patch gives its H3 forward stock's masked timestep plan.

How the call is singled out, without touching Continuity's files:

- The overlay puts a SparkMonarchFaceRoute node between the face node and its
  model. It hands the face node a clone of that model that carries the
  cluster's model handle under ROUTE_KEY. Nothing else ever receives that clone.
- The face node's `execute` is wrapped (ComfyUI copies the class per run, so the
  copy carries the wrap). Only when its model carries a route, and only for the
  duration of that node, does `comfy.sample.sample` lead here. Each call puts
  the original back before dgx-monarch runs, so no other code (dgx-monarch's
  own surface checks included) ever sees the substitute.

The call goes to the cluster only when it is exactly the call the cluster
reproduces: the arguments above and no others, batch 1, a seed, and the noise
the workers regenerate from that seed (checked against the noise given). Any
other call, an image whose dgx-monarch still refuses masks, or a cluster that
fails, samples on this Spark with the local model, as Continuity always does.
A cancel is a cancel and is not retried.

Opt-out: SPARK_FACE_ON_CLUSTER=0.
"""
from __future__ import annotations

import contextlib
import functools
import logging
import os
import threading

log = logging.getLogger("spark.face_route")

NODE_ID = "SparkMonarchFaceRoute"
ROUTE_KEY = "spark_monarch_face_route"
KSAMPLER_NODE = "DGXMonarchKSampler"

_ALLOWED = frozenset({"denoise", "noise_mask", "seed", "callback", "disable_pbar"})
_lock = threading.Lock()


def enabled():
    return os.environ.get("SPARK_FACE_ON_CLUSTER", "1").strip().lower() not in (
        "0", "false", "no", "off")


def mask_capable():
    """Whether this image's dgx-monarch plans masked H3 rows (the mask patch)."""
    try:
        from dgx_monarch.adapters import minimax_h3_packing
    except Exception:
        return False
    return callable(getattr(minimax_h3_packing, "h3_mask_times", None))


class SparkMonarchFaceRoute:
    """The face pass's local model, carrying the cluster's model handle for the
    face node's own sampling call. Built only by the Continuity overlay inside
    its expanded graph."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "cluster": ("DGXM_MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "route"
    CATEGORY = "DGX Monarch/spark"

    def route(self, model, cluster):
        routed = model.clone()
        routed.model_options[ROUTE_KEY] = cluster
        return (routed,)


@contextlib.contextmanager
def routed_face(graph, face_node, cluster):
    """While the face pass is emitted, give the face node a routed model."""
    node = getattr(graph, "node", None)
    had = "node" in vars(graph)
    before = vars(graph).get("node")

    def emit(class_type, *args, **inputs):
        if class_type == face_node and inputs.get("model") is not None:
            inputs["model"] = node(NODE_ID, model=inputs["model"], cluster=cluster).out(0)
        return node(class_type, *args, **inputs)

    graph.node = emit
    try:
        yield
    finally:
        if had:
            graph.node = before
        else:
            del graph.node


def _same_noise(noise, latent_image, seed):
    """Whether `noise` is what every worker regenerates from `seed`."""
    import comfy.sample
    import torch

    expected = comfy.sample.prepare_noise(latent_image, seed)

    def parts(t):
        return list(t.unbind()) if type(t).__name__ == "NestedTensor" else [t]

    a, b = parts(noise), parts(expected)
    return len(a) == len(b) and all(
        x.shape == y.shape and torch.equal(x.cpu(), y.cpu()) for x, y in zip(a, b))


def _unfit(args, kwargs, noise, latent_image):
    """Why this call is not the one the cluster reproduces, or None."""
    if args:
        return "it carries positional arguments past the latent"
    extra = sorted(set(kwargs) - _ALLOWED)
    if extra:
        return f"it carries {', '.join(extra)}"
    if kwargs.get("seed") is None:
        return "it has no seed"
    first = latent_image.unbind()[0] if type(latent_image).__name__ == "NestedTensor" else latent_image
    if int(first.shape[0]) != 1:
        return f"its batch is {int(first.shape[0])}"
    if not _same_noise(noise, latent_image, int(kwargs["seed"])):
        return "its noise is not the seed's own"
    return None


def _cancelled(exc):
    import comfy.model_management as mm

    if isinstance(exc, mm.InterruptProcessingException) or mm.processing_interrupted():
        return True
    return "Cancel" in type(exc).__name__ or "Interrupt" in type(exc).__name__


def _on_cluster(cluster, steps, cfg, sampler_name, scheduler, positive, negative,
                latent_image, kwargs):
    import nodes

    node_cls = nodes.NODE_CLASS_MAPPINGS[KSAMPLER_NODE]
    latent = {"samples": latent_image}
    if kwargs.get("noise_mask") is not None:
        latent["noise_mask"] = kwargs["noise_mask"]
    out = getattr(node_cls(), node_cls.FUNCTION)(
        cluster, int(kwargs["seed"]), int(steps), float(cfg), sampler_name, scheduler,
        positive, negative, latent_image=latent,
        denoise=float(kwargs.get("denoise", 1.0)))
    if isinstance(out, dict):
        out = out["result"]
    return out[0]["samples"]


def _diverting(original, cluster):
    @functools.wraps(original)
    def sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative,
               latent_image, *args, **kwargs):
        import comfy.sample

        # The original back in place for everything below this call.
        comfy.sample.sample = original
        try:
            why = _unfit(args, kwargs, noise, latent_image)
            if why is None:
                try:
                    samples = _on_cluster(cluster, steps, cfg, sampler_name, scheduler,
                                          positive, negative, latent_image, kwargs)
                    log.info("Continuity: face pass window sampled on the cluster")
                    return samples
                except Exception as exc:  # noqa: BLE001 - any failure: sample here
                    if _cancelled(exc):
                        raise
                    log.warning("Continuity: the cluster could not sample the face pass "
                                "(%s: %s); sampling it on this Spark",
                                type(exc).__name__, str(exc).splitlines()[0][:300] if str(exc) else "")
            else:
                log.warning("Continuity: the face pass samples on this Spark: %s", why)
            return original(model, noise, steps, cfg, sampler_name, scheduler,
                            positive, negative, latent_image, *args, **kwargs)
        finally:
            comfy.sample.sample = sample

    return sample


def wrap_face_node(cls):
    """Wrap the face node's execute so a routed model diverts its sampling call."""
    if getattr(cls, "_spark_face_route", False):
        return True
    raw = cls.__dict__.get("execute")
    func = getattr(raw, "__func__", None)
    if not isinstance(raw, classmethod) or func is None:
        log.warning("Continuity: the face node's execute moved; the face pass "
                    "samples on this Spark")
        return False

    @functools.wraps(func)
    def execute(klass, model, *args, **kwargs):
        cluster = (getattr(model, "model_options", None) or {}).get(ROUTE_KEY)
        if cluster is None:
            return func(klass, model, *args, **kwargs)
        import comfy.sample

        with _lock:
            original = comfy.sample.sample
            comfy.sample.sample = _diverting(original, cluster)
            try:
                return func(klass, model, *args, **kwargs)
            finally:
                comfy.sample.sample = original

    cls.execute = classmethod(execute)
    cls._spark_face_route = True
    return True
