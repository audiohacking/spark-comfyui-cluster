"""Continuity's animated taeh3 preview, for a render sampled on the cluster.

On one card Continuity wraps the MODEL in KJNodes' ModelPreviewOverrideKJ,
whose OUTER_SAMPLE wrapper decodes each step's x0 with the picked tiny VAE
(taeh3 for H3) and broadcasts an animated frame as `kj_preview_override`;
Continuity's stage listens for it by node-id prefix. A cluster render has no
MODEL on the driver and dgx-monarch's driver never calls ComfyUI's sampler, so
none of that fires.

This runs that same wrapper on the driver, around dgx-monarch's sampler call:
the result-leader worker streams each step's x0 back on its progress channel
(container/monarch-preview-x0.patch), dgx_monarch.progress hands it to the
sink installed here, and the sink calls the wrapper's own per-step callback.
The decode, the encoding and the event are KJNodes', configured by
Continuity's own `graph_preview()`, so the stage cannot tell the difference.

Previews only: anything that goes wrong here is logged and the render runs on.

It also warms the page cache for the files the render's local passes will
open (the face pass's transformer and detector) while the cluster samples, so
those passes do not start with a 20 GB read.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import sys
import threading
import time
import types

log = logging.getLogger("spark.continuity_monarch")

NODE_ID = "SparkMonarchPreview"
_KJ_MODULE = ".nodes.preview_override_node"


def _kj():
    """KJNodes' preview module, or None when the pack is not loaded."""
    for name, module in list(sys.modules.items()):
        if name.endswith(_KJ_MODULE) and hasattr(module, "_PreviewOverrideWrapper"):
            return module
    return None


def capture_preview_kwargs(core, weights, fps):
    """The kwargs Continuity's `graph_preview` would give KJNodes' node, or
    None when it would add none. Asked of Continuity itself, against a graph
    stand-in that only records, so they can never drift from one card's."""
    recorded = {}

    class _Recorder:
        def node(self, class_type, **kwargs):
            recorded.update(kwargs)
            recorded.pop("model", None)

            class _Out:
                def out(self, index):
                    return ("spark-monarch-preview", index)
            return _Out()

    try:
        core.graph_preview(_Recorder(), ("spark-monarch-model", 0), weights, fps)
    except Exception:
        log.exception("Continuity: cannot read its preview settings; no cluster preview")
        return None
    return recorded or None


def _full_path(name):
    import folder_paths

    for folder in ("diffusion_models", "unet", "checkpoints", "sam3", "sams",
                   "vae_approx", "vae", "text_encoders", "clip"):
        try:
            path = folder_paths.get_full_path(folder, name)
        except Exception:
            path = None
        if path:
            return path
    return None


def _prefetch(names):
    """Ask the kernel to read these files ahead, in the background. Page cache
    only: it holds nothing the kernel cannot drop again under pressure."""
    def run():
        for name in names:
            path = _full_path(name)
            if not path:
                continue
            try:
                fd = os.open(path, os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_WILLNEED)
                finally:
                    os.close(fd)
                log.info("Continuity: prefetching %s for the local pass", name)
            except Exception as exc:
                log.debug("prefetch of %s failed: %r", name, exc)
    threading.Thread(target=run, name="spark-prefetch", daemon=True).start()


class SparkMonarchPreview:
    """Passes a DGXM_MODEL through, carrying the preview settings and the
    files to prefetch to the sampler wrapper below. Built only by the
    Continuity overlay inside its expanded graph."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("DGXM_MODEL",),
                "spec": ("STRING", {"default": "{}"}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("DGXM_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "attach"
    CATEGORY = "DGX Monarch/spark"

    def attach(self, model, spec, unique_id=None):
        settings = json.loads(spec or "{}")
        settings["node_id"] = str(unique_id) if unique_id is not None else None
        out = copy.copy(model)
        # ModelSpec is a frozen dataclass without slots; the attribute rides on
        # this copy only, and the sampler wrapper reads it off its own input.
        object.__setattr__(out, "_spark_preview", settings)
        return (out,)


def _latent_shapes(latent_image):
    samples = (latent_image or {}).get("samples")
    if samples is None:
        return None
    if type(samples).__name__ == "NestedTensor" or getattr(samples, "is_nested", False):
        return [list(t.shape) for t in samples.unbind()]
    return [list(samples.shape)]


def _run_with_preview(settings, latent_image, run):
    """Run dgx-monarch's sampler call inside KJNodes' preview wrapper."""
    kj = _kj()
    shapes = _latent_shapes(latent_image)
    if kj is None or not settings.get("node_id") or not shapes:
        return run()
    import comfy.latent_formats
    import comfy.model_management
    import torch
    from dgx_monarch import progress

    device = comfy.model_management.get_torch_device()
    # What the wrapper reads off a model: the latent format (H3's packed AV
    # format, as on one card) and where to decode. Nothing else is touched for
    # a non-LTX latent without a detail-boost sampler.
    patcher = types.SimpleNamespace(
        model=types.SimpleNamespace(latent_format=comfy.latent_formats.MiniMaxH3AV()),
        load_device=device)
    guider = types.SimpleNamespace(model_patcher=patcher, conds={})

    class _Executor:
        class_obj = guider

        def __call__(self, noise, latent_image_, sampler, sigmas, denoise_mask,
                     callback, disable_pbar, seed, latent_shapes=None):
            def sink(step, total, x0):
                # The pump thread is outside ComfyUI's inference_mode, while
                # the tiny VAE was loaded inside it (on the node's thread):
                # decoding its inference tensors with autograd on fails and
                # KJNodes falls back to latent2rgb. Same mode as the node.
                with torch.inference_mode():
                    callback(int(step) - 1, x0.to(device, torch.float32), None, int(total))
            progress.PREVIEW_SINK = sink
            try:
                return run()
            finally:
                progress.PREVIEW_SINK = None

    wrapper = kj._PreviewOverrideWrapper(
        settings.get("max_resolution", 1024), settings["node_id"],
        settings.get("jpeg_quality", 80), settings.get("suppress_default_preview", True),
        settings.get("preview_frames", 1), settings.get("preview_fps", 24),
        None, settings.get("tiny_vae", "none"))
    # The opening frame is drawn from `noise`; a sample of the latent's own
    # shape, as on one card, rather than nothing.
    noise = torch.randn([1, *shapes[0][1:]], device=device)
    return wrapper(_Executor(), noise, latent_image, None, None, None, None,
                   True, 0, shapes)


def _mem_available_gib():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1048576
    except OSError:
        pass
    return 0.0


def _is_capacity_refusal(exc):
    """dgx-monarch's pre-dispatch capacity refusals (class C): raised before
    anything reaches the workers, so a retry after freeing memory is safe."""
    return (type(exc).__name__ in ("StockLoadCapacityError", "DriverFootprintCapacityError")
            or str(exc).startswith("[dgxm:C "))


def _free_driver_models():
    """Unload this ComfyUI's own models (text encoder, VAEs, detector) and
    return the GiB that gave back to the host. They are idle while the cluster
    samples and reload from the page cache when a later node needs them; on
    unified memory this is memory rank 0 on this Spark can use."""
    import gc

    import comfy.model_management as mm

    before = _mem_available_gib()
    # unload_all_models walks comfy's loaded-model list by index, and another
    # thread can change that list mid-walk during a prompt (an IndexError);
    # a few tries, and never an exception out of here.
    for _ in range(3):
        try:
            mm.unload_all_models()
            break
        except Exception as exc:
            log.debug("unload_all_models raised %r; trying again", exc)
            time.sleep(0.5)
    try:
        gc.collect()
        mm.soft_empty_cache(True)
    except Exception as exc:
        log.debug("cache release raised %r", exc)
    return _mem_available_gib() - before


def wrap_sampler():
    """Wrap DGXMonarchKSampler.sample for a model carrying SparkMonarchPreview
    settings: the page-cache prefetch, the live preview, and one retry after
    freeing this ComfyUI's idle models when dgx-monarch refuses the render for
    capacity. Renders without those settings run exactly as before."""
    import nodes

    cls = nodes.NODE_CLASS_MAPPINGS.get("DGXMonarchKSampler")
    if cls is None or getattr(cls, "_spark_preview_wrapped", False):
        return cls is not None
    orig = getattr(cls, cls.FUNCTION)

    def sample(self, model, *args, **kwargs):
        settings = getattr(model, "_spark_preview", None)
        if settings is None:
            return orig(self, model, *args, **kwargs)
        if settings.get("prefetch"):
            _prefetch(settings["prefetch"])

        def attempt():
            if not settings.get("preview"):
                return orig(self, model, *args, **kwargs)
            started = []

            def run():
                started.append(True)
                return orig(self, model, *args, **kwargs)

            try:
                return _run_with_preview(settings["preview"] | {"node_id": settings.get("node_id")},
                                         kwargs.get("latent_image"), run)
            except Exception:
                from dgx_monarch import progress

                progress.PREVIEW_SINK = None
                # A failure of the render is the render's to report. One of the
                # preview set-up, before the render started, costs only the
                # preview.
                if started:
                    raise
                log.exception("Continuity: the cluster preview could not start; rendering without it")
                return orig(self, model, *args, **kwargs)

        try:
            return attempt()
        except Exception as exc:
            if not _is_capacity_refusal(exc):
                raise
            try:
                freed = _free_driver_models()
            except Exception:
                log.exception("Continuity: freeing this ComfyUI's models failed")
                freed = 0.0
            if freed < 1.0:
                raise exc
            log.warning("Continuity: dgx-monarch refused the render for capacity; freed %.1f GiB "
                        "of this ComfyUI's idle models (they reload when needed) and retrying once",
                        freed)
            return attempt()

    sample.__wrapped__ = orig
    setattr(cls, cls.FUNCTION, sample)
    cls._spark_preview_wrapped = True
    return True


def quiet_notices():
    """Toast only dgx-monarch notices that need attention (warn, error).

    Its informational phase notices (first render, deferred NCCL and load,
    the identity gate starting and passing, queue-time advice) still reach
    the DGX Monarch panel, `dgxm top` and the log; only the canvas toast is
    skipped, the same way dgx-monarch already skips it for the gate's proof
    renders. Faults use another event and are untouched.
    SPARK_MONARCH_NOTICES=all keeps every toast.
    """
    if os.environ.get("SPARK_MONARCH_NOTICES", "").strip().lower() == "all":
        return False
    try:
        from dgx_monarch import first_render
    except Exception:
        return False
    if getattr(first_render.notice, "_spark_quiet", False):
        return True
    orig = first_render.notice

    def notice(phase, message, note, *, toast=True, severity="info", **kwargs):
        if str(severity).lower() not in ("warn", "warning", "error"):
            toast = False
        return orig(phase, message, note, toast=toast, severity=severity, **kwargs)

    notice._spark_quiet = True
    notice.__wrapped__ = orig
    first_render.notice = notice
    return True
