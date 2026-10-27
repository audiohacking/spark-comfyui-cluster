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


# Frames of each step's latent the cluster preview decodes. Continuity asks
# for the whole clip (1024), which for a 6 s clip at 768p is 7-9 s of taeh3 on
# the driver Spark's GPU every step, while rank 0 samples on that same GPU and
# the other ranks wait for it. 17 frames spread over the clip measured 0.4 s.
DEFAULT_PREVIEW_FRAMES = 17


def _preview_rate(frames, fps, video_shape):
    """(frames, fps) for the cluster preview: at most SPARK_MONARCH_PREVIEW_FRAMES
    (0 = Continuity's own count), with the playback rate lowered to match so
    the loop still lasts as long as the clip."""
    try:
        cap = int(os.environ.get("SPARK_MONARCH_PREVIEW_FRAMES", DEFAULT_PREVIEW_FRAMES))
    except ValueError:
        cap = DEFAULT_PREVIEW_FRAMES
    latent_frames = int(video_shape[2]) if len(video_shape) >= 5 else 1
    if cap <= 0 or frames <= 1 or latent_frames <= cap:
        return frames, fps
    # H3 decodes 4 frames per latent frame after the first.
    decoded = 4 * (latent_frames - 1) + 1
    return cap, max(1.0, fps * cap / decoded)


def _run_with_preview(settings, latent_image, run):
    """Run dgx-monarch's sampler call inside KJNodes' preview wrapper."""
    kj = _kj()
    shapes = _latent_shapes(latent_image)
    if kj is None or not settings.get("node_id") or not shapes:
        return run()
    import comfy.latent_formats
    import comfy.model_management
    import comfy.model_management as mm
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
                # A cancel belongs to dgx-monarch's cancel monitor, which polls
                # ComfyUI's interrupt flag to stop the workers. ComfyUI's ops
                # clear that flag when they raise it, so a preview decode
                # running into it would swallow the cancel: skip the decode
                # while one is pending, and put the flag back if a decode
                # still trips it.
                if mm.processing_interrupted():
                    return
                try:
                    # The pump thread is outside ComfyUI's inference_mode while
                    # the tiny VAE was loaded inside it; same mode as the node.
                    with torch.inference_mode():
                        callback(int(step) - 1, x0.to(device, torch.float32), None, int(total))
                except mm.InterruptProcessingException:
                    mm.interrupt_current_processing(True)
            progress.PREVIEW_SINK = sink
            try:
                return run()
            finally:
                progress.PREVIEW_SINK = None

    frames, fps = _preview_rate(settings.get("preview_frames", 1),
                                float(settings.get("preview_fps", 24)), shapes[0])
    wrapper = kj._PreviewOverrideWrapper(
        settings.get("max_resolution", 1024), settings["node_id"],
        settings.get("jpeg_quality", 80), settings.get("suppress_default_preview", True),
        frames, fps, None, settings.get("tiny_vae", "none"))
    # The opening frame is drawn from `noise`; a sample of the latent's own
    # shape, as on one card, rather than nothing.
    noise = torch.randn([1, *shapes[0][1:]], device=device)
    return wrapper(_Executor(), noise, latent_image, None, None, None, None,
                   True, 0, shapes)


def _unload_after_prompt():
    """Have ComfyUI unload this process's models once the prompt is over.

    On the driver Spark a cluster render's rank 0 shares memory with this
    ComfyUI, which otherwise keeps the text encoder, the local passes' copy of
    the model, the detector and the VAEs resident from render to render, until
    dgx-monarch refuses the next one for capacity. This sets the same flag as
    ComfyUI's own "free memory" (POST /free unload_models), which the prompt
    worker applies between prompts, the only time unloading is safe. They
    reload from the page cache when the next render needs them.
    SPARK_MONARCH_UNLOAD_AFTER=0 keeps them resident.
    """
    if os.environ.get("SPARK_MONARCH_UNLOAD_AFTER", "1").strip() == "0":
        return
    try:
        import server

        server.PromptServer.instance.prompt_queue.set_flag("unload_models", True)
    except Exception as exc:
        log.debug("could not ask for an unload after this prompt: %r", exc)


def _release_other_model(model):
    """Reset the workers when they hold a different diffusion model.

    dgx-monarch's H3 activation preflight reads MemAvailable on the driver
    Spark and credits the DiT weights only when this render's model is the one
    the workers already hold. On a switch (fl2va after ref2va, say) it reads
    with the previous model still resident, about 40 GB per Spark under stock
    residency, and charges the new weights on top, although the worker drops
    the old model before loading the new one: a refusal of a render that fits.
    A recycle, the Clear VRAM node's "recycle" level, stops the worker procs
    and returns the model and the allocator pool to the OS; the render then
    brings the fleet up again (about a minute) and loads the model it needs.
    SPARK_MONARCH_RELEASE_ON_SWITCH=0 keeps the previous model resident.
    """
    if os.environ.get("SPARK_MONARCH_RELEASE_ON_SWITCH", "1").strip() == "0":
        return
    wanted = getattr(model, "unet_name", None)
    handle = getattr(getattr(model, "mesh", None), "handle", None)
    if not isinstance(wanted, str) or handle is None:
        return
    try:
        from dgx_monarch.nodes import recycle_drain
        from dgx_monarch.nodes import render_preflight as rp

        # The same memos the preflight credits residency from.
        held = {rp._LAST_RENDERED_UNET, rp._LAST_SUBMITTED_UNET,
                rp._LAST_DRIVER_CHARGED_UNET} - {None}
    except Exception as exc:
        log.debug("cannot read the fleet's resident model: %r", exc)
        return
    if not held or wanted in held:
        return
    log.info("Continuity: the cluster holds %s; resetting the workers to load %s",
             ", ".join(sorted(held)), wanted)
    try:
        outcome = handle.recycle_detailed()
    except Exception:
        log.exception("Continuity: the worker reset failed; the render goes ahead")
        return
    if outcome.ok:
        # As the /dgxm/recycle route does: a new fleet holds none of the old weights.
        recycle_drain.drop_residency_memos()
    else:
        log.warning("Continuity: the workers were not reset (%s: %s); the render goes ahead",
                    outcome.status.value, outcome.detail)


def _mem_available():
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


def _make_room(model):
    """Free memory on every Spark right before the cluster samples.

    dgx-monarch's H3 activation preflight compares this render's activations
    with MemAvailable on the driver Spark, which this ComfyUI shares with
    rank 0. By then Continuity has run the text encoder and the VAEs here,
    and they stay loaded, idle, through the whole cluster sample: about 20 GB
    that a long render or a large reference video needs. They reload from
    the page cache when the local passes or the decode want them. The
    workers drop their allocator caches too (Clear VRAM "soft"); their
    weights stay resident. SPARK_MONARCH_FREE_BEFORE=0 keeps all of it.
    """
    if os.environ.get("SPARK_MONARCH_FREE_BEFORE", "1").strip() == "0":
        return
    import gc

    import comfy.model_management as mm

    before = _mem_available()
    try:
        mm.unload_all_models()
        gc.collect()
        mm.soft_empty_cache()
    except Exception as exc:
        log.debug("could not unload the driver's models: %r", exc)
    handle = getattr(getattr(model, "mesh", None), "handle", None)
    if handle is not None:
        try:
            handle.call_all("clear_vram", "soft", timeout_s=120)
        except Exception as exc:
            log.debug("could not clear the workers' allocator caches: %r", exc)
    after = _mem_available()
    if before is not None and after is not None:
        log.info("Continuity: freed %.1f GiB on this Spark for the cluster render (%.1f GiB available)",
                 (after - before) / 2 ** 30, after / 2 ** 30)


def sampler_wrapped():
    """Whether dgx-monarch's sampler carries the preview wrapper."""
    import nodes

    cls = nodes.NODE_CLASS_MAPPINGS.get("DGXMonarchKSampler")
    return bool(getattr(cls, "_spark_preview_wrapped", False))


def wrap_sampler():
    """Wrap DGXMonarchKSampler.sample for a model carrying SparkMonarchPreview
    settings: the page-cache prefetch and the live preview. Renders without
    those settings run exactly as before."""
    import nodes

    cls = nodes.NODE_CLASS_MAPPINGS.get("DGXMonarchKSampler")
    if cls is None or getattr(cls, "_spark_preview_wrapped", False):
        return cls is not None
    orig = getattr(cls, cls.FUNCTION)

    def sample(self, model, *args, **kwargs):
        settings = getattr(model, "_spark_preview", None)
        if settings is None:
            return orig(self, model, *args, **kwargs)
        _release_other_model(model)
        _make_room(model)
        if settings.get("prefetch"):
            _prefetch(settings["prefetch"])
        _unload_after_prompt()

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
