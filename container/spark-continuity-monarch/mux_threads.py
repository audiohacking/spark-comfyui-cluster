"""The finished video, encoded with x264's frame threads.

Continuity writes its mp4 through PyAV (`creator/mux.py`), and PyAV opens every
codec in slice-threading mode unless told otherwise. x264 honours that with its
sliced-threads mode, which exists for low-latency streaming: it cuts every frame
into slices and is both slower and less efficient than x264's normal frame
threads. On a 20-core Spark a 192-frame 768x1024 clip at the same CRF took 3.8-4.4 s
sliced and 0.5 s with frame threads, at a higher PSNR against its source (41.5
against 40.8 dB) and in an 8% smaller file.

The mode has to be set before the encoder opens, on its first frame. `mux.write`
builds its `_Target` after creating the video stream and before encoding
anything, so the stream is set there, without touching Continuity's files. A
Continuity without that seam is left as it is.
"""
import functools
import logging
import sys

log = logging.getLogger("spark.mux_threads")

_SUFFIX = ".creator.mux"


def frame_threads(stream):
    """Select frame threads on an H.264 stream whose encoder is not open yet."""
    context = stream.codec_context
    if context.name in ("h264", "libx264") and not context.is_open:
        context.thread_type = "FRAME"


def wrap(module):
    """Wrap `module._Target` once. -> True when in place."""
    current = getattr(module, "_Target", None)
    if current is None:
        return False
    if getattr(current, "_spark_frame_threads", False):
        return True

    @functools.wraps(current, updated=())
    def target(output, video, *args, **kwargs):
        try:
            frame_threads(video)
        except Exception:  # noqa: BLE001 - never fail a save over an encoder setting
            log.warning("mux: encoder threading left as it was", exc_info=True)
        return current(output, video, *args, **kwargs)

    target._spark_frame_threads = True
    module._Target = target
    return True


def install():
    """Wrap every loaded Continuity mux module. -> True when one is in place."""
    done = False
    for name, module in list(sys.modules.items()):
        if module is not None and name.endswith(_SUFFIX):
            done = wrap(module) or done
    return done
