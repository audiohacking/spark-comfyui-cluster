"""Continuity's saved files, reported so ComfyUI's queue and assets see them.

Continuity's save nodes report their files under their own UI keys
(`mmc_video`, `mmc_takes`, `mmc_image`, ...) on purpose: the stock `images`
and `video` keys would put a second, stock player on the canvas next to
Continuity's stage. But ComfyUI decides what a job produced from those keys: an
item counts as media only under a known key, or when it carries a `format`
(MIME type). Continuity's items carry none, so the job history shows no media
(its Download is greyed out) and the Assets view finds nothing.

This adds the missing `format` to each saved-file item under an `mmc_*` key,
from the file's extension (`video/mp4`, `audio/...`, `image/png`), when the
item is a file in the output folder and has no format yet. The key, and with it
the canvas, is unchanged. Applied where every node's UI output is created
(`comfy_api.latest.io.NodeOutput`), so it covers every Continuity saver.
"""
import functools
import logging
import mimetypes

log = logging.getLogger("spark.output_formats")

_MEDIA = ("video/", "audio/", "image/")


def annotate(ui):
    """`ui` with a `format` on each output file under an `mmc_*` key. Returns a new
    dict when anything changed, else `ui` itself."""
    if not isinstance(ui, dict):
        return ui
    changed = {}
    for key, items in ui.items():
        if not (isinstance(key, str) and key.startswith("mmc_") and isinstance(items, list)):
            continue
        out, touched = [], False
        for item in items:
            if (isinstance(item, dict) and item.get("type") == "output"
                    and isinstance(item.get("filename"), str) and not item.get("format")):
                mime = mimetypes.guess_type(item["filename"])[0] or ""
                if mime.startswith(_MEDIA):
                    item = {**item, "format": mime}
                    touched = True
            out.append(item)
        if touched:
            changed[key] = out
    return {**ui, **changed} if changed else ui


def install():
    """Wrap NodeOutput.__init__ once. -> True when in place."""
    try:
        from comfy_api.latest import io

        current = io.NodeOutput.__init__
        if getattr(current, "_spark_output_formats", False):
            return True

        @functools.wraps(current)
        def __init__(self, *args, **kwargs):
            current(self, *args, **kwargs)
            try:
                self.ui = annotate(self.ui)
            except Exception:  # noqa: BLE001 - never fail a node over a report detail
                log.debug("output formats: ui left as reported", exc_info=True)

        __init__._spark_output_formats = True
        io.NodeOutput.__init__ = __init__
    except Exception:  # noqa: BLE001
        log.exception("output formats not installed")
        return False
    return True
