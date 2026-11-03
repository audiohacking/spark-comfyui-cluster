"""Checkpoints that carry their own sampling contract, applied for you.

Continuity's sampler row (steps, cfg, sampler, scheduler, the two flow shifts)
and its turbo switch are built for H3 the way the community runs it: the
released checkpoints plus a distillation LoRA picked on the turbo pill. A
checkpoint with the distillation *baked in* is a case it does not have. FastH3
Omni (FastVideo/FastH3-Omni-Comfy) is one: its parallel-decoding (PDD) output
heads were distilled against fixed places on the 32-interval flow grid, so it
renders as it was trained only at 8 steps on `euler` + `simple`, cfg 1.0, the
checkpoints' own shifts (12 video, 3 audio), and with no turbo LoRA or lead-in
on top of it. Any other row quietly samples a different model.

Such a checkpoint says so in its own header: ComfyUI's loader reads
`config.transformer.pdd_steps` from the safetensors metadata to size the head
bank (comfy ce42de5f). That is what this module detects, not a filename, so the
next PDD-baked checkpoint is covered without a code change.

**What it does.** Before Continuity resolves a render, when the checkpoint a
render routes to is PDD-baked, the piece is set the way the user would have set
it by hand:

- the sampler row: steps, cfg, sampler, scheduler and shifts from the contract;
- the accelerators that change the model a step sees (step caches, Spectrum,
  a VDN-H3 stage, SLA sparse attention) off; the exact kernels stay as picked;
- the turbo switch off, and step-distillation LoRAs off every stack (the
  turbo pill's file, and files named as distills: turbo, distill, lightx2v,
  PDD, DMD, N-step). Style and character LoRAs stay;
- on route `auto`, with the baked checkpoint in one routed slot and an
  ordinary one in the other, every segment pinned to the baked one, so one
  render never mixes two sampling contracts. Ref2VA weights take text-only and
  keyframe payloads too (Continuity's own route switch says so).

The face pass, which re-noises the finished pass partway down the schedule,
runs the contract's last k trained steps at denoise k/N, k being the nearest
to the node's face denoise (0.45 of 8 steps: 4 steps at 0.5). ComfyUI's own
partial schedule for 8 steps at 0.45 would fall between the trained intervals.

One log line and one canvas toast name every change, against what the node
said. The node's pills are not rewritten; save the workflow as it is and it
stays that way. SPARK_MODEL_CONTRACT=0 turns this off.

It edits the per-queue parsed blob in `resolve_sampling`, the first thing
Continuity's render reads off it: the route, the turbo block, the stacks and
the sampler row are all read from that same object afterwards.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import re
import struct
from dataclasses import dataclass

log = logging.getLogger("spark.model_contract")

# The H3 checkpoint slots a generation is routed between (Continuity's
# `families/h3/declare.ROUTED`), and the route value that leaves it to the mode.
_ROUTED = ("fl2va", "ref2va")
_AUTO = "auto"

# Files that are a step distillation rather than a style: what a baked
# checkpoint already is. Matched on the file name, case-insensitively.
_DISTILL = re.compile(r"turbo|distill|lightx2v|pdd|dmd|\d+[-_]?steps?(?![a-z])", re.I)

_HEADER_CACHE: dict[tuple, dict] = {}


@dataclass(frozen=True)
class Contract:
    checkpoint: str
    pdd_steps: int
    steps: int
    cfg: float = 1.0
    sampler_name: str = "euler"
    scheduler: str = "simple"
    shift_video: float = 12.0
    shift_audio: float = 3.0

    def row(self):
        return {"steps": self.steps, "cfg": self.cfg,
                "sampler_name": self.sampler_name, "scheduler": self.scheduler,
                "shift_video": self.shift_video, "shift_audio": self.shift_audio}


# The accelerators a baked checkpoint runs without: each changes what the model
# computes at a step, which the PDD heads were not distilled against.
_ACCEL_OFF = {"block_cache": "off", "spectrum": False, "vdn": "off"}


def enabled():
    return os.environ.get("SPARK_MODEL_CONTRACT", "1").strip() != "0"


def _metadata(path):
    """A safetensors file's `__metadata__`, cached on path, size and mtime."""
    try:
        st = os.stat(path)
    except OSError:
        return {}
    key = (path, st.st_size, st.st_mtime_ns)
    if key not in _HEADER_CACHE:
        meta = {}
        try:
            with open(path, "rb") as fh:
                size = struct.unpack("<Q", fh.read(8))[0]
                if 0 < size < 1 << 27:
                    meta = json.loads(fh.read(size)).get("__metadata__") or {}
        except Exception as exc:
            log.debug("cannot read the header of %s: %r", path, exc)
        _HEADER_CACHE[key] = meta
    return _HEADER_CACHE[key]


def contract(name):
    """The sampling contract `name` carries in its header, or None."""
    if not isinstance(name, str) or not name:
        return None
    try:
        import folder_paths

        path = folder_paths.get_full_path("diffusion_models", name)
    except Exception:
        path = None
    if not path:
        return None
    try:
        config = json.loads(_metadata(path).get("config") or "{}")
        pdd = int((config.get("transformer") or {}).get("pdd_steps", 1))
    except (TypeError, ValueError):
        return None
    # Eight fused blocks of four intervals: the count PDD students are run at.
    if pdd <= 1 or pdd % 4:
        return None
    return Contract(checkpoint=name, pdd_steps=pdd, steps=pdd // 4)


def _routed_slot(models):
    """(slot, contract) this render samples a baked checkpoint on, or None.
    Pins route `auto` to the baked slot when the other routed slot is not."""
    route = models.get("route") or _AUTO
    slots = (route,) if route in _ROUTED else tuple(s for s in _ROUTED if models.get(s))
    baked = {slot: contract(models.get(slot)) for slot in slots}
    baked = {slot: c for slot, c in baked.items() if c}
    if not baked:
        return None
    slot = "ref2va" if "ref2va" in baked else next(iter(baked))
    return slot, baked[slot]


def _drop(entries, names):
    kept, dropped = [], []
    for entry in entries or []:
        name = entry.get("name") if isinstance(entry, dict) else None
        if isinstance(name, str) and (name in names or _DISTILL.search(os.path.basename(name))):
            dropped.append(name)
        else:
            kept.append(entry)
    return kept, dropped


def apply(data):
    """Set `data` (Continuity's parsed blob for this queue) to the contract of
    the baked checkpoint it routes to. -> (slot, Contract, changes), or None."""
    models = data.get("models")
    if not isinstance(models, dict):
        return None
    found = _routed_slot(models)
    if found is None:
        return None
    slot, terms = found
    changes = []

    route = models.get("route") or _AUTO
    if route == _AUTO and any(models.get(s) and s != slot for s in _ROUTED):
        models["route"] = slot
        changes.append(f"every segment on {slot} (route was auto)")

    turbo = data.get("turbo") if isinstance(data.get("turbo"), dict) else {}
    distills = {turbo.get("lora")} - {None, ""}
    if turbo.get("on"):
        data["turbo"] = {**turbo, "on": False}
        changes.append("turbo off")

    dropped = []
    if isinstance(data.get("loras"), list):
        data["loras"], gone = _drop(data["loras"], distills)
        dropped += gone
    for segment in data.get("segments") or []:
        if isinstance(segment, dict) and isinstance(segment.get("loras"), list):
            segment["loras"], gone = _drop(segment["loras"], distills)
            dropped += gone
    if dropped:
        changes.append("distill LoRAs off: " + ", ".join(sorted(set(dropped))))

    block = data.get("sampling") if isinstance(data.get("sampling"), dict) else {}
    accel_off = [k for k, v in _ACCEL_OFF.items() if k in block and block[k] != v]
    if block.get("attention") == "sla":
        block = {**block, "attention": "default"}
        accel_off.append("sla attention")
    data["sampling"] = {**block, **_ACCEL_OFF, **terms.row()}
    if accel_off:
        changes.append("accelerators off: " + ", ".join(accel_off))
    return slot, terms, changes


def _announce(terms, before, changes):
    """One log line and one toast: what the render runs, and what moved."""
    moved = [f"{label} {old} -> {new}" for label, old, new in (
        ("steps", before.steps, terms.steps), ("cfg", before.cfg, terms.cfg),
        ("sampler", before.sampler_name, terms.sampler_name),
        ("scheduler", before.scheduler, terms.scheduler),
        ("shifts", f"{before.shift_video:g}/{before.shift_audio:g}",
         f"{terms.shift_video:g}/{terms.shift_audio:g}")) if old != new]
    message = (f"{terms.checkpoint} has its distillation baked in (PDD, "
               f"{terms.pdd_steps} heads): sampling as trained, {terms.steps} steps, "
               f"{terms.sampler_name}/{terms.scheduler}, cfg {terms.cfg:g}, shifts "
               f"{terms.shift_video:g}/{terms.shift_audio:g}")
    detail = moved + changes
    if detail:
        message += ". Changed: " + "; ".join(detail)
    log.info("Continuity: %s", message)
    try:
        from dgx_monarch import first_render

        notice = getattr(first_render.notice, "__wrapped__", first_render.notice)
        notice("model-contract", message, f"{terms.checkpoint}: {terms.steps} steps",
               once_key="spark-contract-" + hashlib.sha1(message.encode()).hexdigest()[:12],
               summary="Model settings")
    except Exception:
        pass  # no dgx-monarch here: the log line is the notice


def on_grid(terms, denoise):
    """(steps, denoise) for a partial pass that lands on the trained grid.

    ComfyUI samples a partial pass as the tail of a `steps / denoise` schedule
    (`KSampler.set_steps`), so 8 steps at 0.45 is the last 8 of a 17-step one,
    whose sigmas sit between the intervals the PDD heads were distilled on. k
    steps at k/N of an N-step contract are exactly its last k trained steps:
    on the grid, and fewer forwards. k is the nearest whole step, at least one.
    """
    k = max(1, min(terms.steps, round(float(denoise) * terms.steps)))
    return k, k / terms.steps


def _checkpoint_of(weights, compiled):
    try:
        return weights.get(compiled.checkpoint)
    except Exception:
        return None


def install(cls):
    """Wrap Continuity's H3 `resolve_sampling` and face pass once. True when
    in place."""
    if getattr(cls, "_spark_contract", False):
        return True
    orig = getattr(cls, "resolve_sampling", None)
    if not callable(orig):
        log.warning("Continuity has no resolve_sampling; checkpoint contracts are off")
        return False
    _install_face(cls)

    def resolve_sampling(self, data, widgets):
        if not enabled() or not isinstance(data, dict):
            return orig(self, data, widgets)
        try:
            before = orig(self, data, widgets)[0]
            # On a copy, committed whole, so a failure cannot half-edit the piece.
            edited = json.loads(json.dumps(data))
            outcome = apply(edited)
        except Exception:
            log.exception("Continuity: the checkpoint contract could not be read; "
                          "the render runs as the node says")
            return orig(self, data, widgets)
        if outcome:
            data.clear()
            data.update(edited)
        result = orig(self, data, widgets)
        if outcome:
            _announce(outcome[1], before, outcome[2])
        return result

    resolve_sampling.__wrapped__ = orig
    cls.resolve_sampling = resolve_sampling
    cls._spark_contract = True
    return True


def _install_face(cls):
    """The face pass re-noises the finished pass partway down the schedule;
    on a baked checkpoint, put that partial schedule on the trained grid."""
    orig = getattr(cls, "emit_face", None)
    if not callable(orig):
        return

    def emit_face(self, graph, links, payload, compiled, face, written, weights,
                  sampling, acceleration, seed):
        terms = contract(_checkpoint_of(weights, compiled)) if enabled() else None
        if terms is not None:
            try:
                steps, denoise = on_grid(terms, face.denoise)
                if (steps, denoise) != (sampling.steps, face.denoise):
                    log.info("Continuity: the face pass runs %s's last %d trained steps "
                             "(denoise %g; the node asked %d steps at %g)",
                             terms.checkpoint, steps, denoise, sampling.steps, face.denoise)
                face = dataclasses.replace(face, denoise=denoise)
                sampling = dataclasses.replace(sampling, steps=steps)
            except Exception:
                log.exception("Continuity: the face pass keeps the node's schedule")
        return orig(self, graph, links, payload, compiled, face, written, weights,
                    sampling, acceleration, seed)

    emit_face.__wrapped__ = orig
    cls.emit_face = emit_face
