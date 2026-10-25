"""Continuity's MiniMax H3 renders, sampled across the dgx-monarch cluster.

ComfyUI-Continuity (https://github.com/roadmaus/ComfyUI-Continuity) builds each
render as an expanded graph, and already has a seam for sampling somewhere
other than this card: its Raylight backend (`creator/raylight.py`). The
segment node encodes the conditioning and the latent on the driver and leaves
its MODEL outputs unwired; the loader and sampler nodes are swapped for
distributed ones; the decode, the seams and the save never learn anything was
distributed. This module adds a third backend on that same seam, "monarch",
whose loader and sampler are dgx-monarch's:

    DGXMonarchInit -> DGXMonarchUNETLoader -> DGXMonarchLoraLoader...
      -> DGXMonarchKSampler -> LATENT

It is an overlay, not a fork: Continuity stays the upstream pack and keeps
updating from ComfyUI-Manager. Nothing in its files changes; its family class
gains wrapped methods at load time. Every hook it wraps is checked for first,
and a Continuity whose internals moved is left completely alone with one log
line, so an upstream refactor can cost the cluster path but never a render.

**When it applies.** In cluster mode (the driver container runs with
SPARK_MONARCH=1) a render whose weights say nothing, Continuity's "one GPU"
default, samples on the cluster; SPARK_CONTINUITY_MONARCH=0 turns that off
for the deployment. A blob that says "monarch" outright gets it anywhere the
dgx-monarch nodes exist, and one that says "raylight" keeps Raylight. Stills
are untouched: Continuity's pre-stage pins its own weights to the local card
(`families/h3/still.py`), which reads the weights outside the family hook
wrapped here.

**What the cluster cannot carry is refused by name**, before anything is
queued, in Continuity's own terms and for its own reason: a render that came
out different from the one asked for is worse than one that did not start.
The mesh holds one transformer with one LoRA stack merged in, and the driver
holds no MODEL at all, so everything that patches or samples a local model is
out: the turbo lead-in, the refine, face and re-detail passes, blended sound
seams, ControlNet guides, the model-patch accelerators, device pins, a
two-checkpoint route, and a sigma shift off the checkpoint's own (dgx-monarch
carries one SD3-style shift; H3's is a video/audio pair).
"""

from __future__ import annotations

import copy
import importlib
import logging
import os
import sys

log = logging.getLogger("spark.continuity_monarch")

MONARCH = "monarch"

INIT_NODE = "DGXMonarchInit"
UNET_NODE = "DGXMonarchUNETLoader"
LORA_NODE = "DGXMonarchLoraLoader"
KSAMPLER_NODE = "DGXMonarchKSampler"
REQUIRED_NODES = (INIT_NODE, UNET_NODE, KSAMPLER_NODE)

# Continuity's own modules, as ComfyUI registers them: under the pack's path.
_RAYLIGHT = ".creator.raylight"
_RENDER = ".creator.families.h3.render"
_PAYLOAD = ".creator.families.h3.payload"
_LORA = ".creator.lora"

# The family methods wrapped below; all must exist or nothing is touched.
_HOOKS = ("weights_from_blob", "preflight", "emit_loaders", "_segment_inputs",
          "emit_control", "emit_sampler", "emit_refine", "emit_face",
          "emit_seam_restore")

_installed = False
# What install() last concluded, for the status route and the logs.
STATE = {"installed": False, "reason": "not attempted"}


def status():
    return {**STATE, "auto": auto(), "backends": _backends()}


def _backends():
    mods = _continuity_modules()
    return list(mods["raylight"].BACKENDS) if mods else []


def auto():
    """Whether a render with no backend of its own samples on the cluster."""
    return (os.environ.get("SPARK_MONARCH") == "1"
            and os.environ.get("SPARK_CONTINUITY_MONARCH", "1") != "0")


def enabled(weights):
    return getattr(weights, "backend", None) == MONARCH


def _require(node_id):
    import nodes

    node = nodes.NODE_CLASS_MAPPINGS.get(node_id)
    if node is None:
        raise ValueError(
            f"Cluster sampling needs dgx-monarch's '{node_id}' node, and this "
            f"ComfyUI does not have it. Run the cluster image "
            f"(spark-comfyui.sh update in cluster mode), or set "
            f"SPARK_CONTINUITY_MONARCH=0 to sample on this card."
        )
    return node


def _off_hint():
    return ("or render on this card alone (SPARK_CONTINUITY_MONARCH=0 for the "
            "deployment)")


# ---- what the cluster cannot carry -----------------------------------------


def refuse_accel(acceleration):
    """The accelerators are patches on a local MODEL, which a cluster render
    does not have. The attention backend is the exception: dgx-monarch picks a
    kernel on its initializer, and 'sage' maps onto that."""
    for on, what in (
            (acceleration.block_cache != "off", f"the '{acceleration.block_cache}' step cache"),
            (acceleration.spectrum, "Spectrum"),
            (acceleration.chunk_ffn, "the chunked feed-forward"),
            (acceleration.fp16_accumulation, "fp16 accumulation"),
            (acceleration.attention == "kitchen", "the kitchen attention kernel")):
        if on:
            raise ValueError(
                f"{what} patches the model on this side of the wire, and a "
                f"cluster render's model lives in the dgx-monarch workers. "
                f"Switch it off under Settings -> Rendering, {_off_hint()}."
            )


def refuse_weights(weights, routes):
    if getattr(weights, "devices", None):
        pinned = ", ".join(sorted(weights.devices))
        raise ValueError(
            f"{pinned} is pinned to a device, and a cluster render's transformer "
            f"is placed by dgx-monarch across the Sparks. Clear the device pins "
            f"in the node's 'weights' control, {_off_hint()}."
        )
    if len(routes) > 1:
        named = " and ".join(sorted(name.upper() for name in routes))
        raise ValueError(
            f"This render routes to both {named}, and the cluster holds one "
            f"transformer: sampling it would reload the weights on every Spark "
            f"between shots. Set the route to one checkpoint in the node's "
            f"'weights' control (Ref2VA takes everything FL2VA does), "
            f"{_off_hint()}."
        )


def refuse_run(compiled, splits, sampling, label=None):
    where = f"{label}: " if label else ""
    if splits:
        raise ValueError(
            f"{where}the turbo lead-in samples the opening steps with the "
            f"distillation held off the same checkpoint, which is the weights "
            f"twice, and the cluster holds one model. Set the turbo lead-in to "
            f"0 steps under Settings -> Rendering, {_off_hint()}."
        )
    if sampling.shifted():
        raise ValueError(
            f"{where}the sigma shift is off the checkpoint's own "
            f"(video {sampling.shift_video}, audio {sampling.shift_audio}). "
            f"dgx-monarch carries a single SD3-style shift and H3's is a "
            f"video/audio pair, so the workers would sample a different "
            f"schedule. Put the shifts back to the checkpoint's defaults, "
            f"{_off_hint()}."
        )
    if compiled.continues_audio or compiled.ends_on_audio:
        raise ValueError(
            f"{where}this seam carries the soundtrack across the cut, anchored "
            f"by a wrapper around the forward pass on this side of the wire, "
            f"which never reaches the cluster's workers. Switch the sound seam "
            f"off for this shot, {_off_hint()}."
        )
    for attribute, setting in (("refine", "the two-pass upscale"),
                               ("face", "the face pass"),
                               ("redetail", "the re-detail pass")):
        if getattr(compiled, attribute, None):
            raise ValueError(
                f"{where}{setting} samples in-process against a model on this "
                f"Spark, and a cluster render has none here. Switch it off for "
                f"this piece, {_off_hint()}."
            )


def refuse_seam(compiled, anchors_anywhere, label=None):
    if anchors_anywhere:
        return
    if compiled.continues or compiled.ends_on:
        where = f"{label}: " if label else ""
        raise ValueError(
            f"{where}this seam pins the previous shot's last frames by a "
            f"wrapper around the forward pass, which does not reach the "
            f"cluster's workers on this ComfyUI. Update ComfyUI (2026-08-14 or "
            f"later anchors them itself), {_off_hint()}."
        )


def refuse_guide():
    raise ValueError(
        "A ControlNet guide is applied by core's forward on a local model, and "
        "the cluster's workers run dgx-monarch's own H3 forward. Switch the "
        f"guide pill off, {_off_hint()}."
    )


def refuse_pass(what):
    raise ValueError(
        f"{what} samples in-process against a model on this Spark, and a "
        f"cluster render has none here. Switch it off, {_off_hint()}."
    )


# ---- the mesh --------------------------------------------------------------


def attention_option(attention):
    """dgx-monarch's kernel for our setting, read off the installed class."""
    if attention != "sage":
        return None
    declared = (_require(INIT_NODE).INPUT_TYPES().get("optional") or {}).get("attention")
    options = [str(option) for option in (declared[0] if declared else [])]
    match = next((option for option in options if "SAGE" in option.upper()), None)
    if match is None:
        raise ValueError(
            f"This dgx-monarch offers no sage attention ({options}). Set the "
            f"attention back to 'default' under Settings -> Rendering."
        )
    return match


class Models:
    """The cluster's model for this render: built once, handed to every shot.

    One mesh, one transformer, one LoRA stack, the same contract Continuity's
    `raylight.Actors` keeps for the same reason: a strip that asks for a
    different combination partway through would reload every Spark between
    shots, which is refused rather than done quietly.
    """

    def __init__(self, graph, weights):
        self._graph = graph
        self._weights = weights
        self._link = None
        self._key = None

    def of(self, checkpoint, loras, acceleration):
        key = (checkpoint, tuple((name, round(float(strength), 6))
                                 for name, strength in loras))
        if self._link is not None:
            if key != self._key:
                raise ValueError(
                    "Every shot in a cluster render shares one set of weights, "
                    "and this strip asks for a different checkpoint or LoRA "
                    "stack partway through. Give the shots the same LoRAs and "
                    "checkpoint, or render them in separate queues."
                )
            return self._link
        self._key = key
        self._link = self._build(checkpoint, loras, acceleration)
        return self._link

    def _build(self, checkpoint, loras, acceleration):
        for node_id in REQUIRED_NODES:
            _require(node_id)
        init = {"topology": "auto", "mode": "auto"}
        chosen = attention_option(acceleration.attention)
        if chosen is not None:
            init["attention"] = chosen
        mesh = self._graph.node(INIT_NODE, **init).out(0)
        filename = self._weights.get(checkpoint)
        dtype = getattr(self._weights, "dtype", None) or "default"
        model = self._graph.node(UNET_NODE, mesh=mesh, unet_name=filename,
                                 weight_dtype=dtype).out(0)
        for name, strength in loras:
            _require(LORA_NODE)
            model = self._graph.node(LORA_NODE, model=model, lora_name=name,
                                     strength_model=float(strength)).out(0)
        return model


# ---- the family hooks ------------------------------------------------------


def _wrap(cls, mods):
    rl = mods["raylight"]
    rm = mods["render"]
    payload_mod = mods["payload"]
    lora_mod = mods["lora"]
    orig = {name: getattr(cls, name) for name in _HOOKS}

    def weights_from_blob(self, data):
        weights = orig["weights_from_blob"](self, data)
        if getattr(weights, "backend", None) == rl.DEFAULT and auto():
            weights.backend = MONARCH
        return weights

    def preflight(self, sampling, acceleration, weights):
        if not enabled(weights):
            return orig["preflight"](self, sampling, acceleration, weights)
        for node_id in REQUIRED_NODES:
            _require(node_id)
        refuse_accel(acceleration)

    def emit_loaders(self, graph, weights, routes):
        if not enabled(weights):
            return orig["emit_loaders"](self, graph, weights, routes)
        # The checkpoint is loaded by the workers from a filename, so the
        # routed loaders are skipped; the text encoder and both VAEs still run
        # here, as they do for Raylight.
        refuse_weights(weights, routes)
        links = rm.core.Links(graph, weights, routes, skip=rm.slots.ROUTED_SLOTS)
        links.monarch = Models(graph, weights)
        return links

    def _segment_inputs(self, links, payload, compiled, weights, seams, splits, run):
        if not enabled(weights):
            return orig["_segment_inputs"](self, links, payload, compiled,
                                           weights, seams, splits, run)
        # The segment node's distributed path is exactly ours: no MODEL
        # sockets, outputs unwired, conditioning and latent only. Its input
        # says 'raylight' because that is the one value it knows for "the
        # transformer is elsewhere".
        remote = copy.copy(weights)
        remote.backend = rl.RAYLIGHT
        return orig["_segment_inputs"](self, links, payload, compiled,
                                       remote, seams, splits, run)

    def emit_control(self, graph, links, segment, compiled, weights, guide):
        if enabled(weights):
            refuse_guide()
        return orig["emit_control"](self, graph, links, segment, compiled,
                                    weights, guide)

    def _loras(self, payload, compiled):
        entries = rm.compiler.active_loras(payload["request"].get("loras"),
                                           compiled.checkpoint, self.id)
        loras = []
        for entry in entries:
            if lora_mod.modality(entry):
                raise ValueError(
                    f"'{entry['name']}' has its audio dial turned down, which "
                    f"slices the adapter's modulation rows: Continuity's own "
                    f"LoRA stack does that, and dgx-monarch's LoRA loader "
                    f"merges at one strength. Put the dial back to 1.0, "
                    f"{_off_hint()}."
                )
            loras.append((entry["name"], float(entry.get("strength", 1.0))))
        return loras

    def emit_sampler(self, graph, links, segment, payload, compiled, sampling,
                     acceleration, weights, seed, run):
        if not enabled(weights):
            return orig["emit_sampler"](self, graph, links, segment, payload,
                                        compiled, sampling, acceleration,
                                        weights, seed, run)
        splits = run.within(sampling, compiled, payload)
        progress = payload.get("progress") or {}
        label = f"Segment {progress['index']}" if progress.get("index") else None
        refuse_run(compiled, splits, sampling, label)
        refuse_seam(compiled, getattr(payload_mod, "CORE_ANCHORS_ANYWHERE", False), label)

        model = links.monarch.of(compiled.checkpoint, _loras(self, payload, compiled),
                                 acceleration)
        # cfg 1.0 on the distilled checkpoints skips the negative, as on one card.
        against = graph.node("ConditioningZeroOut", conditioning=segment.out(1)).out(0)
        # The one-node sampler, as dgx-monarch's own H3 template wires it: its
        # sigmas are computed once the mesh exists, so the Init node's 'auto'
        # topology stands. The custom-sampler trio would need a preset
        # topology, because its scheduler runs before the first attach.
        # denoise 1.0: this is the first pass; the re-noising passes are
        # refused above.
        return graph.node(KSAMPLER_NODE, model=model, seed=seed, steps=sampling.steps,
                          cfg=sampling.cfg, sampler_name=sampling.sampler_name,
                          scheduler=sampling.scheduler, positive=segment.out(1),
                          negative=against, latent_image=segment.out(2),
                          denoise=1.0).out(0)

    def emit_refine(self, graph, links, segment, payload, compiled, weights,
                    seams, latent, sampling, acceleration, seed, run):
        if enabled(weights):
            refuse_pass("The two-pass upscale")
        return orig["emit_refine"](self, graph, links, segment, payload, compiled,
                                   weights, seams, latent, sampling, acceleration,
                                   seed, run)

    def emit_face(self, graph, links, payload, compiled, face, written, weights,
                  sampling, acceleration, seed):
        if enabled(weights):
            refuse_pass("The face pass")
        return orig["emit_face"](self, graph, links, payload, compiled, face,
                                 written, weights, sampling, acceleration, seed)

    def emit_seam_restore(self, graph, links, frames, payload, compiled, denoise,
                          *args, **kwargs):
        if getattr(links, "monarch", None) is not None:
            refuse_pass("The seam restore")
        return orig["emit_seam_restore"](self, graph, links, frames, payload,
                                         compiled, denoise, *args, **kwargs)

    for fn in (weights_from_blob, preflight, emit_loaders, _segment_inputs,
               emit_control, emit_sampler, emit_refine, emit_face, emit_seam_restore):
        fn.__wrapped__ = orig[fn.__name__]
        setattr(cls, fn.__name__, fn)
    cls._spark_monarch = True


def _continuity_modules():
    """Continuity's modules by role, or None while it is not (fully) loaded."""
    for name, module in list(sys.modules.items()):
        if not name.endswith(_RAYLIGHT) or module is None:
            continue
        prefix = name[: -len(_RAYLIGHT)]
        if prefix + _RENDER not in sys.modules:
            # Continuity's registry imports a family's render half on its
            # first render, to keep the sampler out of startup. That is after
            # every on_prompt hook, so it is imported here instead, the way
            # the registry does it.
            try:
                importlib.import_module(prefix + _RENDER)
            except Exception:
                log.exception("Continuity: cannot import its H3 render module")
                continue
        mods = {"raylight": module,
                "render": sys.modules.get(prefix + _RENDER),
                "payload": sys.modules.get(prefix + _PAYLOAD),
                "lora": sys.modules.get(prefix + _LORA)}
        if all(mods.values()) and hasattr(mods["render"], "H3"):
            return mods
    return None


def _compatible(mods):
    rl, rm = mods["raylight"], mods["render"]
    missing = [name for name in _HOOKS if not callable(getattr(rm.H3, name, None))]
    for module, attrs in ((rl, ("DEFAULT", "RAYLIGHT", "BACKENDS")),
                          (rm, ("core", "slots", "compiler")),
                          (mods["lora"], ("modality",))):
        missing += [f"{module.__name__.rsplit('.', 1)[-1]}.{a}"
                    for a in attrs if not hasattr(module, a)]
    if not isinstance(getattr(rl, "BACKENDS", None), list):
        missing.append("raylight.BACKENDS (a list)")
    return missing


def install():
    """Patch Continuity once; True when the cluster backend is in place."""
    global _installed
    if _installed:
        return True
    mods = _continuity_modules()
    if mods is None:
        STATE.update(installed=False, reason="Continuity is not loaded")
        return False
    missing = _compatible(mods)
    if missing:
        log.warning("Continuity's internals moved (%s); its renders stay on "
                    "this card. Update spark-comfyui-cluster's overlay.",
                    ", ".join(missing))
        _installed = True  # answered; do not retry every prompt
        STATE.update(installed=False, reason="internals moved: " + ", ".join(missing))
        return False
    if MONARCH not in mods["raylight"].BACKENDS:
        mods["raylight"].BACKENDS.append(MONARCH)
    if not getattr(mods["render"].H3, "_spark_monarch", False):
        _wrap(mods["render"].H3, mods)
    _installed = True
    STATE.update(installed=True, reason="",
                 continuity=mods["raylight"].__name__[: -len(_RAYLIGHT)])
    log.info("Continuity: dgx-monarch cluster backend installed (%s)",
             "automatic for renders without a backend" if auto()
             else "only where a piece asks for 'monarch'")
    return True
