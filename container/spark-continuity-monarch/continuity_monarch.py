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

**Nothing is refused; the cluster takes what it can carry.** Each
generation's main pass samples on the mesh when it can, and on this Spark,
exactly as Continuity does on one card, when it cannot: a turbo lead-in, a
sigma shift off the checkpoint's (dgx-monarch carries one SD3-style shift,
H3's is a video/audio pair), a sound seam, a ControlNet guide, a model-patch
accelerator, a LoRA with its audio dial down, or a second checkpoint or LoRA
stack in the same strip (the mesh holds one). The face, refine, re-detail and
seam-restore passes always sample here, with Continuity's own LoRA stack and
patches, after the cluster has done the heavy pass. Every local decision is
logged with its reason. LoRAs on the cluster are merged by dgx-monarch's
loader rather than Continuity's own H3 stack.

A cluster sampler carries Continuity's animated preview and a page-cache
prefetch for the local passes' files: see monarch_preview.py.
"""

from __future__ import annotations

import copy
import importlib
import json
import logging
import os
import sys

log = logging.getLogger("spark.continuity_monarch")

from . import monarch_preview  # noqa: E402  (driver-side preview glue)

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
_HOOKS = ("weights_from_blob", "preflight", "emit_loaders", "emit_segment",
          "_segment_inputs", "emit_control", "emit_sampler", "emit_refine",
          "emit_face", "emit_seam_restore")

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


# ---- what goes where ---------------------------------------------------------
#
# Nothing is refused. The cluster takes each generation's main pass when it
# can carry it; anything it cannot is sampled on this Spark exactly as
# Continuity does on one card, with its own LoRA stack and patches, and the
# log says which and why. The local loaders are always in the graph for that,
# and cost nothing unless a local pass uses them: ComfyUI runs an expanded
# graph from its outputs, so a loader nothing reads never executes.


def accel_blocker(acceleration):
    """Why the accelerator row keeps the main pass local, or None. They are
    patches on a local MODEL; the attention backend is the exception, which
    dgx-monarch picks on its initializer ('sage' maps onto that)."""
    if acceleration is None:
        return None
    for on, what in (
            (acceleration.block_cache != "off", f"the '{acceleration.block_cache}' step cache"),
            (acceleration.spectrum, "Spectrum"),
            (acceleration.chunk_ffn, "the chunked feed-forward"),
            (acceleration.fp16_accumulation, "fp16 accumulation"),
            (acceleration.attention == "kitchen", "the kitchen attention kernel")):
        if on:
            return f"{what} patches a local model"
    if acceleration.attention == "sage" and attention_option("sage") is None:
        return "this dgx-monarch offers no sage attention"
    return None


def run_blocker(compiled, splits, sampling, anchors_anywhere):
    """Why one generation's main pass stays local, or None."""
    if splits:
        return "the turbo lead-in samples the opening steps on a second copy of the weights"
    if sampling.shifted():
        return (f"the sigma shift (video {sampling.shift_video}, audio "
                f"{sampling.shift_audio}) is off the checkpoint's, and dgx-monarch "
                f"carries one SD3-style shift, not H3's pair")
    if compiled.continues_audio or compiled.ends_on_audio:
        return "the sound seam is anchored by a wrapper on the local forward pass"
    if not anchors_anywhere and (compiled.continues or compiled.ends_on):
        return "this ComfyUI anchors picture seams with a wrapper on the local forward pass"
    if getattr(compiled, "guide", None):
        return "a ControlNet guide is read by core's forward, not dgx-monarch's"
    return None


# ---- the mesh --------------------------------------------------------------


def attention_option(attention):
    """dgx-monarch's kernel for our setting, read off the installed class.
    None for 'default' (the Init node's own pick), or when there is no sage."""
    if attention != "sage":
        return None
    declared = (_require(INIT_NODE).INPUT_TYPES().get("optional") or {}).get("attention")
    options = [str(option) for option in (declared[0] if declared else [])]
    return next((option for option in options if "SAGE" in option.upper()), None)


def slab_option():
    """The Init node's slab_weights, only when the deployment asks
    (SPARK_MONARCH_SLAB=auto|on|off); None leaves dgx-monarch's own default.
    'on' maps the weights from the file instead of keeping a host copy, but
    its identity gate needs room for both residencies at once, which large
    renders do not have. None as well when this dgx-monarch has no such
    option."""
    wanted = os.environ.get("SPARK_MONARCH_SLAB", "").strip().lower()
    if not wanted:
        return None
    declared = (_require(INIT_NODE).INPUT_TYPES().get("optional") or {}).get("slab_weights")
    options = [str(option) for option in (declared[0] if declared else [])]
    return wanted if wanted in options else None


class Models:
    """The cluster's model for this render: one mesh, one transformer, one
    LoRA stack, built on the first generation that samples there. A later
    generation asking for another checkpoint or stack samples locally rather
    than reloading every Spark mid-strip."""

    def __init__(self, graph, weights):
        self._graph = graph
        self._weights = weights
        self._link = None
        self._key = None

    @staticmethod
    def key(checkpoint, loras):
        return (checkpoint, tuple((name, round(float(strength), 6))
                                  for name, strength in loras))

    def claims(self, key):
        """Whether `key` may sample on the cluster; the first ask claims it."""
        if self._key is None:
            self._key = key
        return key == self._key

    def of(self, checkpoint, loras, acceleration):
        if self._link is None:
            self._link = self._build(checkpoint, loras, acceleration)
        return self._link

    def _build(self, checkpoint, loras, acceleration):
        for node_id in REQUIRED_NODES:
            _require(node_id)
        init = {"topology": "auto", "mode": "auto"}
        slab = slab_option()
        if slab is not None:
            init["slab_weights"] = slab
            if slab == "on":
                # dgx-monarch's 'advanced' residency profile for a homogeneous
                # unified-memory fleet sets these two together; slab needs it.
                init["lora_low_rss"] = "on"
        chosen = attention_option(getattr(acceleration, "attention", "default"))
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
    anchors_anywhere = getattr(payload_mod, "CORE_ANCHORS_ANYWHERE", False)

    def local(weights):
        """The same weights, as Continuity's one-card path reads them."""
        mine = copy.copy(weights)
        mine.backend = rl.DEFAULT
        return mine

    def label_of(payload):
        progress = payload.get("progress") or {}
        return f"segment {progress['index']}" if progress.get("index") else "this generation"

    def cluster_loras(self, payload, compiled):
        """The stack as the workers take it, or (None, why) when they cannot:
        an audio dial slices adaLN rows, which only Continuity's stack does."""
        entries = rm.compiler.active_loras(payload["request"].get("loras"),
                                           compiled.checkpoint, self.id)
        loras = []
        for entry in entries:
            if lora_mod.modality(entry):
                return None, f"'{entry['name']}' has its audio dial turned down"
            loras.append((entry["name"], float(entry.get("strength", 1.0))))
        return loras, None

    def plan(self, links, payload, compiled, sampling, run, acceleration):
        """(loras, None) to sample on the cluster, or (None, why) for local.
        Decided once per payload and remembered, so the segment node and the
        sampler hook cannot disagree."""
        memo = links._spark_plans
        if id(payload) in memo:
            return memo[id(payload)][1]
        loras, why = cluster_loras(self, payload, compiled)
        why = (why or accel_blocker(acceleration)
               or run_blocker(compiled, run.within(sampling, compiled, payload),
                              sampling, anchors_anywhere))
        if why is None and not links.monarch.claims(Models.key(compiled.checkpoint, loras)):
            why = "the cluster already holds another checkpoint or LoRA stack for this render"
        decision = (loras, None) if why is None else (None, why)
        memo[id(payload)] = (payload, decision)  # keep payload alive: id stays unique
        if why is not None:
            log.warning("Continuity: %s samples on this Spark, not the cluster: %s",
                        label_of(payload), why)
        else:
            log.info("Continuity: %s samples on the cluster", label_of(payload))
        return decision

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
        # Local passes may still use the accelerators: let Continuity check
        # their packs exactly as it does for one card.
        weights._spark_accel = acceleration
        return orig["preflight"](self, sampling, acceleration, local(weights))

    def emit_loaders(self, graph, weights, routes):
        if not enabled(weights):
            return orig["emit_loaders"](self, graph, weights, routes)
        # Continuity's full loader table: the text encoder and VAEs run here
        # either way, and the routed checkpoints wait for any local pass.
        links = orig["emit_loaders"](self, graph, local(weights), routes)
        links.monarch = Models(graph, weights)
        links._spark_plans = {}
        return links

    def emit_segment(self, graph, links, payload, compiled, weights, sampling,
                     seams, run):
        if not enabled(weights):
            return orig["emit_segment"](self, graph, links, payload, compiled,
                                        weights, sampling, seams, run)
        loras, _ = plan(self, links, payload, compiled, sampling, run,
                        getattr(weights, "_spark_accel", None))
        if loras is None:
            return orig["emit_segment"](self, graph, links, payload, compiled,
                                        local(weights), sampling, seams, run)
        # The segment node's distributed path is exactly the cluster's: no
        # MODEL sockets, conditioning and latent only. 'raylight' is the one
        # value it knows for "the transformer is elsewhere".
        remote = copy.copy(weights)
        remote.backend = rl.RAYLIGHT
        return orig["emit_segment"](self, graph, links, payload, compiled,
                                    remote, sampling, seams, run)

    def _segment_inputs(self, links, payload, compiled, weights, seams, splits, run):
        # Every caller hands the weights its pass runs on (see emit_segment);
        # a cluster render reaching here as such is a local pass.
        if enabled(weights):
            weights = local(weights)
        return orig["_segment_inputs"](self, links, payload, compiled,
                                       weights, seams, splits, run)

    def emit_control(self, graph, links, segment, compiled, weights, guide):
        # A guided generation is planned local, so its segment has a model.
        return orig["emit_control"](self, graph, links, segment, compiled,
                                    local(weights) if enabled(weights) else weights,
                                    guide)

    def emit_sampler(self, graph, links, segment, payload, compiled, sampling,
                     acceleration, weights, seed, run):
        if not enabled(weights):
            return orig["emit_sampler"](self, graph, links, segment, payload,
                                        compiled, sampling, acceleration,
                                        weights, seed, run)
        loras, _ = plan(self, links, payload, compiled, sampling, run, acceleration)
        if loras is None:
            return orig["emit_sampler"](self, graph, links, segment, payload,
                                        compiled, sampling, acceleration,
                                        local(weights), seed, run)
        model = links.monarch.of(compiled.checkpoint, loras, acceleration)
        model = _with_preview(graph, model, weights, compiled)
        # cfg 1.0 on the distilled checkpoints skips the negative, as on one card.
        against = graph.node("ConditioningZeroOut", conditioning=segment.out(1)).out(0)
        # The one-node sampler, as dgx-monarch's own H3 template wires it: its
        # sigmas are computed once the mesh exists, so the Init node's 'auto'
        # topology stands (the custom-sampler trio needs a preset topology,
        # because its scheduler runs before the first attach). denoise 1.0:
        # this is the first pass; the re-noising passes run locally.
        return graph.node(KSAMPLER_NODE, model=model, seed=seed, steps=sampling.steps,
                          cfg=sampling.cfg, sampler_name=sampling.sampler_name,
                          scheduler=sampling.scheduler, positive=segment.out(1),
                          negative=against, latent_image=segment.out(2),
                          denoise=1.0).out(0)

    def _with_preview(graph, model, weights, compiled):
        """The sampler's model, carrying Continuity's animated preview settings
        (read off its own graph_preview) and the files its local passes will
        open, for monarch_preview to act on while the cluster samples."""
        preview = None
        if rm.core.preview_available():
            preview = monarch_preview.capture_preview_kwargs(
                rm.core, weights, rm.declare.RULES.fps)
        prefetch = []
        if any(getattr(compiled, name, None) for name in ("face", "refine", "redetail")):
            for slot in (compiled.checkpoint, "sam3"):
                try:
                    name = weights.get(slot)
                except Exception:
                    name = None
                if isinstance(name, str) and name:
                    prefetch.append(name)
        if not preview and not prefetch:
            return model
        spec = json.dumps({"preview": preview, "prefetch": prefetch}, sort_keys=True)
        return graph.node(monarch_preview.NODE_ID, model=model, spec=spec).out(0)

    # The passes after the first sample a crop or a re-noised latent against a
    # real model, with Continuity's own stack and patches: always on this
    # Spark, after the cluster has done the heavy pass.
    def emit_refine(self, graph, links, segment, payload, compiled, weights,
                    seams, latent, sampling, acceleration, seed, run):
        if enabled(weights):
            log.info("Continuity: the two-pass upscale samples on this Spark")
            weights = local(weights)
        return orig["emit_refine"](self, graph, links, segment, payload, compiled,
                                   weights, seams, latent, sampling, acceleration,
                                   seed, run)

    def emit_face(self, graph, links, payload, compiled, face, written, weights,
                  sampling, acceleration, seed):
        if enabled(weights):
            log.info("Continuity: the face pass samples on this Spark")
            weights = local(weights)
        return orig["emit_face"](self, graph, links, payload, compiled, face,
                                 written, weights, sampling, acceleration, seed)

    def emit_seam_restore(self, graph, links, frames, payload, compiled, denoise,
                          *args, **kwargs):
        args = list(args)
        for i, value in enumerate(args):
            if enabled(value):
                args[i] = local(value)
        kwargs = {key: local(value) if enabled(value) else value
                  for key, value in kwargs.items()}
        return orig["emit_seam_restore"](self, graph, links, frames, payload,
                                         compiled, denoise, *args, **kwargs)

    for fn in (weights_from_blob, preflight, emit_loaders, emit_segment,
               _segment_inputs, emit_control, emit_sampler, emit_refine,
               emit_face, emit_seam_restore):
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
                          (rm, ("core", "slots", "compiler", "declare")),
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
    if not monarch_preview.wrap_sampler():
        log.warning("Continuity: dgx-monarch's sampler is not loaded; cluster renders will not preview")
    _installed = True
    STATE.update(installed=True, reason="",
                 continuity=mods["raylight"].__name__[: -len(_RAYLIGHT)])
    log.info("Continuity: dgx-monarch cluster backend installed (%s)",
             "automatic for renders without a backend" if auto()
             else "only where a piece asks for 'monarch'")
    return True
