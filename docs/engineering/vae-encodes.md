# VAE encodes on the cluster

Traced 2026-11-07 on the 3-Spark ring, from the log of a 5 s MiniMax H3 ref2va
job (506.6 s): 2 reference images, a reference video with its
soundtrack, face pass on.

## Where a job's time went

| Phase | Seconds | Where |
|---|---|---|
| reference video, VAE encode (first) | 21.4 | driver GPU |
| text encode | 8.5 | driver GPU |
| reference video, VAE encode (**again**) | 21.6 | driver GPU |
| main pass sampling | 207.1 | 3 Sparks |
| VAE decode (split over 3) | 14.4 | 3 Sparks |
| SAM3 face detection | 9.4 | driver GPU |
| face crop, VAE encode | ~22 | driver GPU |
| face pass sampling | 169.9 | 3 Sparks |
| VAE decode (split over 3) | 11.7 | 3 Sparks |
| save | ~8 | driver |

Sampling is 74% of the job. From the model's shapes (50 layers, hidden 5376,
56 heads x 128, FFN 14336; 16,862 of 50,586 rows per rank) attention is about
twice the linear layers' work per rank, and Ulysses moves ~30 GB per rank per
step, which matches the RDMA counters (~34 GB per 26 s step). Sampling is
compute-bound; nothing exact was found there without a per-step profile.

## The double encode

Continuity keys its reference-latent cache (`creator/latents.py`, `_ref_key`)
on the generation's canvas under `ref_size: match`. A render builds its
conditioning at two canvases, 768x768 (the face pass) and 768x1024 (the main
pass). The clip's own canvas (480x864) is smaller than both, so it is resized
to the same pixels twice, and both lookups miss. Proof:

- The two entries' keys were reproduced from their parts: identical except
  `gen_w`/`gen_h` (768x768 and 768x1024).
- Their contents are byte-identical: latent `fe049e5c…` in both (and
  `9126b589…` in both for the previous job's clip), presentation too.
- The text encoder's cache hit on the second pass confirms identical tokens.

Fix, without touching Continuity: **`vae_cache.py`** (overlay pack), an exact
cache on `comfy.sd.VAE.encode` for every workflow. Key: the loader recorded on
the patcher (`cached_patcher_init`, files by path, size and mtime), the VAE's
class, dtypes, devices and tiling settings, object patches and model options,
and every byte of the pixels (an 8-chunk SHA-256 tree: 0.04 s for a 617 MB clip
on Grace, 0.25 s single-threaded). Single-batch encodes only: ComfyUI splits a
batch by free memory, and the split is not in the key. Patched VAEs and VAEs of
unknown origin are not cached. Results live on the CPU and are handed out as
copies on the encode's own device and dtype. It stays the outermost wrapper on
`VAE.encode` (re-installed on every prompt), so a hit never opens a cluster
session. `SPARK_VAE_CACHE=0` turns it off; `SPARK_VAE_CACHE_MB` (1024) bounds
it. The status route shows it under `vae_cache`.

Tests: `scripts/vae/test_vae_cache.py`, 28 CPU tests (hits across new VAE
objects, bit-identity, copies, a miss on one flipped bit, frame count, dtypes,
VAE class, loader metadata, weights mtime; never cached: unknown origin, weight
patches, code object patches, batches; layering under dgx-monarch's wrapper).

## Encodes split over the Sparks

The H3 video VAE encodes in independent clips (`encode_temporal`): each 17
frames, the last padded with copies of its final frame, each spatially tiled
and blended inside `tiled_encode`, with no state carried between clips. So a
clip encoded on another Spark with the same file, dtype and software is the
driver's clip, byte for byte, as with dgx-monarch's decode split.

**`container/monarch-vae-encode-split.patch`** (dgx-monarch, applied after the
other five):
- A keyed wrapper on `VAE.encode`, armed by the same `worker_vae_decode = "auto"`
  knob, shadows `encode_temporal` for MiniMax H3 encodes of more than one clip.
- `h3_encode_plan` cuts the clips as the stock loop does; `H3EncodeReplay`
  stands in for `_adaptive_encode` during the stock loop. A Worker's result is
  used only when the normalized clip the loop hands over equals the planned
  clip normalized by the model's own `_normalize_pixels` on the same device,
  and the result matches the driver's own clip 0 in dtype and full shape.
  Anything else encodes on the driver (fail-open, as decode).
- Workers: unit kind `h3_encode` on the existing `vae_decode` endpoint runs
  `_adaptive_encode(_normalize_pixels(clip))` on the VAE they already hold for
  decode, priced with the stock `memory_used_encode`.
- Log: `VAE encode split over 3 ranks (minimax_h3, VAE Encode): …`.
  `DGXM_VAE_ENCODE=off` keeps encodes on the driver.

Expected: 8 clips (124 frames) over 3 GPUs in 3 rounds, ~2.7 s per clip, so
~22 s becomes ~8.5 s for the face crop and for a new reference clip; ~42-60 MB
of pixels per remote clip over the fabric.

Tests (CPU, real ComfyUI classes, simulated Worker ranks running the Worker
body's computation): `tests/test_vae_encode_split_comfy.py` in the patch, 21
tests: byte-identical to the stock encode for 22/39/124 frames, fp32/fp16, one
and two remote ranks; a rank that raises, declines, answers in the wrong dtype
or a wrong shape; one clip and a still stay here; a plan the loop leaves; the
Worker body; the stock seams the replay depends on. dgx-monarch's own suite:
16,651 pass; every remaining failure also fails without this patch (the
Ulysses-any-world patch's intended changes) or comes from running a `docker cp`
copy (git ownership, Studio file ownership).

## A note on standalone GPU scripts

A standalone script that loaded the H3 video VAE in the driver container and
encoded a 124-frame clip ran the host out of memory twice (the second time with
22 frames), killing the driver ComfyUI and a queued job. The script's
VAE took ComfyUI's non-dynamic "loaded completely" path, unlike the server's
dynamic-VRAM load. Don't encode with the video VAE outside the server process;
test VAE code on CPU, and verify on a real render.
