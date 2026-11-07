# Text encoders on the GB10

Measured 2026-11-07 on one DGX Spark (GB10, driver 580.178.04, kernel
7.0.0-1019-nvidia, ComfyUI v0.43.0), with MiniMax H3's text encoder
`qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` and a real ref2va presentation:
2 reference images, a reference video at 2 fps (5 two-frame blocks), an audio
slot, a long prompt. 5,760 tokens.

## Where the time went

The encoder is Qwen3-VL-32B truncated to 50 LM layers (hidden 5120, 64 query
heads, 8 KV heads, head dim 128, MLP 25600). The conditioning is the hidden
state after layer 50, `[1, 5760, 5120]`, 112 MB in fp32.

| Part | Warm time | Share |
|---|---|---|
| Vision tower (2 images + 5 video blocks, ~0.5 s each) | 3.4 s | 7% |
| LM: 50 layers × 0.86 s | 43.1 s | 91% |
| Whole encode | 47.2 s | |

A repeat encode is bit-identical. Peak GPU memory 17.4 GB.

Kernel profile of one encode (`p1_kernels.py`):

| Kernel | GPU time | |
|---|---|---|
| `magma_sgemmEx` + `cutlass…simt_sgemm` | 37.1 s (77%) | fp32 matrix multiplies on CUDA cores |
| `fmha_cutlassF_f32` | 7.1 s (15%) | fp32 attention |
| `dequantize_nvfp4` | 0.5 s (1%) | weight unpacking |

At 0.86 s per layer (~6.7 TFLOP) the LM ran at about 8 TFLOPS.

## Why fp32

- **Stock ComfyUI runs every text encoder with fp32 activations.**
  `comfy/sd1_clip.py` hard-codes `dtype=torch.float32` in
  `SDClipModel.forward` and `out_dtype=torch.float32` for the embeddings.
  The CLI flags (`--bf16-text-enc`, `--fp16-text-enc`, `--fp8_e4m3fn-text-enc`,
  `--fp32-text-enc`) only choose how weights are *stored*. No option changes
  the activations.
- **The H3 checkpoint is weight-only fp4.** Despite "awq" in its name, every
  layer's `comfy_quant` config has `"full_precision_matrix_mult": true`, and
  the layers carry no `input_scale` or `pre_quant_scale`. ComfyUI builds
  quantized text encoders with `full_precision_mm=True` anyway
  (`sd1_clip.py`), so the weights are dequantized and the multiply runs in the
  activation dtype. A native fp4 × fp4 (W4A4) path is not supported by this
  file; it would need activation calibration.
- **MiniMax runs this encoder in bf16.** The release ships Qwen3-VL-32B in
  bf16; Comfy-Org repackages it as bf16 (51.5 GB), int8_convrot (27.1 GB) and
  this nvfp4 (15.7 GB, a community NVFP4 quant of Qwen3-VL-32B-Instruct,
  credited to cybermotaz).

## Options measured

| Option | Encode | Result vs today | Outcome |
|---|---|---|---|
| fp32, split over 3 Sparks (Ulysses) | ~18 s (estimated) | **bit-identical** (proven on one GPU) | not built: bf16 on one Spark is faster |
| **bf16 activations** (weights dequantized to bf16, bf16 tensor cores) | **10.5 s** | rel L2 4.3%, cosine mean 0.998 | **shipped, default** |
| W4A4 native nvfp4 | n/a | n/a | not supported by the checkpoint |
| bf16 weights + bf16 compute (MiniMax's reference) | not run | n/a | 51.5 GB download skipped |

### fp32 Ulysses split: bit-exact on one GPU (`p2_invariance.py`)

On the real fp32 input of LM layers 0, 10 and 49, each piece was computed
whole and split the way three Sparks would, and compared bit for bit:

| Piece | Split | Max difference |
|---|---|---|
| RMSNorm, q/k/v/o projections, SwiGLU MLP | rows in thirds | 0.0 |
| q/k RMSNorm and RoPE | rows in thirds (positions sliced with the rows) | 0.0 |
| causal GQA attention | KV groups 3/3/2 (query heads 24/24/16), full sequence | 0.0 |
| whole layer, emulated end to end | as above | 0.0 |

So the fp32 kernels return the same bits for any row count and any head
subset, and a distributed version would only add NCCL byte movement. Pitfall:
`TransformerBlock.forward` writes its result into its input tensor
(`output = x … torch.add(residual, x, out=output)`); a harness must clone the
input before calling it.

### bf16 activations (`p3_precision.py`, `p4_fp4probe.py`)

| | Encode (warm) | vs fp32 (stock) |
|---|---|---|
| stock fp32 | 47.9 s | n/a |
| bf16 | 10.5 s | rel L2 4.34%, per-token cosine mean 0.99837, 1st percentile 0.97212, min 0.72042 |

The 5120 × 25600 MLP projection over 5,760 rows runs on cuBLAS's
`nvjet_sm121_tst_mma_128x208x64…` in 16.6 ms (~91 TFLOPS), plus ~1.5 ms of
`dequantize_nvfp4`. Setting `_full_precision_mm=False` changed nothing: the
checkpoint's own config forces full-precision matmul.

### Not concluded

A layer-by-layer comparison against MiniMax's bf16 weights (range-fetched from
Comfy-Org's file, layers 0/25/49) was attempted to separate the fp4-weight
error from the bf16-compute error. Its harness did not reproduce ComfyUI's own
layer output (attention 168% and MLP 65% apart with the same weights), most
likely because of in-place writes in ComfyUI's fused ops, so its numbers were
discarded. Quality was judged on renders instead, side by side with fp32, and the bf16
results were accepted.

## What shipped

**Mod `gb10-te-bf16`** (`mods/gb10-te-bf16/`, patches `comfy/sd1_clip.py`):
`SDClipModel.forward` casts the embeddings and passes `dtype` from
`_spark_te_act_dtype(transformer, device)`. That returns bf16 for an encoder
whose transformer is `comfy.text_encoders.llama.BaseLlama` (Qwen3-VL/H3,
Qwen, Llama) on a bf16-capable CUDA device, else fp32.
- Outputs stay fp32 (`z = outputs[0].float()`), as stock.
- LLM text generation (`process_tokens`) is untouched.
- CLIP, T5 and UMT5 keep fp32.
- `SPARK_TE_COMPUTE=fp32` restores stock. Verified bit-identical on the real
  H3 presentation (conditioning and `minimax_token_tags`) with `t_encode.py`.

**`te_cache.py`** (overlay pack): an exact cache on
`comfy.sd.CLIP.encode_from_tokens_scheduled` for every workflow. Key:
- the loader ComfyUI records on the patcher (`cached_patcher_init`), with
  weight files by path, size and mtime;
- LoRA patches by a digest of their tensors and strengths, once per
  `patches_uuid`;
- object patches the digest can read (ComfyUI sets `manual_cast_dtype` on
  every text encoder);
- `model_options`, the clip layer, tokenizer options;
- call arguments and `SPARK_TE_COMPUTE`;
- a digest of the whole token structure, every tensor's bytes included.

Anything unreadable (unknown origin, hooks, weight-wrapper patches, closures)
is not cached, and the status route (`/spark/continuity-monarch`, `te_cache`)
counts why. Results are handed out as deep copies, kept in RAM, LRU within
`SPARK_TE_CACHE_MB` (4096).

Why it was needed:
- ComfyUI's loaders run again on every prompt and hand out a new CLIP object,
  so the previous per-object memo (`encode_reuse.py`) never hit across renders.
- Since `video_ref_size=match`, the face pass's vision tokens differ from the
  main pass's, so it missed within a render too.

`test_te_cache.py`: 21 CPU tests:
- a hit across new CLIP objects, bit-identical in every tensor;
- the cache unaffected when a caller mutates its result;
- a miss on one flipped bit in an image tensor, a token, a LoRA patch or its
  strength, the clip layer, a call argument, a changed weight file, a
  different `manual_cast_dtype`, or a different `SPARK_TE_COMPUTE`;
- never cached: an unknown origin, hooks, a closure in `model_options`, a
  code object patch;
- the on/off switch, the budget and the status.

## In renders (3 Sparks, H3 ref2va, 5 s)

| | Main-pass encode | Face-pass encode | Job |
|---|---|---|---|
| stock fp32 (16:19 run) | ~52 s | 48 s | 10:46 steady state |
| bf16, first run after a restart | 15.1 s | 13.1 s | 12:07 (incl. cluster bring-up and cold DiT load) |
| bf16 + cache, new job after a restart | ~10.7 s | 0 s (cache hit on the main pass's tokens) | 9:18 (incl. bring-up) |

On a seed-only re-roll, ComfyUI's own node cache skips both encodes (the
segment nodes' inputs do not include the seed). The text-encoder cache covers
what that cache misses: other changes in the graph, alternating workflows, a
pass whose tokens equal another pass's.

Not run: re-queueing an earlier job exactly, to show bit-identical latents
across restarts (bf16 determinism) and through a cache hit (cache exactness).
The test: re-queue a job whose encodes run fresh, then one whose encodes must
come from the cache, comparing latent hashes with the original renders.
