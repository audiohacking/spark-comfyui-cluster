"""Phase 3: the H3 text encoder at native precisions (one GPU, no render).

V0  today: nvfp4 weights dequantized, fp32 activations (CUDA-core SGEMM)
V1  bf16 activations, weights dequantized to bf16 (bf16 tensor cores)
V2  bf16 activations, native nvfp4 W4A4 GEMMs (fp4 tensor cores, AWQ scales)
Warm time per variant, and each one's drift from V0 on the real conditioning.
"""
import os, sys, json, time
sys.path.insert(0, "/opt/ComfyUI"); os.chdir("/opt/ComfyUI")
import comfy.options; comfy.options.enable_args_parsing(); sys.argv = [sys.argv[0]]
import torch
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "p0_inputs.py")).read())               # clip, tokens
tr = clip.cond_stage_model.qwen3vl_32b.transformer
orig_forward = tr.forward
linears = [m for m in tr.model.modules() if hasattr(m, "_full_precision_mm")]
print("quantized-capable linears in the LM:", len(linears),
      "layout:", {getattr(m, "layout_type", None) for m in linears} , "quant_format:", {getattr(m, "quant_format", None) for m in linears})

def set_variant(v):
    if v == "V0":
        tr.forward = orig_forward
        for m in linears: m._full_precision_mm = True
        return
    def fwd(*a, **k):
        if k.get("embeds") is not None: k["embeds"] = k["embeds"].to(torch.bfloat16)
        if k.get("deepstack_embeds") is not None: k["deepstack_embeds"] = [d.to(torch.bfloat16) for d in k["deepstack_embeds"]]
        k["dtype"] = torch.bfloat16
        return orig_forward(*a, **k)
    tr.forward = fwd
    for m in linears: m._full_precision_mm = (v == "V1")

def run(v):
    set_variant(v)
    clip.encode_from_tokens_scheduled(tokens); torch.cuda.synchronize()       # warm
    s = time.monotonic(); out = clip.encode_from_tokens_scheduled(tokens); torch.cuda.synchronize()
    return time.monotonic() - s, out[0][0].float().cpu(), out[0][1]

R = {}; base = None
for v in ("V0", "V1", "V2"):
    try:
        t, c, extra = run(v)
    except Exception as e:
        R[v] = {"error": repr(e)[:300]}; continue
    r = {"warm_s": round(t, 2), "dtype_out": None}
    if base is None:
        base = c
    else:
        d = (c - base)
        cos = torch.nn.functional.cosine_similarity(c[0], base[0], dim=-1)
        r.update({"rel_l2_vs_V0": float(d.norm() / base.norm()), "max_abs_vs_V0": float(d.abs().max()),
                  "base_absmax": float(base.abs().max()), "cos_mean": float(cos.mean()), "cos_min": float(cos.min()),
                  "cos_p01": float(cos.kthvalue(max(1, int(0.01 * cos.numel()))).values), "finite": bool(torch.isfinite(c).all())})
    R[v] = r
    print(v, json.dumps(r), flush=True)
os.makedirs("/tmp/te3out", exist_ok=True)
print(json.dumps(R, indent=1))
