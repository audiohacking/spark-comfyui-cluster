"""Phase 1: which kernels the H3 LM layers spend their time in (one GPU, encoder only)."""
import os, sys, time, json
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/te3out"
sys.path.insert(0, "/opt/ComfyUI"); os.chdir("/opt/ComfyUI")
import comfy.options; comfy.options.enable_args_parsing(); sys.argv = [sys.argv[0]]
import torch
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "p0_inputs.py")).read())          # builds clip, tokens
tr = clip.cond_stage_model.qwen3vl_32b.transformer
clip.encode_from_tokens_scheduled(tokens); torch.cuda.synchronize()     # warm
layer = tr.model.layers[10]
print("layer modules:", [(n, type(m).__name__) for n, m in layer.named_children()])
for n, m in layer.named_modules():
    if hasattr(m, "weight") and m.weight is not None and n.count(".") == 1:
        w = m.weight; print(f"  {n}: {type(m).__module__}.{type(m).__qualname__} weight {type(w).__name__} {getattr(w,'dtype',None)} {tuple(getattr(w,'shape',()))} {getattr(m,'comfy_cast_weights',None)}")
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    clip.encode_from_tokens_scheduled(tokens); torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=22, max_name_column_width=70))
