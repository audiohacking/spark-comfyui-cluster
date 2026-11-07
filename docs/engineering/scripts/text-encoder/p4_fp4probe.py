import os, sys, json, time
sys.path.insert(0, "/opt/ComfyUI"); os.chdir("/opt/ComfyUI")
import comfy.options; comfy.options.enable_args_parsing(); sys.argv = [sys.argv[0]]
import torch, comfy.sd, comfy.model_management as mm
clip = comfy.sd.load_clip(ckpt_paths=["/opt/ComfyUI/models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"], embedding_directory=[], clip_type=comfy.sd.CLIPType.MINIMAX)
mm.load_models_gpu([clip.patcher])
tr = clip.cond_stage_model.qwen3vl_32b.transformer
m = tr.model.layers[10].mlp.gate_proj
print({k: getattr(m, k, "<none>") for k in ("layout_type", "_full_precision_mm", "_full_precision_mm_config", "comfy_force_cast_weights", "comfy_cast_weights", "quant_format")},
      "weight_function", len(m.weight_function), "weight", type(m.weight).__name__, m.weight.device,
      "has input_scale", getattr(m, "input_scale", None) is not None, "pre_quant_scale", getattr(m, "pre_quant_scale", None) is not None)
x = torch.randn(5760, 5120, device="cuda", dtype=torch.bfloat16)
from torch.profiler import profile, ProfilerActivity
for fp in (True, False):
    m._full_precision_mm = fp
    for _ in range(2): y = m(x)
    torch.cuda.synchronize(); s = time.monotonic(); y = m(x); torch.cuda.synchronize(); t = time.monotonic() - s
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        m(x); torch.cuda.synchronize()
    ks = sorted(((e.key, e.device_time_total) for e in prof.key_averages() if e.device_time_total > 0), key=lambda kv: -kv[1])[:4]
    print(f"full_precision_mm={fp}: {t*1000:.1f} ms out {y.dtype}", [ (k[:70], round(v/1000,2)) for k, v in ks])
