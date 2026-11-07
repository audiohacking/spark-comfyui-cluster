"""Phase 0: where the H3 text encoder's time goes (one GPU, no render).

Loads the real Qwen3-VL-32B H3 encoder the way ComfyUI's CLIPLoader does,
builds an H3 ref2va presentation like Continuity's (2 reference images, a
reference video at 2 fps, an audio slot, the H3_00191 prompt), and times:
tokenize, each vision item, each LM layer, the whole encode (cold and warm).
Saves the tokens and the conditioning for the identity tests that follow.
"""
import json, os, sys, time
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/te3out"
sys.path.insert(0, "/opt/ComfyUI"); os.chdir("/opt/ComfyUI")
import comfy.options; comfy.options.enable_args_parsing(); sys.argv = [sys.argv[0]]
import torch
import comfy.sd, comfy.model_management as mm

T = {}

t0 = time.monotonic()
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "p0_inputs.py")).read())   # clip, tokens
T["load_and_tokenize_s"] = time.monotonic() - t0
ent = tokens["qwen3vl_32b"][0]
T["entries"] = len(ent); T["vision_items"] = sum(1 for e in ent if isinstance(e[0], dict))

tr = clip.cond_stage_model.qwen3vl_32b.transformer
layers = tr.model.layers
T["lm_layers"] = len(layers)
sync = torch.cuda.synchronize
per_layer, vis = [], []
_start = {}
def pre(m, a, kw=None):
    sync(); _start[id(m)] = time.monotonic()
def post(m, a, out, kw=None):
    sync(); per_layer.append(time.monotonic() - _start[id(m)])
for l in layers:
    l.register_forward_pre_hook(pre); l.register_forward_hook(post)
orig_pre = tr.preprocess_embed
def timed_pre(embed, device):
    sync(); s = time.monotonic(); r = orig_pre(embed, device); sync(); vis.append(time.monotonic() - s); return r
tr.preprocess_embed = timed_pre

def encode(tag):
    per_layer.clear(); vis.clear(); sync(); s = time.monotonic()
    out = clip.encode_from_tokens_scheduled(tokens); sync()
    T[tag] = {"total_s": round(time.monotonic() - s, 2), "vision_s": round(sum(vis), 2),
              "vision_items_s": [round(v, 2) for v in vis], "lm_s": round(sum(per_layer), 2),
              "lm_layer_s_mean": round(sum(per_layer) / max(1, len(per_layer)), 3),
              "seq": int(out[0][0].shape[1]), "cond_shape": list(out[0][0].shape),
              "cond_mb": round(out[0][0].numel() * out[0][0].element_size() / 2**20, 1)}
    return out

o1 = encode("cold")
o2 = encode("warm")
T["warm_repeat_bit_identical"] = bool(torch.equal(o1[0][0], o2[0][0]))
T["gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
T["dtype"] = str(o1[0][0].dtype)
print(json.dumps(T, indent=1), flush=True)
os.makedirs(OUT, exist_ok=True)
torch.save({"tokens": tokens, "cond": o1[0][0].cpu(), "extra": {k: v for k, v in o1[0][1].items() if torch.is_tensor(v)}},
           os.path.join(OUT, "ref.pt"))
