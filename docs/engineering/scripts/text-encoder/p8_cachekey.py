"""Why does the cache skip the real H3 encoder? Load it exactly as the CLIPLoader node does."""
import os, sys, json, importlib.util
sys.path.insert(0, "/opt/ComfyUI"); os.chdir("/opt/ComfyUI")
import comfy.options; comfy.options.enable_args_parsing(); sys.argv = [sys.argv[0]]
import torch
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "p0_inputs.py")).read().split("clip = comfy.sd.load_clip")[0]
exec(src)
import nodes, folder_paths
clip = nodes.CLIPLoader().load_clip("qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", type="minimax")[0]
tokens = clip.tokenize(prompt, minimax_ref_items=items)
spec = importlib.util.spec_from_file_location("te_cache", "/opt/monarch-nodes/spark-continuity-monarch/te_cache.py"); tc = importlib.util.module_from_spec(spec); spec.loader.exec_module(tc)
p = clip.patcher
print("origin:", p.cached_patcher_init[0].__name__ if p.cached_patcher_init else None, [type(a).__name__ for a in (p.cached_patcher_init[1] if p.cached_patcher_init else [])])
print("model_options:", {k: type(v).__name__ for k, v in p.model_options.items()}, p.model_options.get("transformer_options"))
print("patches:", len(p.patches), "object_patches:", list(p.object_patches)[:5], "wrapper:", list(p.weight_wrapper_patches)[:5], "hooks:", p.forced_hooks)
print("key:", tc._key(clip, tokens, False, {}), "skips:", tc.SKIPS)
ent = tokens["qwen3vl_32b"][0]
print("token entry types:", sorted({(type(e[0]).__name__, type(e[1]).__name__) for e in ent}), {k: type(v).__name__ for e in ent if isinstance(e[0], dict) for k, v in e[0].items()})
