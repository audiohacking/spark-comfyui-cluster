"""Encode the H3 test presentation once through ComfyUI's normal path; save or compare."""
import os, sys, json, time
mode, path = sys.argv[1], sys.argv[2]            # save|compare <file>
sys.path.insert(0, "/opt/ComfyUI"); os.chdir("/opt/ComfyUI")
import comfy.options; comfy.options.enable_args_parsing(); sys.argv = [sys.argv[0]]
import torch
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "p0_inputs.py")).read())
import comfy.sd1_clip
clip.encode_from_tokens_scheduled(tokens); torch.cuda.synchronize()
s = time.monotonic(); out = clip.encode_from_tokens_scheduled(tokens); torch.cuda.synchronize(); t = time.monotonic() - s
cond, extra = out[0][0].cpu(), {k: v.cpu() for k, v in out[0][1].items() if torch.is_tensor(v)}
info = {"SPARK_TE_COMPUTE": os.environ.get("SPARK_TE_COMPUTE"), "patched": hasattr(comfy.sd1_clip, "_spark_te_act_dtype"),
        "warm_s": round(t, 2), "cond_dtype": str(cond.dtype), "shape": list(cond.shape)}
if mode == "save":
    torch.save({"cond": cond, "extra": extra}, path); info["saved"] = path
else:
    ref = torch.load(path)
    info["bit_identical_cond"] = bool(torch.equal(cond, ref["cond"]))
    info["bit_identical_extra"] = all(torch.equal(extra[k], ref["extra"][k]) for k in ref["extra"])
    d = cond.float() - ref["cond"].float(); cos = torch.nn.functional.cosine_similarity(cond[0].float(), ref["cond"][0].float(), dim=-1)
    info.update({"rel_l2": round(float(d.norm() / ref["cond"].float().norm()), 5), "cos_mean": round(float(cos.mean()), 5),
                 "cos_p01": round(float(cos.kthvalue(max(1, int(0.01 * cos.numel()))).values), 5), "cos_min": round(float(cos.min()), 5)})
print("RESULT", json.dumps(info), flush=True)
