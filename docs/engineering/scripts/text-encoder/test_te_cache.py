import os, sys, tempfile, importlib.util, torch
sys.path.insert(0, "/opt/ComfyUI"); os.chdir("/opt/ComfyUI")
_te = sys.argv[1]; sys.argv = [sys.argv[0], "--cpu"]
import comfy.options; comfy.options.enable_args_parsing()
import comfy.sd
spec = importlib.util.spec_from_file_location("te_cache", _te); tc = importlib.util.module_from_spec(spec); spec.loader.exec_module(tc)
assert tc.install() and tc.install()                       # idempotent
assert getattr(comfy.sd.CLIP.encode_from_tokens_scheduled, "_spark_te_cache")
wfile = tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors"); wfile.write(b"x"*100); wfile.close()

class P:  # minimal ModelPatcher surface
    def __init__(s, origin=True):
        s.cached_patcher_init = (comfy.sd.load_clip_model_patcher, ([wfile.name], None, comfy.sd.CLIPType.STABLE_DIFFUSION, {"dtype": torch.bfloat16})) if origin else None
        s.forced_hooks=None; s.object_patches={}; s.weight_wrapper_patches={}; s.patches={}; s.patches_uuid="u0"; s.model_options={"transformer_options": {}}
class C(comfy.sd.CLIP):
    calls = 0
    def __init__(s, **kw): super().__init__(no_init=True); s.patcher=P(**kw); s.layer_idx=None; s.tokenizer_options={}; s.use_clip_schedule=False; s.apply_hooks_to_conds=None; s.cond_stage_model=object()
    def encode_from_tokens(s, tokens, return_pooled=False, return_dict=False):
        C.calls += 1
        g = torch.Generator().manual_seed(int(tokens["qwen"][0][0][0]))
        return {"cond": torch.randn(1, 8, 4, generator=g), "pooled_output": torch.randn(1, 4, generator=g), "attention_mask": torch.ones(1, 8)}
def toks(v=7, img=None): return {"qwen": [[(v, 1.0), (3, 1.0)]], **({"images": [img]} if img is not None else {})}
def run(clip, t, **kw):
    before = C.calls; out = clip.encode_from_tokens_scheduled(t, **kw); return out, C.calls - before
ok = lambda c, m: print(("PASS " if c else "FAIL ") + m) or c
res = []
img = torch.rand(3, 16, 16)
a, n1 = run(C(), toks(7, img)); b, n2 = run(C(), toks(7, img.clone()))     # NEW clip object, same content
res.append(ok(n1 == 1 and n2 == 0, "hit across new CLIP objects with identical weights+tokens"))
res.append(ok(torch.equal(a[0][0], b[0][0]) and all(torch.equal(a[0][1][k], b[0][1][k]) for k in a[0][1]), "hit is bit-identical (cond and every pooled-dict tensor)"))
b[0][0].add_(1.0); c, n3 = run(C(), toks(7, img))
res.append(ok(n3 == 0 and torch.equal(c[0][0], a[0][0]), "caller mutating its result does not touch the cache"))
img2 = img.clone(); img2.view(torch.int32).view(-1)[5] ^= 1                   # flip one bit
_, n = run(C(), toks(7, img2)); res.append(ok(n == 1, "one flipped bit in an image tensor -> miss"))
_, n = run(C(), toks(8, img)); res.append(ok(n == 1, "different token id -> miss"))
cl = C(); cl.patcher.patches = {"w": [(1.0, torch.ones(2), 1.0, None, None)]}; cl.patcher.patches_uuid = "u1"
_, n = run(cl, toks(7, img)); res.append(ok(n == 1, "LoRA patch on the encoder -> miss"))
cl2 = C(); cl2.patcher.patches = {"w": [(0.5, torch.ones(2), 1.0, None, None)]}; cl2.patcher.patches_uuid = "u2"
_, n = run(cl2, toks(7, img)); res.append(ok(n == 1, "same LoRA, different strength -> miss"))
cl3 = C(); cl3.layer_idx = -2; _, n = run(cl3, toks(7, img)); res.append(ok(n == 1, "different clip layer -> miss"))
_, n = run(C(), toks(7, img), unprojected=True); res.append(ok(n == 1, "different call argument (unprojected) -> miss"))
_, n = run(C(), toks(7, img), add_dict={"guidance": 3.5}); res.append(ok(n == 1, "different add_dict -> miss"))
os.utime(wfile.name, ns=(1, 1)); _, n = run(C(), toks(7, img)); res.append(ok(n == 1, "weight file changed (mtime) -> miss"))
_, n = run(C(origin=False), toks(7, img)); _, n2 = run(C(origin=False), toks(7, img)); res.append(ok(n == 1 and n2 == 1, "CLIP of unknown origin -> never cached"))
cl4 = C(); cl4.patcher.forced_hooks = object(); _, n = run(cl4, toks(7, img)); _, n2 = run(cl4, toks(7, img)); res.append(ok(n == 1 and n2 == 1, "hooks -> never cached"))
cl5 = C(); cl5.patcher.model_options = {"transformer_options": {"f": (lambda x: x)}}; _, n = run(cl5, toks(7, img)); _, n2 = run(cl5, toks(7, img)); res.append(ok(n == 1 and n2 == 1, "closure in model_options -> never cached"))
cl6 = C(); cl6.patcher.object_patches = {"manual_cast_dtype": torch.bfloat16}; _, n = run(cl6, toks(7, img)); _, n2 = run(cl6, toks(7, img)); res.append(ok(n == 1 and n2 == 0, "readable object patch (manual_cast_dtype) -> cached, keyed"))
cl7 = C(); cl7.patcher.object_patches = {"diffusion_model.forward": (lambda *a: a)}; _, n = run(cl7, toks(7, img)); _, n2 = run(cl7, toks(7, img)); res.append(ok(n == 1 and n2 == 1, "code object patch -> never cached"))
cl8 = C(); cl8.patcher.object_patches = {"manual_cast_dtype": torch.float16}; _, n = run(cl8, toks(7, img)); res.append(ok(n == 1, "different manual_cast_dtype -> miss"))
os.environ["SPARK_TE_CACHE"] = "0"; _, n = run(C(), toks(7, img)); res.append(ok(n == 1, "SPARK_TE_CACHE=0 -> off")); del os.environ["SPARK_TE_CACHE"]
tc.clear(); os.environ["SPARK_TE_CACHE_MB"] = "0"; _, n = run(C(), toks(9, img)); _, n2 = run(C(), toks(9, img)); res.append(ok(n == 1 and n2 == 1, "entry larger than budget -> not kept")); del os.environ["SPARK_TE_CACHE_MB"]
tc.clear(); [run(C(), toks(v, img)) for v in range(20, 25)]; st = tc.status(); res.append(ok(st["entries"] == 5, f"status {st}"))
print(f"{sum(res)}/{len(res)} passed")
tc.clear(); _, n1 = run(C(), toks(31, img)); os.environ["SPARK_TE_COMPUTE"] = "fp32"; _, n2 = run(C(), toks(31, img)); del os.environ["SPARK_TE_COMPUTE"]; _, n3 = run(C(), toks(31, img))
print(("PASS " if (n1, n2, n3) == (1, 1, 0) else "FAIL ") + f"changing SPARK_TE_COMPUTE re-encodes; switching back hits ({n1},{n2},{n3})")
