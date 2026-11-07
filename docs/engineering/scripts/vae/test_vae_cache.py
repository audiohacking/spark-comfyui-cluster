"""CPU unit tests of vae_cache.py. Run inside the driver container:
  CUDA_VISIBLE_DEVICES= python test_vae_cache.py /opt/monarch-nodes/spark-continuity-monarch
The stock VAE.encode is replaced by a stand-in before the cache wraps it, so
no model loads and nothing touches the GPU."""
import os, sys, tempfile, types, importlib, torch
pack = sys.argv[1]
sys.path.insert(0, "/opt/ComfyUI"); os.chdir("/opt/ComfyUI"); sys.argv = [sys.argv[0], "--cpu"]
import comfy.options; comfy.options.enable_args_parsing()
import comfy.sd

calls = {"n": 0}
def stand_in(self, pixel_samples):                     # a deterministic "encode"
    calls["n"] += 1
    x = pixel_samples.float()
    return (x.mean(dim=-1, keepdim=True)[..., ::4, ::4, :] * 3.0 + self._salt).movedim(-1, 1)
comfy.sd.VAE.encode = stand_in
pkg = types.ModuleType("ovl"); pkg.__path__ = [pack]; sys.modules["ovl"] = pkg
vc = importlib.import_module("ovl.vae_cache")
assert vc.install() and vc.install()                   # idempotent
assert getattr(comfy.sd.VAE.encode, "_spark_vae_cache")

wfile = tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors"); wfile.write(b"x" * 100); wfile.close()
class P:
    def __init__(s, origin=True):
        s.cached_patcher_init = (comfy.sd.load_vae_patcher, (wfile.name, {"m": "1"}, None)) if origin else None
        s.patches = {}; s.weight_wrapper_patches = {}; s.object_patches = {}; s.model_options = {}
class FSM: tiling = True; tile_size = 256; tile_overlap_min = 64
def V(origin=True, latent_dim=3, not_video=False, dtype=torch.float16, salt=0.0):
    v = comfy.sd.VAE.__new__(comfy.sd.VAE)
    v.patcher = P(origin); v.first_stage_model = FSM(); v.latent_dim = latent_dim; v.not_video = not_video
    v.vae_dtype = dtype; v.device = torch.device("cpu"); v.output_device = torch.device("cpu")
    v.handles_tiling = True; v.format_encoded = None; v._salt = salt
    v.vae_output_dtype = lambda: torch.float32
    return v
def run(v, px):
    before = calls["n"]; out = v.encode(px); return out, calls["n"] - before
ok = lambda c, m: print(("PASS " if c else "FAIL ") + m) or c
res = []
clip = torch.rand(22, 64, 48, 3)
a, n1 = run(V(), clip); b, n2 = run(V(), clip.clone())                       # NEW VAE object, same content
res.append(ok(n1 == 1 and n2 == 0, "hit across new VAE objects with identical weights+pixels"))
res.append(ok(torch.equal(a, b) and a.dtype == b.dtype and a.device == b.device, "hit is bit-identical, same dtype and device"))
res.append(ok(a.data_ptr() != b.data_ptr(), "hit is a copy, not the caller's earlier tensor"))
b.add_(1.0); c, n3 = run(V(), clip)
res.append(ok(n3 == 0 and torch.equal(c, a), "caller mutating its result does not touch the cache"))
a.add_(1.0); c, _ = run(V(), clip); res.append(ok(not torch.equal(c, a), "caller mutating the miss's result does not touch the cache"))
c2 = clip.clone(); c2.view(-1).view(torch.int32)[777] ^= 1
_, n = run(V(), c2); res.append(ok(n == 1, "one flipped bit in the pixels -> miss"))
_, n = run(V(), clip[:17]); res.append(ok(n == 1, "different frame count -> miss"))
_, n = run(V(), clip.double()); res.append(ok(n == 1, "different pixel dtype -> miss"))
_, n = run(V(dtype=torch.bfloat16), clip); res.append(ok(n == 1, "different VAE dtype -> miss"))
v = V(); v.first_stage_model = type("F2", (), {"tiling": False})(); _, n = run(v, clip); res.append(ok(n == 1, "different VAE class/tiling -> miss"))
v = V(); v.patcher.cached_patcher_init = (comfy.sd.load_vae_patcher, (wfile.name, {"m": "2"}, None)); _, n = run(v, clip); res.append(ok(n == 1, "different loader metadata -> miss"))
os.utime(wfile.name, ns=(1, 1)); _, n = run(V(), clip); res.append(ok(n == 1, "weights file changed (mtime) -> miss"))
_, n = run(V(origin=False), clip); _, n2 = run(V(origin=False), clip); res.append(ok(n == 1 and n2 == 1, "VAE of unknown origin -> never cached"))
v = V(); v.patcher.patches = {"w": [(1.0, torch.ones(2), 1.0)]}; _, n = run(v, clip); _, n2 = run(v, clip); res.append(ok(n == 1 and n2 == 1, "weight patches -> never cached"))
v = V(); v.patcher.object_patches = {"f": (lambda x: x)}; _, n = run(v, clip); _, n2 = run(v, clip); res.append(ok(n == 1 and n2 == 1, "code object patch -> never cached"))
img = torch.rand(2, 32, 32, 3)
_, n = run(V(latent_dim=2), img); _, n2 = run(V(latent_dim=2), img); res.append(ok(n == 1 and n2 == 1, "batch of 2 images -> never cached (split depends on free memory)"))
_, n = run(V(latent_dim=2), img[:1]); _, n2 = run(V(latent_dim=2), img[:1]); res.append(ok(n == 1 and n2 == 0, "single image -> cached"))
_, n = run(V(not_video=True), clip); _, n2 = run(V(not_video=True), clip); res.append(ok(n == 1 and n2 == 1, "frames as a batch (not_video) -> never cached"))
d1 = vc.pixels_digest(clip); res.append(ok(d1 == vc.pixels_digest(clip.clone()) and d1 != vc.pixels_digest(clip.reshape(22, 48, 64, 3)), "digest: content-equal tensors agree, a reshape differs"))
res.append(ok(vc.pixels_digest(clip.permute(0, 2, 1, 3)) == vc.pixels_digest(clip.permute(0, 2, 1, 3).contiguous()), "digest: non-contiguous view equals its contiguous copy"))
os.environ["SPARK_VAE_CACHE"] = "0"; _, n = run(V(), clip); _, n2 = run(V(), clip); res.append(ok(n == 1 and n2 == 1, "SPARK_VAE_CACHE=0 -> off")); del os.environ["SPARK_VAE_CACHE"]
vc.clear(); os.environ["SPARK_VAE_CACHE_MB"] = "0"; _, n = run(V(), clip); _, n2 = run(V(), clip); res.append(ok(n == 1 and n2 == 1, "entry larger than budget -> not kept")); del os.environ["SPARK_VAE_CACHE_MB"]
vc.clear(); [run(V(salt=0.0), torch.rand(5, 16, 16, 3)) for _ in range(4)]; st = vc.status(); res.append(ok(st["entries"] == 4 and st["misses"] > 0, f"status {st}"))
# Layering: dgx-monarch's encode split wraps VAE.encode with functools.wraps when its mesh arms.
import functools
through = {"n": 0}
def monarch_like(original):
    @functools.wraps(original)
    def encode(self, px):
        through["n"] += 1
        return original(self, px)
    encode._dgxm_vae_decode_route = True
    return encode
vc.clear(); comfy.sd.VAE.encode = monarch_like(comfy.sd.VAE.encode)      # monarch on top
assert vc.install()                                                      # cache back on top
top = comfy.sd.VAE.encode
res.append(ok(top is vc._WRAPPER and getattr(top.__wrapped__, "_dgxm_vae_decode_route", False), "install() puts the cache back above the monarch layer"))
res.append(ok(vc.install() and comfy.sd.VAE.encode is top, "install() is idempotent while on top"))
lay = torch.rand(22, 40, 40, 3); through["n"] = 0
_, n = run(V(), lay); t1 = through["n"]; _, n2 = run(V(), lay); t2 = through["n"]
res.append(ok((n, t1) == (1, 1) and (n2, t2) == (0, 1), f"miss passes the monarch layer once, a hit never reaches it ({n},{t1},{n2},{t2})"))
st = vc.status(); res.append(ok(st["hits"] >= 1, "the inner cache layer steps aside (no double entries)"))
res.append(ok(len(vc._store) == 1, f"one entry for one encode across two cache layers ({len(vc._store)})"))
print(f"{sum(res)}/{len(res)} passed")
