"""Phase 2: can a 3-way Ulysses split of the H3 LM layer be bit-exact? (one GPU)

Captures one real LM layer's fp32 input during an encode, then compares the
layer's own forward with an emulation of what three Sparks would compute:
rows split for every per-token op, KV groups split (3/3/2) for attention over
the full sequence. Each op is also checked on its own so a mismatch is located.
"""
import os, sys, json
sys.path.insert(0, "/opt/ComfyUI"); os.chdir("/opt/ComfyUI")
import comfy.options; comfy.options.enable_args_parsing(); sys.argv = [sys.argv[0]]
import torch
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "p0_inputs.py")).read())               # clip, tokens
from comfy.text_encoders.llama import apply_rope

tr = clip.cond_stage_model.qwen3vl_32b.transformer
LAYERS = [int(v) for v in os.environ.get("LAYERS", "10").split(",")]
for LAYER in LAYERS:
    cap = {}
    def grab(m, args, kwargs):
        if "x" not in cap:
            cap["x"] = (args[0] if args else kwargs["x"]).detach().clone()
            cap["kw"] = {k: v for k, v in kwargs.items() if k != "x"}
            cap["args"] = args[1:]
    h = tr.model.layers[LAYER].register_forward_pre_hook(grab, with_kwargs=True)
    clip.encode_from_tokens_scheduled(tokens); torch.cuda.synchronize(); h.remove()
    layer = tr.model.layers[LAYER]
    x = cap["x"]; kw = cap["kw"]
    print("captured", tuple(x.shape), x.dtype, "kwargs", sorted(kw), "extra args", len(cap["args"]))
    mask = kw.get("attention_mask"); freqs = kw.get("freqs_cis"); opt = kw.get("optimized_attention")
    print("mask", None if mask is None else (tuple(mask.shape), mask.dtype), "freqs", type(freqs).__name__,
          [tuple(f.shape) for f in freqs] if isinstance(freqs, (list, tuple)) else getattr(freqs, "shape", None),
          "attn", getattr(opt, "__name__", opt))

    R = {}
    def same(a, b): return bool(torch.equal(a, b)), float((a - b).abs().max())
    L = x.shape[1]; W = 3
    rows = [slice(r * L // W, (r + 1) * L // W) for r in range(W)]

    with torch.no_grad():
        full, _ = layer(x.clone(), *cap["args"], **kw)
        # determinism of the layer itself
        R["layer_repeat"] = same(full, layer(x.clone(), *cap["args"], **kw)[0])

        att = layer.self_attn; mlp = layer.mlp
        xn = layer.input_layernorm(x)
        R["rmsnorm_rows"] = same(xn, torch.cat([layer.input_layernorm(x[:, s]) for s in rows], 1))
        for name in ("q_proj", "k_proj", "v_proj"):
            m = getattr(att, name)
            R[f"{name}_rows"] = same(m(xn), torch.cat([m(xn[:, s]) for s in rows], 1))
        big = torch.randn(1, L, att.inner_size, device=x.device, dtype=x.dtype)
        R["o_proj_rows"] = same(att.o_proj(big), torch.cat([att.o_proj(big[:, s]) for s in rows], 1))
        xm = layer.post_attention_layernorm(x)
        R["mlp_rows"] = same(mlp(xm), torch.cat([mlp(xm[:, s]) for s in rows], 1))

        # attention: all heads at once vs KV groups split 3/3/2 (GQA: 8 q heads per kv head)
        B = 1; H, KV, D = att.num_heads, att.num_kv_heads, att.head_dim
        q = att.q_proj(xn).view(B, L, H, D).transpose(1, 2)
        k = att.k_proj(xn).view(B, L, KV, D).transpose(1, 2)
        v = att.v_proj(xn).view(B, L, KV, D).transpose(1, 2)
        if att.q_norm is not None: q = att.q_norm(q)
        if att.k_norm is not None: k = att.k_norm(k)
        q, k = apply_rope(q, k, freqs_cis=freqs)
        o_full = opt(q, k, v, H, mask=mask, skip_reshape=True, enable_gqa=True)
        groups = [(0, 3), (3, 6), (6, 8)]; per = H // KV
        parts = [opt(q[:, a * per:b * per], k[:, a:b], v[:, a:b], (b - a) * per, mask=mask, skip_reshape=True,
                     enable_gqa=True) for a, b in groups]
        # optimized_attention returns (B, L, heads*D): stitch the head blocks back in order
        stitched = torch.cat([p.view(B, L, -1, D) for p in parts], 2).reshape(B, L, H * D)
        R["attention_kv_groups_332"] = same(o_full, stitched)
        # rope / q-k norm on row shards (positions sliced with the rows)
        def rope_rows(s):
            fr = tuple(f[:, s] for f in freqs) if isinstance(freqs, (list, tuple)) else freqs[:, s]
            qq = att.q_proj(xn[:, s]).view(B, -1, H, D).transpose(1, 2); kk = att.k_proj(xn[:, s]).view(B, -1, KV, D).transpose(1, 2)
            if att.q_norm is not None: qq = att.q_norm(qq)
            if att.k_norm is not None: kk = att.k_norm(kk)
            return apply_rope(qq, kk, freqs_cis=fr)
        try:
            sh = [rope_rows(s) for s in rows]
            R["qnorm_rope_rows_q"] = same(q, torch.cat([a for a, _ in sh], 2))
            R["qnorm_rope_rows_k"] = same(k, torch.cat([b for _, b in sh], 2))
        except Exception as e:
            R["qnorm_rope_rows"] = f"slicing freqs failed: {e!r}"[:200]

        # whole layer, emulated split end to end
        o_rows = [att.o_proj(stitched[:, s]) for s in rows]
        h1 = [x[:, s] + o for s, o in zip(rows, o_rows)]
        out = torch.cat([hh + mlp(layer.post_attention_layernorm(hh)) for hh in h1], 1)
        R["LAYER_split_emulated"] = same(full, out)
    print("LAYER", LAYER, json.dumps({k: v for k, v in R.items()}))
