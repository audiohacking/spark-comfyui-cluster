#!/usr/bin/env python3
"""Run Llama-family text encoders in bf16 on the GB10 instead of fp32.

Stock ComfyUI hands every text encoder fp32 activations (SDClipModel.forward
passes dtype=torch.float32 and fp32 embeddings), so on a GB10 their matrix
multiplies run as fp32 SGEMM on CUDA cores. The Qwen3-VL-32B behind MiniMax H3
spends 77% of a 48 s encode there; MiniMax runs that encoder in bf16, and in
bf16 the same multiplies run on cuBLAS's sm_121 tensor-core kernels (10.5 s).

Scope: encoders whose transformer is comfy.text_encoders.llama.BaseLlama (the
decoder-LLM encoders, normally run in bf16), on a CUDA device with bf16. Other
encoders (CLIP, T5, UMT5) keep stock fp32. Embeddings are cast in forward()
only, so LLM text generation through process_tokens is untouched; outputs are
still returned as fp32 (z = outputs[0].float()), as stock.

SPARK_TE_COMPUTE=fp32 restores stock behaviour bit for bit.

Reads source on stdin, writes patched source to stdout; echoes the input
unchanged when an anchor is missing, so the caller reports anchor-not-found.
"""
import os
import sys

MARKER = os.environ["MARKER"]
src = sys.stdin.read()

call_old = ("        outputs = self.transformer(None, attention_mask_model, embeds=embeds, num_tokens=num_tokens, "
            "intermediate_output=intermediate_output, final_layer_norm_intermediate=self.layer_norm_hidden_state, "
            "dtype=torch.float32, embeds_info=embeds_info)\n")
if src.count(call_old) != 1 or "def _spark_te_act_dtype" in src:
    sys.stdout.write(src); sys.exit(0)

helper = f'''

{MARKER} - GB10: Llama-family text encoders compute in bf16 (tensor cores).
def _spark_te_act_dtype(transformer, device):
    """bf16 for a Llama-family encoder on a bf16-capable CUDA device, else fp32."""
    if os.environ.get("SPARK_TE_COMPUTE", "bf16").strip().lower() in ("fp32", "float32", "stock", "0", "off"):
        return torch.float32
    try:
        from comfy.text_encoders.llama import BaseLlama
        if (isinstance(transformer, BaseLlama) and torch.device(device).type == "cuda"
                and torch.cuda.is_bf16_supported()):
            return torch.bfloat16
    except Exception:
        pass
    return torch.float32

'''
anchor = "\nclass ClipTokenWeightEncoder:"
if anchor not in src:
    sys.stdout.write(src); sys.exit(0)
src = src.replace(anchor, helper + anchor[1:], 1)

call_new = (
    f"        {MARKER}\n"
    "        _act = _spark_te_act_dtype(self.transformer, device)\n"
    "        if _act != torch.float32:\n"
    "            embeds = embeds.to(_act)\n"
    + call_old.replace("dtype=torch.float32, embeds_info", "dtype=_act, embeds_info")
)
src = src.replace(call_old, call_new, 1)
sys.stdout.write(src)
