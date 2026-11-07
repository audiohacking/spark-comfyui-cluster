# shellcheck shell=bash
# =============================================================================
#  mod: gb10-te-bf16
#  Llama-family text encoders (Qwen3-VL/MiniMax H3, Qwen, Llama) compute in
#  bf16 on the GB10's tensor cores instead of ComfyUI's hard-coded fp32
#  activations. SPARK_TE_COMPUTE=fp32 restores stock behaviour exactly.
#  Patches: comfy/sd1_clip.py (SDClipModel.forward)
# =============================================================================
MOD_TAG="te_bf16"
MOD_FILE="comfy/sd1_clip.py"

mod_describe() {
  echo "bf16 tensor-core compute for Llama-family text encoders (SPARK_TE_COMPUTE=fp32 for stock)"
}

mod_apply() {
  py_patch_file "$MOD_FILE" "$MOD_TAG" "$MOD_DIR/transform.py"
}

mod_verify() {
  py_marker_present "$MOD_FILE" "$MOD_TAG"
}
