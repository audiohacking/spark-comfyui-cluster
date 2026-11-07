# Shared by the text-encoder scripts (exec'd): builds an H3 ref2va presentation
# like Continuity's and loads the H3 encoder the way CLIPLoader does.
#   H3_PROMPT        prompt text (or H3_PROMPT_FILE: a text file)
#   H3_IMAGES        reference images in ComfyUI's input/, separated by '|'
#   H3_VIDEO         reference video in ComfyUI's input/ (sampled at 2 fps)
#   H3_TE            text-encoder file in models/text_encoders/
import json, os, time, numpy as np, av
from PIL import Image
import comfy.sd

def img(path, short=768):
    im = Image.open(path).convert("RGB"); w, h = im.size; s = short / min(w, h)
    im = im.resize((round(w * s), round(h * s)), Image.BICUBIC)
    return torch.from_numpy(np.asarray(im).astype(np.float32) / 255.0)[None]

def video(path, seconds=5.0, fps=2.0, size=(768, 1024)):
    c = av.open(path); frames = []; nxt = 0.0
    for f in c.decode(video=0):
        t = float(f.pts * f.time_base)
        if t + 1e-6 >= nxt:
            frames.append(np.asarray(f.to_image().convert("RGB").resize(size, Image.BICUBIC)).astype(np.float32) / 255.0)
            nxt += 1.0 / fps
        if t >= seconds or len(frames) >= int(seconds * fps):
            break
    return torch.from_numpy(np.stack(frames))

prompt = os.environ.get("H3_PROMPT") or open(os.environ["H3_PROMPT_FILE"]).read()
inp = "/opt/ComfyUI/input/"
items = [{"type": "image", "data": img(inp + f)} for f in os.environ["H3_IMAGES"].split("|") if f]
items += [{"type": "video", "data": video(inp + os.environ["H3_VIDEO"])}, {"type": "audio"}]

clip = comfy.sd.load_clip(ckpt_paths=["/opt/ComfyUI/models/text_encoders/" + os.environ.get("H3_TE", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")],
                          embedding_directory=[], clip_type=comfy.sd.CLIPType.MINIMAX)
tokens = clip.tokenize(prompt, minimax_ref_items=items)
