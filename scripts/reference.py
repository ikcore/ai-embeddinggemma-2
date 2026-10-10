"""Reference embeddings for EmbeddingGemma 2 from transformers (float32, eager).

usage: python -I reference.py MODEL_DIR OUT_DIR
Writes OUT_DIR/test.png, OUT_DIR/test.wav and OUT_DIR/reference.json.
"""
import json
import math
import struct
import sys
import wave

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoModel, AutoProcessor

model_dir, out_dir = sys.argv[1], sys.argv[2]
torch.manual_seed(0)

# --- inputs -----------------------------------------------------------------
img = Image.new("RGB", (640, 480))
px = img.load()
for y in range(480):
    for x in range(640):
        px[x, y] = (x * 255 // 639, y * 255 // 479, (x * y) % 256)
d = ImageDraw.Draw(img)
d.ellipse((200, 120, 440, 360), fill=(220, 30, 40))
d.rectangle((40, 40, 160, 200), fill=(20, 200, 60))
img.save(f"{out_dir}/test.png")
img = Image.open(f"{out_dir}/test.png").convert("RGB")

sr = 16000
n = int(2.5 * sr)
t = np.arange(n) / sr
sig = 0.4 * np.sin(2 * np.pi * (220 + 300 * t) * t) + 0.2 * np.sin(2 * np.pi * 880 * t) * (t > 1.0)
pcm = np.clip(np.round(sig * 32767), -32768, 32767).astype(np.int16)
with wave.open(f"{out_dir}/test.wav", "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(pcm.tobytes())
audio = pcm.astype(np.float32) / 32768.0

processor = AutoProcessor.from_pretrained(model_dir)
model = AutoModel.from_pretrained(model_dir, dtype=torch.float32, attn_implementation="eager").eval()


def embed(inputs):
    with torch.no_grad():
        out = model(**inputs).last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1).to(out.dtype)
    pooled = (out * mask).sum(1) / mask.sum(1)
    return torch.nn.functional.normalize(pooled, dim=-1).tolist()


result = {}
texts = [
    "task: search result | query: how much protein should a female eat",
    "title: none | text: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon.",
    "task: sentence similarity | query: summit define",
    "",
]
tok = processor.tokenizer(texts, padding=True, return_tensors="pt")
result["texts"] = texts
result["text_ids"] = [ids[: int(m.sum())].tolist() for ids, m in zip(tok["input_ids"], tok["attention_mask"])]
result["text"] = embed(dict(tok))
# Each text alone too (padding independence).
result["text_single"] = [embed(dict(processor.tokenizer([x], return_tensors="pt")))[0] for x in texts]

inp = processor(images=[[img]], return_tensors="pt")
result["image_ids"] = inp["input_ids"][0].tolist()
pv = inp["pixel_values"][0]
pos = inp["image_position_ids"][0]
valid = (pos != -1).all(-1)
result["image_pixels_head"] = pv[valid][:2, :12].tolist()
result["image_num_patches"] = int(valid.sum())
result["image"] = embed({k: v for k, v in inp.items() if k != "num_soft_tokens_per_image"})

# 768×768 needs no resize: isolates the vision tower from the resampler.
exact = img.resize((768, 768), Image.BICUBIC)
exact.save(f"{out_dir}/test768.png")
exact = Image.open(f"{out_dir}/test768.png").convert("RGB")
inp = processor(images=[[exact]], return_tensors="pt")
result["image_exact"] = embed({k: v for k, v in inp.items() if k != "num_soft_tokens_per_image"})

inp = processor(audio=[audio], return_tensors="pt")
result["audio_ids"] = inp["input_ids"][0].tolist()
feats = inp["input_features"][0]
fmask = inp["input_features_mask"][0]
result["audio_frames"] = int(feats.shape[0])
result["audio_valid_frames"] = int(fmask.sum())
result["audio_mel_head"] = feats[:3, :8].tolist()
result["audio_mel_mid"] = feats[100, 40:48].tolist()
result["audio"] = embed(dict(inp))

inp = processor(text=["a photo of a red circle <|image|>"], images=[[img]], return_tensors="pt")
result["mixed_ids_head"] = inp["input_ids"][0][:12].tolist()
result["mixed_len"] = int(inp["attention_mask"][0].sum())
result["mixed"] = embed({k: v for k, v in inp.items() if k != "num_soft_tokens_per_image"})

json.dump(result, open(f"{out_dir}/reference.json", "w"))
print("ok", {k: (len(v) if isinstance(v, list) else v) for k, v in result.items()})
