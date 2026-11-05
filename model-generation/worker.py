import os
import sys
import subprocess
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from rembg import remove, new_session

root = Path(__file__).resolve().parent
source = root / "source" / "TripoSR"
weights = root / "weights"
input_path = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
output_dir.mkdir(parents=True, exist_ok=True)
(output_dir / "0").mkdir(parents=True, exist_ok=True)

# Segment the person, crop to its alpha bounds, then composite onto neutral gray.
print("ROVE_STAGE:segment", flush=True)
human_session = new_session("u2net_human_seg")
portrait_session = new_session("birefnet-portrait")
with Image.open(input_path) as uploaded:
    if uploaded.mode == "RGBA" and uploaded.getchannel("A").getextrema()[0] < 255:
        rgba = uploaded.convert("RGBA")
        human_alpha = portrait_alpha = np.asarray(rgba.getchannel("A"), dtype=np.float32)
    else:
        rgb = uploaded.convert("RGB")
        human_alpha = np.asarray(remove(rgb, session=human_session).getchannel("A"), dtype=np.float32)
        portrait_alpha = np.asarray(remove(rgb, session=portrait_session).getchannel("A"), dtype=np.float32)
        rgba = Image.new("RGBA", rgb.size)
        rgba.paste(rgb)
Image.fromarray(human_alpha.clip(0, 255).astype(np.uint8)).save(output_dir / "subject-mask-human.png")
Image.fromarray(portrait_alpha.clip(0, 255).astype(np.uint8)).save(output_dir / "subject-mask-portrait.png")
fused_alpha = np.sqrt((human_alpha / 255) * (portrait_alpha / 255)) * 255
alpha = Image.fromarray(fused_alpha.clip(0, 255).astype(np.uint8))
alpha.save(output_dir / "subject-mask-fused.png")
rgba.putalpha(alpha)
bounds = alpha.getbbox()
if not bounds:
    raise RuntimeError("未检测到人物主体；请换一张主体清晰的全身图。")
alpha_array = np.asarray(alpha)
count, labels, stats, _ = cv2.connectedComponentsWithStats((alpha_array >= 128).astype(np.uint8), 8)
if count <= 1:
    raise RuntimeError("人体分割未检出清晰前景；已停止生成，避免把背景送入模型。")
largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
keep = cv2.dilate((labels == largest).astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1).astype(bool)
alpha_array = np.where(keep, alpha_array, 0).astype(np.uint8)
rgba.putalpha(Image.fromarray(alpha_array))
rgba.getchannel("A").save(output_dir / "subject-mask.png")
bounds = rgba.getchannel("A").getbbox()
rgba = rgba.crop(bounds)
rgba.thumbnail((430, 430), Image.Resampling.LANCZOS)
rgba.save(output_dir / "subject-cutout.png")
print("ROVE_STAGE:prepare", flush=True)
canvas = Image.new("RGBA", (512, 512), (128, 128, 128, 255))
canvas.alpha_composite(rgba, ((512-rgba.width)//2, (512-rgba.height)//2))
prepared = output_dir / "input-prepared.png"
canvas.convert("RGB").save(output_dir / "model-input.png")
canvas.convert("RGB").save(prepared)

env = os.environ.copy()
env["HF_HOME"] = str(root / "cache" / "huggingface")
args = [sys.executable, str(source / "run.py"), str(prepared),
        "--pretrained-model-name-or-path", str(weights),
        "--output-dir", str(output_dir), "--model-save-format", "glb",
        "--chunk-size", "2048", "--mc-resolution", "256", "--no-remove-bg"]
print("ROVE_STAGE:geometry", flush=True)
try:
    subprocess.run(args, cwd=source, env=env, check=True)
finally:
    prepared.unlink(missing_ok=True)
