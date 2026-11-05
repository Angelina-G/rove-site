"""Batch-cut out newly captured JPGs and write local QC artifacts."""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rembg import new_session, remove

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "test-images"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "u2net_human_seg"
OUTPUT = INPUT / f"processed-{MODEL.replace('_', '-')}"
OUTPUT.mkdir(parents=True, exist_ok=True)

files = sorted(INPUT.glob("*.jpg")) + sorted(INPUT.glob("*.jpeg"))
if not files:
    raise SystemExit("test-images 根目录没有 JPG/JPEG 真人照片。")

print(f"加载分割模型：{MODEL}", flush=True)
session = new_session(MODEL)
report = []
tiles = []
font_path = Path("C:/Windows/Fonts/msyh.ttc")
font = ImageFont.truetype(str(font_path), 13) if font_path.exists() else ImageFont.load_default()

for index, source in enumerate(files, 1):
    started = time.perf_counter()
    with Image.open(source) as image:
        original = image.convert("RGB")
        rgba = remove(original, session=session).convert("RGBA")
    raw_alpha = rgba.getchannel("A")
    bbox = raw_alpha.getbbox()
    if bbox is None:
        raise RuntimeError(f"第 {index} 张没有分出前景：{source.name}")

    alpha_array = np.asarray(raw_alpha)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats((alpha_array >= 128).astype(np.uint8), 8)
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    largest = (labels == largest_label).astype(np.uint8)
    keep = cv2.dilate(largest, np.ones((7, 7), np.uint8), iterations=1).astype(bool)
    alpha_array = np.where(keep, alpha_array, 0).astype(np.uint8)
    rgba.putalpha(Image.fromarray(alpha_array))
    mask = rgba.getchannel("A")
    bbox = mask.getbbox()
    raw_mask_path = OUTPUT / f"{source.stem}-raw-mask.png"
    raw_alpha.save(raw_mask_path)
    cutout_path = OUTPUT / f"{source.stem}-cutout.png"
    mask_path = OUTPUT / f"{source.stem}-mask.png"
    preview_path = OUTPUT / f"{source.stem}-gray-preview.png"
    rgba.save(cutout_path)
    mask.save(mask_path)

    gray = Image.new("RGBA", rgba.size, (128, 128, 128, 255))
    gray.alpha_composite(rgba)
    gray.convert("RGB").save(preview_path, quality=95)

    pixels = list(mask.resize((256, 256), Image.Resampling.NEAREST).getdata())
    soft = sum(0 < value < 255 for value in pixels) / len(pixels)
    foreground = sum(value >= 128 for value in pixels) / len(pixels)
    w, h = rgba.size
    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    view_role = "面部精修参考" if foreground > .3 else "全身/转身角度候选"
    border = Image.new("L", rgba.size, 0)
    d = ImageDraw.Draw(border)
    d.rectangle((0, 0, w - 1, h - 1), outline=255, width=max(2, round(min(w, h) * .01)))
    border_px = list(border.resize((256, 256), Image.Resampling.NEAREST).getdata())
    border_fg = sum(1 for a, b in zip(pixels, border_px) if a >= 128 and b > 0) / max(1, sum(1 for b in border_px if b > 0))
    raw_small_components = int(sum(stats[label, cv2.CC_STAT_AREA] for label in range(1, component_count) if label != largest_label))

    thumb = gray.convert("RGB")
    thumb.thumbnail((260, 360), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (280, 405), "white")
    tile.paste(thumb, ((280 - thumb.width) // 2, 5))
    caption = f"{index}. {source.name[:13]}\n{view_role} · fg {foreground:.1%} · edge {border_fg:.1%}"
    ImageDraw.Draw(tile).text((6, 370), caption, fill="#17241f", font=font)
    tiles.append(tile)

    row = {
        "source": source.name,
        "model": MODEL,
        "size": [w, h],
        "bbox": list(bbox),
        "suggested_role": view_role,
        "foreground_ratio_256": round(foreground, 4),
        "soft_alpha_ratio_256": round(soft, 4),
        "foreground_on_outer_border_ratio_256": round(border_fg, 4),
        "discarded_non_person_components_px": raw_small_components,
        "raw_component_count": component_count - 1,
        "cutout": cutout_path.name,
        "mask": mask_path.name,
        "raw_mask": raw_mask_path.name,
        "gray_preview": preview_path.name,
        "seconds": round(time.perf_counter() - started, 2),
    }
    report.append(row)
    print(f"[{index}/{len(files)}] {source.name}: fg={foreground:.1%}, soft={soft:.1%}, border={border_fg:.1%}", flush=True)

cols = 4
sheet = Image.new("RGB", (cols * 280, ((len(tiles) + cols - 1) // cols) * 405), "#dfe4df")
for i, tile in enumerate(tiles):
    sheet.paste(tile, ((i % cols) * 280, (i // cols) * 405))
sheet.save(OUTPUT / "qc-contact-sheet.jpg", quality=92)
(OUTPUT / "quality-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"已保存 {len(files)} 张透明抠图、mask、灰底预览和质量报告：{OUTPUT}", flush=True)
