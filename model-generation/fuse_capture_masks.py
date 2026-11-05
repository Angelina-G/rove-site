"""Fuse two pretrained rembg masks, remove detached background islands, and package approved cutouts."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "test-images"
HUMAN = INPUT / "processed-u2net-human"
PORTRAIT = INPUT / "processed-birefnet-portrait"
OUTPUT = INPUT / "processed-approved"
OUTPUT.mkdir(parents=True, exist_ok=True)
font_path = Path("C:/Windows/Fonts/msyh.ttc")
font = ImageFont.truetype(str(font_path), 13) if font_path.exists() else ImageFont.load_default()
rows, tiles = [], []

for index, source in enumerate(sorted(INPUT.glob("*.jpg")) + sorted(INPUT.glob("*.jpeg")), 1):
    with Image.open(source) as original:
        rgba = original.convert("RGBA")
        w, h = rgba.size
    human_path = HUMAN / f"{source.stem}-raw-mask.png"
    portrait_path = PORTRAIT / f"{source.stem}-raw-mask.png"
    with Image.open(human_path) as image:
        human = np.asarray(image.convert("L"), dtype=np.float32)
    with Image.open(portrait_path) as image:
        portrait = np.asarray(image.convert("L"), dtype=np.float32)

    # Agreement fusion rejects foreground hallucinated by only one model while preserving shared soft edges.
    fused = np.sqrt((human / 255) * (portrait / 255))
    raw = np.clip(fused * 255, 0, 255).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats((raw >= 128).astype(np.uint8), 8)
    if count <= 1:
        raise RuntimeError(f"融合蒙版没有保留主体：{source.name}")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    keep = cv2.dilate((labels == largest).astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1).astype(bool)
    final_alpha = np.where(keep, raw, 0).astype(np.uint8)
    alpha = Image.fromarray(final_alpha)
    bbox = alpha.getbbox()
    if bbox is None:
        raise RuntimeError(f"融合蒙版为空：{source.name}")
    rgba.putalpha(alpha)

    foreground_ratio = float((final_alpha >= 128).mean())
    role = "面部精修参考" if foreground_ratio > .3 else "全身/转身角度候选"
    cutout = OUTPUT / f"{source.stem}-cutout.png"
    mask_path = OUTPUT / f"{source.stem}-mask.png"
    preview = OUTPUT / f"{source.stem}-gray-preview.png"
    rgba.save(cutout)
    alpha.save(mask_path)
    bg = Image.new("RGBA", (w, h), (128, 128, 128, 255))
    bg.alpha_composite(rgba)
    bg.convert("RGB").save(preview)

    border = np.zeros_like(final_alpha, dtype=np.uint8)
    thickness = max(2, round(min(w, h) * .01))
    border[:thickness, :] = border[-thickness:, :] = 1
    border[:, :thickness] = border[:, -thickness:] = 1
    border_pixels = max(1, int(border.sum()))
    border_leak = float(((final_alpha >= 128) & (border > 0)).sum() / border_pixels)
    areas_removed = int(sum(stats[k, cv2.CC_STAT_AREA] for k in range(1, count) if k != largest))

    thumb = bg.convert("RGB")
    thumb.thumbnail((260, 360), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (280, 405), "white")
    tile.paste(thumb, ((280 - thumb.width) // 2, 5))
    warning = "· 边缘需补拍" if border_leak > .05 else ""
    ImageDraw.Draw(tile).text((6, 370), f"{index}. {source.stem[:13]}\n{role} · edge {border_leak:.1%} {warning}", fill="#17241f", font=font)
    tiles.append(tile)

    rows.append({
        "source": source.name,
        "role": role,
        "mask_fusion": "sqrt(u2net_human_seg * birefnet-portrait)",
        "size": [w, h],
        "bbox": list(bbox),
        "foreground_ratio": round(foreground_ratio, 4),
        "foreground_on_outer_border_ratio": round(border_leak, 4),
        "touches_frame_edge": border_leak > .05,
        "removed_detached_components_pixels": areas_removed,
        "cutout": cutout.name,
        "mask": mask_path.name,
        "gray_preview": preview.name,
    })

sheet = Image.new("RGB", (1120, ((len(tiles) + 3) // 4) * 405), "#dfe4df")
for i, tile in enumerate(tiles):
    sheet.paste(tile, ((i % 4) * 280, (i // 4) * 405))
sheet.save(OUTPUT / "qc-contact-sheet.jpg", quality=92)
(OUTPUT / "quality-report.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"已生成 {len(rows)} 张融合蒙版及质量检查表：{OUTPUT}")
