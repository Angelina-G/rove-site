"""Keep face-detail captures separate and generate portrait cutouts for review."""
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rembg import new_session, remove

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "test-images" / "face-detail" / "source"
OUTPUT = ROOT / "test-images" / "face-detail" / "processed"
OUTPUT.mkdir(parents=True, exist_ok=True)
MODEL = "birefnet-portrait"

files = sorted(INPUT.glob("*.jpg")) + sorted(INPUT.glob("*.jpeg"))
if not files:
    raise SystemExit(f"面部素材目录没有 JPG/JPEG：{INPUT}")

session = new_session(MODEL)
font_path = Path("C:/Windows/Fonts/msyh.ttc")
font = ImageFont.truetype(str(font_path), 15) if font_path.exists() else ImageFont.load_default()
rows, tiles = [], []
for index, source in enumerate(files, 1):
    start = time.perf_counter()
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        rgba = remove(rgb, session=session).convert("RGBA")
    alpha = rgba.getchannel("A")
    if alpha.getbbox() is None:
        raise RuntimeError(f"没有识别到人像：{source.name}")
    cutout = OUTPUT / f"{source.stem}-cutout.png"
    mask_path = OUTPUT / f"{source.stem}-mask.png"
    preview = OUTPUT / f"{source.stem}-preview.jpg"
    rgba.save(cutout)
    alpha.save(mask_path)
    bg = Image.new("RGBA", rgba.size, (128, 128, 128, 255))
    bg.alpha_composite(rgba)
    bg.convert("RGB").save(preview, quality=94)
    thumb = bg.convert("RGB")
    thumb.thumbnail((320, 360), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (340, 405), "white")
    tile.paste(thumb, ((340 - thumb.width) // 2, 4))
    ImageDraw.Draw(tile).text((8, 371), f"{index}. {source.stem[:14]}", font=font, fill="#17241f")
    tiles.append(tile)
    rows.append({"source": source.name, "model": MODEL, "size": list(rgb.size),
                 "cutout": cutout.name, "mask": mask_path.name,
                 "preview": preview.name, "seconds": round(time.perf_counter() - start, 2)})

sheet = Image.new("RGB", (340 * 3, 405 * ((len(tiles) + 2) // 3)), "#dfe4df")
for i, tile in enumerate(tiles):
    sheet.paste(tile, ((i % 3) * 340, (i // 3) * 405))
sheet.save(OUTPUT / "face-detail-contact-sheet.jpg", quality=92)
(OUTPUT / "quality-report.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"已整理并处理 {len(rows)} 张面部角度参考图：{OUTPUT}")
