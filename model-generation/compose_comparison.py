"""Compose model preview sheets into one comparison image."""
import json
import sys
from pathlib import Path
from PIL import Image, ImageDraw


def main():
    if len(sys.argv) != 3:
        print('usage: compose_comparison.py <jobs-root> <comparison-dir>', file=sys.stderr)
        return 2
    jobs_root, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    previews = []
    for job in sorted(jobs_root.iterdir()):
        if not job.is_dir() or job.name.startswith('.'):
            continue
        images = sorted(job.glob('*-preview.png'))
        if images:
            previews.append((job.name, max(images, key=lambda p: p.stat().st_mtime_ns)))
    if not previews:
        raise RuntimeError('没有可用的模型预览图；请先为至少一个 GLB 生成 *-preview.png。')
    tile_w, tile_h = 720, 180
    sheet = Image.new('RGB', (tile_w, tile_h * len(previews)), (20, 24, 30))
    draw = ImageDraw.Draw(sheet)
    manifest = []
    for i, (name, source) in enumerate(previews):
        img = Image.open(source).convert('RGB')
        img.thumbnail((tile_w - 12, tile_h - 28))
        x, y = 6, i * tile_h + 24
        sheet.paste(img, (x, y))
        draw.text((8, i * tile_h + 5), name, fill=(235, 240, 248))
        manifest.append({'id': name, 'label': name, 'previewUrl': f'/api/jobs/{name}/files/{source.name}', 'previewMtimeNs': source.stat().st_mtime_ns})
    output = out_dir / 'contact-sheet.png'
    sheet.save(output)
    (out_dir / 'manifest.json').write_text(json.dumps({'models': manifest, 'contactSheet': output.name}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'COMPARISON_OK {output} models={len(manifest)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
