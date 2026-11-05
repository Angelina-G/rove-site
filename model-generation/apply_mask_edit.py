"""Apply a hand-edited alpha mask and atomically refresh derived QC files."""
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "test-images"


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: apply_mask_edit.py <version> <stem> <mask.png>", file=sys.stderr)
        return 2
    version, stem, mask_path = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    src = SRC / f"{stem}.jpg"
    if not src.is_file():
        print(f"source photo missing: {src}", file=sys.stderr)
        return 3
    out_dir = SRC / version
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb = Image.open(src).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if mask.size != rgb.size:
        mask = mask.resize(rgb.size, Image.Resampling.LANCZOS)
    binary = (np.asarray(mask, dtype=np.uint8) >= 128).astype(np.uint8) * 255
    blurred = np.asarray(Image.fromarray(binary).filter(ImageFilter.GaussianBlur(1.2)), dtype=np.uint8)
    soft = np.minimum(blurred, binary)

    rgba = rgb.convert("RGBA")
    rgba.putalpha(Image.fromarray(soft))
    gray = Image.new("RGBA", rgba.size, (128, 128, 128, 255))
    gray.alpha_composite(rgba)

    suffix = f".tmp-{os.getpid()}"
    targets = {
        "cutout": out_dir / f"{stem}-cutout.png",
        "mask": out_dir / f"{stem}-mask.png",
        "gray": out_dir / f"{stem}-gray-preview.png",
    }
    temps = {key: path.with_name(path.name + suffix) for key, path in targets.items()}
    backups = {key: path.with_name(path.name + f".bak-{os.getpid()}") for key, path in targets.items()}
    replaced = []
    try:
        rgba.save(temps["cutout"], format="PNG")
        Image.fromarray(soft).save(temps["mask"], format="PNG")
        gray.convert("RGB").save(temps["gray"], format="PNG", quality=95)
        for key, target in targets.items():
            backup = backups[key]
            if target.exists():
                os.replace(target, backup)
            os.replace(temps[key], target)
            replaced.append(key)
        for backup in backups.values():
            backup.unlink(missing_ok=True)
    except Exception:
        for key in replaced:
            targets[key].unlink(missing_ok=True)
        for key, backup in backups.items():
            if backup.exists():
                os.replace(backup, targets[key])
        for temp in temps.values():
            temp.unlink(missing_ok=True)
        raise
    finally:
        for temp in temps.values():
            temp.unlink(missing_ok=True)

    kept = int((soft >= 128).sum())
    print(f"APPLIED version={version} stem={stem} kept_px={kept}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
