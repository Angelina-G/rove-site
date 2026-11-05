"""Prototype: erase handheld cloth by inpainting *before* segmentation.

Why: cutting the cloth out of the alpha mask leaves rectangular holes wherever the
cloth overlapped the skirt.  Inpainting the cloth away first, then letting rembg
segment the cleaned photo, avoids holes and gives the 3D model a plausible silhouette.

Usage: python proto_inpaint_props.py <stem> [<stem> ...]
Outputs go to test-images/processed-v2/_work/inpaint/<stem>-*.jpg
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "test-images"
CACHE = SRC / "processed-v2" / "_work" / "cache"
OUT = SRC / "processed-v2" / "_work" / "inpaint"


def skin_mask(rgb):
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    m1 = (cr >= 133) & (cr <= 180) & (cb >= 70) & (cb <= 135) & (y >= 60)
    r, g, b = rgb[..., 0].astype(int), rgb[..., 1].astype(int), rgb[..., 2].astype(int)
    m2 = (r > 80) & (g > 30) & (b > 15) & (r > g) & (r > b) & ((r - g) > 8)
    m = (m1 & m2).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return m.astype(bool)


def prop_region(rgb, mask):
    h, w = mask.shape
    skin = skin_mask(rgb)
    ys, _ = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    span = y1 - y0
    low = skin.copy()
    low[: y0 + int(0.45 * span), :] = False
    n, _, stats, cents = cv2.connectedComponentsWithStats(low.astype(np.uint8), 8)
    hands = []
    for side in (0, 1):
        best, best_area = None, 0
        for k in range(1, n):
            area = stats[k, cv2.CC_STAT_AREA]
            cx, _ = cents[k]
            if (cx < w / 2) != (side == 0) or area < 800 or area <= best_area:
                continue
            best, best_area = k, area
        if best is not None:
            hands.append((int(stats[best, cv2.CC_STAT_LEFT]), int(stats[best, cv2.CC_STAT_TOP]),
                          int(stats[best, cv2.CC_STAT_WIDTH]), int(stats[best, cv2.CC_STAT_HEIGHT])))
    remove = np.zeros_like(mask, dtype=np.uint8)
    skin_keep = cv2.dilate(skin.astype(np.uint8), np.ones((13, 13), np.uint8)).astype(bool)
    for (hx, hy, hw, hh) in hands:
        x0 = max(0, hx - int(1.2 * hw))
        x1 = min(w, hx + hw + int(1.2 * hw))
        yy0 = max(0, hy - int(0.4 * hh))
        yy1 = min(h, hy + hh + int(2.7 * hh))
        win = np.zeros_like(mask)
        win[yy0:yy1, x0:x1] = True
        remove |= (win & mask & ~skin_keep).astype(np.uint8)
    remove = cv2.dilate(remove, np.ones((9, 9), np.uint8))
    return remove.astype(bool), hands


def main(stems):
    OUT.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        rgb = np.asarray(Image.open(SRC / f"{stem}.jpg").convert("RGB"))
        alpha = np.load(CACHE / stem / "fused-alpha.npy")
        mask = alpha >= 128
        remove, hands = prop_region(rgb, mask)
        inpainted = cv2.inpaint(rgb, remove.astype(np.uint8) * 255, 7, cv2.INPAINT_TELEA)
        Image.fromarray(inpainted).save(OUT / f"{stem}-inpainted.jpg", quality=94)
        vis = rgb.copy()
        vis[remove] = (255, 0, 255)
        comp = np.concatenate([vis, inpainted], axis=1)
        comp = cv2.resize(comp, (comp.shape[1] // 2, comp.shape[0] // 2))
        Image.fromarray(comp).save(OUT / f"{stem}-compare.jpg", quality=90)
        print(f"{stem[:8]}: hands={len(hands)} inpaint_px={int(remove.sum())}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["9ca2f019a28896cbef7fc86359863ddc"])
