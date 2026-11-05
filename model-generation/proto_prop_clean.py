"""Prototype: remove handheld cloth from a cutout using hand (skin) anchors.

Idea
----
The only thing the character *should* keep below the wrist is the hand itself, and
hands are skin-coloured in these photos while the prop cloth is not.  So:
  1. find skin clusters in the lower half of the subject (hands),
  2. for each hand, take a window that covers the hanging cloth,
  3. inside that window delete mask pixels that are not skin (and not a few px
     around skin), i.e. the cloth,
  4. close + keep significant components.

This is a prototype for visual inspection only; it writes previews to _work/.
Usage: python proto_prop_clean.py <stem> [<stem> ...]
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
APP = ROOT / "test-images" / "processed-v2"
CACHE = APP / "_work" / "cache"
OUT = APP / "_work" / "prop-clean"


def skin_mask(rgb: np.ndarray) -> np.ndarray:
    """YCbCr + RGB rule-of-thumb skin detector (no DL, tuned for these photos)."""
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    m1 = (cr >= 133) & (cr <= 180) & (cb >= 70) & (cb <= 135) & (y >= 60)
    r, g, b = rgb[..., 0].astype(int), rgb[..., 1].astype(int), rgb[..., 2].astype(int)
    m2 = (r > 80) & (g > 30) & (b > 15) & (r > g) & (r > b) & ((r - g) > 8)
    m = (m1 & m2).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return m.astype(bool)


def clean_one(stem: str, debug=True):
    rgb = np.asarray(Image.open(ROOT / "test-images" / f"{stem}.jpg").convert("RGB"))
    alpha = np.load(CACHE / stem / "fused-alpha.npy")
    mask = alpha >= 128
    h, w = mask.shape
    skin = skin_mask(rgb)

    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    span = y1 - y0
    skin_low = skin.copy()
    skin_low[: y0 + int(0.45 * span), :] = False

    n, labels, stats, cents = cv2.connectedComponentsWithStats(skin_low.astype(np.uint8), 8)
    cand = []
    for k in range(1, n):
        area = stats[k, cv2.CC_STAT_AREA]
        if area < 800:
            continue
        cx, cy = cents[k]
        side = 0 if cx < w / 2 else 1
        cand.append((area, side, k))
    # 每侧只取最大的一个皮肤连通块（≈手），避免把腿/脸当手
    hands = []
    for side in (0, 1):
        best = max((c for c in cand if c[1] == side), default=None, key=lambda c: c[0])
        if not best:
            continue
        k = best[2]
        hands.append((int(stats[k, cv2.CC_STAT_LEFT]), int(stats[k, cv2.CC_STAT_TOP]),
                      int(stats[k, cv2.CC_STAT_WIDTH]), int(stats[k, cv2.CC_STAT_HEIGHT]), int(best[0])))

    # 身体主体：腐蚀后重建，保住裙子/腿的轮廓，只把“挂在身体外面的附件”当作可删区域
    r = max(9, int(0.012 * min(h, w)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    eroded = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
    core = eroded.copy()
    while True:
        grown = cv2.dilate(core.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & mask
        if grown.sum() == core.sum():
            break
        core = grown

    remove = np.zeros_like(mask)
    skin_keep = cv2.dilate(skin.astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)
    for (hx, hy, hw, hh, area) in hands:
        x0 = max(0, hx - int(1.2 * hw))
        x1 = min(w, hx + hw + int(1.2 * hw))
        yy0 = max(0, hy - int(0.4 * hh))
        yy1 = min(h, hy + hh + int(2.6 * hh))
        win = np.zeros_like(mask)
        win[yy0:yy1, x0:x1] = True
        remove |= win & mask & (~skin_keep) & (~core)

    new = mask & (~remove)
    new = cv2.morphologyEx(new.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)).astype(bool)
    # keep main body + any sizeable component
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(new.astype(np.uint8), 8)
    if n2 > 1:
        areas = st2[1:, cv2.CC_STAT_AREA]
        biggest = 1 + int(np.argmax(areas))
        keep = lab2 == biggest
        for k in range(1, n2):
            if k != biggest and st2[k, cv2.CC_STAT_AREA] >= 1500:
                keep |= lab2 == k
        new = keep
    new = cv2.morphologyEx(new.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)).astype(bool)

    print(f"{stem[:8]}: hands={len(hands)} removed_px={int(remove.sum())} "
          f"mask_before={int(mask.sum())} after={int(new.sum())}")

    if debug:
        OUT.mkdir(parents=True, exist_ok=True)
        vis = rgb.copy()
        tint = np.zeros_like(vis)
        tint[remove & mask] = (255, 0, 0)
        tint[skin & mask] = (0, 160, 255)
        vis = cv2.addWeighted(vis, 0.72, tint, 0.55, 0)
        cut = np.full_like(vis, 128)
        a3 = np.repeat(new[..., None], 3, axis=2)
        cut = np.where(a3, vis, cut)
        comp = np.concatenate([vis, cut], axis=1)
        Image.fromarray(comp).save(OUT / f"{stem}-clean-check.jpg", quality=90)
    return new


if __name__ == "__main__":
    stems = sys.argv[1:] or ["9ca2f019a28896cbef7fc86359863ddc"]
    for s in stems:
        clean_one(s)
