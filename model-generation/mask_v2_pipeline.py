"""Rove capture QC v2: human-parsing-guided mask repair for the 10 test photos.

Stages (cached in test-images/processed-v2/_work/cache):
  masks  -> per-model rembg alpha + segformer-clothes parsing maps
  fuse   -> repaired alpha per image (rules, fast to iterate)
  qc     -> processed-v2 deliverables, zoom crops, contact sheet, manifest

Usage:
  python mask_v2_pipeline.py masks
  python mask_v2_pipeline.py fuse
  python mask_v2_pipeline.py qc
  python mask_v2_pipeline.py all
"""
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "test-images"
OUTPUT = INPUT / "processed-v2"
WORK = OUTPUT / "_work"
CACHE = WORK / "cache"
os.environ.setdefault("U2NET_HOME", str(ROOT / "cache" / "u2net"))

PARSE_DIR = ROOT / "cache" / "huggingface" / "segformer-b2-clothes"
CLASSES = [
    "Background", "Hat", "Hair", "Sunglasses", "Upper-clothes", "Skirt", "Pants", "Dress",
    "Belt", "Left-shoe", "Right-shoe", "Face", "Left-leg", "Right-leg", "Left-arm",
    "Right-arm", "Bag", "Scarf",
]
C = {name: i for i, name in enumerate(CLASSES)}
PALETTE = np.array([
    [0, 0, 0], [255, 215, 0], [160, 82, 45], [0, 255, 255], [0, 128, 255], [255, 0, 255],
    [128, 0, 255], [255, 105, 180], [255, 255, 0], [0, 200, 0], [0, 255, 120],
    [255, 200, 150], [255, 0, 0], [200, 0, 0], [0, 100, 255], [0, 60, 200],
    [128, 128, 0], [255, 140, 0],
], dtype=np.uint8)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

REMBG_MODELS = ["u2net_human_seg", "birefnet-portrait", "isnet-general-use", "birefnet-general"]


def sources():
    return sorted(INPUT.glob("*.jpg")) + sorted(INPUT.glob("*.jpeg"))


def image_size():
    with Image.open(sources()[0]) as im:
        return im.size


# ---------------------------------------------------------------- stage: masks
def load_parse_session():
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    so.log_severity_level = 3
    return ort.InferenceSession(str(PARSE_DIR / "onnx-model.onnx"), sess_options=so,
                                providers=["CPUExecutionProvider"])


def parse_person(sess, rgb: Image.Image, box, short_side=512):
    """Segformer clothes parsing on a person crop; returns full-size class map + confidence."""
    w, h = rgb.size
    crop = rgb.crop(box)
    cw, ch = crop.size
    scale = short_side / min(cw, ch)
    nw = max(64, int(round(cw * scale)))
    nh = max(64, int(round(ch * scale)))
    nw -= nw % 32
    nh -= nh % 32
    arr = np.asarray(crop.resize((nw, nh), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    logits = np.asarray(sess.run(None, {"pixel_values": arr.transpose(2, 0, 1)[None].astype(np.float32)})[0][0],
                        dtype=np.float32)
    prob = np.exp(logits - logits.max(0, keepdims=True))
    prob /= prob.sum(0, keepdims=True)
    # per-class probability maps upsampled to crop resolution (uint8 keeps memory sane)
    prob_crop = np.empty((len(CLASSES), ch, cw), dtype=np.uint8)
    for c in range(len(CLASSES)):
        prob_crop[c] = np.clip(np.asarray(Image.fromarray(prob[c]).resize((cw, ch), Image.Resampling.BILINEAR)) * 255, 0, 255)
    cls_crop = np.argmax(prob_crop, axis=0).astype(np.uint8)
    conf_crop = prob_crop.max(axis=0)
    cls_full = np.zeros((h, w), dtype=np.uint8)
    conf_full = np.zeros((h, w), dtype=np.uint8)
    cls_full[box[1]:box[3], box[0]:box[2]] = cls_crop
    conf_full[box[1]:box[3], box[0]:box[2]] = conf_crop
    return cls_full, conf_full, prob_crop


def stage_masks(force=False):
    from rembg import new_session, remove
    CACHE.mkdir(parents=True, exist_ok=True)
    parse_sess = load_parse_session()
    sessions = {}
    for m in REMBG_MODELS:
        if m == "birefnet-general" and not (ROOT / "cache" / "u2net" / "birefnet-general.onnx").exists():
            print(f"跳过 {m}：权重尚未下载完成", flush=True)
            continue
        t = time.perf_counter()
        sessions[m] = new_session(m)
        print(f"session {m} {time.perf_counter()-t:.1f}s", flush=True)
    for index, src in enumerate(sources(), 1):
        stem = src.stem
        d = CACHE / stem
        d.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as im:
            rgb = im.convert("RGB")
        w, h = rgb.size
        if force or not (d / "u2net_human_seg.png").exists():
            for name, sess in sessions.items():
                t = time.perf_counter()
                alpha = remove(rgb, session=sess, only_mask=True, post_process_mask=False).convert("L")
                alpha.save(d / f"{name}.png")
                print(f"[{index}/10] {stem[:4]} {name} {time.perf_counter()-t:.1f}s", flush=True)
        if force or not (d / "parse-cls.npy").exists():
            t = time.perf_counter()
            with Image.open(d / "u2net_human_seg.png") as m:
                a = np.asarray(m)
            ys, xs = np.where(a > 20)
            pad = 40
            box = (int(max(0, xs.min() - pad)), int(max(0, ys.min() - pad)),
                   int(min(w, xs.max() + pad)), int(min(h, ys.max() + pad)))
            cls_full, conf_full, prob_crop = parse_person(parse_sess, rgb, box)
            np.save(d / "parse-cls.npy", cls_full)
            np.save(d / "parse-conf.npy", conf_full)
            np.savez_compressed(d / "parse-prob.npz", prob=prob_crop, box=np.array(box))
            print(f"[{index}/10] {stem[:4]} parse {time.perf_counter()-t:.1f}s box={box}", flush=True)


# --------------------------------------------------------------- image config
# 人工确认过的图级配置：angle 供多视图生成选图；prop_rois 是人工标注的道具区域
# （用户红标处：手持格纹布 / 灰色布料袋），单位是原图像素，多边形点序列。
#
# 2026-11-05 尝试过用矩形 ROI 去掉手持布料，结果失败并已回退：
#   - 格纹布在图像上与裙子重叠，且 segformer 把它判成 Skirt/Dress，矩形 ROI
#     会把裙子一起切掉，导致蒙版出现矩形缺口（processed-v2 因此回退到无 ROI 版本）。
#   - 杯子/布料袋在部分视角被判成 Background/Bag，矩形仍然会覆盖腿部。
# 正确做法：逐像素标注多边形（或接 SAM/手部-道具分割模型），先只删布料像素，
# 再重跑 `mask_v2_pipeline.py fuse` 与 `qc`，最后重跑 multiview_worker。
IMAGE_CONFIG = {
    "01139015a4830e26611025860662f5c9": {"angle": "back", "primary": True, "prop_rois": []},
    "34f5400320cf3f25383d7901751a14dd": {"angle": "front-left", "primary": False, "prop_rois": []},
    "3c723083098d0d1812a9bccc238db489": {"angle": "face-detail", "primary": True, "prop_rois": []},
    "458df2ea1d31efc1b1fc6d787eef11d3": {"angle": "back", "primary": False, "prop_rois": []},
    "52a9f704b1daa15643d7e6d0d13e16d0": {"angle": "left", "primary": True, "prop_rois": []},
    "697021aea8e185fa59c60203cc0d5232": {"angle": "back", "primary": True, "prop_rois": []},
    "9ca2f019a28896cbef7fc86359863ddc": {"angle": "front", "primary": True, "prop_rois": []},
    "9e01afc1d631e3f3d42de46bd77545ac": {"angle": "left", "primary": False, "prop_rois": []},
    "f33c6a450c352daf3cfba6552bf350d9": {"angle": "front", "primary": False, "prop_rois": []},
    "fc555d3e063b0ee9d2b0e5e7a7124353": {"angle": "right", "primary": True, "prop_rois": []},
}

PERSON_CLASSES = [
    "Hat", "Hair", "Sunglasses", "Upper-clothes", "Skirt", "Pants", "Dress", "Belt",
    "Left-shoe", "Right-shoe", "Face", "Left-leg", "Right-leg", "Left-arm", "Right-arm",
]
TORSO_CLASSES = ["Upper-clothes", "Skirt", "Dress", "Pants", "Belt", "Face", "Hair"]
LEG_CLASSES = ["Left-leg", "Right-leg", "Left-shoe", "Right-shoe", "Pants"]
SKIN_CLASSES = ["Face", "Left-leg", "Right-leg", "Left-arm", "Right-arm"]


def _load_rgb(stem):
    with Image.open(INPUT / f"{stem}.jpg") as im:
        return im.convert("RGB")


def _load_alpha(stem, model):
    p = CACHE / stem / f"{model}.png"
    if not p.exists():
        return None
    with Image.open(p) as im:
        return np.asarray(im.convert("L"), dtype=np.float32) / 255.0


def _parse_maps(stem):
    d = CACHE / stem
    cls = np.load(d / "parse-cls.npy")
    conf = np.load(d / "parse-conf.npy").astype(np.float32) / 255.0
    return cls, conf


def _largest_cc(binary):
    binary = binary.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if n <= 1:
        return np.zeros_like(binary, dtype=bool), labels, stats, n
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == idx, labels, stats, n


def _fill_holes(binary, max_ratio=0.02):
    """Fill interior holes; keeps the leg gap (open to the bottom) untouched."""
    inv = (~binary).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, 4)
    filled = binary.copy()
    total = binary.sum()
    h, w = binary.shape
    for k in range(1, n):
        x, y, bw, bh, area = stats[k]
        if x == 0 or y == 0 or x + bw >= w or y + bh >= h:
            continue  # 与画面边界相连 = 外部背景
        if area <= max_ratio * total:
            filled[labels == k] = True
    return filled


def build_parse_prior(cls, conf, min_conf=0.30):
    """人体先验：只保留与躯干核心连通的人体部位，排除背景/手持道具/Bag。"""
    known = np.isin(cls, [C[name] for name in PERSON_CLASSES])
    prior = known & (conf >= min_conf)
    torso = np.isin(cls, [C[name] for name in TORSO_CLASSES]) & (conf >= 0.45)
    core, _, _, _ = _largest_cc(torso)
    if not core.any():
        core = prior
    # 连通性扩展：保留与躯干核心相连的人体部位（臂/腿/鞋），丢弃漂浮布料
    grow = core.copy()
    for _ in range(24):
        grown = cv2.dilate(grow.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        nxt = grow | (prior & grown)
        if nxt.sum() == grow.sum():
            break
        grow = nxt
    prior = grow & prior
    prior = _fill_holes(prior, max_ratio=0.004)
    return prior, core


def fuse_image(stem):
    cfg = IMAGE_CONFIG.get(stem, {})
    rgb = _load_rgb(stem)
    w, h = rgb.size
    alphas = {}
    for model in ["u2net_human_seg", "birefnet-portrait", "isnet-general-use", "birefnet-general"]:
        a = _load_alpha(stem, model)
        if a is not None:
            alphas[model] = a
    if not alphas:
        raise RuntimeError(f"{stem} 没有缓存任何 rembg 蒙版")
    cls, conf = _parse_maps(stem)
    prior, core = build_parse_prior(cls, conf)

    stack = np.stack(list(alphas.values()), axis=0)
    base = np.median(stack, axis=0) if stack.shape[0] >= 3 else np.mean(stack, axis=0)

    # 1) 人体解析先验做硬门控：背景布料/道具（解析判为背景或 Bag）被剔除
    gate = cv2.GaussianBlur(prior.astype(np.float32), (0, 0), 2.0) > 0.45
    alpha = base * gate

    # 2) 补回解析模型确认的腿/脚/鞋（用户绿标：9ca2 左腿+鞋缺失）
    #    注意：两条腿在解析图里通常是彼此分离的两块，不能只取“最大连通块”，
    #    否则会稳定丢掉一条腿（2026-11-05 实测 9ca2 左腿被丢掉就是这个原因）。
    leg_region = np.isin(cls, [C[name] for name in LEG_CLASSES]) & (conf >= 0.40)
    n_leg, leg_labels, leg_stats, _ = cv2.connectedComponentsWithStats(leg_region.astype(np.uint8), 8)
    leg_keep = np.zeros_like(leg_region)
    for k in range(1, n_leg):
        if leg_stats[k, cv2.CC_STAT_AREA] >= 300:
            leg_keep |= leg_labels == k
    leg_region = leg_keep
    recover = leg_region & (alpha < 0.35)
    recover = cv2.morphologyEx(recover.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)).astype(bool)
    recovered_px = int(recover.sum())
    alpha = np.where(recover, np.maximum(alpha, 0.98), alpha)

    # 3) 硬二值化 + 保留主体连通域（去掉地面鞋/背景残片等独立小块）
    binary = alpha >= 0.5
    binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)).astype(bool)
    binary = _fill_holes(binary, max_ratio=0.01)
    main, labels, stats, n = _largest_cc(binary)
    # 允许保留其它与主体重叠足够多的人体部位块（例如被解析割断的脚），但面积必须可观
    keep = main.copy()
    main_area = max(1, int(main.sum()))
    for k in range(1, n):
        area = stats[k, cv2.CC_STAT_AREA]
        if area < 0.002 * main_area:
            continue
        comp = labels == k
        if (comp & core).sum() / area > 0.05 or (comp & prior).sum() / area > 0.6:
            keep |= comp
    binary = keep

    # 4) 手工道具 ROI（人工标注，仅用于模型无法区分的手持布料）
    prop_mask = np.zeros((h, w), dtype=np.uint8)
    for roi in cfg.get("prop_rois", []):
        cv2.fillPoly(prop_mask, [np.array(roi, dtype=np.int32)], 1)
    props_removed_px = int(((prop_mask > 0) & binary).sum())
    if prop_mask.any():
        binary &= prop_mask == 0
        binary = _fill_holes(binary, max_ratio=0.02)
        # 去掉被 ROI 切断后的小碎块
        main, labels, stats, n = _largest_cc(binary)
        keep = main.copy()
        main_area = max(1, int(main.sum()))
        for k in range(1, n):
            area = stats[k, cv2.CC_STAT_AREA]
            if area >= 0.02 * main_area and ((labels == k) & prior).sum() / area > 0.5:
                keep |= labels == k
        binary = keep

    # 5) 边界软化：二值形状 + rembg 羽化 alpha 带
    band = cv2.dilate(binary.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~cv2.erode(
        binary.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    final = np.zeros((h, w), dtype=np.float32)
    final[binary] = 1.0
    final[band] = np.clip(base[band], 0.0, 1.0)
    final_u8 = np.clip(final * 255, 0, 255).astype(np.uint8)
    return {
        "rgb": rgb, "alpha": final_u8, "binary": binary, "prior": prior, "core": core,
        "base": base, "cls": cls, "conf": conf, "alphas": alphas,
        "props_removed_px": props_removed_px, "recovered_px": recovered_px,
    }


def stage_fuse(only=None):
    WORK.mkdir(parents=True, exist_ok=True)
    fdir = CACHE
    for src in sources():
        stem = src.stem
        if only and stem not in only:
            continue
        t = time.perf_counter()
        res = fuse_image(stem)
        np.save(fdir / stem / "fused-alpha.npy", res["alpha"])
        print(f"[fuse] {stem[:4]} {time.perf_counter()-t:.1f}s props_removed={res['props_removed_px']} "
              f"recovered={res['recovered_px']} fg={float((res['alpha']>=128).mean()):.4f}", flush=True)


def _zoom_regions(binary):
    """从最终形状推出 左手 / 右手 / 双脚 / 头发 四个放大自查区域（原图像素）。"""
    h, w = binary.shape
    ys, xs = np.where(binary)
    if len(ys) == 0:
        return {}
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    bh = max(1, y1 - y0)
    bw = max(1, x1 - x0)
    hand_top = int(y0 + 0.42 * bh)
    hand_bot = int(y0 + 0.78 * bh)
    band = binary[hand_top:hand_bot]
    cols = np.where(band.any(axis=0))[0]
    hand_l = int(cols.min()) if len(cols) else x0
    hand_r = int(cols.max()) if len(cols) else x1
    pad_h = max(70, int(0.10 * bw))
    pad_v = max(80, int(0.09 * bh))
    regions = {
        "left-hand": (hand_l - pad_h, hand_top - pad_v, hand_l + pad_h, hand_bot + pad_v),
        "right-hand": (hand_r - pad_h, hand_top - pad_v, hand_r + pad_h, hand_bot + pad_v),
        "feet": (x0 - pad_h, int(y1 - 0.14 * bh) - pad_v, x1 + pad_h, y1 + pad_v),
        "hair": (x0 - pad_h, y0 - pad_v, x1 + pad_h, int(y0 + 0.20 * bh) + pad_v),
    }
    out = {}
    for name, (a, b, c, d) in regions.items():
        out[name] = (max(0, a), max(0, b), min(w, c), min(h, d))
    return out


def render_cutout(rgb_img, alpha_u8, bg=(255, 0, 255)):
    rgba = rgb_img.convert("RGBA")
    rgba.putalpha(Image.fromarray(alpha_u8))
    back = Image.new("RGBA", rgba.size, tuple(bg) + (255,))
    back.alpha_composite(rgba)
    return back.convert("RGB")


def stage_diag(names=None, limit=4):
    WORK.mkdir(parents=True, exist_ok=True)
    for src in sources():
        stem = src.stem
        if names and stem not in names:
            continue
        rgb = _load_rgb(stem)
        alpha = np.load(CACHE / stem / "fused-alpha.npy")
        binary = alpha >= 128
        cut = render_cutout(rgb, alpha)
        lines = []
        for name, (a, b, c, d) in _zoom_regions(binary).items():
            cw, chh = c - a, d - b
            if cw <= 4 or chh <= 4:
                continue
            up = max(1.0, min(3.0, 520 / max(cw, 1)))
            og = rgb.crop((a, b, c, d)).resize((int(cw * up), int(chh * up)), Image.Resampling.LANCZOS)
            cu = cut.crop((a, b, c, d)).resize((int(cw * up), int(chh * up)), Image.Resampling.LANCZOS)
            strip = Image.new("RGB", (og.width * 2 + 8, og.height), "#222222")
            strip.paste(og, (0, 0))
            strip.paste(cu, (og.width + 8, 0))
            d2 = ImageDraw.Draw(strip)
            d2.text((4, 4), name, fill="#00ff00")
            lines.append(strip)
        if not lines:
            continue
        width = max(s.width for s in lines)
        total_h = sum(s.height for s in lines) + 6 * (len(lines) - 1)
        sheet = Image.new("RGB", (width, total_h), "#222222")
        y = 0
        for s in lines:
            sheet.paste(s, (0, y))
            y += s.height + 6
        if sheet.width > 1300:
            sheet = sheet.resize((1300, int(sheet.height * 1300 / sheet.width)), Image.Resampling.LANCZOS)
        out = WORK / f"zoom-{stem[:4]}.jpg"
        sheet.save(out, quality=90)
        print("diag", out, sheet.size, flush=True)


def _metrics(stem, res):
    rgb, alpha = res["rgb"], res["alpha"]
    w, h = rgb.size
    binary = alpha >= 128
    ys, xs = np.where(binary)
    if len(ys) == 0:
        raise RuntimeError(f"{stem} 蒙版为空")
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    thickness = max(2, round(min(w, h) * 0.01))
    border = np.zeros((h, w), dtype=bool)
    border[:thickness, :] = border[-thickness:, :] = True
    border[:, :thickness] = border[:, -thickness:] = True
    border_leak = float((binary & border).sum() / max(1, border.sum()))
    soft_ratio = float(((alpha > 8) & (alpha < 247)).sum() / max(1, binary.sum()))
    # 底部 10% 区域连通块数量（检查双脚是否齐全）
    b0 = int(bbox[3] - 0.10 * (bbox[3] - bbox[1]))
    band = binary[b0:bbox[3]]
    n_band, _, band_stats, _ = cv2.connectedComponentsWithStats(band.astype(np.uint8), 8)
    big = [int(s[cv2.CC_STAT_AREA]) for s in band_stats[1:] if s[cv2.CC_STAT_AREA] > 400]
    return {
        "size": [w, h],
        "bbox": bbox,
        "foreground_ratio": round(float(binary.mean()), 4),
        "foreground_px": int(binary.sum()),
        "soft_alpha_ratio": round(soft_ratio, 4),
        "foreground_on_outer_border_ratio": round(border_leak, 4),
        "touches_frame_edge": bool(border_leak > 0.05),
        "bottom_band_feet_components": len(big),
        "bottom_band_feet_px": big,
        "props_removed_px": int(res["props_removed_px"]),
        "recovered_leg_shoe_px": int(res["recovered_px"]),
        "model_masks_used": sorted(res["alphas"].keys()),
    }


def _zoom_sheet(stem, rgb, alpha, binary, up_scale=None):
    cut = render_cutout(rgb, alpha)
    strips = []
    for name, (a, b, c, d) in _zoom_regions(binary).items():
        cw, chh = c - a, d - b
        if cw <= 4 or chh <= 4:
            continue
        up = up_scale or max(1.0, min(2.4, 460 / max(cw, 1)))
        og = rgb.crop((a, b, c, d)).resize((int(cw * up), int(chh * up)), Image.Resampling.LANCZOS)
        cu = cut.crop((a, b, c, d)).resize((int(cw * up), int(chh * up)), Image.Resampling.LANCZOS)
        strip = Image.new("RGB", (og.width * 2 + 8, og.height + 18), "#1b1b1b")
        strip.paste(og, (0, 18))
        strip.paste(cu, (og.width + 8, 18))
        d2 = ImageDraw.Draw(strip)
        d2.text((4, 3), f"{name}  原图 | 抠图(洋红=透明)  box={a},{b}-{c},{d}", fill="#7CFC00")
        strips.append(strip)
    if not strips:
        return None
    width = max(s.width for s in strips)
    sheet = Image.new("RGB", (width, sum(s.height for s in strips) + 6 * (len(strips) - 1)), "#1b1b1b")
    y = 0
    for s in strips:
        sheet.paste(s, (0, y))
        y += s.height + 6
    if sheet.width > 1500:
        sheet = sheet.resize((1500, int(sheet.height * 1500 / sheet.width)), Image.Resampling.LANCZOS)
    return sheet


def stage_qc():
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font = ImageFont.truetype(str(font_path), 14) if font_path.exists() else ImageFont.load_default()
    zoom_dir = OUTPUT / "qc-zoom"
    zoom_dir.mkdir(parents=True, exist_ok=True)
    rows, tiles = [], []
    for index, src in enumerate(sources(), 1):
        stem = src.stem
        rgb = _load_rgb(stem)
        alpha = np.load(CACHE / stem / "fused-alpha.npy")
        binary = alpha >= 128
        cutout = rgb.convert("RGBA")
        cutout.putalpha(Image.fromarray(alpha))
        mpath = OUTPUT / f"{stem}-mask.png"
        cpath = OUTPUT / f"{stem}-cutout.png"
        gpath = OUTPUT / f"{stem}-gray-preview.png"
        cutout.save(cpath)
        Image.fromarray(alpha).save(mpath)
        gray = Image.new("RGBA", rgb.size, (128, 128, 128, 255))
        gray.alpha_composite(cutout)
        gray.convert("RGB").save(gpath, quality=95)
        sheet = _zoom_sheet(stem, rgb, alpha, binary)
        zpath = zoom_dir / f"{stem}-zoom.jpg"
        if sheet is not None:
            sheet.save(zpath, quality=91)

        cfg = IMAGE_CONFIG.get(stem, {})
        met = _metrics(stem, {
            "rgb": rgb, "alpha": alpha, "props_removed_px": 0, "recovered_px": 0,
            "alphas": {m: True for m in ["u2net_human_seg", "birefnet-portrait", "isnet-general-use"]
                       if (CACHE / stem / f"{m}.png").exists()},
        })
        # 覆写为 fuse 阶段记录的真实数字
        fused_note = {}
        for k in ("props_removed_px", "recovered_px"):
            side = CACHE / stem / "fuse-info.json"
            if side.exists():
                fused_note = json.loads(side.read_text(encoding="utf-8"))
        if fused_note:
            met["props_removed_px"] = fused_note.get("props_removed_px", met["props_removed_px"])
            met["recovered_leg_shoe_px"] = fused_note.get("recovered_px", met["recovered_leg_shoe_px"])
        row = {
            "source": src.name,
            "suggested_angle": cfg.get("angle"),
            "primary_suggestion": cfg.get("primary", False),
            "manual_prop_roi_count": len(cfg.get("prop_rois", [])),
            "model_chain": "rembg[median(u2net_human_seg,birefnet-portrait,isnet-general-use)] "
                           "× segformer_b2_clothes 人体解析先验 + 解析腿/鞋补回 + 连通域清理 + 孔洞填充",
            **met,
            "cutout": cpath.name,
            "mask": mpath.name,
            "gray_preview": gpath.name,
            "zoom_qc": zpath.name,
            "notes": cfg.get("notes", ""),
        }
        rows.append(row)
        thumb = gray.convert("RGB")
        thumb.thumbnail((250, 345), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (268, 400), "white")
        tile.paste(thumb, ((268 - thumb.width) // 2, 5))
        caption = (f"{index}. {stem[:13]}  {cfg.get('angle','?')}\n"
                   f"fg {met['foreground_ratio']:.1%} edge {met['foreground_on_outer_border_ratio']:.1%}")
        ImageDraw.Draw(tile).text((6, 356), caption, fill="#17241f", font=font)
        tiles.append(tile)
        print(f"[qc] {index}/10 {stem[:4]} fg={met['foreground_ratio']:.3f} "
              f"feet={met['bottom_band_feet_components']} props={met['props_removed_px']} "
              f"recovered={met['recovered_leg_shoe_px']}", flush=True)

    cols = 5
    sheet = Image.new("RGB", (cols * 268, ((len(tiles) + cols - 1) // cols) * 400), "#dfe4df")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % cols) * 268, (i // cols) * 400))
    sheet.save(OUTPUT / "qc-contact-sheet.jpg", quality=92)
    (OUTPUT / "mask-manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写出 {len(rows)} 张成品与 mask-manifest.json / qc-contact-sheet.jpg -> {OUTPUT}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("masks", "all"):
        stage_masks(force="--force" in sys.argv)
    if cmd in ("fuse", "all"):
        stage_fuse()
    if cmd == "diag":
        stage_diag([a for a in sys.argv[2:] if not a.startswith("--")])
    if cmd in ("qc", "all"):
        stage_qc()
