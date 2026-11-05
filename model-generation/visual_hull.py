"""Build a complete closed mesh from front/left/back/right silhouettes (visual hull).

Classic multi-view silhouette carving: a voxel belongs to the object when it projects
inside every silhouette.  No neural network, no GPU; gives a guaranteed complete body
volume that respects all supplied views (blobby but whole, unlike single-view guessing).

Usage:
    python visual_hull.py --front <png> --back <png> --left <png> --right <png>
        --out-dir <dir> [--resolution 160] [--size 512]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image


def load_mask(path: Path, size: int, target_height: float) -> np.ndarray:
    """Return a (size, size) float mask; subject scaled to target_height, feet on bottom."""
    image = Image.open(path).convert("RGBA")
    alpha = image.getchannel("A")
    bbox = alpha.getbbox()
    if bbox is None:
        raise RuntimeError(f"空蒙版：{path}")
    crop = image.crop(bbox)
    scale = target_height / crop.height
    new_w = max(1, min(size, int(round(crop.width * scale))))
    new_h = max(1, min(size, int(round(crop.height * scale))))
    crop = crop.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (size, size), 0)
    left = (size - new_w) // 2
    top = size - new_h  # anchor feet at the bottom edge
    canvas.paste(crop.getchannel("A"), (left, top))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return arr


def sample(mask: np.ndarray, u: np.ndarray, v: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour sample; outside the image counts as background."""
    ui = np.clip(np.rint(u).astype(np.int32), 0, size - 1)
    vi = np.clip(np.rint(v).astype(np.int32), 0, size - 1)
    inside = (u >= 0) & (u <= size - 1) & (v >= 0) & (v <= size - 1)
    return (mask[vi, ui] > 0.5) & inside


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--front", required=True)
    parser.add_argument("--back", required=True)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--resolution", type=int, default=160)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--subject-height", type=float, default=0.94)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    px = args.subject_height * args.size  # pixel height of the subject
    masks = {}
    for view in ("front", "back", "left", "right"):
        masks[view] = load_mask(Path(getattr(args, view)), args.size, px)
        Image.fromarray((masks[view] * 255).astype(np.uint8)).save(out_dir / f"silhouette-{view}.png")

    res = args.resolution
    axis = np.linspace(-1.0, 1.0, res, dtype=np.float32)
    # world: x right, y up (0 at feet), z depth.  canvas mapping:
    #   u = size/2 + x * (size/2 * s), v = size - y * (px / 2)
    half = args.size / 2.0
    scale_world = half  # world ±1 fills the canvas width
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")  # gy in [-1,1]

    y_world = (gy + 1.0) / 2.0  # 0..1 feet->head
    v = args.size - y_world * px
    u_x = half + gx * scale_world
    u_z = half + gz * scale_world

    occ = np.ones((res, res, res), dtype=bool)
    occ &= sample(masks["front"], u_x, v, args.size)
    occ &= sample(masks["back"], args.size - u_x, v, args.size)
    occ &= sample(masks["left"], u_z, v, args.size)
    occ &= sample(masks["right"], args.size - u_z, v, args.size)

    # Drop tiny floating islands so the downloaded mesh stays one body.
    from scipy import ndimage

    labels, count = ndimage.label(occ)
    if count > 1:
        sizes = ndimage.sum(occ, labels, range(1, count + 1))
        keep = int(np.argmax(sizes)) + 1
        removed = int(occ.sum() - sizes[keep - 1])
        occ = labels == keep
    else:
        removed = 0

    volume = float(occ.mean())
    print(f"occupied voxels: {int(occ.sum())} ({volume:.3%} of grid), removed islands: {removed}")

    import trimesh
    from skimage import measure

    padded = np.pad(occ.astype(np.float32), 1, mode="constant")
    verts, faces, _, _ = measure.marching_cubes(padded, level=0.5)
    verts = verts - 1.0  # undo padding
    # index space -> [-1, 1] on every axis
    verts = verts / (res - 1) * 2.0 - 1.0
    # y was sampled through y01 = (gy + 1) / 2 in [0, 1]; convert back to world height
    # where x/z span 2 world units across the canvas width and the subject fills `px` pixels.
    height_world = 2.0 * px / args.size
    verts[:, 0] *= 1.0
    verts[:, 2] *= 1.0
    verts[:, 1] = (verts[:, 1] + 1.0) / 2.0 * height_world
    mesh = trimesh.Trimesh(vertices=verts.astype(np.float64), faces=faces.astype(np.int64), process=True)
    mesh.remove_unreferenced_vertices()
    glb = out_dir / "visual-hull.glb"
    mesh.export(glb)
    report = {
        "method": "multi-view silhouette carving (visual hull)",
        "views": ["front", "back", "left", "right"],
        "resolution": res,
        "occupied_ratio": round(volume, 5),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "seconds": round(time.perf_counter() - started, 2),
        "glb": glb.name,
        "note": "Complete closed volume honouring all four silhouettes; no texture, no skeleton, low detail.",
    }
    (out_dir / "generation-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("VISUAL_HULL_OK", json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
