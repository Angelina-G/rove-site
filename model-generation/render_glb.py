"""Render front/left/right/back previews of a GLB without OpenGL.

Pure numpy software rasterizer (orthographic, z-buffer, flat Lambert shading).
Usage:
    python render_glb.py <mesh.glb> <out-dir> [--size 640] [--label "Job A"]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_mesh(path: Path):
    import trimesh

    scene = trimesh.load(str(path), force="scene")
    if isinstance(scene, trimesh.Trimesh):
        return scene
    # dump(concatenate=True) applies scene-graph node transforms before merging,
    # which matters because GLB exporters often store the axis flip on the root node.
    merged = scene.dump(concatenate=True)
    if isinstance(merged, trimesh.Trimesh) and len(merged.vertices):
        return merged
    meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
    if not meshes:
        raise RuntimeError(f"GLB 中没有可渲染网格：{path}")
    return trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]


def rotation(view: str) -> np.ndarray:
    """World is Y-up (glTF). Return rotation that maps world -> camera space."""
    views = {
        # camera looks down -Z in camera space; rotate world into that frame
        "front": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
        "back": np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64),
        "left": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64),
        "right": np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float64),
        "three_quarter": np.array(
            [[0.7071, 0, 0.7071], [0, 1, 0], [-0.7071, 0, 0.7071]], dtype=np.float64
        ),
    }
    return views[view]


def rasterize(vertices, faces, normals, size, light_dir):
    """Return (rgb uint8, depth) for one orthographic view."""
    half = 0.98
    xy = vertices[:, :2] * (size / 2) / half
    px = xy[:, 0] + size / 2
    py = size / 2 - xy[:, 1]
    depth = vertices[:, 2]

    color = np.zeros((size, size, 3), dtype=np.float32)
    zbuf = np.full((size, size), -np.inf, dtype=np.float64)
    shade = 0.30 + 0.70 * np.clip(normals @ light_dir, 0.0, 1.0)
    base = np.array([0.62, 0.66, 0.72], dtype=np.float32)

    spans = []
    for index in range(len(faces)):
        tri = faces[index]
        x = px[tri]
        y = py[tri]
        x0, x1 = int(np.floor(x.min())), int(np.ceil(x.max()))
        y0, y1 = int(np.floor(y.min())), int(np.ceil(y.max()))
        x0 = max(x0, 0); x1 = min(x1, size - 1)
        y0 = max(y0, 0); y1 = min(y1, size - 1)
        if x0 > x1 or y0 > y1:
            continue
        spans.append((x0, x1, y0, y1, index))

    grid_x = np.arange(size, dtype=np.float64) + 0.5
    for x0, x1, y0, y1, index in spans:
        tri = faces[index]
        x = px[tri]; y = py[tri]; z = depth[tri]
        denom = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2])
        if abs(denom) < 1e-12:
            continue
        xs = grid_x[x0:x1 + 1][None, :]
        ys = grid_x[y0:y1 + 1][:, None]
        w0 = ((y[1] - y[2]) * (xs - x[2]) + (x[2] - x[1]) * (ys - y[2])) / denom
        w1 = ((y[2] - y[0]) * (xs - x[2]) + (x[0] - x[2]) * (ys - y[2])) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        zi = w0 * z[0] + w1 * z[1] + w2 * z[2]
        sub_z = zbuf[y0:y1 + 1, x0:x1 + 1]
        take = inside & (zi > sub_z)
        if not take.any():
            continue
        sub_z[take] = zi[take]
        sub_c = color[y0:y1 + 1, x0:x1 + 1]
        sub_c[take] = base * float(shade[index])
    return color, zbuf


def decimate_for_preview(mesh, max_faces=90000):
    """Reduce face count for a fast preview; keeps topology with quadric decimation."""
    import trimesh

    faces = np.asarray(mesh.faces)
    if len(faces) <= max_faces:
        return mesh
    try:
        import fast_simplification

        v, f = fast_simplification.simplify(
            np.asarray(mesh.vertices, dtype=np.float32),
            faces.astype(np.int32),
            target_reduction=float(1.0 - max_faces / len(faces)),
        )
        return trimesh.Trimesh(vertices=v, faces=f, process=True)
    except Exception:
        pass
    # fallback: grid-quantise vertices
    v = np.asarray(mesh.vertices)
    lo, hi = v.min(axis=0), v.max(axis=0)
    span = float((hi - lo).max()) or 1.0
    grid = int(np.clip(round((max_faces / max(1, len(faces))) ** (1 / 2) * 160), 60, 420))
    q = np.floor((v - lo) / span * grid).astype(np.int64)
    _, first, inverse = np.unique(q, axis=0, return_index=True, return_inverse=True)
    new_v = v[first]
    new_f = inverse[faces]
    keep = (new_f[:, 0] != new_f[:, 1]) & (new_f[:, 1] != new_f[:, 2]) & (new_f[:, 0] != new_f[:, 2])
    return trimesh.Trimesh(vertices=new_v, faces=new_f[keep], process=True)


def render(mesh_path: Path, out_dir: Path, size: int, label: str):
    import trimesh

    mesh = decimate_for_preview(load_mesh(mesh_path))
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces)
    if len(v) == 0 or len(f) == 0:
        raise RuntimeError("网格为空，无法渲染。")

    lo, hi = v.min(axis=0), v.max(axis=0)
    extent_vec = hi - lo
    up_axis = int(np.argmax(extent_vec))
    up_names = {0: "X-up (rotated to Y-up)", 1: "Y-up", 2: "Z-up (rotated to Y-up)"}
    face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
    if up_axis == 2:  # Z-up -> Y-up
        rot = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
        v = v @ rot.T
        face_normals = face_normals @ rot.T
    elif up_axis == 0:  # X-up -> Y-up
        rot = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64)
        v = v @ rot.T
        face_normals = face_normals @ rot.T

    # Decide which end is the head: a human head is much narrower than shoulders/feet,
    # so compare the horizontal spread of the bottom 6% and top 6% slices.
    y = v[:, 1]
    lo_y, hi_y = float(y.min()), float(y.max())
    span = max(hi_y - lo_y, 1e-9)
    margin = span * 0.06
    bottom = v[y <= lo_y + margin]
    top = v[y >= hi_y - margin]

    def spread(points):
        if len(points) == 0:
            return 0.0
        radial = np.hypot(points[:, 0], points[:, 2])
        return float(np.percentile(radial, 95))

    bottom_spread, top_spread = spread(bottom), spread(top)
    flipped = bottom_spread < top_spread * 0.75  # narrow end is at the bottom -> upside down
    if flipped:
        v[:, 1] = -v[:, 1]
        face_normals[:, 1] = -face_normals[:, 1]

    lo, hi = v.min(axis=0), v.max(axis=0)
    center = (lo + hi) / 2.0
    extent = float((hi - lo).max()) or 1.0
    v = (v - center) / (extent / 2.0)

    views = ["front", "left", "back", "right"]
    tiles = []
    stats = {
        "mesh": str(mesh_path),
        "vertices": int(len(v)),
        "faces": int(len(f)),
        "bounds": [lo.tolist(), hi.tolist()],
        "up_axis": up_names.get(up_axis, str(up_axis)),
        "auto_flipped": bool(flipped),
        "views": views,
        "watertight": bool(mesh.is_watertight),
    }
    for view in views:
        r = rotation(view)
        cam_v = v @ r.T
        cam_n = face_normals @ r.T
        rgb, zbuf = rasterize(cam_v, f, cam_n, size, np.array([0.55, 0.62, 0.56]))
        img = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        tile = Image.new("RGB", (size + 8, size + 30), (240, 240, 240))
        tile.paste(Image.fromarray(img), (4, 4))
        ImageDraw.Draw(tile).text((8, size + 9), f"{label} · {view}", fill=(20, 20, 20))
        tiles.append(tile)

    sheet = Image.new("RGB", (len(tiles) * (size + 8), size + 30), (240, 240, 240))
    for i, t in enumerate(tiles):
        sheet.paste(t, (i * (size + 8), 0))
    out_dir.mkdir(parents=True, exist_ok=True)
    preview = out_dir / f"{mesh_path.stem}-preview.png"
    sheet.save(preview)
    for index, tile in enumerate(tiles, 1):
        # 用 viewN 命名，避免被工作台的“拍摄视角”扫描误当成输入视图
        tile.save(out_dir / f"{mesh_path.stem}-view{index}.png")
    stats["preview"] = preview.name
    (out_dir / f"{mesh_path.stem}-render-stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"RENDER_OK {preview}  vertices={stats['vertices']} faces={stats['faces']} watertight={stats['watertight']}")
    return preview


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh")
    parser.add_argument("out_dir")
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    preview = render(Path(args.mesh), Path(args.out_dir), args.size, args.label or Path(args.mesh).stem)
    print(preview)


if __name__ == "__main__":
    main()
