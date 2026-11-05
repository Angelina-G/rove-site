"""Generate a shape-only GLB from named front/left/right/back cutouts.

Model : Tencent Hunyuan3D-2mv (multi-view shape DiT), turbo variant, fp16.
Weights: ModelScope mirror ``Tencent-Hunyuan/Hunyuan3D-2mv`` -> downloaded by
         ``cache/download_hunyuan3d_ms.py`` into ``weights/hunyuan3d-2mv/``.
Code   : community low-VRAM fork ``source/Hunyuan3D-2GP`` (DeepBeepMeep) with
         mmgp offloading (profile 4 = LowRAM_LowVRAM, <6 GB VRAM per README).
Host   : RTX 4070 Laptop 8 GB (single task at a time).

Stages emit ``ROVE_STAGE:<name>`` lines to stdout and ``stage-log.txt``; a failed
stage keeps the original traceback in the report instead of pretending success.

Usage (full 4 views, octree 256):
    python multiview_worker.py --octree 256 --steps 5 --profile 4

Usage (smoke: front view only, octree 128):
    python multiview_worker.py --preset smoke
"""
import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "source" / "Hunyuan3D-2GP"
CACHE = ROOT / "cache"
WEIGHTS = ROOT / "weights"
APPROVED_V2 = ROOT / "test-images" / "processed-v2"
APPROVED = APPROVED_V2 if (APPROVED_V2 / "mask-manifest.json").exists() else ROOT / "test-images" / "processed-approved"
DEFAULT_OUT = ROOT / "output" / "jobs" / "hunyuan3d-2mv-turntable"
SMOKE_OUT = ROOT / "output" / "jobs" / "hunyuan3d-2mv-smoke"

MODEL_DIR = "hunyuan3d-2mv"                     # dir under HY3DGEN_MODELS
MODEL_SUBFOLDER = "hunyuan3d-dit-v2-mv-turbo"   # turbo variant (5 step)

# front / left / right / back picked from the repaired v2 masks
# (test-images/processed-v2/mask-manifest.json, one person / one outfit):
#   9ca2f019 = front (facing camera, both shoes), 9e01afc1 = left 侧 (鼻子朝画面左),
#   52a9f704 = right 侧 (鼻子朝画面右), 01139015 = back。
# 之前用的是 52a9 + fc55，二者其实是同一侧，导致左腿缺少真实侧向约束、被模型截断。
VIEW_FILES = {
    "front": "9ca2f019a28896cbef7fc86359863ddc-cutout.png",
    "left": "9e01afc1d631e3f3d42de46bd77545ac-cutout.png",
    "right": "52a9f704b1daa15643d7e6d0d13e16d0-cutout.png",
    "back": "01139015a4830e26611025860662f5c9-cutout.png",
}


class StageLog:
    """Append-only ROVE_STAGE log: stdout + stage-log.txt + structured records."""

    def __init__(self, path: Path):
        self.path = path
        self.records = []
        self._t0 = time.time()
        path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, line: str):
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def stage(self, name: str, message: str = ""):
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] ROVE_STAGE:{name}" + (f" {message}" if message else "")
        self._write(line)
        self.records.append(
            {"stage": name, "message": message, "at": stamp,
             "elapsed_since_start_s": round(time.time() - self._t0, 2)}
        )
        return time.perf_counter()

    def info(self, message: str):
        stamp = time.strftime("%H:%M:%S")
        self._write(f"[{stamp}] {message}")

    def result(self, message: str):
        stamp = time.strftime("%H:%M:%S")
        self._write(f"[{stamp}] ROVE_RESULT:{message}")


def prepare(image_path: Path, output_path: Path, box: int = 440, canvas_size: int = 512) -> Path:
    """Crop to alpha bounds, keep aspect, paste on a transparent square canvas."""
    with Image.open(image_path) as source:
        rgba = source.convert("RGBA")
    bbox = rgba.getchannel("A").getbbox()
    if bbox is None:
        raise RuntimeError(f"空蒙版，停止生成：{image_path.name}")
    rgba = rgba.crop(bbox)
    rgba.thumbnail((box, box), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (canvas_size, canvas_size), (255, 255, 255, 0))
    canvas.alpha_composite(rgba, ((canvas_size - rgba.width) // 2, (canvas_size - rgba.height) // 2))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def replace_property_getter(instance, property_name, new_getter):
    """Same trick as the fork's gradio_app: force _execution_device to cuda for mmgp."""
    original_class = type(instance)
    original_property = getattr(original_class, property_name)
    custom_class = type(f"Custom{original_class.__name__}", (original_class,), {})
    new_property = property(new_getter, original_property.fset)
    setattr(custom_class, property_name, new_property)
    instance.__class__ = custom_class
    return instance


def file_sha256(path: Path, chunk=16 * 1024 * 1024) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def gb(value: float) -> float:
    return round(value / (1024 ** 3), 2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["smoke", "full"], default="full")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--octree", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--num-chunks", type=int, default=200000)
    parser.add_argument("--profile", type=int, default=4)
    parser.add_argument("--views", default=None, help="comma separated subset, default all 4")
    parser.add_argument("--mask-version", default=None,
                        help="test-images 下的蒙版版本目录，如 processed-v3；默认 processed-v2")
    parser.add_argument("--tag", default="")
    parser.add_argument("--cleanup", dest="cleanup", action="store_true", default=True)
    parser.add_argument("--no-cleanup", dest="cleanup", action="store_false")
    args = parser.parse_args()

    if args.preset == "smoke":
        out_dir = Path(args.out_dir) if args.out_dir else SMOKE_OUT
        octree = args.octree if args.octree is not None else 128
        steps = args.steps if args.steps is not None else 5
        views = (args.views or "front").split(",")
    else:
        out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT
        octree = args.octree if args.octree is not None else 256
        steps = args.steps if args.steps is not None else 5
        views = (args.views or "front,left,right,back").split(",")

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = ROOT / "test-images" / (args.mask_version or "processed-v2")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"蒙版版本目录不存在：{mask_dir}")
    log = StageLog(out_dir / "stage-log.txt")
    # Fresh log per run.
    log.path.write_text("", encoding="utf-8")
    run_started = time.time()
    report = {
        "status": "running",
        "mask_version": args.mask_version or "processed-v2",
        "model": {
            "name": "Tencent Hunyuan3D-2mv (turbo, fp16, shape only)",
            "code": "rove-site/model-generation/source/Hunyuan3D-2GP (mmgp low-VRAM fork)",
            "weights_repo": "ModelScope Tencent-Hunyuan/Hunyuan3D-2mv",
            "weights_subfolder": MODEL_SUBFOLDER,
        },
        "host": {},
        "views": {},
        "params": {
            "octree_resolution": octree, "steps": steps, "guidance_scale": args.guidance,
            "seed": args.seed, "num_chunks": args.num_chunks, "mmgp_profile": args.profile,
            "dtype": "float16", "shape_only": True, "preset": args.preset,
        },
        "timings_seconds": {},
        "vram": {},
        "mesh": {},
        "stages": log.records,
        "command": " ".join([sys.executable] + sys.argv),
    }
    report_path = out_dir / "generation-report.json"

    def save_report():
        report["stages"] = log.records
        report["elapsed_total_seconds"] = round(time.time() - run_started, 2)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        # ---------------------------------------------------------------- inputs
        log.stage("inputs_ready", f"preset={args.preset} views={views}")
        inputs_dir = out_dir / "inputs"
        resolved = {}
        for view in views:
            if view not in VIEW_FILES:
                raise KeyError(f"未知视角 {view}，可选 {list(VIEW_FILES)}")
            src = mask_dir / VIEW_FILES[view]
            if not src.is_file():
                raise FileNotFoundError(f"缺少多视图抠图：{src}")
            dst = prepare(src, inputs_dir / f"{view}.png")
            resolved[view] = {"source": str(src), "prepared": str(dst)}
            log.info(f"view {view}: {src.name} -> {dst.name}")
        report["views"] = resolved
        (out_dir / "input-views.json").write_text(
            json.dumps({k: v["source"] for k, v in resolved.items()}, ensure_ascii=False, indent=2),
            encoding="utf-8")

        # ------------------------------------------------------------------- env
        log.stage("env_setup")
        os.environ["HY3DGEN_MODELS"] = str(WEIGHTS)
        os.environ.setdefault("HY3DGEN_DEBUG", "1")
        os.environ.setdefault("HF_HOME", str(CACHE / "huggingface"))
        sys.path.insert(0, str(REPO))
        log.info(f"HY3DGEN_MODELS={os.environ['HY3DGEN_MODELS']}")
        log.info(f"REPO={REPO}")

        import torch

        torch.set_default_device("cpu")
        from mmgp import offload, profile_type
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        from hy3dgen.shapegen.pipelines import export_to_trimesh

        # Route the fork's internal stage timings into our stage log too.
        hy_logger = logging.getLogger("hy3dgen.shapgen")
        fh = logging.FileHandler(out_dir / "stage-log.txt", encoding="utf-8", mode="a")
        fh.setFormatter(logging.Formatter("[%(asctime)s] %(name)s %(message)s", "%H:%M:%S"))
        hy_logger.addHandler(fh)

        report["host"] = {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda",
            "gpu_total_gb": gb(torch.cuda.get_device_properties(0).total_memory) if torch.cuda.is_available() else None,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "python": sys.version.split()[0],
            "platform": sys.platform,
        }
        log.info(f"torch={torch.__version__} cuda={torch.version.cuda} gpu={report['host']['gpu']}")

        # ------------------------------------------------------------ model load
        log.stage("model_load", f"{MODEL_DIR}/{MODEL_SUBFOLDER}")
        t_load = time.perf_counter()
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            MODEL_DIR, subfolder=MODEL_SUBFOLDER,
            use_safetensors=True, device="cuda", dtype=torch.float16,
        )
        replace_property_getter(pipeline, "_execution_device", lambda self: "cuda")
        report["timings_seconds"]["model_load"] = round(time.perf_counter() - t_load, 2)
        log.info(f"model_load {report['timings_seconds']['model_load']}s")
        weight_file = WEIGHTS / MODEL_DIR / MODEL_SUBFOLDER / "model.fp16.safetensors"
        report["model"]["weights_file"] = str(weight_file)
        if weight_file.is_file():
            report["model"]["weights_bytes"] = weight_file.stat().st_size
            report["model"]["weights_sha256"] = file_sha256(weight_file)
        log.info(f"weights {weight_file} ({report['model'].get('weights_bytes')} bytes)")
        save_report()

        # --------------------------------------------------------- mmgp offload
        log.stage("mmgp_offload", f"profile={args.profile}")
        t_off = time.perf_counter()
        pipe = offload.extract_models("i23d_worker", pipeline)
        offload_kwargs = {}
        if args.profile not in (1, 3):
            offload_kwargs["budgets"] = {"*": 2200}
        if args.profile < 5:
            offload_kwargs["pinnedMemory"] = "i23d_worker/model"
        offload.default_verboseLevel = 1
        offload.profile(pipe, profile_no=profile_type(args.profile), verboseLevel=1, **offload_kwargs)
        report["timings_seconds"]["mmgp_offload"] = round(time.perf_counter() - t_off, 2)
        log.info(f"mmgp profile {args.profile} ready in {report['timings_seconds']['mmgp_offload']}s")
        save_report()

        pil_views = {view: Image.open(resolved[view]["prepared"]).convert("RGBA") for view in views}

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline_alloc = torch.cuda.memory_allocated()

        # ------------------------------------------------------------- generate
        log.stage("generating_multiview_shape",
                  f"octree={octree} steps={steps} guidance={args.guidance} views={views}")
        t_gen = time.perf_counter()
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        outputs = pipeline(
            image=pil_views,
            num_inference_steps=steps,
            guidance_scale=args.guidance,
            generator=generator,
            octree_resolution=octree,
            num_chunks=args.num_chunks,
            output_type="mesh",
        )
        torch.cuda.synchronize()
        report["timings_seconds"]["generate"] = round(time.perf_counter() - t_gen, 2)
        peak_alloc = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        report["vram"] = {
            "peak_allocated_gb": gb(peak_alloc), "peak_reserved_gb": gb(peak_reserved),
            "baseline_allocated_gb": gb(baseline_alloc),
            "device_total_gb": report["host"]["gpu_total_gb"],
        }
        log.stage("shape_generated", f"{report['timings_seconds']['generate']}s vram_peak={report['vram']['peak_allocated_gb']}GB")
        save_report()

        # --------------------------------------------------------------- export
        log.stage("export_to_trimesh")
        t_ex = time.perf_counter()
        mesh = export_to_trimesh(outputs)[0]
        if args.cleanup:
            from hy3dgen.shapegen import FloaterRemover, DegenerateFaceRemover  # noqa: F401  (kept for reference)

            before = (len(mesh.vertices), len(mesh.faces))
            # pymeshlab 的 FloaterRemover 在本机（中文路径 / pymeshlab 2025 版）会抛
            # PyMeshLabException，这里改用 trimesh 自身完成同样目的：合并重复顶点、
            # 删退化面、只保留最大连通块，避免后处理把整条流水线打断。
            mesh = mesh.copy()
            mesh.merge_vertices()
            mesh.remove_unreferenced_vertices()
            mesh.update_faces(mesh.nondegenerate_faces())
            parts = mesh.split(only_watertight=False)
            if parts:
                mesh = max(parts, key=lambda m: len(m.faces))
            log.info(f"cleanup {before} -> ({len(mesh.vertices)}, {len(mesh.faces)})")
        glb_path = out_dir / "hunyuan3d-2mv.glb"
        mesh.export(str(glb_path))
        report["timings_seconds"]["export"] = round(time.perf_counter() - t_ex, 2)
        log.info(f"glb {glb_path} ({glb_path.stat().st_size} bytes)")

        # --------------------------------------------------------------- verify
        log.stage("verify_glb")
        import trimesh

        reloaded = trimesh.load(str(glb_path), force="scene")
        merged = reloaded.dump(concatenate=True) if not isinstance(reloaded, trimesh.Trimesh) else reloaded
        bounds = np.asarray(merged.bounds, dtype=float)
        report["mesh"] = {
            "path": str(glb_path),
            "bytes": glb_path.stat().st_size,
            "vertices": int(len(merged.vertices)),
            "faces": int(len(merged.faces)),
            "watertight": bool(merged.is_watertight),
            "winding_consistent": bool(merged.is_winding_consistent),
            "bounds": bounds.round(6).tolist(),
            "extents": (bounds[1] - bounds[0]).round(6).tolist(),
            "sha256": file_sha256(glb_path),
        }
        log.stage("done", f"vertices={report['mesh']['vertices']} faces={report['mesh']['faces']}")
        report["status"] = "ok"
        report["elapsed_total_seconds"] = round(time.time() - run_started, 2)
        save_report()
        log.result(json.dumps({"status": "ok", **report["mesh"]}, ensure_ascii=False))
        print("ROVE_DONE", glb_path, flush=True)
        return 0
    except Exception as exc:  # keep the original error text, never fake success
        tb = traceback.format_exc()
        log.path.parent.mkdir(parents=True, exist_ok=True)
        with open(log.path, "a", encoding="utf-8") as fh:
            fh.write(f"[{time.strftime('%H:%M:%S')}] ROVE_STAGE:failed {type(exc).__name__}: {exc}\n")
            fh.write(tb + "\n")
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": tb}
        try:
            report["vram"] = report.get("vram") or {}
            save_report()
        except Exception:
            pass
        print(f"ROVE_STAGE:failed {type(exc).__name__}: {exc}", flush=True)
        print(tb, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
