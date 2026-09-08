#!/usr/bin/env python3
"""RunPod serverless job handler.

Lives inside the package so its imports resolve from any working directory.
The entrypoint that actually calls runpod.serverless.start() is handler.py at
the repo root: RunPod's GitHub builder scans the repository for that call and
only recognises it at the top level of a root-level file -- nested in a
subdirectory, or guarded by __name__ == "__main__", it reports
"runpod.serverless.start() handler not found in your repo".

Design notes that matter for cost and reliability:

* Models load ONCE at import, not per job. A warm worker skips ~15s that you
  would otherwise pay for on every request.
* Video moves over presigned URLs, never through the job payload.
* Hard caps on input size and duration, because a serverless endpoint will
  happily bill you for a 40-minute clip somebody submitted by accident.
* CUDA OOM returns refresh_worker so RunPod recycles the container instead of
  leaving a poisoned worker to fail every subsequent job.

Local smoke test (no GPU needed with backend "median"):
    PRELOAD=0 python handler.py               # picks up test_input.json
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
import traceback
from pathlib import Path

import runpod

from . import masking, storage, video_io
from .pipeline import Options, run

DEVICE = os.environ.get("DEVICE", "cuda")
SAM2_CKPT = os.environ.get("SAM2_CKPT", masking.DEFAULT_SAM2_CKPT)
SAM2_CFG = os.environ.get("SAM2_CFG", masking.DEFAULT_SAM2_CFG)

MAX_INPUT_BYTES = int(float(os.environ.get("MAX_INPUT_MB", 300)) * 1e6)
MAX_INPUT_SECONDS = float(os.environ.get("MAX_INPUT_SECONDS", 60))
MAX_INLINE_BYTES = int(float(os.environ.get("MAX_INLINE_MB", 8)) * 1e6)
MAX_WORK_RES = int(os.environ.get("MAX_WORK_RES", 1080))

# ---------------------------------------------------------------------------
# cold start
# ---------------------------------------------------------------------------
if os.environ.get("PRELOAD", "1") == "1":
    _t = time.time()
    print(f"[boot] loading models on {DEVICE} ...", flush=True)
    masking.preload(cfg=SAM2_CFG, ckpt=SAM2_CKPT, device=DEVICE)
    print(f"[boot] ready in {time.time() - _t:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------
def _build_options(inp: dict) -> Options:
    if not any(inp.get(k) for k in ("prompt", "points", "box")):
        raise ValueError("give 'prompt' (text), 'points' ([[x,y,label],...]) or 'box'")

    points = [tuple(p) if len(p) == 3 else (p[0], p[1], 1)
              for p in (inp.get("points") or [])]

    box = inp.get("box")
    if box is not None:
        if len(box) != 4:
            raise ValueError("'box' must be [x0, y0, x1, y1]")
        box = tuple(int(v) for v in box)

    work_res = int(inp.get("work_res", 720))
    if work_res > MAX_WORK_RES:
        raise ValueError(f"work_res {work_res} exceeds cap {MAX_WORK_RES}")

    backend_opts = {}
    if inp.get("backend_cmd"):
        # Arbitrary shell on your own GPU. Fine for a private endpoint; gate it
        # behind ALLOW_BACKEND_CMD before you expose this to anyone else.
        if os.environ.get("ALLOW_BACKEND_CMD", "0") != "1":
            raise ValueError("backend_cmd is disabled; set ALLOW_BACKEND_CMD=1 to enable")
        backend_opts["cmd"] = inp["backend_cmd"]

    return Options(
        prompt=inp.get("prompt"),
        points=points,
        static_box=box,
        init_frame=int(inp.get("init_frame", 0)),
        backend=inp.get("backend", "propainter"),
        backend_opts=backend_opts,
        work_res=work_res,
        composite=bool(inp.get("composite", True)),
        lossless=bool(inp.get("lossless", True)),
        crf=int(inp.get("crf", 16)),
        feather=int(inp.get("feather", 4)),
        dilate=int(inp.get("dilate", 12)),
        temporal_pad=int(inp.get("temporal_pad", 2)),
        close=int(inp.get("close", 5)),
        min_area=int(inp.get("min_area", 64)),
        device=DEVICE,
        sam2_cfg=SAM2_CFG,
        sam2_ckpt=SAM2_CKPT,
        box_threshold=float(inp.get("box_threshold", 0.30)),
        text_threshold=float(inp.get("text_threshold", 0.25)),
        preview=bool(inp.get("preview", False)),
        keep_work=False,
    )


def _fetch_input(inp: dict, dest: Path) -> Path:
    if inp.get("video_url"):
        return storage.download(inp["video_url"], dest, max_bytes=MAX_INPUT_BYTES)
    if inp.get("video_base64"):
        return storage.from_base64(inp["video_base64"], dest)
    raise ValueError("give either 'video_url' (presigned GET) or 'video_base64'")


def _deliver(inp: dict, path: Path) -> dict:
    if inp.get("output_url"):
        return {"output_url": storage.upload_presigned(path, inp["output_url"])}
    return {"video_base64": storage.to_base64(path, max_bytes=MAX_INLINE_BYTES),
            "note": "inline delivery is for smoke tests; pass output_url for real jobs"}


# ---------------------------------------------------------------------------
# handler
# ---------------------------------------------------------------------------
def handler(job: dict) -> dict:
    started = time.time()
    inp = job.get("input") or {}
    work = Path(tempfile.mkdtemp(prefix="vremove-"))

    def progress(stage, pct):
        try:
            runpod.serverless.progress_update(job, {"stage": stage, "percent": pct})
        except Exception:
            pass  # never let telemetry kill a job

    try:
        opts = _build_options(inp)

        src = _fetch_input(inp, work / "input.mp4")

        info = video_io.probe(src)
        if info.duration > MAX_INPUT_SECONDS:
            raise ValueError(
                f"clip is {info.duration:.1f}s, cap is {MAX_INPUT_SECONDS:.0f}s. "
                f"Split it, or raise MAX_INPUT_SECONDS on the endpoint.")

        out = work / ("preview.mp4" if opts.preview else "output.mp4")
        result = run(src, out, work / "scratch", opts, progress=progress)

        payload = {
            "status": "ok",
            "preview": result["preview"],
            "coverage": result["coverage"],
            "frames": result["frames"],
            "width": result["width"],
            "height": result["height"],
            "fps": result["fps"],
            "duration": result["duration"],
            "backend": result["backend"],
            "warnings": result["warnings"],
            "seconds": round(time.time() - started, 1),
        }
        payload.update(_deliver(inp, Path(result["output"])))
        return payload

    except Exception as e:
        traceback.print_exc()
        oom = "out of memory" in str(e).lower()
        err = {"status": "error", "error": f"{type(e).__name__}: {e}",
               "seconds": round(time.time() - started, 1)}
        if oom:
            err["hint"] = ("lower work_res, shorten the clip, or move to a bigger GPU; "
                           "worker is being recycled")
            # A worker that has OOM'd often stays broken. Force a fresh one.
            return {"refresh_worker": True, "job_results": err}
        return err

    finally:
        shutil.rmtree(work, ignore_errors=True)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
