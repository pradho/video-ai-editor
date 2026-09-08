"""End-to-end orchestration: video in -> object gone -> video out."""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from . import inpaint, masking, video_io
from .composite import composite_hires


@dataclass
class Options:
    prompt: str | None = None
    points: list[tuple[int, int, int]] = field(default_factory=list)
    static_box: tuple[int, int, int, int] | None = None  # plumbing test only
    init_frame: int = 0

    backend: str = "propainter"
    backend_opts: dict = field(default_factory=dict)

    work_res: int = 720
    composite: bool = True
    # PNG intermediates by default. Measured on a 640x360 clip: JPEG q2 raised
    # the error on pixels OUTSIDE the mask from mean 0.79 / max 23 to mean 1.72
    # / max 91. Those pixels are the entire point of the hi-res composite, so
    # paying ~2x scratch disk to keep them is the right trade.
    lossless: bool = True
    crf: int = 16
    feather: int = 4

    dilate: int = 12
    temporal_pad: int = 2
    close: int = 5
    min_area: int = 64

    device: str = "cuda"
    sam2_cfg: str = masking.DEFAULT_SAM2_CFG
    sam2_ckpt: str = masking.DEFAULT_SAM2_CKPT
    sam2_offload: bool = True
    box_threshold: float = 0.30
    text_threshold: float = 0.25

    preview: bool = False
    keep_work: bool = False


def run(src, out, work_dir, opts: Options, progress=None) -> dict:
    """Returns {output, coverage, frames, backend, warnings, width, height, fps}.

    `progress` is an optional callable(stage: str, pct: float) -- the serverless
    worker uses it to stream status back to the caller during long jobs.
    """
    def tick(stage, pct):
        print(f"[{pct:>3.0f}%] {stage}")
        if progress:
            progress(stage, pct)

    src, out, work = Path(src), Path(out), Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    info = video_io.probe(src)
    print(f"[in ] {src.name}  {info.width}x{info.height}  {info.fps:.3f}fps  "
          f"{info.nframes} frames  audio={info.has_audio}")
    tick("extracting frames", 5)

    # 1. frames -----------------------------------------------------------
    work_frames, _ = video_io.extract_frames(src, work / "work", max_side=opts.work_res)
    n_work = len(video_io.frame_paths(work_frames, "jpg"))
    print(f"[work] {n_work} frames at <= {opts.work_res}px")

    full_frames = full_ext = None
    if opts.composite:
        full_frames, full_ext = video_io.extract_frames(
            src, work / "full", lossless=opts.lossless)

    # 2. masks ------------------------------------------------------------
    tick("detecting object", 15)
    mask_dir = work / "masks"

    if opts.static_box:
        # No model at all -- lets the whole pipeline be exercised offline.
        masking.static_box_masks(work_frames, mask_dir, opts.static_box)
        warnings.append("static_box is a plumbing test, not real segmentation")
        print("[mask] static box (no model)")
    else:
        boxes = None
        if opts.prompt:
            init = video_io.frame_paths(work_frames, "jpg")[opts.init_frame]
            boxes = masking.detect_boxes(
                init, opts.prompt, device=opts.device,
                box_threshold=opts.box_threshold, text_threshold=opts.text_threshold)
            print(f"[mask] '{opts.prompt}' -> {len(boxes)} box(es) on frame {opts.init_frame}")
            if len(boxes) == 0:
                raise ValueError(
                    f"nothing matched {opts.prompt!r}. Lower box_threshold, try a "
                    f"different init_frame, or point at it with an explicit point")

        tick("tracking through video", 25)
        masking.track_masks(work_frames, mask_dir, boxes=boxes, points=opts.points or None,
                            cfg=opts.sam2_cfg, ckpt=opts.sam2_ckpt, device=opts.device,
                            init_frame=opts.init_frame, offload=opts.sam2_offload)

    masking.postprocess_masks(mask_dir, dilate=opts.dilate, temporal_pad=opts.temporal_pad,
                              close=opts.close, min_area=opts.min_area)

    cov = masking.coverage(mask_dir)
    print(f"[mask] mean coverage {cov:.1%}")

    meta = {"coverage": round(cov, 4), "frames": n_work, "width": info.width,
            "height": info.height, "fps": round(info.fps, 3),
            "duration": round(info.duration, 2), "backend": opts.backend,
            "warnings": warnings}

    # 3. preview escape hatch --------------------------------------------
    if opts.preview:
        ov = masking.write_overlay(work_frames, mask_dir, work / "overlay")
        tick("encoding preview", 90)
        p = video_io.encode(ov, out, info.fps, ext="png", crf=20)
        print(f"[out ] mask preview -> {p}")
        tick("done", 100)
        return {"output": p, "preview": True, **meta}

    # 4. inpaint ----------------------------------------------------------
    backend = inpaint.get(opts.backend, **opts.backend_opts)
    if cov > 0.12 and backend.large_mask_score <= 2:
        w = (f"mask covers {cov:.1%} of frame and backend '{backend.name}' scores "
             f"{backend.large_mask_score}/5 on large masks -- expect blur in the fill")
        warnings.append(w)
        print(f"[warn] {w}")
    if not backend.commercial_ok:
        w = f"backend '{backend.name}' licence: {backend.licence}"
        warnings.append(w)
        print(f"[warn] {w}")

    tick("inpainting", 40)
    inp_dir = work / "inpainted"
    backend.run(work_frames, mask_dir, inp_dir, fps=info.fps)

    # 5. paste back + encode ----------------------------------------------
    final = inp_dir
    if opts.composite:
        tick("compositing to full resolution", 85)
        final = composite_hires(full_frames, inp_dir, mask_dir, work / "final",
                                orig_ext=full_ext, feather=opts.feather)

    tick("encoding", 95)
    p = video_io.encode(final, out, info.fps, ext="png",
                        audio_from=src if info.has_audio else None, crf=opts.crf)
    print(f"[out ] {p}")

    if not opts.keep_work:
        shutil.rmtree(work, ignore_errors=True)
    tick("done", 100)
    return {"output": p, "preview": False, **meta}
