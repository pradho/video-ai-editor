#!/usr/bin/env python3
"""Remove an object from a video by name.

    python remove.py in.mp4 -o out.mp4 --prompt "person"
    python remove.py in.mp4 -o preview.mp4 --prompt "person" --preview
    python remove.py in.mp4 -o out.mp4 --point 640,380 --backend median
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from vremove import inpaint
from vremove.pipeline import Options, run


def _point(s: str):
    parts = s.split(",")
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError("--point wants x,y or x,y,label")
    x, y = int(parts[0]), int(parts[1])
    return (x, y, int(parts[2]) if len(parts) == 3 else 1)


def _box(s: str):
    parts = s.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--box wants x0,y0,x1,y1")
    return tuple(int(p) for p in parts)


def list_backends():
    print(f"{'name':<12} {'commercial':<11} {'vram':<7} {'large-mask':<11} licence")
    print("-" * 78)
    for name, cls in sorted(inpaint.REGISTRY.items()):
        vram = f"~{cls.vram_gb:.0f}GB" if cls.vram_gb else "-"
        print(f"{name:<12} {'yes' if cls.commercial_ok else 'NO':<11} {vram:<7} "
              f"{cls.large_mask_score}/5{'':<8} {cls.licence}")
        if cls.notes:
            print(f"{'':<12} {cls.notes}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", help="source video")
    ap.add_argument("-o", "--output", help="destination mp4")
    ap.add_argument("--list-backends", action="store_true")

    g = ap.add_argument_group("what to remove")
    g.add_argument("--prompt", help='text query, e.g. "person" or "red car"')
    g.add_argument("--point", type=_point, action="append", default=[],
                   metavar="X,Y[,LABEL]", help="click a pixel; label 1=object 0=not (repeatable)")
    g.add_argument("--box", type=_box, metavar="X0,Y0,X1,Y1",
                   help="fixed rectangle, no model -- for testing the pipeline offline")
    g.add_argument("--init-frame", type=int, default=0,
                   help="frame the prompt refers to; pick one where the object is fully visible")

    g = ap.add_argument_group("mask shaping")
    g.add_argument("--dilate", type=int, default=12,
                   help="px of margin (default 12). Raise it if you see a halo")
    g.add_argument("--temporal-pad", type=int, default=2, help="union with +/- N frames")
    g.add_argument("--close", type=int, default=5, help="fill pinholes")
    g.add_argument("--min-area", type=int, default=64, help="drop blobs smaller than this")

    g = ap.add_argument_group("inpainting")
    g.add_argument("--backend", default="propainter", help="see --list-backends")
    g.add_argument("--backend-cmd", help="command template for --backend external")
    g.add_argument("--propainter-dir", help="ProPainter checkout (or set PROPAINTER_DIR)")
    g.add_argument("--work-res", type=int, default=720,
                   help="longest side fed to the models (default 720)")
    g.add_argument("--no-composite", action="store_true",
                   help="skip hi-res paste-back; output stays at --work-res")

    g = ap.add_argument_group("output")
    g.add_argument("--preview", action="store_true",
                   help="render the mask overlay and stop -- check this before burning GPU time")
    g.add_argument("--crf", type=int, default=16, help="x264 quality, lower is better")
    g.add_argument("--feather", type=int, default=4, help="seam softness in px")
    g.add_argument("--jpeg-frames", action="store_true",
                   help="lossy intermediates: ~half the scratch disk, but it degrades "
                        "the pixels outside the mask that you were trying to preserve")
    g.add_argument("--work-dir", help="scratch dir (kept if given)")
    g.add_argument("--device", default="cuda")

    g = ap.add_argument_group("models")
    g.add_argument("--sam2-cfg", default=Options.sam2_cfg)
    g.add_argument("--sam2-ckpt", default=Options.sam2_ckpt)
    g.add_argument("--box-threshold", type=float, default=0.30)
    g.add_argument("--text-threshold", type=float, default=0.25)

    a = ap.parse_args(argv)

    if a.list_backends:
        list_backends()
        return 0
    if not a.input or not a.output:
        ap.error("input and -o/--output are required")
    if not a.prompt and not a.point and not a.box:
        ap.error("say what to remove: --prompt TEXT, --point X,Y, or --box X0,Y0,X1,Y1")

    backend_opts = {}
    if a.backend_cmd:
        backend_opts["cmd"] = a.backend_cmd
    if a.propainter_dir:
        backend_opts["repo"] = a.propainter_dir

    opts = Options(
        prompt=a.prompt, points=a.point, static_box=a.box, init_frame=a.init_frame,
        backend=a.backend, backend_opts=backend_opts,
        work_res=a.work_res, composite=not a.no_composite, lossless=not a.jpeg_frames,
        crf=a.crf, feather=a.feather,
        dilate=a.dilate, temporal_pad=a.temporal_pad, close=a.close, min_area=a.min_area,
        device=a.device, sam2_cfg=a.sam2_cfg, sam2_ckpt=a.sam2_ckpt,
        box_threshold=a.box_threshold, text_threshold=a.text_threshold,
        preview=a.preview, keep_work=bool(a.work_dir),
    )

    try:
        if a.work_dir:
            run(a.input, a.output, a.work_dir, opts)
        else:
            with tempfile.TemporaryDirectory(prefix="vremove-") as tmp:
                run(a.input, a.output, Path(tmp), opts)
    except (ValueError, RuntimeError, KeyError) as e:
        # These carry actionable messages; a traceback would just bury them.
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
