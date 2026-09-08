"""ProPainter adapter (flow-guided propagation + transformer).

Not a pip package -- it is a research repo run through its own CLI, so this
shells out to it. Clone it and point PROPAINTER_DIR at the checkout.

LICENCE: S-Lab License 1.0, research / non-commercial. Fine for personal use,
but it cannot ship in a paid product. See `external` for the commercial path.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .base import Inpainter, register


@register
class ProPainter(Inpainter):
    name = "propainter"
    licence = "S-Lab 1.0 (NON-COMMERCIAL)"
    commercial_ok = False
    vram_gb = 10.0
    large_mask_score = 2
    notes = "best quality/VRAM ratio, but blurs out on wide or long-lived masks"

    def run(self, frames_dir, masks_dir, out_dir, *, fps: float) -> str:
        repo = Path(self.opts.get("repo") or os.environ.get("PROPAINTER_DIR", "")).expanduser()
        script = repo / "inference_propainter.py"
        if not script.is_file():
            raise RuntimeError(
                "ProPainter not found. Clone it and set PROPAINTER_DIR:\n"
                "  git clone https://github.com/sczhou/ProPainter\n"
                "  export PROPAINTER_DIR=$PWD/ProPainter"
            )

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp:
            cmd = [
                sys.executable, str(script),
                "--video", str(Path(frames_dir).resolve()),
                "--mask", str(Path(masks_dir).resolve()),
                "--output", str(Path(tmp).resolve()),
                "--save_frames",
                # We already dilated deliberately in masking.py; letting
                # ProPainter dilate again on top would over-erase.
                "--mask_dilation", "0",
                "--subvideo_length", str(self.opts.get("subvideo_length", 80)),
                "--neighbor_length", str(self.opts.get("neighbor_length", 10)),
                "--ref_stride", str(self.opts.get("ref_stride", 10)),
                "--save_fps", str(int(round(fps)) or 25),
            ]
            if self.opts.get("fp16", True):
                cmd.append("--fp16")

            print("[propainter]", " ".join(cmd))
            # Captured, not inherited: a subprocess crash used to leave nothing
            # but "ProPainter exited 1" in the job result, sending you digging
            # through RunPod's log viewer for a traceback that may not even be
            # there (tqdm/child-process stdout doesn't always reach RunPod's
            # log shipper). Printed here so it still lands in container logs,
            # AND the tail comes back in the exception so it reaches the
            # client directly.
            p = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
            if p.stdout:
                print("[propainter stdout]\n" + p.stdout[-4000:])
            if p.stderr:
                print("[propainter stderr]\n" + p.stderr[-4000:])
            if p.returncode != 0:
                tail = "\n".join(p.stderr.strip().splitlines()[-40:]) or "(no stderr captured)"
                raise RuntimeError(f"ProPainter exited {p.returncode}\n--- stderr tail ---\n{tail}")

            # It nests results under <output>/<input basename>/frames.
            produced = sorted(Path(tmp).rglob("*.png")) or sorted(Path(tmp).rglob("*.jpg"))
            if not produced:
                raise RuntimeError(f"ProPainter produced no frames under {tmp}")
            for i, f in enumerate(produced):
                shutil.copyfile(f, out_dir / f"{i:05d}.png")
        return "png"


@register
class External(Inpainter):
    """Escape hatch for any repo with a frames-in / frames-out CLI.

    The commercial-safe upgrades (DiffuEraser, MiniMax-Remover, VACE/Wan2.1)
    each ship their own script with their own flags, and those flags move
    between releases -- so rather than guess them, supply the command yourself:

        --backend external --backend-cmd "python /workspace/DiffuEraser/run.py \\
            --input {frames} --mask {masks} --output {out}"

    Placeholders: {frames} {masks} {out} {fps}. Any *.png/*.jpg written
    anywhere under {out} is collected in sorted order.
    """
    name = "external"
    licence = "depends on the repo you point it at"
    commercial_ok = True
    vram_gb = 0.0
    large_mask_score = 4
    notes = "bring your own model; the route to an Apache-2.0 backend"

    def run(self, frames_dir, masks_dir, out_dir, *, fps: float) -> str:
        tpl = self.opts.get("cmd")
        if not tpl:
            raise RuntimeError("backend 'external' requires --backend-cmd")

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp:
            cmd = tpl.format(frames=Path(frames_dir).resolve(),
                             masks=Path(masks_dir).resolve(),
                             out=Path(tmp).resolve(), fps=fps)
            print("[external]", cmd)
            p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if p.stdout:
                print("[external stdout]\n" + p.stdout[-4000:])
            if p.stderr:
                print("[external stderr]\n" + p.stderr[-4000:])
            if p.returncode != 0:
                tail = "\n".join(p.stderr.strip().splitlines()[-40:]) or "(no stderr captured)"
                raise RuntimeError(f"external backend exited {p.returncode}\n--- stderr tail ---\n{tail}")

            produced = sorted(Path(tmp).rglob("*.png")) or sorted(Path(tmp).rglob("*.jpg"))
            if not produced:
                raise RuntimeError(f"external backend produced no frames under {tmp}")
            for i, f in enumerate(produced):
                shutil.copyfile(f, out_dir / f"{i:05d}.png")
        return "png"
