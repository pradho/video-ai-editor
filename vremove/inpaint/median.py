"""Temporal-median background plate. No GPU, no model, no licence.

Only valid when the camera does not move. When it applies it beats every
learned model outright, because it recovers the *actual* background pixels
rather than hallucinating plausible ones -- and a long-lived mask helps here
instead of hurting, since more frames means more chances to observe what sits
behind the object.
"""
from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from ..video_io import frame_paths
from .base import Inpainter, register


@register
class MedianInpainter(Inpainter):
    name = "median"
    licence = "none (first-party code)"
    commercial_ok = True
    vram_gb = 0.0
    large_mask_score = 5
    notes = "static camera only; output is exact background, not a guess"

    def run(self, frames_dir, masks_dir, out_dir, *, fps: float) -> str:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        frames = frame_paths(frames_dir)
        masks = frame_paths(masks_dir, "png")
        if not frames:
            raise RuntimeError(f"no frames in {frames_dir}")
        n = len(frames)
        h, w = cv2.imread(str(frames[0])).shape[:2]

        budget = int(self.opts.get("ram_budget_bytes", 2 << 30))
        rows = max(1, min(h, budget // max(1, n * w * 3 * 4)))

        with tempfile.TemporaryDirectory() as tmp:
            # One decode pass into a memmap; the median then streams over
            # row-bands so peak RAM stays at `budget` regardless of clip length.
            buf = np.memmap(Path(tmp) / "f.dat", np.uint8, "w+", shape=(n, h, w, 3))
            mbuf = np.memmap(Path(tmp) / "m.dat", np.bool_, "w+", shape=(n, h, w))
            for i, (fp, mp) in enumerate(tqdm(list(zip(frames, masks)), desc="load")):
                buf[i] = cv2.imread(str(fp))
                m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                if m.shape != (h, w):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                mbuf[i] = m > 127

            plate = np.zeros((h, w, 3), np.uint8)
            never_seen = np.zeros((h, w), np.uint8)
            for y0 in tqdm(range(0, h, rows), desc="median"):
                y1 = min(h, y0 + rows)
                band = buf[:, y0:y1].astype(np.float32)
                band[mbuf[:, y0:y1]] = np.nan
                with warnings.catch_warnings():
                    # All-NaN columns are expected and handled below; the
                    # warning would fire once per band and drown the log.
                    warnings.simplefilter("ignore", RuntimeWarning)
                    med = np.nanmedian(band, axis=0)
                bad = np.isnan(med).any(-1)
                med[bad] = 0
                plate[y0:y1] = np.clip(med, 0, 255).astype(np.uint8)
                never_seen[y0:y1] = bad.astype(np.uint8) * 255

            del buf, mbuf

        # Pixels the object covered in *every* frame have no observation at all.
        if never_seen.any():
            plate = cv2.inpaint(plate, never_seen, 5, cv2.INPAINT_TELEA)
            frac = float((never_seen > 0).mean())
            print(f"[median] {frac:.2%} of pixels were occluded in every frame "
                  f"-- filled by diffusion, expect softness there")

        feather = int(self.opts.get("feather", 3))
        for fp, mp in tqdm(list(zip(frames, masks)), desc="composite"):
            img = cv2.imread(str(fp))
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            a = m.astype(np.float32) / 255.0
            if feather > 0:
                a = cv2.GaussianBlur(a, (feather * 2 + 1,) * 2, 0)
            a = a[..., None]
            cv2.imwrite(str(out_dir / f"{int(fp.stem):05d}.png"),
                        (img * (1 - a) + plate * a).astype(np.uint8))
        return "png"
