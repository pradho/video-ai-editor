"""Hi-res paste-back.

The trick that keeps output sharp without renting an 80 GB card: inpaint at
512-720p, then blend *only the masked region* back into the untouched
full-resolution frames. Everything outside the mask stays original pixels, so
a 4K source comes back 4K everywhere the object never was.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .video_io import frame_paths


def composite_hires(orig_dir, inpainted_dir, mask_dir, out_dir, *,
                    orig_ext=None, feather=4):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    origs = frame_paths(orig_dir, orig_ext)
    inps = frame_paths(inpainted_dir, "png")
    masks = frame_paths(mask_dir, "png")
    if not (len(origs) == len(inps) == len(masks)):
        raise RuntimeError(
            f"frame count mismatch: orig={len(origs)} inpainted={len(inps)} masks={len(masks)}")

    for op, ip, mp in tqdm(list(zip(origs, inps, masks)), desc="composite"):
        orig = cv2.imread(str(op))
        h, w = orig.shape[:2]

        inp = cv2.imread(str(ip))
        if inp.shape[:2] != (h, w):
            inp = cv2.resize(inp, (w, h), interpolation=cv2.INTER_LANCZOS4)

        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m.shape != (h, w):
            # Linear, not nearest: we want a soft ramp at the seam.
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)

        a = m.astype(np.float32) / 255.0
        if feather > 0:
            a = cv2.GaussianBlur(a, (feather * 2 + 1,) * 2, 0)
        a = a[..., None]

        cv2.imwrite(str(out_dir / f"{int(op.stem):05d}.png"),
                    (orig * (1 - a) + inp * a).astype(np.uint8))
    return out_dir
