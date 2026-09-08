"""Inpainter plugin contract + registry.

Every backend is swappable behind one method, because the licence question is
still open: ProPainter is S-Lab (non-commercial), so a commercial launch means
switching to an Apache-2.0 model without touching the rest of the pipeline.
"""
from __future__ import annotations

REGISTRY: dict[str, type] = {}


def register(cls):
    REGISTRY[cls.name] = cls
    return cls


class Inpainter:
    # --- metadata, surfaced by `remove.py --list-backends` ---
    name = "base"
    licence = "unknown"
    commercial_ok = False
    vram_gb = 0.0
    # How well it holds up when the mask is large or on screen for a long
    # stretch, 1 (falls apart) .. 5 (solid). This is the axis that matters here.
    large_mask_score = 0
    notes = ""

    def __init__(self, **opts):
        self.opts = opts

    def run(self, frames_dir, masks_dir, out_dir, *, fps: float) -> str:
        """Fill masked regions. Writes 00000.png frames into out_dir.

        Input frames and masks are already at working resolution and share
        filenames; the caller handles the hi-res paste-back.
        """
        raise NotImplementedError


def get(name: str, **opts) -> Inpainter:
    if name not in REGISTRY:
        raise KeyError(f"unknown backend {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name](**opts)
