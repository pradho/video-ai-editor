"""Text prompt -> boxes (Grounding DINO) -> per-frame masks (SAM 2 video predictor).

Mask post-processing lives here too. It matters more than people expect: a mask
that hugs the object exactly leaves shadows, motion blur and hair edges behind,
and every inpainter will faithfully reconstruct that residue as a halo.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .video_io import frame_paths

DEFAULT_GDINO = "IDEA-Research/grounding-dino-base"
DEFAULT_SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_SAM2_CKPT = "weights/sam2.1_hiera_large.pt"

# Process-wide model cache. The CLI loads once and exits, so this changes
# nothing there -- but a serverless worker handles many jobs in one process,
# and re-loading SAM 2 + Grounding DINO per job would add ~15s of billed time
# to every single request.
_CACHE: dict = {}


def load_gdino(model_id=DEFAULT_GDINO, device="cuda"):
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    key = ("gdino", model_id, device)
    if key not in _CACHE:
        proc = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
        _CACHE[key] = (proc, model)
    return _CACHE[key]


def load_sam2(cfg=DEFAULT_SAM2_CFG, ckpt=DEFAULT_SAM2_CKPT, device="cuda"):
    from sam2.build_sam import build_sam2_video_predictor

    key = ("sam2", cfg, ckpt, device)
    if key not in _CACHE:
        _CACHE[key] = build_sam2_video_predictor(cfg, ckpt, device=device)
    return _CACHE[key]


def preload(*, gdino=DEFAULT_GDINO, cfg=DEFAULT_SAM2_CFG,
            ckpt=DEFAULT_SAM2_CKPT, device="cuda"):
    """Warm both models. Call at container start, not per request."""
    load_gdino(gdino, device)
    load_sam2(cfg, ckpt, device)


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #
def detect_boxes(image_path, prompt, *, model_id=DEFAULT_GDINO, device="cuda",
                 box_threshold=0.30, text_threshold=0.25, max_boxes=10):
    """Return xyxy boxes in pixel coords for `prompt` on a single frame."""
    import torch
    from PIL import Image

    # Grounding DINO wants lowercase phrases terminated by a period.
    text = prompt.strip().lower()
    if not text.endswith("."):
        text += "."

    proc, model = load_gdino(model_id, device)

    img = Image.open(image_path).convert("RGB")
    inputs = proc(images=img, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # transformers renamed box_threshold -> threshold around 4.51; support both.
    kw = dict(input_ids=inputs.input_ids, text_threshold=text_threshold,
              target_sizes=[img.size[::-1]])
    try:
        res = proc.post_process_grounded_object_detection(outputs, threshold=box_threshold, **kw)
    except TypeError:
        res = proc.post_process_grounded_object_detection(outputs, box_threshold=box_threshold, **kw)

    boxes = res[0]["boxes"].detach().cpu().numpy()
    scores = res[0]["scores"].detach().cpu().numpy()
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)

    order = np.argsort(-scores)[:max_boxes]
    return boxes[order].astype(np.float32)


# --------------------------------------------------------------------------- #
# tracking
# --------------------------------------------------------------------------- #
def track_masks(frames_dir, out_dir, *, boxes=None, points=None,
                cfg=DEFAULT_SAM2_CFG, ckpt=DEFAULT_SAM2_CKPT,
                device="cuda", init_frame=0, offload=True):
    """Propagate prompts from `init_frame` across the clip, writing 00000.png masks.

    `points` is [(x, y, label)] with label 1 = foreground, 0 = background.
    `offload` keeps decoded frames and per-frame state on CPU -- slower per
    frame, but SAM 2 otherwise pins the whole clip in VRAM and long videos OOM.
    """
    import torch

    frames_dir, out_dir = Path(frames_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = frame_paths(frames_dir, "jpg")
    if not files:
        raise RuntimeError(f"no jpg frames in {frames_dir} -- SAM 2 requires jpg")
    h, w = cv2.imread(str(files[0])).shape[:2]

    predictor = load_sam2(cfg, ckpt, device)

    amp = (torch.autocast("cuda", dtype=torch.bfloat16) if device.startswith("cuda")
           else contextlib.nullcontext())
    with torch.inference_mode(), amp:
        state = predictor.init_state(video_path=str(frames_dir),
                                     offload_video_to_cpu=offload,
                                     offload_state_to_cpu=offload)

        n_obj = 0
        for i, box in enumerate(boxes if boxes is not None else []):
            predictor.add_new_points_or_box(state, frame_idx=init_frame,
                                            obj_id=i + 1, box=np.asarray(box, np.float32))
            n_obj += 1
        if points:
            pts = np.array([[p[0], p[1]] for p in points], np.float32)
            lbl = np.array([p[2] for p in points], np.int32)
            predictor.add_new_points_or_box(state, frame_idx=init_frame,
                                            obj_id=n_obj + 1, points=pts, labels=lbl)
            n_obj += 1
        if n_obj == 0:
            raise ValueError("no boxes and no points -- nothing to track")

        seen = set()
        for idx, _obj_ids, logits in tqdm(predictor.propagate_in_video(state),
                                          total=len(files), desc="sam2 track"):
            m = (logits > 0).squeeze(1).any(0).cpu().numpy()
            cv2.imwrite(str(out_dir / f"{idx:05d}.png"), (m * 255).astype(np.uint8))
            seen.add(idx)

        # The predictor is cached across jobs; the per-video state is not.
        predictor.reset_state(state)
        del state

    # Frames before init_frame are never visited by forward propagation.
    blank = np.zeros((h, w), np.uint8)
    for i in range(len(files)):
        if i not in seen:
            cv2.imwrite(str(out_dir / f"{i:05d}.png"), blank)
    return out_dir


def static_box_masks(frames_dir, out_dir, box):
    """A fixed rectangle on every frame. No model, no GPU, no download.

    Not for real work -- it exists so the rest of the pipeline (extract, mask
    shaping, inpaint, composite, encode) can be exercised end to end on a
    laptop before anything is rented.
    """
    frames_dir, out_dir = Path(frames_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = frame_paths(frames_dir, "jpg") or frame_paths(frames_dir)
    if not files:
        raise RuntimeError(f"no frames in {frames_dir}")

    h, w = cv2.imread(str(files[0])).shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, x1 = sorted((max(0, x0), min(w, x1)))
    y0, y1 = sorted((max(0, y0), min(h, y1)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"box {tuple(box)} is empty inside a {w}x{h} frame")

    m = np.zeros((h, w), np.uint8)
    m[y0:y1, x0:x1] = 255
    for f in files:
        cv2.imwrite(str(out_dir / f"{int(f.stem):05d}.png"), m)
    return out_dir


# --------------------------------------------------------------------------- #
# post-processing
# --------------------------------------------------------------------------- #
def postprocess_masks(mask_dir, *, dilate=12, temporal_pad=2, close=5, min_area=64):
    """Grow masks in space and time so the inpainter never sees leftover edges.

    dilate       px of margin, absorbs shadow / motion blur / hair
    temporal_pad union with +/- N neighbouring frames, covers tracking lag
    close        fills pinholes inside the object
    min_area     drops speckle blobs
    """
    mask_dir = Path(mask_dir)
    files = frame_paths(mask_dir, "png")
    masks = [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in files]

    if close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close * 2 + 1,) * 2)
        masks = [cv2.morphologyEx(m, cv2.MORPH_CLOSE, k) for m in masks]

    if min_area > 0:
        cleaned = []
        for m in masks:
            n, lab, stats, _ = cv2.connectedComponentsWithStats((m > 127).astype(np.uint8), 8)
            keep = np.zeros_like(m)
            for i in range(1, n):
                if stats[i, cv2.CC_STAT_AREA] >= min_area:
                    keep[lab == i] = 255
            cleaned.append(keep)
        masks = cleaned

    if temporal_pad > 0:
        padded = []
        for i in range(len(masks)):
            lo, hi = max(0, i - temporal_pad), min(len(masks), i + temporal_pad + 1)
            acc = masks[lo]
            for j in range(lo + 1, hi):
                acc = cv2.bitwise_or(acc, masks[j])
            padded.append(acc)
        masks = padded

    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate * 2 + 1,) * 2)
        masks = [cv2.dilate(m, k) for m in masks]

    for f, m in zip(files, masks):
        cv2.imwrite(str(f), m)
    return mask_dir


def coverage(mask_dir):
    """Mean fraction of pixels masked -- used to warn about weak backends."""
    files = frame_paths(mask_dir, "png")
    if not files:
        return 0.0
    tot = 0.0
    for f in files:
        m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        tot += float((m > 127).mean())
    return tot / len(files)


def write_overlay(frames_dir, mask_dir, out_dir, ext="jpg"):
    """Red-tinted mask preview so you can sanity-check before paying for GPU."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in tqdm(frame_paths(frames_dir, ext), desc="overlay"):
        img = cv2.imread(str(f))
        m = cv2.imread(str(Path(mask_dir) / f"{int(f.stem):05d}.png"), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            if m.shape[:2] != img.shape[:2]:
                m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            tint = np.zeros_like(img)
            tint[..., 2] = 255
            a = (m > 127)[..., None] * 0.45
            img = (img * (1 - a) + tint * a).astype(np.uint8)
        cv2.imwrite(str(out_dir / f"{int(f.stem):05d}.png"), img)
    return out_dir
