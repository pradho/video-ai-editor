"""ffmpeg/ffprobe wrappers: probe, frame extraction, re-encode, audio remux."""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"


def _run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{Path(cmd[0]).name} failed:\n{p.stderr[-4000:]}")
    return p.stdout


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    nframes: int
    has_audio: bool
    duration: float


def probe(path: str | Path) -> VideoInfo:
    out = _run([FFPROBE, "-v", "error", "-print_format", "json",
                "-show_streams", "-show_format", str(path)])
    d = json.loads(out)
    v = next(s for s in d["streams"] if s["codec_type"] == "video")
    has_audio = any(s["codec_type"] == "audio" for s in d["streams"])

    num, _, den = v.get("avg_frame_rate", "0/1").partition("/")
    fps = float(num) / float(den) if float(den or 0) else 0.0

    duration = float(d.get("format", {}).get("duration") or 0.0)
    nframes = int(v.get("nb_frames") or 0)
    if not nframes and fps and duration:
        nframes = int(round(fps * duration))

    return VideoInfo(int(v["width"]), int(v["height"]), fps, nframes, has_audio, duration)


def _scale_filter(max_side: int) -> str:
    """Fit the longest side to max_side, never upscale, keep both dims even."""
    return (f"scale=w='if(gt(iw,ih),min(iw,{max_side}),-2)'"
            f":h='if(gt(iw,ih),-2,min(ih,{max_side}))'")


def extract_frames(src: str | Path, out_dir: str | Path, *,
                   max_side: int | None = None, lossless: bool = False) -> tuple[Path, str]:
    """Dump frames as 00000.jpg / 00000.png.

    Pure-integer filenames are required: SAM 2's video predictor sorts a frame
    directory with int(basename), so any prefix makes it throw.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = "png" if lossless else "jpg"

    cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", str(src)]
    if max_side:
        cmd += ["-vf", _scale_filter(max_side)]
    if ext == "jpg":
        cmd += ["-q:v", "2"]
    cmd += ["-start_number", "0", str(out_dir / f"%05d.{ext}")]
    _run(cmd)
    return out_dir, ext


def frame_paths(d: str | Path, ext: str | None = None) -> list[Path]:
    d = Path(d)
    pats = [ext] if ext else ["png", "jpg"]
    for p in pats:
        files = sorted(d.glob(f"*.{p}"), key=lambda f: int(f.stem))
        if files:
            return files
    return []


def encode(frames_dir: str | Path, out_path: str | Path, fps: float, *,
           ext: str = "png", audio_from: str | Path | None = None,
           crf: int = 16, preset: str = "slow") -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [FFMPEG, "-y", "-loglevel", "error",
           "-framerate", f"{fps:.6f}", "-start_number", "0",
           "-i", str(Path(frames_dir) / f"%05d.{ext}")]
    if audio_from:
        cmd += ["-i", str(audio_from)]
    cmd += ["-c:v", "libx264", "-crf", str(crf), "-preset", preset,
            "-pix_fmt", "yuv420p"]
    if audio_from:
        # `0:a?` would grab nothing -- audio lives on input 1. The trailing ?
        # keeps it optional so a silent source still encodes.
        cmd += ["-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac", "-b:a", "192k",
                "-shortest"]
    cmd += [str(out_path)]
    _run(cmd)
    return out_path
