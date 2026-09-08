"""Video in/out for the serverless worker.

RunPod's job payload is small and not meant for media, so video travels over
presigned URLs: the client signs a GET for the input and a PUT for the output,
and the worker holds no bucket credentials at all. Base64 exists only as a
smoke-test path for short clips.
"""
from __future__ import annotations

import base64
from pathlib import Path

import requests

CONTENT_TYPE = "video/mp4"


class TooLarge(ValueError):
    pass


def download(url: str, dest, *, max_bytes: int, timeout: int = 120) -> Path:
    dest = Path(dest)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()

        declared = int(r.headers.get("content-length") or 0)
        if declared and declared > max_bytes:
            raise TooLarge(f"input is {declared / 1e6:.1f} MB, limit is {max_bytes / 1e6:.0f} MB")

        seen = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                seen += len(chunk)
                # Re-check while streaming: content-length can be absent or lie.
                if seen > max_bytes:
                    raise TooLarge(f"input exceeded {max_bytes / 1e6:.0f} MB while downloading")
                f.write(chunk)
    return dest


def upload_presigned(path, url: str, *, timeout: int = 600) -> str:
    """PUT to a presigned URL. Content-Type must match what the client signed."""
    with open(path, "rb") as f:
        r = requests.put(url, data=f, headers={"Content-Type": CONTENT_TYPE}, timeout=timeout)
    r.raise_for_status()
    return url.split("?", 1)[0]


def from_base64(b64: str, dest) -> Path:
    dest = Path(dest)
    dest.write_bytes(base64.b64decode(b64))
    return dest


def to_base64(path, *, max_bytes: int) -> str:
    data = Path(path).read_bytes()
    if len(data) > max_bytes:
        raise TooLarge(
            f"result is {len(data) / 1e6:.1f} MB, too big to inline "
            f"(limit {max_bytes / 1e6:.0f} MB) -- pass output_url instead")
    return base64.b64encode(data).decode()
