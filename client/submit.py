#!/usr/bin/env python3
"""Submit a job to the RunPod serverless endpoint from your laptop.

    python client/submit.py in.mp4 -o preview.mp4 --prompt "person" --preview
    python client/submit.py in.mp4 -o out.mp4 --prompt "person"

Two transport modes:

  presigned (default)  uploads to your S3/R2 bucket, hands the worker signed
                       GET/PUT URLs. The worker never sees your credentials.
  --inline             base64 through the job payload. Short clips only; use it
                       to prove the endpoint works before wiring up a bucket.

Environment:
    RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID
    S3_ENDPOINT, S3_BUCKET, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY  (presigned)
    S3_REGION   default "auto"  (correct for Cloudflare R2)
"""
from __future__ import annotations

import argparse
import base64
import os
import sys
import time
import uuid
from pathlib import Path

import requests

API = "https://api.runpod.ai/v2"
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}


def env(name, required=True, default=None):
    v = os.environ.get(name, default)
    if required and not v:
        sys.exit(f"missing env {name}")
    return v


# --------------------------------------------------------------------------- #
# bucket
# --------------------------------------------------------------------------- #
def s3_client():
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=env("S3_ENDPOINT"),
        aws_access_key_id=env("S3_ACCESS_KEY_ID"),
        aws_secret_access_key=env("S3_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("S3_REGION", "auto"),
        config=Config(signature_version="s3v4"),
    )


def stage(src: Path, ttl: int):
    """Upload the source, return (signed GET for input, signed PUT for output)."""
    s3, bucket = s3_client(), env("S3_BUCKET")
    job = uuid.uuid4().hex[:12]
    in_key, out_key = f"vremove/{job}/in{src.suffix}", f"vremove/{job}/out.mp4"

    print(f"[up  ] {src.name} -> s3://{bucket}/{in_key}")
    s3.upload_file(str(src), bucket, in_key, ExtraArgs={"ContentType": "video/mp4"})

    get_url = s3.generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": in_key}, ExpiresIn=ttl)
    # ContentType must match the header the worker sends, or the signature fails.
    put_url = s3.generate_presigned_url(
        "put_object",
        Params={"Bucket": bucket, "Key": out_key, "ContentType": "video/mp4"},
        ExpiresIn=ttl)
    return get_url, put_url, (s3, bucket, out_key)


# --------------------------------------------------------------------------- #
# runpod
# --------------------------------------------------------------------------- #
def submit(payload: dict):
    eid, key = env("RUNPOD_ENDPOINT_ID"), env("RUNPOD_API_KEY")
    h = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    # Async /run, never /runsync: these jobs run for minutes and a synchronous
    # HTTP call will time out long before the video is done.
    r = requests.post(f"{API}/{eid}/run", json={"input": payload}, headers=h, timeout=60)
    r.raise_for_status()
    job_id = r.json()["id"]
    print(f"[job ] {job_id}")
    return job_id, eid, h


def poll(job_id, eid, headers, interval=3.0):
    last = None
    t0 = time.time()
    try:
        while True:
            r = requests.get(f"{API}/{eid}/status/{job_id}", headers=headers, timeout=30)
            r.raise_for_status()
            d = r.json()
            st = d.get("status")

            note = st
            out = d.get("output")
            if isinstance(out, dict) and "stage" in out:
                note = f"{st}  {out.get('percent', 0):>3.0f}%  {out['stage']}"
            elif isinstance(out, list) and out and isinstance(out[-1], dict):
                p = out[-1]
                note = f"{st}  {p.get('percent', 0):>3.0f}%  {p.get('stage', '')}"

            if note != last:
                print(f"[{time.time() - t0:>6.0f}s] {note}")
                last = note
            if st in TERMINAL:
                return d
            time.sleep(interval)
    except KeyboardInterrupt:
        # Do NOT just exit -- an abandoned job keeps burning GPU seconds.
        requests.post(f"{API}/{eid}/cancel/{job_id}", headers=headers, timeout=30)
        sys.exit("\ncancelled")


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--prompt")
    ap.add_argument("--point", action="append", default=[], metavar="X,Y[,LABEL[,GROUP]]",
                    help="click a pixel; label 1=object 0=not-object (repeatable). "
                         "Different GROUP values track as separate SAM 2 objects, e.g. "
                         "two different people -- default group is 0 for all points")
    ap.add_argument("--no-bidirectional", action="store_true",
                    help="only track forward from --init-frame instead of both directions")
    ap.add_argument("--box", metavar="X0,Y0,X1,Y1",
                    help="fixed rectangle, no model -- smoke-test the endpoint")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--backend", default="propainter")
    ap.add_argument("--work-res", type=int, default=720)
    ap.add_argument("--dilate", type=int, default=12)
    ap.add_argument("--init-frame", type=int, default=0)
    ap.add_argument("--no-composite", action="store_true")
    ap.add_argument("--inline", action="store_true", help="base64 transport, short clips only")
    ap.add_argument("--ttl", type=int, default=3600, help="presigned URL lifetime (s)")
    a = ap.parse_args(argv)

    src, dst = Path(a.input), Path(a.output)
    if not src.is_file():
        sys.exit(f"no such file: {src}")
    if not a.prompt and not a.point and not a.box:
        ap.error("say what to remove: --prompt TEXT, --point X,Y, or --box X0,Y0,X1,Y1")

    points = []
    for p in a.point:
        parts = [int(x) for x in p.split(",")]
        if len(parts) == 2:
            parts = parts + [1]          # default label: foreground
        points.append(parts)

    payload = {
        "prompt": a.prompt, "points": points, "init_frame": a.init_frame,
        "backend": a.backend, "work_res": a.work_res, "dilate": a.dilate,
        "preview": a.preview, "composite": not a.no_composite,
        "bidirectional": not a.no_bidirectional,
    }
    if a.box:
        payload["box"] = [int(x) for x in a.box.split(",")]

    sink = None
    if a.inline:
        mb = src.stat().st_size / 1e6
        if mb > 8:
            sys.exit(f"{mb:.1f} MB is too big for --inline; drop the flag and use a bucket")
        payload["video_base64"] = base64.b64encode(src.read_bytes()).decode()
    else:
        get_url, put_url, sink = stage(src, a.ttl)
        payload["video_url"] = get_url
        payload["output_url"] = put_url

    job_id, eid, headers = submit(payload)
    res = poll(job_id, eid, headers)

    if res.get("status") != "COMPLETED":
        print(f"[fail] {res.get('status')}: {res.get('error') or res.get('output')}")
        return 1

    out = res.get("output") or {}
    # An OOM is reported as {"refresh_worker": True, "job_results": {...}} --
    # RunPod itself still calls the JOB "COMPLETED" since the handler returned
    # normally instead of raising. Without unwrapping this, the code below
    # falls through to the success path and 404s trying to download a file
    # that was never produced -- confirmed live, that 404 is what actually
    # sent us looking for this shape in the first place.
    if isinstance(out.get("job_results"), dict):
        out = out["job_results"]
    if out.get("status") == "error":
        print(f"[fail] {out['error']}")
        if out.get("hint"):
            print(f"       hint: {out['hint']}")
        return 1

    for w in out.get("warnings") or []:
        print(f"[warn] {w}")
    print(f"[stat] coverage={out.get('coverage')}  frames={out.get('frames')}  "
          f"gpu={out.get('seconds')}s")

    dst.parent.mkdir(parents=True, exist_ok=True)
    if out.get("video_base64"):
        dst.write_bytes(base64.b64decode(out["video_base64"]))
    elif sink:
        s3, bucket, key = sink
        s3.download_file(bucket, key, str(dst))
    else:
        print(f"[out ] {out.get('output_url')}")
        return 0
    print(f"[out ] {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
