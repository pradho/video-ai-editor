#!/usr/bin/env python3
"""Serverless entrypoint.

This file exists at the repo ROOT, and the runpod.serverless.start() call below
is at module top level rather than behind `if __name__ == "__main__":`, because
RunPod's GitHub builder statically scans the repository for that call. With the
handler nested in a subdirectory or wrapped in a __main__ guard the build is
rejected with:

    runpod.serverless.start() handler not found in your repo

All the actual logic is in vremove/serverless.py. Keep this file thin.

Local smoke test (no models loaded, no GPU):
    PRELOAD=0 python handler.py       # reads test_input.json from the cwd
"""
import runpod

from vremove.serverless import handler

runpod.serverless.start({"handler": handler})
