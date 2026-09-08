#!/usr/bin/env bash
# usage: fetch_weights.sh [weights_dir] [propainter_dir]
# Shared by scripts/setup_pod.sh (pod) and worker/Dockerfile (serverless image).
set -euo pipefail

W="${1:-weights}"
PP="${2:-ProPainter}"
mkdir -p "$W" "$PP/weights"

fetch() {  # url dest
  if [ -s "$2" ]; then echo "have    $(basename "$2")"; return 0; fi
  echo "fetch   $(basename "$2")"
  wget -q --show-progress -O "$2" "$1"
}

# --- SAM 2.1 ---------------------------------------------------------------
fetch https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \
      "$W/sam2.1_hiera_large.pt"

# --- ProPainter ------------------------------------------------------------
# Release assets. If a fetch 404s the tag has moved -- check
# https://github.com/sczhou/ProPainter/releases and bump PP_TAG. A miss is not
# fatal: ProPainter downloads its own weights on first use, you just pay for it
# on every cold start instead of once at build time.
PP_TAG="${PP_TAG:-v0.1.0}"
PP_BASE="https://github.com/sczhou/ProPainter/releases/download/${PP_TAG}"

for f in ProPainter.pth raft-things.pth recurrent_flow_completion.pth; do
  if ! fetch "$PP_BASE/$f" "$PP/weights/$f"; then
    rm -f "$PP/weights/$f"
    echo "WARN    $f not baked -- will download at runtime (slow cold start)"
  fi
done

echo
echo "weights in $W and $PP/weights:"
du -sh "$W" "$PP/weights" 2>/dev/null || true
