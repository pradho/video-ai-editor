#!/usr/bin/env bash
# Bootstrap on a RunPod GPU pod (PyTorch template, CUDA 12.x).
# Torch is already installed there -- reinstalling it is the #1 way to break the pod.
#
#   bash scripts/setup_pod.sh
#   source scripts/env.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "== system deps =="
apt-get update -qq
apt-get install -y -qq ffmpeg git wget libgl1 libglib2.0-0

echo "== python deps =="
pip install -q -r requirements.txt
pip install -q "git+https://github.com/facebookresearch/sam2.git"

echo "== ProPainter (S-Lab licence, non-commercial) =="
[ -d ProPainter ] || git clone -q --depth 1 https://github.com/sczhou/ProPainter

echo "== weights =="
bash scripts/fetch_weights.sh "$ROOT/weights" "$ROOT/ProPainter"

echo "== warming Grounding DINO into the HF cache =="
python - <<'PY'
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
m = "IDEA-Research/grounding-dino-base"
AutoProcessor.from_pretrained(m); AutoModelForZeroShotObjectDetection.from_pretrained(m)
print("ok")
PY

cat > "$ROOT/scripts/env.sh" <<EOF
export PROPAINTER_DIR="$ROOT/ProPainter"
export HF_HOME="\${HF_HOME:-/workspace/.cache/huggingface}"
EOF

echo
echo "done. next:"
echo "  source scripts/env.sh"
echo "  python remove.py in.mp4 -o preview.mp4 --prompt 'person' --preview"
