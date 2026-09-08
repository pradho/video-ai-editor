#!/usr/bin/env python3
"""Build-time compatibility check + Hugging Face cache warm.

Two failure modes, handled deliberately differently:

* A torch/transformers mismatch is our own bug and fully deterministic. Fail
  the build loudly with both versions printed -- far better than shipping an
  image whose every job dies inside from_pretrained with "PyTorch was not
  found", which is what transformers reports when it silently disables every
  model class.

* The model download is network-dependent and transient. Warn and carry on;
  the worker fetches it on the first job, paying one slow cold start instead
  of failing a 15-minute build.
"""
import sys

import torch
import transformers
from transformers.utils import is_torch_available

MODEL = "IDEA-Research/grounding-dino-base"

print(f"torch        {torch.__version__}")
print(f"transformers {transformers.__version__}")

if not is_torch_available():
    sys.exit(
        f"\nFATAL: transformers {transformers.__version__} refuses to use "
        f"torch {torch.__version__}.\n"
        f"transformers 5.x requires torch >= 2.5.0 and disables every model\n"
        f"class below that (see transformers/utils/import_utils.py).\n\n"
        f"Fix one of:\n"
        f"  - bump the FROM tag in the Dockerfile to a torch >= 2.5 image\n"
        f"  - pin transformers < 5 in requirements.txt\n")

# Grounding DINO's image processor needs one of two backends. Check it here,
# hard: without either, from_pretrained raises "Could not load any image
# processor class", and that is a deterministic packaging bug -- not something
# to let slide into a runtime failure on the first job.
backends = []
for name in ("PIL", "torchvision"):
    try:
        __import__(name)
        backends.append(name)
    except ImportError:
        pass
print(f"backends     {', '.join(backends) or 'NONE'}")
if not backends:
    sys.exit("\nFATAL: Grounding DINO needs Pillow or torchvision for its image\n"
             "processor and neither is importable. Add Pillow to requirements.txt.\n")

try:
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    AutoProcessor.from_pretrained(MODEL)
    AutoModelForZeroShotObjectDetection.from_pretrained(MODEL)
    print(f"cached       {MODEL}")
except Exception as e:
    print(f"WARN: could not pre-cache {MODEL}: {type(e).__name__}: {e}")
    print("      the worker will download it on the first job (slow cold start)")
    print("      if this is an HF rate limit, pass a token at build time:")
    print("        ARG HF_TOKEN / ENV HF_TOKEN, then --build-arg HF_TOKEN=hf_...")
