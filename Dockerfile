# Build context is the REPO ROOT. This file sits here rather than in worker/ so
# it works whether the builder uses the repo root or the Dockerfile's own
# directory as the context -- RunPod's GitHub builder picks one, and a mismatch
# would break every COPY below.
#
#   docker build -t <user>/video-removal:v1 .
#
# -runtime, not -devel: 3.3GB vs 7GB, and nothing here compiles CUDA. SAM 2
# installs with SAM2_BUILD_CUDA=0 (skips its one optional CUDA op) and
# ProPainter is pure PyTorch.
#
# torch 2.6, NOT 2.4.1: transformers 5.x gates every model class behind
# `is_torch_available()`, which returns False for torch < 2.5.0
# (transformers/utils/import_utils.py). On 2.4.1 the Grounding DINO warm-up
# below dies with an opaque "PyTorch was not found" and fails the whole build.
#
# CUDA stays at 12.4 -- fine for Ada/Ampere (4090, L4, A5000, A100). If you
# ever target Blackwell (5090, sm_120) switch to a -cuda12.8- tag.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    PROPAINTER_DIR=/opt/ProPainter \
    SAM2_CKPT=/app/weights/sam2.1_hiera_large.pt \
    SAM2_BUILD_CUDA=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# build-essential is insurance: the runtime base has no compiler, and a single
# sdist-only dependency would otherwise fail the build. ~80MB compressed
# against the 3.9GB saved by dropping -devel.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git wget ca-certificates libgl1 libglib2.0-0 build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- python deps (cached layer; changes rarely) -----------------------------
COPY requirements.txt ./
RUN pip install -r requirements.txt runpod requests \
 && pip install "git+https://github.com/facebookresearch/sam2.git"

# --- ProPainter: a repo, not a package --------------------------------------
# Its own requirements.txt is itself incomplete -- it lists imageio-ffmpeg but
# not imageio, and imageio-ffmpeg does not pull imageio in as a dependency.
# Confirmed live: a real removal job failed with
#   ModuleNotFoundError: No module named 'imageio'
# torch/torchvision lines are stripped before install -- the base image
# already has a matched torch 2.6.0 CUDA build, and letting pip "satisfy"
# ProPainter's loose `torch>=1.7.1` could pull a mismatched CPU/CUDA wheel.
# opencv-python is stripped too: we already installed opencv-python-headless,
# and having both installed can fight over which cv2 wins.
RUN git clone --depth 1 https://github.com/sczhou/ProPainter /opt/ProPainter \
 && grep -viE '^(torch|opencv-python)' /opt/ProPainter/requirements.txt > /tmp/pp-reqs.txt \
 && pip install -r /tmp/pp-reqs.txt imageio

# --- weights baked into the image -------------------------------------------
# Deliberately not a network volume: a volume pins the endpoint to a single
# datacenter, and GPU availability there collapses at peak. ~2GB of layer is a
# cheaper price than a job that cannot find a worker.
COPY scripts/fetch_weights.sh /tmp/fetch_weights.sh
RUN bash /tmp/fetch_weights.sh /app/weights /opt/ProPainter

# --- compat check + Grounding DINO into the image's HF cache ----------------
COPY scripts/warm_cache.py /tmp/warm_cache.py
RUN python /tmp/warm_cache.py

# --- app code last: the only layer that rebuilds while you iterate ----------
COPY vremove/ /app/vremove/
COPY remove.py handler.py /app/

WORKDIR /app
CMD ["python", "-u", "handler.py"]
