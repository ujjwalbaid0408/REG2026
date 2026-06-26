FROM --platform=linux/amd64 pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime AS reg2026_algorithm_amd64
# PyTorch + CUDA base so GPU is available at inference time.

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1
# Ensure imports like "from core import ..." and the vendored "conch" package work
ENV PYTHONPATH=/opt/app
# Fully offline at inference: weights are bundled in model.tar.gz, no network is used.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# OpenSlide is a system C library required by openslide-python (WSI tiling). Install as root
# before dropping privileges.
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends libopenslide0 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/

# You can add any Python dependencies to requirements.txt
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

COPY --chown=user:user core.py      /opt/app/
COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user conch/       /opt/app/conch/
COPY --chown=user:user src/         /opt/app/src/

ENTRYPOINT ["python", "inference.py"]
