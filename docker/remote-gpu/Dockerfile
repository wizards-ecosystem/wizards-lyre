# The Wizard's Lyre - Remote GPU host, as a container.
#
# The one place Lyre uses Docker (SPEC.md section 3.1). The local app is never
# containerized; this image is what runs on a GPU you rent or own.
#
# Everything is installed here, once, on a build machine: on rented hardware,
# installing dependencies at startup is a network call and a resolver run on a
# machine billed by the second. Weights are NOT baked in. The host fetches
# the profiles it needs on first use (worker/remote_host/models.py), so the
# image stays publishable and a session pays only for what it renders.
#
# No secrets are baked in either: LYRE_REMOTE_SECRET arrives as deploy-time
# environment. That is what makes the image safe to publish.
#
# Build from the repository root:
#   ./scripts/lyre remote-image
#
# --platform=linux/amd64 matters: Runpod hosts are x86-64, and an arm64 image
# built on Apple Silicon fails with "exec format error" only after the push.
#
# Never tag this `latest`. Hosts cache images per machine, so a mutable tag
# serves stale code; the launcher tags with the build id /health reports.

# A plain Python base, not nvidia/cuda: torch's cu128 wheels carry their own
# CUDA runtime, and the driver comes from the host through the container
# runtime. A CUDA base would be a second copy of the same libraries.
ARG BASE_IMAGE=python:3.12-slim-bookworm
FROM ghcr.io/astral-sh/uv:0.11.17 AS uv
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PYTHON_DOWNLOADS=never \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# git: fetch ACE-Step at its pinned revision. libgomp1: torch's OpenMP runtime,
# absent from slim images and only missed at import. ffmpeg/libsndfile1: audio
# decoding for cover and repaint sources.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git libgomp1 ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /usr/local/bin/uv

# ACE-Step, exactly as ./scripts/lyre install sets it up locally: the pinned
# revision, its own checked-in lock, its unused web UI removed, and Lyre's
# tested security overlay applied last.
COPY ACE_STEP_REVISION requirements/ace-step-security.txt /tmp/lyre/
RUN revision="$(tr -d '\n' < /tmp/lyre/ACE_STEP_REVISION)" \
    && git init -q /opt/ace-step \
    && git -C /opt/ace-step fetch -q --depth=1 \
        https://github.com/ACE-Step/ACE-Step-1.5.git "$revision" \
    && git -C /opt/ace-step checkout -q --detach FETCH_HEAD \
    && rm -rf /opt/ace-step/.git \
    && uv venv --python /usr/local/bin/python3.12 /opt/venv \
    && uv sync --project /opt/ace-step --active --locked --inexact --no-dev \
    && uv pip uninstall --python /opt/venv/bin/python gradio gradio-client \
    && uv pip install --python /opt/venv/bin/python \
        --requirement /tmp/lyre/ace-step-security.txt \
    && rm -rf /tmp/lyre

# Only the modules the host runs: its own package, the ACE-Step adapter it
# wraps, and the mock backend for GPU-free smoke tests. Nothing from server/.
WORKDIR /app
# The build id (worker/remote_host/__init__.py) also hashes these, so the image
# keeps them where it can find them and reports the same id as its source.
COPY ACE_STEP_REVISION /app/
COPY requirements/ace-step-security.txt /app/requirements/
COPY docker/remote-gpu/Dockerfile /app/docker/remote-gpu/
COPY worker/__init__.py worker/mock_worker.py /app/worker/
COPY worker/acestep_worker /app/worker/acestep_worker
COPY worker/remote_host /app/worker/remote_host

# Weights and caches live under LYRE_REMOTE_ROOT. The in-image default is lost
# with the container; point it at a volume to keep weights between pods.
ENV LYRE_REMOTE_HOST=0.0.0.0 \
    LYRE_REMOTE_PORT=8000 \
    LYRE_REMOTE_ROOT=/app/data

EXPOSE 8000
CMD ["python", "-m", "worker.remote_host"]
