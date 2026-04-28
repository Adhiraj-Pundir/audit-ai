# syntax=docker/dockerfile:1.6
# BuildKit required. Build with:
#   DOCKER_BUILDKIT=1 docker build \
#     --secret id=hf_token,env=HF_TOKEN \
#     -t gcr.io/$PROJECT_ID/auditai-backend .
#
# The HF token is injected via a secret mount — it never appears in any image
# layer or in `docker history`. The final runtime image has no trace of it.

# ── Stage 1: download model weights ───────────────────────────────────────────
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS model-fetcher

RUN apt-get update && apt-get install -y python3.12 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir huggingface_hub transformers

# Secret is mounted read-only at /run/secrets/hf_token for this RUN only.
# It is never written to the filesystem and never cached in an image layer.
RUN --mount=type=secret,id=hf_token \
    HUGGING_FACE_HUB_TOKEN=$(cat /run/secrets/hf_token) \
    python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('google/gemma-2-2b', local_dir='/weights', \
  ignore_patterns=['*.msgpack','flax_model*','tf_model*'])"


# ── Stage 2: runtime image ─────────────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.12 python3-pip && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/python3.12 /usr/local/bin/python3 && \
    ln -s /usr/bin/pip3 /usr/local/bin/pip

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Weights copied from fetcher stage — no token needed here
COPY --from=model-fetcher /weights /weights

ENV MODEL_LOCAL_PATH=/weights
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HOME=/weights

COPY app/ ./app/

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
