FROM --platform=linux/amd64 docker.io/library/python:3.11-slim AS reg2-evaluation-amd64

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

USER user

WORKDIR /opt/app

ENV PATH=/home/user/.local/bin:${PATH}

COPY --chown=user:user requirements.txt /opt/app/requirements.txt

# GPU-ready PyTorch (linux/amd64). Still runs evaluations on CPU when JUDGE_DEVICE=cpu / no GPU allocated.
RUN python -m pip install --user --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --user --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu124 \
        torch \
    && python -m pip install --user --no-cache-dir -r /opt/app/requirements.txt

# Sci spaCy model matching KeywordEvaluator(default) in evaluate_metrics.py
RUN python -m pip install --user --no-cache-dir \
    https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz

# Defaults match evaluation_config.env (overridden at build time via do_build.sh --build-arg)
ARG GRAND_CHALLENGE_MAX_WORKERS=
ARG JUDGE_DEVICE=auto
ARG JUDGE_MODEL_PATH=/opt/ml/input/data/ground_truth/Qwen3-8B
ARG EMBEDDING_MODEL=NeuML/pubmedbert-base-embeddings

# Pre-cache embedding weights for offline evaluation (--network none at runtime).
# transformers cache alone is insufficient: SemanticScorer uses SentenceTransformer (modules.json).
RUN python -c "from transformers import AutoModel, AutoTokenizer; from sentence_transformers import SentenceTransformer; m='${EMBEDDING_MODEL}'; AutoTokenizer.from_pretrained(m); AutoModel.from_pretrained(m); SentenceTransformer(m)"

COPY --chown=user:user helpers.py evaluate.py evaluate_metrics.py /opt/app/
ARG INCLUDE_PER_CASE_RESULTS=0
ARG INCLUDE_EVALUATION_DETAILS=0

ENV GRAND_CHALLENGE_MAX_WORKERS=${GRAND_CHALLENGE_MAX_WORKERS}
ENV JUDGE_DEVICE=${JUDGE_DEVICE}
ENV JUDGE_MODEL_PATH=${JUDGE_MODEL_PATH}
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL}
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV INCLUDE_PER_CASE_RESULTS=${INCLUDE_PER_CASE_RESULTS}
ENV INCLUDE_EVALUATION_DETAILS=${INCLUDE_EVALUATION_DETAILS}

ENTRYPOINT ["python", "evaluate.py"]
