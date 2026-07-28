FROM --platform=linux/amd64 pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

LABEL org.opencontainers.image.title="ORena FOCUS Procedure - Qwen3-VL-4B LoRA epoch 8 v2"
LABEL org.opencontainers.image.source="submissions/procedure/qwen3-vl-4b-lora-both-official-epoch8-v2-20260727"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    FOCUS_MODEL_DTYPE=bfloat16 \
    FOCUS_MAX_FRAMES=96

RUN groupadd --system algorithm \
    && useradd --system --create-home --gid algorithm algorithm

WORKDIR /opt/app
COPY --chown=algorithm:algorithm resources/python-vendor.tar /opt/app/resources/python-vendor.tar
COPY --chown=algorithm:algorithm resources/base_model/ /opt/app/resources/base_model/
COPY --chown=algorithm:algorithm resources/adapter/ /opt/app/resources/adapter/
COPY --chown=algorithm:algorithm inference.py /opt/app/inference.py

USER algorithm
ENTRYPOINT ["python", "/opt/app/inference.py"]
