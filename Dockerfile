FROM nvcr.io/nvidia/vllm:26.06-py3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/AdDhakhiraCorpusAI/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/AdDhakhiraCorpusAI/.cache/huggingface/transformers \
    SENTENCE_TRANSFORMERS_HOME=/workspace/AdDhakhiraCorpusAI/.cache/sentence_transformers

WORKDIR /workspace/AdDhakhiraCorpusAI

COPY requirements.txt /tmp/requirements.txt

RUN python3 -m pip install --upgrade pip \
    && PIP_CONSTRAINT= python3 -m pip install -r /tmp/requirements.txt

CMD ["bash"]
