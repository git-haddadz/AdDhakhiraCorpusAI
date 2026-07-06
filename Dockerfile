FROM nvcr.io/nvidia/vllm:26.06-py3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/AdDhakhiraCorpusAI/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/AdDhakhiraCorpusAI/.cache/huggingface/transformers \
    SENTENCE_TRANSFORMERS_HOME=/workspace/AdDhakhiraCorpusAI/.cache/sentence_transformers

WORKDIR /workspace/AdDhakhiraCorpusAI

COPY requirements.txt /tmp/requirements.txt

# The NVIDIA vLLM image provides the validated GPU/runtime stack already.
# Keep requirements.txt as the source of truth, but do not reinstall packages
# that are pinned by the base image or by architecture-specific GPU wheels.
RUN grep -vE '^(torch==|numpy==|https://github.com/vllm-project/vllm|flashinfer-python==|triton==)' \
        /tmp/requirements.txt > /tmp/requirements.runtime.txt \
    && python3 -m pip install --upgrade pip \
    && python3 -m pip install -r /tmp/requirements.runtime.txt

CMD ["bash"]
