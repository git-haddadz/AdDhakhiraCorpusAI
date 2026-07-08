import subprocess
import multiprocessing as mp
import os
from pathlib import Path, PurePosixPath

import modal


APP_NAME = "addhakhira"
APP_DIR = PurePosixPath("/workspace/AdDhakhiraCorpusAI")
PERSIST_DIR = APP_DIR / "persist"
MODELS_DIR = PERSIST_DIR / "models"
VECTOR_INDEX_DIR = PERSIST_DIR / "vector_indexes"
PORT = 7860

GPU = "A100-80GB"
VOLUME_NAME = "addhakhira-persist"

app = modal.App(APP_NAME)
persist_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_dockerfile("Dockerfile")
    .add_local_dir("src", remote_path=str(APP_DIR / "src"), copy=True)
    .add_local_dir("database", remote_path=str(APP_DIR / "database"), copy=True)
    .add_local_file("main.py", remote_path=str(APP_DIR / "main.py"), copy=True)
    .add_local_file("requirements.txt", remote_path=str(APP_DIR / "requirements.txt"), copy=True)
    .env(
        {
            "PYTHONPATH": str(APP_DIR),
            "HF_HOME": str(PERSIST_DIR / "huggingface"),
            "TRANSFORMERS_CACHE": str(PERSIST_DIR / "huggingface" / "transformers"),
            "SENTENCE_TRANSFORMERS_HOME": str(PERSIST_DIR / "sentence_transformers"),
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        }
    )
)


def _path(path: PurePosixPath) -> Path:
    return Path(str(path))


def _write_modal_config() -> None:
    config_path = _path(APP_DIR / "src" / "config.py")
    config_path.write_text(
        f"""
from pathlib import Path

LLM_BACKEND = "default"
GEMINI_API_KEY = ""
OPENAI_API_KEY = ""
ANTHROPIC_API_KEY = ""

REPO_ROOT = Path("{APP_DIR}")
JSON_INPUT_PATH = REPO_ROOT / "database"

MODEL_EXTRACTOR_PATH = "{MODELS_DIR / "gemma-4-12B-it"}"
MODEL_REASONER_PATH = "{MODELS_DIR / "Qwen3.6-35B-A3B"}"

TOP_K_CHUNKS = 8
TOP_K_PAGES = 5

MAX_MODEL_LEN_EXTRACTOR = 4096
MIN_MODEL_LEN_REASONER = 8192
NUM_GPUS_EXTRACTOR = 1
NUM_GPUS_REASONER = 1
VLLM_GPU_MEMORY_UTILIZATION = 0.90
VLLM_MAX_NUM_BATCHED_TOKENS = None

EMBEDDING_MODEL = "{MODELS_DIR / "Qwen__Qwen3-Embedding-4B"}"
EMBEDDING_INDEX_DIR = Path("{VECTOR_INDEX_DIR}")
VECTOR_INDEX_BACKEND = "faiss"
DENSE_TOP_K = 40
HYBRID_DENSE_WEIGHT = 0.40
HYBRID_LEXICAL_WEIGHT = 0.60
ENABLE_DENSE_RETRIEVAL = True
ENABLE_HYBRID_RETRIEVAL = False

TRANSLATE_TOP_CHUNKS_TO_FRENCH = False
TRANSLATION_MAX_TOKENS = 4096
AUTO_TRANSLATE_QUESTION_TO_ARABIC = False

REASONER_TEMPERATURE = 0.2
REASONER_TOP_P = 0.9
REASONER_OUTPUT_MAX_TOKENS = 1200
REASONER_CONTEXT_SAFETY_TOKENS = 8000
JSON_GENERATION_MAX_RETRIES = 4
JSON_GENERATION_MAX_TOKEN_MULTIPLIER = 4
""".lstrip(),
        encoding="utf-8",
    )


@app.function(
    image=image,
    gpu=GPU,
    volumes={str(PERSIST_DIR): persist_volume},
    timeout=60 * 60 * 6,
)
def prepare_models_and_index(force_index: bool = False) -> None:
    _write_modal_config()
    _path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    _path(VECTOR_INDEX_DIR).mkdir(parents=True, exist_ok=True)

    downloads = [
        ("google/gemma-4-12B-it", MODELS_DIR / "gemma-4-12B-it"),
        ("Qwen/Qwen3.6-35B-A3B", MODELS_DIR / "Qwen3.6-35B-A3B"),
        ("Qwen/Qwen3-Embedding-4B", MODELS_DIR / "Qwen__Qwen3-Embedding-4B"),
    ]
    for repo_id, target_dir in downloads:
        target_path = _path(target_dir)
        if target_path.exists() and any(target_path.iterdir()):
            print(f"Model already present: {target_dir}")
            continue
        subprocess.run(
            ["hf", "download", repo_id, "--local-dir", str(target_dir)],
            cwd=str(APP_DIR),
            check=True,
        )

    index_cmd = [
        "python3",
        "-m",
        "src.vector_index",
        "--model",
        str(MODELS_DIR / "Qwen__Qwen3-Embedding-4B"),
        "--backend",
        "faiss",
        "--json-input",
        str(APP_DIR / "database"),
        "--output-dir",
        str(VECTOR_INDEX_DIR),
        "--show-progress",
    ]
    if force_index:
        index_cmd.append("--force")
    subprocess.run(index_cmd, cwd=str(APP_DIR), check=True)
    persist_volume.commit()


@app.function(
    image=image,
    gpu=GPU,
    volumes={str(PERSIST_DIR): persist_volume},
    timeout=60 * 60,
    max_containers=1,
    scaledown_window=30,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app(label="addhakhira-webapp")
def gradio_webapp():
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    _write_modal_config()
    from fastapi import FastAPI
    import gradio as gr
    from src.web_app import _allowed_paths, build_demo

    demo = build_demo()
    demo.queue(default_concurrency_limit=1)
    return gr.mount_gradio_app(
        FastAPI(),
        demo,
        path="/",
        allowed_paths=_allowed_paths(),
    )


@app.local_entrypoint()
def main(prepare: bool = False, force_index: bool = False) -> None:
    if prepare:
        prepare_models_and_index.remote(force_index=force_index)
        return
    print("Use `modal run modal_app.py --prepare` first, then `modal deploy modal_app.py`.")
