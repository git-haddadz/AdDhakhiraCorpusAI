import argparse
import importlib
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

from src import config as base_config
from src.reporting import write_output_with_timing


BACKEND_CHOICES = [
    ("Local vLLM (config.py)", "default"),
    ("Gemini API", "gemini_api"),
    ("ChatGPT / OpenAI API", "openai_api"),
    ("Claude / Anthropic API", "anthropic_api"),
]

API_KEY_BY_BACKEND = {
    "gemini_api": "GEMINI_API_KEY",
    "openai_api": "OPENAI_API_KEY",
    "anthropic_api": "ANTHROPIC_API_KEY",
}

DEFAULT_API_MODELS = {
    "gemini_api": "gemini-2.5-flash",
    "openai_api": "gpt-4.1",
    "anthropic_api": "claude-sonnet-4-20250514",
}

PIPELINE_MODULE_PREFIXES = (
    "src.pipeline",
    "src.llm_ops",
    "src.llm_backend",
    "src.retrieval",
)

_pipeline_lock = threading.Lock()


def _config_value(name: str, default=None):
    return getattr(base_config, name, default)


def _model_for_backend(backend: str) -> str:
    current_backend = _config_value("LLM_BACKEND", "default")
    if current_backend == backend and backend != "default":
        return str(_config_value("MODEL_REASONER_PATH", DEFAULT_API_MODELS[backend]))
    return DEFAULT_API_MODELS[backend]


def _config_api_key_for_backend(backend: str) -> str:
    key_name = API_KEY_BY_BACKEND.get(backend)
    if not key_name:
        return ""
    return str(_config_value(key_name, "") or "").strip()


def _output_dir() -> Path:
    directory = Path(_config_value("REPO_ROOT", Path.cwd())) / "outputs" / "web_app"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _api_key_for_backend(backend: str, api_key: str) -> str:
    key_name = API_KEY_BY_BACKEND.get(backend)
    if not key_name:
        return ""
    return (
        (api_key or "").strip()
        or str(_config_value(key_name, "") or "").strip()
        or os.environ.get(key_name, "").strip()
    )


def _runtime_config(
    backend: str,
    api_key: str,
    dense_retrieval: bool,
) -> Dict[str, object]:
    cfg = {
        "LLM_BACKEND": backend,
        "ENABLE_DENSE_RETRIEVAL": bool(dense_retrieval),
        "GEMINI_API_KEY": "",
        "OPENAI_API_KEY": "",
        "ANTHROPIC_API_KEY": "",
        "MODEL_EXTRACTOR_PATH": _config_value("MODEL_EXTRACTOR_PATH"),
        "MODEL_REASONER_PATH": _config_value("MODEL_REASONER_PATH"),
    }
    if backend != "default":
        model_name = _model_for_backend(backend)
        key_name = API_KEY_BY_BACKEND[backend]
        cfg.update(
            {
                key_name: _api_key_for_backend(backend, api_key),
                "MODEL_EXTRACTOR_PATH": model_name,
                "MODEL_REASONER_PATH": model_name,
            }
        )
    return cfg


def _apply_runtime_config(cfg: Dict[str, object]) -> None:
    for key, value in cfg.items():
        setattr(base_config, key, value)


def _clear_pipeline_modules() -> None:
    for name in list(sys.modules):
        if name == "src.config":
            continue
        if name.startswith(PIPELINE_MODULE_PREFIXES):
            sys.modules.pop(name, None)


def _run_question(
    backend: str,
    api_key: str,
    dense_retrieval: bool,
    translate_pages: bool,
    diagnostic_coherence: bool,
    auto_translate_question: bool,
    question: str,
):
    question = (question or "").strip()
    if not question:
        return "Saisis une question.", "", None
    if backend != "default" and not _api_key_for_backend(backend, api_key):
        return "La clé API est requise pour ce backend.", "", None

    cfg = _runtime_config(
        backend=backend,
        api_key=api_key,
        dense_retrieval=dense_retrieval,
    )

    # The pipeline imports config values as module constants. Keep one run at a
    # time so requests cannot overwrite each other's runtime backend/key.
    with _pipeline_lock:
        try:
            _apply_runtime_config(cfg)
            _clear_pipeline_modules()
            pipeline = importlib.import_module("src.pipeline")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = _output_dir() / f"answer_{timestamp}.html"
            t0 = time.time()
            report = pipeline.build_final_report(
                question=question,
                translate_to_french=translate_pages,
                diagnostic_coherence=diagnostic_coherence,
                auto_translate_question_to_arabic=auto_translate_question,
            )
            elapsed = time.time() - t0
            write_output_with_timing(
                report,
                elapsed_seconds=elapsed,
                output_path=str(output_path),
            )
            status = f"Terminé en {elapsed:.1f}s avec {backend}. HTML sauvegardé : {output_path}"
            return status, report, str(output_path)
        except Exception as exc:
            return f"Erreur: {exc}", "", None


def _toggle_backend_fields(backend: str):
    import gradio as gr

    is_api = backend != "default"
    return gr.update(visible=is_api, value=_config_api_key_for_backend(backend) if is_api else "")


def build_demo():
    import gradio as gr

    initial_backend = str(_config_value("LLM_BACKEND", "default"))
    if initial_backend not in {value for _, value in BACKEND_CHOICES}:
        initial_backend = "default"

    with gr.Blocks(title="AdDhakhiraCorpusAI") as demo:
        gr.Markdown("## Assistant de recherche Malikite")
        with gr.Row():
            backend = gr.Dropdown(
                choices=BACKEND_CHOICES,
                value=initial_backend,
                label="Backend",
            )
            dense_retrieval = gr.Checkbox(
                label="Retrieval dense",
                value=bool(_config_value("ENABLE_DENSE_RETRIEVAL", True)),
            )

        api_key = gr.Textbox(
            label="Clé API",
            value=_config_api_key_for_backend(initial_backend) if initial_backend != "default" else "",
            visible=(initial_backend != "default"),
        )

        with gr.Row():
            translate_pages = gr.Checkbox(
                label="Traduire les pages en français",
                value=bool(_config_value("TRANSLATE_TOP_CHUNKS_TO_FRENCH", False)),
            )
            diagnostic_coherence = gr.Checkbox(label="Diagnostic cohérence", value=True)
            auto_translate_question = gr.Checkbox(
                label="Traduire la question en arabe",
                value=bool(_config_value("AUTO_TRANSLATE_QUESTION_TO_ARABIC", False)),
            )

        question = gr.Textbox(label="Question", lines=4)
        submit = gr.Button("Envoyer", variant="primary")
        status = gr.Markdown()
        answer = gr.HTML()
        download = gr.File(label="Télécharger la réponse HTML")

        backend.change(
            _toggle_backend_fields,
            inputs=backend,
            outputs=api_key,
        )
        submit.click(
            _run_question,
            inputs=[
                backend,
                api_key,
                dense_retrieval,
                translate_pages,
                diagnostic_coherence,
                auto_translate_question,
                question,
            ],
            outputs=[status, answer, download],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interface Gradio pour AdDhakhiraCorpusAI.")
    parser.add_argument("--host", default="0.0.0.0", help="Adresse d'écoute Gradio.")
    parser.add_argument("--port", type=int, default=7860, help="Port d'écoute Gradio.")
    parser.add_argument("--share", action="store_true", help="Créer un lien public Gradio temporaire.")
    parser.add_argument("--debug", action="store_true", help="Activer le mode debug Gradio.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo = build_demo()
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        allowed_paths=[str(_output_dir())],
    )


if __name__ == "__main__":
    main()
