import subprocess
import multiprocessing as mp
import html
import hmac
import os
import secrets
import urllib.parse
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
AUTH_ENV_VAR = "ADDHAKHIRA_AUTH"
AUTH_COOKIE_NAME = "addhakhira_session"
_ACTIVE_SESSIONS = {}

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


def _auth_credentials():
    raw_auth = os.environ.get(AUTH_ENV_VAR, "").strip()
    if not raw_auth:
        return None

    credentials = []
    for raw_pair in raw_auth.replace("\n", ",").split(","):
        pair = raw_pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise ValueError(
                f"{AUTH_ENV_VAR} must use the format user:password,user2:password2"
            )
        username, password = pair.split(":", 1)
        username = username.strip()
        password = password.strip()
        if not username or not password:
            raise ValueError(
                f"{AUTH_ENV_VAR} contains an empty username or password."
            )
        credentials.append((username, password))

    return credentials or None


def _auth_credentials_dict():
    credentials = _auth_credentials()
    if not credentials:
        return None
    return dict(credentials)


def _login_page(error: str = "") -> str:
    error_html = (
        f'<p class="error">{html.escape(error)}</p>'
        if error
        else ""
    )
    return f"""
<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Connexion - AdDhakhiraCorpusAI</title>
  <style>
    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f8fafc;
      color: #0f172a;
    }}
    main {{
      width: min(420px, calc(100vw - 32px));
      padding: 28px;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      background: white;
      box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
    }}
    h1 {{
      margin: 0 0 20px;
      font-size: 1.35rem;
      line-height: 1.25;
    }}
    label {{
      display: block;
      margin: 14px 0 6px;
      font-weight: 650;
    }}
    input {{
      width: 100%;
      box-sizing: border-box;
      min-height: 44px;
      padding: 9px 11px;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      font: inherit;
    }}
    button {{
      width: 100%;
      min-height: 48px;
      margin-top: 20px;
      border: 0;
      border-radius: 6px;
      background: #0f766e;
      color: white;
      font: inherit;
      font-weight: 750;
      cursor: pointer;
    }}
    button:hover {{
      background: #115e59;
    }}
    .error {{
      margin: 0 0 14px;
      color: #b91c1c;
      font-weight: 650;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Connexion à l'assistant</h1>
    {error_html}
    <form method="post" action="/login">
      <label for="username">Utilisateur</label>
      <input id="username" name="username" autocomplete="username" required>
      <label for="password">Mot de passe</label>
      <input id="password" name="password" type="password" autocomplete="current-password" required>
      <button type="submit">Se connecter</button>
    </form>
  </main>
</body>
</html>
""".strip()


def _build_authenticated_app(demo, allowed_paths):
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, RedirectResponse
    import gradio as gr

    credentials = _auth_credentials_dict()
    web_app = FastAPI()

    if credentials:

        @web_app.middleware("http")
        async def require_login(request: Request, call_next):
            if request.url.path == "/login":
                return await call_next(request)
            session_id = request.cookies.get(AUTH_COOKIE_NAME, "")
            if session_id in _ACTIVE_SESSIONS:
                return await call_next(request)
            return RedirectResponse("/login", status_code=303)

        @web_app.get("/login", response_class=HTMLResponse)
        async def login_form():
            return HTMLResponse(_login_page())

        @web_app.post("/login", response_class=HTMLResponse)
        async def login_submit(request: Request):
            body = (await request.body()).decode("utf-8", errors="replace")
            form = urllib.parse.parse_qs(body, keep_blank_values=True)
            username = form.get("username", [""])[0].strip()
            password = form.get("password", [""])[0]
            expected_password = credentials.get(username)

            if not expected_password or not hmac.compare_digest(password, expected_password):
                return HTMLResponse(_login_page("Identifiants incorrects."), status_code=401)

            session_id = secrets.token_urlsafe(32)
            _ACTIVE_SESSIONS[session_id] = username
            response = RedirectResponse("/", status_code=303)
            response.set_cookie(
                AUTH_COOKIE_NAME,
                session_id,
                httponly=True,
                secure=True,
                samesite="lax",
            )
            return response

    return gr.mount_gradio_app(
        web_app,
        demo,
        path="/",
        allowed_paths=allowed_paths,
    )


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
    secrets=[modal.Secret.from_name("addhakhira-auth")],
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
    from src.web_app import _allowed_paths, build_demo

    demo = build_demo()
    demo.queue(default_concurrency_limit=1)
    return _build_authenticated_app(demo, _allowed_paths())


@app.local_entrypoint()
def main(prepare: bool = False, force_index: bool = False) -> None:
    if prepare:
        prepare_models_and_index.remote(force_index=force_index)
        return
    print("Use `modal run modal_app.py --prepare` first, then `modal deploy modal_app.py`.")
