from pathlib import Path

# === Modèles (Hugging Face IDs) ===
MODEL_EXTRACTOR_ID = "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
MODEL_REASONER_ID = "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"

# === Chemins ===
REPO_ROOT = Path(__file__).resolve().parent.parent  # remonte de src/ vers la racine
JSON_INPUT_PATH = REPO_ROOT / "database"

# === Paramètres RAG ===
TOP_K_CHUNKS = 8
TOP_K_PAGES = 5

# === Modèles ===
MAX_MODEL_LEN_EXTRACTOR = 4096
MIN_MODEL_LEN_REASONER = 8192
NUM_GPUS_EXTRACTOR = 2
NUM_GPUS_REASONER = 2

# === Embeddings ===
EMBEDDING_MODEL = None

# === Traduction ===
TRANSLATE_TOP_CHUNKS_TO_FRENCH = False
TRANSLATION_MAX_TOKENS = 4096

# === Reasoner ===
REASONER_TEMPERATURE = 0.2
REASONER_TOP_P = 0.9
REASONER_OUTPUT_MAX_TOKENS = 1200
REASONER_CONTEXT_SAFETY_TOKENS = 8000