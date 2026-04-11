from pathlib import Path

# === Backend LLM ===
# custom: exécution locale via vLLM
# gemini_api: exécution via API Gemini
LLM_BACKEND = "custom"
GEMINI_API_KEY = ""

# === Modèles (Extractor / Reasoner) ===
# custom: chemins locaux
# gemini_api: IDs de modèles Gemini
MODEL_EXTRACTOR_PATH = "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"  # modèle par défaut
MODEL_REASONER_PATH = "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"  # modèle par défaut

# === Chemins ===
REPO_ROOT = Path(__file__).resolve().parent.parent
JSON_INPUT_PATH = REPO_ROOT / "database"

# === Paramètres RAG ===
TOP_K_CHUNKS = 8
TOP_K_PAGES = 5

# === Hyperparamètres Modèles ===
MAX_MODEL_LEN_EXTRACTOR = 4096
MIN_MODEL_LEN_REASONER = 8192
NUM_GPUS_EXTRACTOR = 2
NUM_GPUS_REASONER = 2

# === Embeddings ===
EMBEDDING_MODEL = None

# === Traduction ===
TRANSLATE_TOP_CHUNKS_TO_FRENCH = False
TRANSLATION_MAX_TOKENS = 4096
AUTO_TRANSLATE_QUESTION_TO_ARABIC = False

# === Hyperparamètres de Génération / Vérification ===
REASONER_TEMPERATURE = 0.2
REASONER_TOP_P = 0.9
REASONER_OUTPUT_MAX_TOKENS = 1200
REASONER_CONTEXT_SAFETY_TOKENS = 8000
