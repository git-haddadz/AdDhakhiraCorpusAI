from pathlib import Path

# Copy this file to src/config.py, then edit the values for your machine.

# === Backend LLM ===
# default: local inference through vLLM
# gemini_api/openai_api/anthropic_api: external API inference
LLM_BACKEND = "default"
GEMINI_API_KEY = ""
OPENAI_API_KEY = ""
ANTHROPIC_API_KEY = ""

# === Models (Extractor / Reasoner) ===
# For LLM_BACKEND="default", use local model directories or HF model IDs.
# For API backends, use API model IDs.
MODEL_EXTRACTOR_PATH = "/path/to/extractor-model"
MODEL_REASONER_PATH = "/path/to/reasoner-model"

# === Paths ===
REPO_ROOT = Path(__file__).resolve().parent.parent
JSON_INPUT_PATH = REPO_ROOT / "database"

# === RAG parameters ===
TOP_K_CHUNKS = 8
TOP_K_PAGES = 5

# === Model runtime parameters ===
MAX_MODEL_LEN_EXTRACTOR = 4096
MIN_MODEL_LEN_REASONER = 8192
NUM_GPUS_EXTRACTOR = 1
NUM_GPUS_REASONER = 1
VLLM_GPU_MEMORY_UTILIZATION = 0.90
VLLM_MAX_NUM_BATCHED_TOKENS = None

# === Embeddings / Vector retrieval ===
EMBEDDING_MODEL = "/path/to/Qwen__Qwen3-Embedding-4B"
EMBEDDING_INDEX_DIR = REPO_ROOT / "database" / "vector_indexes"
VECTOR_INDEX_BACKEND = "faiss"
DENSE_TOP_K = 40
HYBRID_DENSE_WEIGHT = 0.40
HYBRID_LEXICAL_WEIGHT = 0.60
ENABLE_DENSE_RETRIEVAL = True
ENABLE_HYBRID_RETRIEVAL = False

# === Translation ===
TRANSLATE_TOP_CHUNKS_TO_FRENCH = False
TRANSLATION_MAX_TOKENS = 4096
AUTO_TRANSLATE_QUESTION_TO_ARABIC = False

# === Generation / verification parameters ===
REASONER_TEMPERATURE = 0.2
REASONER_TOP_P = 0.9
REASONER_OUTPUT_MAX_TOKENS = 1200
REASONER_CONTEXT_SAFETY_TOKENS = 8000
JSON_GENERATION_MAX_RETRIES = 4
JSON_GENERATION_MAX_TOKEN_MULTIPLIER = 4
