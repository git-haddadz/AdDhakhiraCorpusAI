from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


def _apply_transformers_compatibility_shims(model_name: str) -> None:
    lower = (model_name or "").lower()

    if "jina-embeddings-v3" in lower:
        try:
            from transformers import PreTrainedModel

            if not hasattr(PreTrainedModel, "all_tied_weights_keys"):

                @property
                def all_tied_weights_keys(self):
                    keys = getattr(self, "_tied_weights_keys", None) or []
                    return {key: key for key in keys}

                PreTrainedModel.all_tied_weights_keys = all_tied_weights_keys
        except Exception:
            pass

    if "gte-qwen2-7b-instruct" in lower:
        try:
            from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

            if not hasattr(Qwen2Config, "rope_theta"):
                Qwen2Config.rope_theta = 1000000.0
        except Exception:
            pass


@dataclass(frozen=True)
class EmbeddingAdapter:
    model_name: str
    query_instruction: Optional[str] = None
    passage_instruction: Optional[str] = None

    def format_query(self, text: str) -> str:
        return self._format(text, self.query_instruction)

    def format_passage(self, text: str) -> str:
        return self._format(text, self.passage_instruction)

    @staticmethod
    def _format(text: str, instruction: Optional[str]) -> str:
        text = (text or "").strip()
        if not instruction:
            return text
        return f"{instruction.strip()}\n{text}"


def get_embedding_adapter(model_name: str) -> EmbeddingAdapter:
    lower = (model_name or "").lower()
    if "qwen3-embedding" in lower:
        return EmbeddingAdapter(
            model_name=model_name,
            query_instruction="Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery:",
            passage_instruction="Passage:",
        )
    if "jina-embeddings-v3" in lower:
        return EmbeddingAdapter(
            model_name=model_name,
            query_instruction="Represent the query for retrieving supporting passages:",
            passage_instruction="Represent the passage for retrieval:",
        )
    if "multilingual-e5-large-instruct" in lower:
        return EmbeddingAdapter(
            model_name=model_name,
            query_instruction="Instruct: Retrieve relevant passages for the question.\nQuery:",
            passage_instruction="Passage:",
        )
    if "gte-qwen2-7b-instruct" in lower:
        return EmbeddingAdapter(
            model_name=model_name,
            query_instruction="Instruct: Retrieve relevant passages for the query.\nQuery:",
            passage_instruction="Passage:",
        )
    if "multilingual-e5" in lower:
        return EmbeddingAdapter(
            model_name=model_name,
            query_instruction="query:",
            passage_instruction="passage:",
        )
    return EmbeddingAdapter(model_name=model_name)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (arr / norms).astype(np.float32, copy=False)


class EmbeddingModel:
    def __init__(
        self,
        model_name: str,
        cache_folder: Optional[str] = None,
        devices: Optional[Sequence[str]] = None,
        show_progress: bool = False,
    ):
        if not model_name:
            raise ValueError("An embedding model name/path is required.")
        self.model_name = model_name
        self.adapter = get_embedding_adapter(model_name)
        self.devices = [d for d in (devices or []) if d]
        self.show_progress = show_progress
        _apply_transformers_compatibility_shims(model_name)
        device = "cpu" if len(self.devices) > 1 else None
        self.model = SentenceTransformer(
            model_name,
            cache_folder=cache_folder,
            trust_remote_code=True,
            device=device,
        )

    def encode_chunks(self, texts: Iterable[str], batch_size: int = 32) -> np.ndarray:
        formatted = [self.adapter.format_passage(t) for t in texts]
        if len(self.devices) > 1:
            return self._encode_multi_process(formatted, batch_size=batch_size)
        return self._encode(formatted, batch_size=batch_size)

    def encode_query(self, query: str) -> np.ndarray:
        return self._encode([self.adapter.format_query(query)], batch_size=1)

    def _encode(self, texts: List[str], batch_size: int) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=self.show_progress,
            normalize_embeddings=False,
        )
        return normalize_vectors(np.asarray(vectors, dtype=np.float32))

    def _encode_multi_process(self, texts: List[str], batch_size: int) -> np.ndarray:
        pool = self.model.start_multi_process_pool(target_devices=list(self.devices))
        try:
            vectors = self.model.encode_multi_process(
                texts,
                pool,
                batch_size=batch_size,
                show_progress_bar=self.show_progress,
                normalize_embeddings=False,
            )
        finally:
            self.model.stop_multi_process_pool(pool)
        return normalize_vectors(np.asarray(vectors, dtype=np.float32))

    def close(self) -> None:
        try:
            del self.model
        except AttributeError:
            pass
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
