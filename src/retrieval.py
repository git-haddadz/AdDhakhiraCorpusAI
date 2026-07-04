from typing import Dict, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from src.config import (
    DENSE_TOP_K,
    EMBEDDING_INDEX_DIR,
    ENABLE_DENSE_RETRIEVAL,
    ENABLE_HYBRID_RETRIEVAL,
    HYBRID_DENSE_WEIGHT,
    HYBRID_LEXICAL_WEIGHT,
    VECTOR_INDEX_BACKEND,
)
from src.embeddings import EmbeddingModel
from src.models import PageDoc, TextChunk
from src.text_utils import is_editorial_noise_page, normalize_arabic, tokenize_for_bm25
from src.vector_index import load_compatible_index


class HybridRetriever:
    def __init__(
        self,
        chunks: List[TextChunk],
        embedding_model_name: Optional[str] = None,
        enable_dense: bool = ENABLE_DENSE_RETRIEVAL,
        enable_hybrid: bool = ENABLE_HYBRID_RETRIEVAL,
        vector_backend: str = VECTOR_INDEX_BACKEND,
        index_root=EMBEDDING_INDEX_DIR,
        dense_top_k: int = DENSE_TOP_K,
        lexical_weight: float = HYBRID_LEXICAL_WEIGHT,
        dense_weight: float = HYBRID_DENSE_WEIGHT,
    ):
        self.chunks = chunks
        self.bm25_corpus = [tokenize_for_bm25(c.normalized_lexical_text) for c in chunks]
        self.bm25 = BM25Okapi(self.bm25_corpus) if self.bm25_corpus else None
        self.embedding_model_name = embedding_model_name
        self.enable_dense = bool(enable_dense and embedding_model_name)
        self.enable_hybrid = bool(enable_hybrid)
        self.vector_backend = vector_backend
        self.index_root = index_root
        self.dense_top_k = int(dense_top_k)
        self.lexical_weight = float(lexical_weight)
        self.dense_weight = float(dense_weight)
        self.embedder = None
        self.vector_index = None
        self.dense_enabled = False
        self._load_dense_if_enabled()

    def _load_dense_if_enabled(self):
        if not self.enable_dense:
            return
        if not self.chunks:
            return
        self.vector_index = load_compatible_index(
            self.index_root,
            self.embedding_model_name,
            self.vector_backend,
            self.chunks,
        )
        self.embedder = EmbeddingModel(self.embedding_model_name)
        self.dense_enabled = True

    def close(self) -> None:
        if self.embedder is not None and hasattr(self.embedder, "close"):
            self.embedder.close()
        self.embedder = None
        self.vector_index = None

    @staticmethod
    def _minmax(arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return arr
        vmin = float(arr.min())
        vmax = float(arr.max())
        if vmax - vmin < 1e-9:
            return np.zeros_like(arr, dtype=np.float32)
        return (arr - vmin) / (vmax - vmin)

    def search(self, query: str, keywords: List[str], top_k: int = 8) -> List[Dict]:
        if not self.chunks:
            return []

        normalized_keywords = [normalize_arabic(k).lower() for k in keywords if k.strip()]
        query_norm = normalize_arabic(query).lower()
        lexical_query = " ".join([query_norm] + normalized_keywords).strip()
        query_tokens = tokenize_for_bm25(lexical_query)

        bm25_scores = np.zeros(len(self.chunks), dtype=np.float32)
        if self.bm25 and query_tokens:
            bm25_scores = np.asarray(self.bm25.get_scores(query_tokens), dtype=np.float32)

        keyword_boost = np.zeros(len(self.chunks), dtype=np.float32)
        title_boost = np.zeros(len(self.chunks), dtype=np.float32)
        if normalized_keywords:
            for i, c in enumerate(self.chunks):
                keyword_boost[i] = sum(c.normalized_text.count(kw) for kw in normalized_keywords)
                if c.normalized_section_path:
                    title_boost[i] = sum(c.normalized_section_path.count(kw) for kw in normalized_keywords)

        bm25_norm = self._minmax(bm25_scores)
        kw_norm = self._minmax(keyword_boost)
        title_norm = self._minmax(title_boost)
        lexical = 0.65 * bm25_norm + 0.20 * kw_norm + 0.15 * title_norm

        if self.dense_enabled:
            q_emb = self.embedder.encode_query(query)
            dense_results = self.vector_index.search(
                q_emb,
                top_k=max(int(top_k), self.dense_top_k),
            )
            dense_scores = np.zeros(len(self.chunks), dtype=np.float32)
            for rec in dense_results:
                dense_scores[int(rec["chunk_index"])] = float(rec["dense_score"])
            dense_norm = self._minmax(dense_scores)
            if self.enable_hybrid:
                final_scores = self.lexical_weight * lexical + self.dense_weight * dense_norm
            else:
                final_scores = dense_norm
        else:
            dense_scores = np.zeros(len(self.chunks), dtype=np.float32)
            final_scores = lexical

        top_idx = np.argsort(final_scores)[::-1][:top_k]
        results = []
        for idx in top_idx:
            c = self.chunks[int(idx)]
            results.append(
                {
                    "score": float(final_scores[idx]),
                    "lexical_score": float(lexical[idx]),
                    "dense_score": float(dense_scores[idx]),
                    "vector_backend": self.vector_backend if self.dense_enabled else None,
                    "page_number": c.page_number,
                    "page_id": c.page_id,
                    "part_index": c.part_index,
                    "page_key": c.page_key,
                    "source_id": c.source_id,
                    "section_path": c.section_path,
                    "text": c.text,
                }
            )
        return results


def top_pages_from_chunks(
    top_chunks: List[Dict],
    pages_by_key: Dict[str, PageDoc],
    top_k_pages: int,
) -> List[Dict]:
    per_page = {}
    for c in top_chunks:
        key = c["page_key"]
        if key not in per_page:
            per_page[key] = {
                "scores": [],
                "lexical_scores": [],
                "dense_scores": [],
                "vector_backend": c.get("vector_backend"),
                "page_key": key,
            }
        per_page[key]["scores"].append(float(c["score"]))
        per_page[key]["lexical_scores"].append(float(c["lexical_score"]))
        per_page[key]["dense_scores"].append(float(c.get("dense_score") or 0.0))

    aggregated_pages = []
    for rec in per_page.values():
        scores = sorted(rec["scores"], reverse=True)
        lexical_scores = sorted(rec["lexical_scores"], reverse=True)
        dense_scores = sorted(rec["dense_scores"], reverse=True)
        top_n = min(3, len(scores))
        top_n_lexical = min(3, len(lexical_scores))
        top_n_dense = min(3, len(dense_scores))
        best_score = scores[0]
        mean_top_score = float(np.mean(scores[:top_n]))
        best_lexical = lexical_scores[0]
        mean_top_lexical = float(np.mean(lexical_scores[:top_n_lexical]))
        best_dense = dense_scores[0]
        mean_top_dense = float(np.mean(dense_scores[:top_n_dense]))
        aggregated_pages.append(
            {
                "page_key": rec["page_key"],
                "score": 0.75 * best_score + 0.25 * mean_top_score,
                "lexical_score": 0.75 * best_lexical + 0.25 * mean_top_lexical,
                "dense_score": 0.75 * best_dense + 0.25 * mean_top_dense,
                "vector_backend": rec.get("vector_backend"),
            }
        )

    sorted_pages = sorted(aggregated_pages, key=lambda x: x["score"], reverse=True)
    results = []
    for rec in sorted_pages:
        if len(results) >= top_k_pages:
            break
        page_doc = pages_by_key.get(rec["page_key"])
        if page_doc is None:
            continue
        if is_editorial_noise_page(page_doc.full_text, page_doc.section_path):
            continue
        results.append(
            {
                "score": rec["score"],
                "lexical_score": rec["lexical_score"],
                "dense_score": rec.get("dense_score", 0.0),
                "vector_backend": rec.get("vector_backend"),
                "page_number": page_doc.page_number,
                "page_id": page_doc.page_id,
                "part_index": page_doc.part_index,
                "source_id": page_doc.source_id,
                "section_path": page_doc.section_path,
                "text": page_doc.full_text,
                "author": page_doc.author,
                "book_title": page_doc.title,
            }
        )
    return results
