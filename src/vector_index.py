import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.data_loader import load_chunks
from src.embeddings import EmbeddingModel, normalize_vectors
from src.models import TextChunk

INDEX_VERSION = 1


def log(message: str) -> None:
    print(f"[VectorIndex] {time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def model_slug(model_name: str) -> str:
    model_path = Path(model_name)
    if model_path.exists():
        model_name = model_path.name
    slug = re.sub(r"[^A-Za-z0-9._-]+", "__", model_name.strip())
    return slug.strip("._-") or "none"


def index_dir_for(base_dir: Path, model_name: str, backend: str) -> Path:
    return Path(base_dir) / backend / model_slug(model_name)


def chunks_signature(chunks: Iterable[TextChunk]) -> str:
    h = hashlib.sha256()
    for chunk in chunks:
        payload = {
            "page_key": chunk.page_key,
            "page_id": chunk.page_id,
            "page_number": chunk.page_number,
            "part_index": chunk.part_index,
            "source_id": chunk.source_id,
            "section_path": chunk.section_path,
            "text": chunk.text,
        }
        h.update(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def chunk_metadata(chunks: List[TextChunk], pages_by_key: Dict) -> List[Dict]:
    metadata = []
    for idx, chunk in enumerate(chunks):
        page = pages_by_key.get(chunk.page_key)
        metadata.append(
            {
                "chunk_index": idx,
                "page_key": chunk.page_key,
                "page_id": chunk.page_id,
                "page_number": chunk.page_number,
                "part_index": chunk.part_index,
                "source_id": chunk.source_id,
                "section_path": chunk.section_path,
                "text": chunk.text,
                "author": getattr(page, "author", "Auteur inconnu") if page else "Auteur inconnu",
                "book_title": getattr(page, "title", "Livre inconnu") if page else "Livre inconnu",
            }
        )
    return metadata


def build_signature(model_name: str, backend: str, chunks: List[TextChunk]) -> Dict:
    return {
        "index_version": INDEX_VERSION,
        "embedding_model": model_name,
        "backend": backend,
        "normalized": True,
        "similarity": "inner_product",
        "chunks_sha256": chunks_signature(chunks),
        "chunk_count": len(chunks),
    }


class VectorIndex:
    def __init__(self, backend: str, signature: Dict, metadata: List[Dict], index):
        self.backend = backend
        self.signature = signature
        self.metadata = metadata
        self.metadata_by_chunk_index = {int(m["chunk_index"]): m for m in metadata}
        self.index = index

    @classmethod
    def build(
        cls,
        embeddings: np.ndarray,
        metadata: List[Dict],
        signature: Dict,
    ) -> "VectorIndex":
        backend = signature["backend"]
        vectors = normalize_vectors(embeddings)
        if backend == "faiss":
            import faiss

            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            return cls(backend, signature, metadata, index)
        if backend == "turbovec":
            from turbovec import IdMapIndex

            index = IdMapIndex(dim=vectors.shape[1], bit_width=4)
            ids = np.asarray([m["chunk_index"] for m in metadata], dtype=np.uint64)
            index.add_with_ids(vectors, ids)
            return cls(backend, signature, metadata, index)
        raise ValueError(f"Unsupported vector backend: {backend}")

    @classmethod
    def load(cls, directory: Path) -> "VectorIndex":
        directory = Path(directory)
        signature = json.loads((directory / "signature.json").read_text(encoding="utf-8"))
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        backend = signature["backend"]
        if backend == "faiss":
            import faiss

            index = faiss.read_index(str(directory / "index.faiss"))
            return cls(backend, signature, metadata, index)
        if backend == "turbovec":
            from turbovec import IdMapIndex

            index = IdMapIndex.load(str(directory / "index.tvim"))
            return cls(backend, signature, metadata, index)
        raise ValueError(f"Unsupported vector backend: {backend}")

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "signature.json").write_text(
            json.dumps(self.signature, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (directory / "metadata.json").write_text(
            json.dumps(self.metadata, ensure_ascii=False),
            encoding="utf-8",
        )
        if self.backend == "faiss":
            import faiss

            faiss.write_index(self.index, str(directory / "index.faiss"))
            return
        if self.backend == "turbovec":
            self.index.write(str(directory / "index.tvim"))
            return
        raise ValueError(f"Unsupported vector backend: {self.backend}")

    def assert_compatible(self, expected_signature: Dict) -> None:
        for key in ("index_version", "embedding_model", "backend", "chunks_sha256", "chunk_count"):
            if self.signature.get(key) != expected_signature.get(key):
                raise ValueError(
                    f"Vector index mismatch for {key}: "
                    f"expected {expected_signature.get(key)!r}, found {self.signature.get(key)!r}"
                )

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Dict]:
        q = normalize_vectors(query_vector)
        if self.backend == "faiss":
            scores, indices = self.index.search(q, int(top_k))
            pairs = zip(scores[0], indices[0])
        elif self.backend == "turbovec":
            scores, ids = self.index.search(q, k=int(top_k))
            pairs = zip(scores[0], ids[0])
        else:
            raise ValueError(f"Unsupported vector backend: {self.backend}")

        results = []
        for score, idx in pairs:
            idx = int(idx)
            if idx < 0:
                continue
            source = self.metadata_by_chunk_index.get(idx)
            if source is None:
                continue
            rec = dict(source)
            rec["dense_score"] = float(score)
            rec["vector_backend"] = self.backend
            results.append(rec)
        return results


def build_and_save_index(
    json_input_path: Path,
    output_dir: Path,
    model_name: str,
    backend: str,
    batch_size: int = 32,
    force: bool = False,
    devices: Optional[Sequence[str]] = None,
    show_progress: bool = False,
) -> Path:
    start = time.time()
    log(f"loading chunks json_input={json_input_path}")
    chunks, pages_by_key = load_chunks(str(json_input_path))
    log(f"loaded chunks={len(chunks)} pages={len(pages_by_key)}")
    signature = build_signature(model_name, backend, chunks)
    directory = index_dir_for(Path(output_dir), model_name, backend)
    log(f"target directory={directory}")
    sig_path = directory / "signature.json"
    if sig_path.exists() and not force:
        existing = json.loads(sig_path.read_text(encoding="utf-8"))
        if all(existing.get(k) == signature.get(k) for k in ("index_version", "embedding_model", "backend", "chunks_sha256", "chunk_count")):
            log("compatible index already exists; skipping build")
            return directory

    device_list = [d.strip() for d in (devices or []) if d and d.strip()]
    log(
        "loading embedding model "
        f"model={model_name} backend={backend} batch_size={batch_size} "
        f"devices={','.join(device_list) if device_list else 'default'}"
    )
    embedder = EmbeddingModel(model_name, devices=device_list, show_progress=show_progress)
    log("encoding chunks")
    embeddings = embedder.encode_chunks([c.text for c in chunks], batch_size=batch_size)
    log(f"encoded embeddings shape={embeddings.shape}")
    metadata = chunk_metadata(chunks, pages_by_key)
    log(f"building {backend} index")
    index = VectorIndex.build(embeddings, metadata, signature)
    log("saving index and metadata")
    index.save(directory)
    log(f"done path={directory} elapsed_seconds={int(time.time() - start)}")
    return directory


def parse_devices(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def load_compatible_index(
    index_root: Path,
    model_name: str,
    backend: str,
    chunks: List[TextChunk],
) -> VectorIndex:
    directory = index_dir_for(Path(index_root), model_name, backend)
    if not directory.exists():
        raise FileNotFoundError(
            f"Vector index not found: {directory}. Build it before enabling dense retrieval."
        )
    index = VectorIndex.load(directory)
    index.assert_compatible(build_signature(model_name, backend, chunks))
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a persistent vector index for AdDhakhiraCorpusAI.")
    parser.add_argument("--model", required=True, help="Embedding model name or path.")
    parser.add_argument("--backend", default="faiss", choices=["faiss", "turbovec"])
    parser.add_argument("--json-input", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--devices",
        default=None,
        help="Comma-separated devices for multi-process encoding, e.g. cuda:0,cuda:1,cuda:2,cuda:3.",
    )
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    from src.config import EMBEDDING_INDEX_DIR, JSON_INPUT_PATH

    output_dir = Path(args.output_dir) if args.output_dir else Path(EMBEDDING_INDEX_DIR)
    json_input = Path(args.json_input) if args.json_input else Path(JSON_INPUT_PATH)
    path = build_and_save_index(
        json_input_path=json_input,
        output_dir=output_dir,
        model_name=args.model,
        backend=args.backend,
        batch_size=args.batch_size,
        force=args.force,
        devices=parse_devices(args.devices),
        show_progress=args.show_progress,
    )
    print(path)


if __name__ == "__main__":
    main()
