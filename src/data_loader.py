import json
import os
from typing import Dict, List, Optional, Tuple

from src.models import PageDoc, TextChunk
from src.text_utils import clean_text, is_editorial_noise_page, normalize_arabic, split_into_chunks


def _make_page_key(source_id: str, part_index: int, page_id: Optional[str], page_number: Optional[int]) -> str:
    return f"source={source_id}|part={part_index}|page_id={page_id}|page_number={page_number}"


def extract_book_info(data: Dict) -> Dict[str, str]:
    title = (data.get("title") or "").strip() or "Livre inconnu"
    authors = data.get("authors") or []
    author_name = ""
    if authors:
        main_author = next((a for a in authors if a.get("is_main_author")), authors[0])
        author_name = (main_author.get("name") or "").strip()
    if not author_name:
        author_name = "Auteur inconnu"
    return {"author": author_name, "title": title}


def _resolve_json_files(json_input_path: str) -> List[str]:
    if os.path.isdir(json_input_path):
        files = [
            os.path.join(json_input_path, name)
            for name in sorted(os.listdir(json_input_path))
            if name.lower().endswith(".json")
        ]
        return [p for p in files if os.path.isfile(p)]
    if os.path.isfile(json_input_path):
        return [json_input_path]
    raise FileNotFoundError(f"JSON path not found: {json_input_path}")


def load_chunks(json_input_path: str) -> Tuple[List[TextChunk], Dict[str, PageDoc]]:
    json_files = _resolve_json_files(json_input_path)
    if not json_files:
        raise ValueError(f"No JSON files found in: {json_input_path}")

    chunks: List[TextChunk] = []
    pages: Dict[str, PageDoc] = {}

    seen_chunk_signatures = set()

    for json_path in json_files:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(
                f"Unsupported JSON structure in {json_path}: expected dict with parts/pages."
            )

        source_id = os.path.splitext(os.path.basename(json_path))[0]
        book_info = extract_book_info(data)
        for p_idx, part in enumerate(data.get("parts", [])):
            for page in part.get("pages", []):
                body = page.get("body", "")
                if not body:
                    continue
                cleaned = clean_text(body)
                if not cleaned:
                    continue
                section_path = clean_text(str(page.get("title") or ""))
                if is_editorial_noise_page(cleaned, section_path):
                    continue
                page_key = _make_page_key(source_id, p_idx, page.get("page_id"), page.get("page_number"))
                pages[page_key] = PageDoc(
                    page_number=page.get("page_number"),
                    page_id=page.get("page_id"),
                    part_index=p_idx,
                    page_key=page_key,
                    author=book_info.get("author", "Auteur inconnu"),
                    title=book_info.get("title", "Livre inconnu"),
                    full_text=cleaned,
                    section_path=section_path or None,
                    source_id=source_id,
                )
                page_chunks = split_into_chunks(cleaned)
                for chunk_text in page_chunks:
                    chunk_text = chunk_text.strip()
                    if not chunk_text:
                        continue
                    lexical_text = f"{section_path} {chunk_text}".strip() if section_path else chunk_text
                    normalized_text = normalize_arabic(chunk_text).lower()
                    signature = f"{source_id}|{normalized_text}"
                    if signature in seen_chunk_signatures:
                        continue
                    seen_chunk_signatures.add(signature)
                    chunks.append(
                        TextChunk(
                            page_number=page.get("page_number"),
                            page_id=page.get("page_id"),
                            part_index=p_idx,
                            page_key=page_key,
                            text=chunk_text,
                            normalized_text=normalized_text,
                            normalized_lexical_text=normalize_arabic(lexical_text).lower(),
                            normalized_section_path=normalize_arabic(section_path).lower() if section_path else "",
                            section_path=section_path or None,
                            source_id=source_id,
                        )
                    )
    return chunks, pages
