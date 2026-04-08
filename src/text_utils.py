import re
from typing import List

from transformers import AutoTokenizer

ARABIC_DIACRITICS = re.compile(
    """
    ّ    |
    َ    |
    ً    |
    ُ    |
    ٌ    |
    ِ    |
    ٍ    |
    ْ    |
    ـ
    """,
    re.VERBOSE,
)
ARABIC_WORD_RE = re.compile(r"[\u0600-\u06FF]+")
TOKEN_RE = re.compile(r"[\w\u0600-\u06FF]+", re.UNICODE)


def normalize_arabic(text: str) -> str:
    text = re.sub(ARABIC_DIACRITICS, "", text)
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "و", text)
    text = re.sub("ئ", "ي", text)
    text = re.sub("ة", "ه", text)
    return text


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def truncate_text_by_tokens(text: str, tokenizer: AutoTokenizer, max_tokens: int) -> str:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return text
    return tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=True)


def split_into_chunks(text: str, chunk_chars: int = 1800, overlap: int = 250) -> List[str]:
    if len(text) <= chunk_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def tokenize_for_bm25(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())
