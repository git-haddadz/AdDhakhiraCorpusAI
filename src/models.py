from dataclasses import dataclass
from typing import Optional


@dataclass
class TextChunk:
    page_number: Optional[int]
    page_id: Optional[str]
    part_index: int
    page_key: str
    text: str
    normalized_text: str
    normalized_lexical_text: str
    normalized_section_path: str
    section_path: Optional[str] = None
    source_id: Optional[str] = None


@dataclass
class PageDoc:
    page_number: Optional[int]
    page_id: Optional[str]
    part_index: int
    page_key: str
    author: str
    title: str
    full_text: str
    section_path: Optional[str] = None
    source_id: Optional[str] = None
