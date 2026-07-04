import json
import os
import re
from bisect import bisect_left
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass
class SectionRecord:
    path: str
    text_norm: str
    char_len: int


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


def _flatten_sections(nodes: List[Dict], parents: List[str], out: List[SectionRecord]) -> None:
    for node in nodes:
        if not isinstance(node, dict):
            continue
        title = clean_text(str(node.get("title") or "")).strip()
        current = parents + ([title] if title else [])
        section_path = " > ".join([p for p in current if p]).strip()
        text = clean_text(str(node.get("text") or "")).strip()
        if text:
            text_norm = normalize_arabic(text).lower()
            out.append(SectionRecord(path=section_path, text_norm=text_norm, char_len=len(text_norm)))
        sub = node.get("sub") or []
        if isinstance(sub, list):
            _flatten_sections(sub, current, out)


def _pick_old_json_path(legacy_dir: str, source_id_padded: str) -> Optional[str]:
    candidates = [os.path.join(legacy_dir, f"{source_id_padded}.json")]
    try:
        candidates.append(os.path.join(legacy_dir, f"{int(source_id_padded)}.json"))
    except ValueError:
        pass
    for cand in candidates:
        if os.path.isfile(cand):
            return cand
    return None


def _make_prefix_probes(text: str, probe_len: int = 80) -> List[str]:
    if not text:
        return []
    probes = []
    if len(text) <= probe_len:
        return [text]
    probes.append(text[:probe_len])
    mid = max(0, (len(text) // 2) - (probe_len // 2))
    probes.append(text[mid : mid + probe_len])
    probes.append(text[-probe_len:])
    return [p for p in probes if p.strip()]


def _assign_sections_to_pages(sections: List[SectionRecord], old_data: Dict) -> None:
    if not sections:
        return

    section_ends: List[int] = []
    csum = 0
    for s in sections:
        csum += s.char_len
        section_ends.append(csum)
    total_new = section_ends[-1]

    pages: List[Dict] = []
    for part in old_data.get("parts", []):
        for page in part.get("pages", []):
            body = clean_text(str(page.get("body") or "")).strip()
            body_norm = normalize_arabic(body).lower() if body else ""
            page["_tmp_body_norm"] = body_norm
            page["_tmp_len"] = len(body_norm)
            pages.append(page)

    total_old = sum(int(p["_tmp_len"]) for p in pages)
    if total_old <= 0:
        for p in pages:
            p["title"] = ""
            p.pop("_tmp_body_norm", None)
            p.pop("_tmp_len", None)
        return

    old_cursor = 0
    for page in pages:
        page_text = page.get("_tmp_body_norm", "")
        page_len = int(page.get("_tmp_len", 0))
        if not page_text or page_len == 0:
            page["title"] = ""
            old_cursor += page_len
            continue

        old_mid = old_cursor + (page_len // 2)
        mapped_mid = int((old_mid / total_old) * total_new)
        est_idx = bisect_left(section_ends, mapped_mid)
        est_idx = min(max(est_idx, 0), len(sections) - 1)

        best_idx = est_idx
        best_score = -1
        probes = _make_prefix_probes(page_text)
        lo = max(0, est_idx - 4)
        hi = min(len(sections), est_idx + 5)
        for i in range(lo, hi):
            sec_text = sections[i].text_norm
            score = 0
            for pr in probes:
                if pr in sec_text:
                    score += 1
            if score > best_score:
                best_score = score
                best_idx = i

        page["title"] = sections[best_idx].path
        old_cursor += page_len

    for p in pages:
        p.pop("_tmp_body_norm", None)
        p.pop("_tmp_len", None)


def build_hybrid_database(
    section_dir: str,
    legacy_dir: str,
    output_dir: str,
) -> Tuple[int, int]:
    os.makedirs(output_dir, exist_ok=True)
    built = 0
    skipped = 0

    for name in sorted(os.listdir(section_dir)):
        if not name.lower().endswith(".json"):
            continue
        source_id = os.path.splitext(name)[0]
        new_path = os.path.join(section_dir, name)
        old_path = _pick_old_json_path(legacy_dir, source_id)
        if old_path is None:
            skipped += 1
            continue

        with open(new_path, "r", encoding="utf-8") as f:
            new_data = json.load(f)
        with open(old_path, "r", encoding="utf-8") as f:
            old_data = json.load(f)

        if not isinstance(new_data, list) or not isinstance(old_data, dict):
            skipped += 1
            continue

        sections: List[SectionRecord] = []
        _flatten_sections(new_data, [], sections)
        _assign_sections_to_pages(sections, old_data)

        out_path = os.path.join(output_dir, name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(old_data, f, ensure_ascii=False)
        built += 1

    return built, skipped


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    section_dir = os.path.join(root, "misc", "database_paginated")
    legacy_dir = os.path.join(root, "misc", "database")
    output_dir = os.path.join(root, "database")

    built, skipped = build_hybrid_database(section_dir, legacy_dir, output_dir)
    print(f"Hybrid database generated in {output_dir}: built={built}, skipped={skipped}")


if __name__ == "__main__":
    main()
