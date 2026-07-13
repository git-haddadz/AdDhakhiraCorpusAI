import json
import re
from typing import Dict, List, Optional

from src.config import (
    ANTHROPIC_API_KEY,
    GEMINI_API_KEY,
    LLM_BACKEND,
    OPENAI_API_KEY,
    REASONER_OUTPUT_MAX_TOKENS,
    REASONER_TEMPERATURE,
    REASONER_TOP_P,
    TRANSLATION_MAX_TOKENS,
)
try:
    from src.config import KEYWORD_GENERATION_MAX_ATTEMPTS
except ImportError:
    # Keep existing generated/local config.py files compatible after upgrading.
    KEYWORD_GENERATION_MAX_ATTEMPTS = 3

from src.llm_backend import (
    AnthropicBackend,
    CustomBackend,
    GeminiBackend,
    LLMBackend,
    OpenAIBackend,
    _extract_json_object,
)
from src.text_utils import ARABIC_WORD_RE


DEBUG_RESPONSE_MAX_CHARS = 4000


def _safe_debug_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    # Provider errors should not contain credentials, but mask common key shapes
    # before placing any diagnostic text in a downloadable HTML file.
    text = re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b", "[OPENAI_KEY_REDACTED]", text)
    text = re.sub(r"\bAIza[A-Za-z0-9_-]{20,}\b", "[GEMINI_KEY_REDACTED]", text)
    text = re.sub(
        r"(?i)(api[_ -]?key\s*[:=]\s*)[^\s,;]+",
        r"\1[API_KEY_REDACTED]",
        text,
    )
    if len(text) > DEBUG_RESPONSE_MAX_CHARS:
        return text[:DEBUG_RESPONSE_MAX_CHARS] + "\n… [réponse tronquée pour le debug]"
    return text


def build_generation_debug_event(
    stage: str,
    *,
    response: object = None,
    error: Optional[BaseException] = None,
    cycle: Optional[int] = None,
) -> Dict[str, object]:
    event: Dict[str, object] = {"stage": stage}
    if cycle is not None:
        event["cycle"] = cycle
    if error is not None:
        event["error_type"] = getattr(error, "cause_type", None) or type(error).__name__
        event["error_message"] = _safe_debug_text(
            getattr(error, "cause_message", None) or str(error)
        )
        attempts = getattr(error, "attempts", None)
        if attempts is not None:
            event["json_attempts"] = attempts
        provider = getattr(error, "provider", None)
        if provider:
            event["provider"] = provider
        raw_response = getattr(error, "raw_response", None)
        if raw_response:
            event["response"] = _safe_debug_text(raw_response)
    elif response is not None:
        event["response"] = _safe_debug_text(response)
    return event


def _parse_jsonish(raw: str):
    text = (raw or "").strip()
    if not text:
        return None
    for candidate in (text, f"{{{text}}}"):
        try:
            return _extract_json_object(candidate)
        except Exception:
            pass
    return None


def _dedupe_arabic_terms(text: str, limit: int = 10) -> List[str]:
    terms: List[str] = []
    seen = set()
    for term in ARABIC_WORD_RE.findall(text or ""):
        term = term.strip()
        if len(term) < 2 or term in seen:
            continue
        seen.add(term)
        terms.append(term)
        if len(terms) >= limit:
            break
    return terms


def _coerce_keyword_candidates(raw) -> List[str]:
    if isinstance(raw, dict):
        if isinstance(raw.get("keywords"), list):
            return [str(item) for item in raw["keywords"]]
        for value in raw.values():
            if isinstance(value, list):
                return [str(item) for item in value]
            if isinstance(value, str) and ARABIC_WORD_RE.search(value):
                return _dedupe_arabic_terms(value)
        return []
    if isinstance(raw, list):
        return [str(item) for item in raw]
    if isinstance(raw, str):
        parsed = _parse_jsonish(raw)
        if parsed is not None:
            return _coerce_keyword_candidates(parsed)
        lines = [line.strip("-* \t\r\n") for line in raw.splitlines() if line.strip()]
        if len(lines) > 1:
            return lines
        return _dedupe_arabic_terms(raw)
    return []


def _filter_arabic_keywords(candidates: List[str]) -> List[str]:
    filtered = []
    seen = set()
    for candidate in candidates:
        keyword = re.sub(r"^\d+[\).\-\s]*", "", str(candidate)).strip()
        if not keyword or not ARABIC_WORD_RE.search(keyword) or keyword in seen:
            continue
        seen.add(keyword)
        filtered.append(keyword)
    return filtered


def _coerce_translation(raw, fallback: str) -> str:
    if isinstance(raw, dict):
        for key in ("question_ar", "translation", "translated_question", "arabic", "text"):
            value = raw.get(key)
            if isinstance(value, str) and ARABIC_WORD_RE.search(value):
                return value.strip()
        for value in raw.values():
            if isinstance(value, str) and ARABIC_WORD_RE.search(value):
                return value.strip()
        return fallback
    if not isinstance(raw, str):
        return fallback
    parsed = _parse_jsonish(raw)
    if parsed is not None:
        return _coerce_translation(parsed, fallback)
    if not ARABIC_WORD_RE.search(raw):
        return fallback
    cleaned = re.sub(
        r'^\s*["\']?(?:question_ar|translation|translated_question|arabic|text)["\']?\s*:\s*',
        "",
        raw.strip(),
        flags=re.IGNORECASE,
    )
    return cleaned.strip().strip("{}").strip().strip('"').strip("'") or fallback


def instantiate_model(
    model_path: str,
    num_gpus: int = 1,
    max_model_len: int = 1024,
):
    if LLM_BACKEND == "gemini_api":
        backend = GeminiBackend(model_name=model_path, api_key=GEMINI_API_KEY or None)
        return backend, backend.get_tokenizer()
    if LLM_BACKEND == "openai_api":
        backend = OpenAIBackend(model_name=model_path, api_key=OPENAI_API_KEY or None)
        return backend, backend.get_tokenizer()
    if LLM_BACKEND == "anthropic_api":
        backend = AnthropicBackend(model_name=model_path, api_key=ANTHROPIC_API_KEY or None)
        return backend, backend.get_tokenizer()

    if LLM_BACKEND != "default":
        raise ValueError(
            "Unsupported LLM_BACKEND="
            f"{LLM_BACKEND!r}. Expected one of: default, gemini_api, openai_api, anthropic_api."
        )

    backend = CustomBackend(
        model_path=model_path,
        num_gpus=num_gpus,
        max_model_len=max_model_len,
    )
    return backend, backend.get_tokenizer()


def generate_json_output(
    model: LLMBackend,
    messages,
    schema,
    max_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
):
    return model.generate_json(
        messages=messages,
        schema=schema,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )


def extract_keywords(
    model: LLMBackend,
    question: str,
    diagnostic: Optional[Dict[str, object]] = None,
) -> List[str]:
    schema = {
        "type": "object",
        "properties": {
            "keywords": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 10,
            }
        },
        "required": ["keywords"],
        "additionalProperties": False,
    }
    system_prompt = """You are an Islamic research assistant.
Return at least 10 Arabic keywords in JSON.
Rules:
- Keep only Arabic keywords.
- Prefer short base-form terms useful for retrieval.
- Do not answer the question.
"""
    collected: List[str] = []
    attempts = max(1, int(KEYWORD_GENERATION_MAX_ATTEMPTS))
    structured_calls = 0
    text_calls = 0
    modes_with_keywords = set()
    debug_events: List[Dict[str, object]] = []

    if diagnostic is not None:
        diagnostic.clear()
        diagnostic.update(
            {
                "keyword_extraction_mode": None,
                "keyword_extraction_attempt": None,
                "keyword_structured_calls": 0,
                "keyword_text_calls": 0,
                "keyword_valid_before_fallback": 0,
                "keyword_count": 0,
                "llm_debug_events": debug_events,
            }
        )

    for attempt in range(attempts):
        retry_instruction = ""
        if attempt:
            retry_instruction = (
                "\nA previous response was empty or invalid. "
                "You must return a JSON list containing at least 10 Arabic keywords."
            )
        messages = [
            {"role": "system", "content": system_prompt + retry_instruction},
            {"role": "user", "content": question},
        ]
        candidates: List[str] = []
        candidate_mode = "structured_json"
        structured_calls += 1
        try:
            data = generate_json_output(
                model,
                messages,
                schema,
                max_tokens=256,
                temperature=0.0,
                top_p=1.0,
            )
            candidates = _coerce_keyword_candidates(data)
            debug_events.append(
                build_generation_debug_event(
                    "keywords_json",
                    response=data,
                    cycle=attempt + 1,
                )
            )
        except Exception as exc:
            debug_events.append(
                build_generation_debug_event(
                    "keywords_json",
                    error=exc,
                    cycle=attempt + 1,
                )
            )
            candidate_mode = "text_fallback"
            text_calls += 1
            try:
                raw = model.generate_text(
                    messages,
                    max_tokens=128,
                    temperature=0.0,
                    top_p=1.0,
                )
                debug_events.append(
                    build_generation_debug_event(
                        "keywords_text",
                        response=raw,
                        cycle=attempt + 1,
                    )
                )
                candidates = _coerce_keyword_candidates(raw)
            except Exception as exc:
                debug_events.append(
                    build_generation_debug_event(
                        "keywords_text",
                        error=exc,
                        cycle=attempt + 1,
                    )
                )
                candidates = []

        valid_candidates = _filter_arabic_keywords(candidates)
        added_count = 0
        for keyword in valid_candidates:
            if keyword not in collected:
                collected.append(keyword)
                added_count += 1
        if added_count:
            modes_with_keywords.add(candidate_mode)
        if len(collected) >= 10:
            selected = list(collected)
            if len(modes_with_keywords) == 1:
                extraction_mode = next(iter(modes_with_keywords))
            else:
                extraction_mode = "mixed_model_outputs"
            if diagnostic is not None:
                diagnostic.update(
                    {
                        "keyword_extraction_mode": extraction_mode,
                        "keyword_extraction_attempt": attempt + 1,
                        "keyword_structured_calls": structured_calls,
                        "keyword_text_calls": text_calls,
                        "keyword_valid_before_fallback": len(collected),
                        "keyword_count": len(selected),
                    }
                )
            return selected

    # The processed question remains the dense/BM25 query. These broad Arabic
    # terms only guarantee a usable lexical fallback when every model call fails.
    valid_before_fallback = len(collected)
    fallback_terms = _dedupe_arabic_terms(question) + [
        "مسألة",
        "حكم",
        "دليل",
        "قول",
        "فقه",
        "مذهب",
        "علماء",
        "نص",
        "بحث",
        "شرعي",
    ]
    for keyword in fallback_terms:
        if keyword not in collected:
            collected.append(keyword)
        if len(collected) >= 10:
            break
    selected = list(collected)
    if diagnostic is not None:
        diagnostic.update(
            {
                "keyword_extraction_mode": "deterministic_fallback",
                "keyword_extraction_attempt": attempts,
                "keyword_structured_calls": structured_calls,
                "keyword_text_calls": text_calls,
                "keyword_valid_before_fallback": valid_before_fallback,
                "keyword_count": len(selected),
            }
        )
    return selected


def translate_question_to_arabic(model: LLMBackend, question: str) -> str:
    schema = {
        "type": "object",
        "properties": {
            "question_ar": {"type": "string", "minLength": 1},
        },
        "required": ["question_ar"],
        "additionalProperties": False,
    }
    system_prompt = """You are a translator for Islamic research queries.
Translate the user question into clear Modern Standard Arabic.
Rules:
- Preserve all facts, conditions, negations, and temporal/order details.
- Do not answer the question.
- Return only JSON following the schema.
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    try:
        data = generate_json_output(
            model,
            messages,
            schema,
            max_tokens=220,
            temperature=0.0,
            top_p=1.0,
        )
        translated = _coerce_translation(data, question)
        return translated or question
    except Exception:
        raw = model.generate_text(messages, max_tokens=220, temperature=0.0, top_p=1.0).strip()
        return _coerce_translation(raw, question)


def generate_pedagogical_answer(
    model: LLMBackend,
    question: str,
    context: str,
    extra_system_rules: Optional[str] = None,
) -> Dict:
    schema = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["enough_context", "not_enough_context"],
            },
            "reponse_courte": {"type": "string"},
            "points": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "titre": {"type": "string"},
                        "citation_arabe": {"type": "string"},
                        "explication_fr": {"type": "string"},
                        "source": {"type": "string"},
                    },
                    "required": ["titre", "citation_arabe", "explication_fr", "source"],
                    "additionalProperties": False,
                },
                "minItems": 0,
                "maxItems": 5,
            },
            "limites": {"type": "string"},
        },
        "required": ["status", "reponse_courte", "points", "limites"],
        "additionalProperties": False,
    }

    system_prompt = """You are a specialist in Maliki fiqh.
Answer in French.
Use strictly and only the provided excerpts.
Rules:
- No external knowledge.
- If at least one excerpt gives a directly applicable rule, set status=enough_context.
- Use status=not_enough_context only when no applicable rule exists in the excerpts.
- Each point must include a verbatim Arabic quote and a direct source reference like "Page 238 (page_id=237)", with no tags and no numeric index.
- The source field must be formatted exactly as: "Page <number> (page_id=<id>)".
- You must only cite page_number/page_id that exist in the provided [page_ref] headers.
- If you cannot map a quote to an existing [page_ref], do not use that quote.
- Arabic quotes must never be truncated and must not contain ellipsis.
- All non-quote fields must be French only.
- Keep explanations faithful and concise.
- If the answer depends on a condition not explicitly verified in the question, answer conditionally (use "si... alors..."), not with an unconditional yes/no.
- If the question already gives explicit facts that invalidate a required condition, do not answer "yes if ..."; answer negatively and explain why based on those facts.
"""
    if extra_system_rules:
        system_prompt = f"{system_prompt}\n\nAdditional constraints:\n{extra_system_rules.strip()}\n"
    user_prompt = f"Question:\n{question}\n\nExcerpts:\n{context}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    answer = generate_json_output(
        model,
        messages,
        schema,
        max_tokens=REASONER_OUTPUT_MAX_TOKENS,
        temperature=REASONER_TEMPERATURE,
        top_p=REASONER_TOP_P,
    )

    non_fr = False
    if re.search(r"[\u0600-\u06FF]", answer.get("reponse_courte", "")):
        non_fr = True
    if re.search(r"[\u0600-\u06FF]", answer.get("limites", "")):
        non_fr = True
    for pt in answer.get("points", []):
        if re.search(r"[\u0600-\u06FF]", pt.get("explication_fr", "")) or re.search(r"[\u0600-\u06FF]", pt.get("titre", "")):
            non_fr = True
            break
    if non_fr:
        rewrite_messages = [
            {
                "role": "system",
                "content": (
                    "Rewrite the JSON in French only for non-quote fields. "
                    "Never modify citation_arabe or source."
                ),
            },
            {"role": "user", "content": json.dumps(answer, ensure_ascii=False)},
        ]
        try:
            answer = generate_json_output(
                model,
                rewrite_messages,
                schema,
                max_tokens=REASONER_OUTPUT_MAX_TOKENS,
                temperature=0.0,
                top_p=1.0,
            )
        except Exception:
            pass
    return answer


def assess_answer_consistency(
    model: LLMBackend,
    question: str,
    context: str,
    answer: Dict,
    extra_verifier_rules: Optional[str] = None,
) -> Dict:
    schema = {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["supported", "contradicted", "insufficient"],
            },
            "issues": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 0,
                "maxItems": 6,
            },
            "reasoning": {"type": "string"},
        },
        "required": ["verdict", "issues", "reasoning"],
        "additionalProperties": False,
    }

    system_prompt = """You are a strict verifier.
Task: verify whether the proposed answer is supported by the provided excerpts.
Rules:
- Use only provided excerpts.
- Mark contradicted if the answer asserts something opposite to excerpts.
- Mark insufficient if excerpts do not allow a firm conclusion.
- Mark supported only when key claims are directly grounded in excerpts.
- You must also use explicit facts stated in the question itself.
- If the answer is conditional ("if X then Y"), check whether X is already satisfied or contradicted by the question facts.
- If question facts already negate X but the answer still gives a permissive/positive conclusion, mark contradicted.
- If the answer contains internal inconsistency (e.g., short answer says yes while limits deny the same claim), mark contradicted.
- For time/order statements in the question (before/after, earlier/later, hours), enforce logical consistency.
- Be conservative and brief.
"""
    if extra_verifier_rules:
        system_prompt = f"{system_prompt}\n\nAdditional verifier constraints:\n{extra_verifier_rules.strip()}\n"
    user_payload = {
        "question": question,
        "answer": answer,
        "excerpts": context,
    }
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]
    try:
        return generate_json_output(
            model,
            messages,
            schema,
            max_tokens=420,
            temperature=0.0,
            top_p=1.0,
        )
    except Exception:
        return {
            "verdict": "insufficient",
            "issues": ["Consistency checker failed; fallback to conservative mode."],
            "reasoning": "The verifier could not produce a structured response.",
        }


def translate_pages_to_french(model: LLMBackend, top_pages: List[Dict]) -> List[str]:
    if not top_pages:
        return []

    system_prompt = """You are an Arabic -> French translator.
Translate the provided text faithfully and clearly.
Rules:
- Simple and readable French.
- No analysis, no additions, no commentary.
- Keep bracketed references if present.
"""
    schema = {
        "type": "object",
        "properties": {"translation_fr": {"type": "string"}},
        "required": ["translation_fr"],
        "additionalProperties": False,
    }

    translations: List[str] = []
    for p in top_pages:
        user_text = (
            "Translate the following page into French. "
            "Return JSON only with key translation_fr.\n\n"
            f"{p['text']}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        src_tokens = model.count_tokens(p["text"])
        max_tokens = min(TRANSLATION_MAX_TOKENS, max(700, int(src_tokens * 1.2) + 200))
        try:
            data = generate_json_output(
                model,
                messages,
                schema,
                max_tokens=max_tokens,
                temperature=0.0,
                top_p=1.0,
            )
            tr = (data.get("translation_fr") or "").strip()
            translations.append(tr if tr else "[Traduction indisponible]")
        except Exception:
            raw = model.generate_text(messages, max_tokens=max_tokens, temperature=0.0, top_p=1.0).strip()
            translations.append(raw if raw else "[Traduction indisponible]")

    return translations
