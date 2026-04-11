import json
import logging
import re
from typing import Dict, List, Optional

from src.config import (
    GEMINI_API_KEY,
    LLM_BACKEND,
    REASONER_OUTPUT_MAX_TOKENS,
    REASONER_TEMPERATURE,
    REASONER_TOP_P,
    TRANSLATION_MAX_TOKENS,
)
from src.llm_backend import CustomBackend, GeminiBackend, LLMBackend
from src.text_utils import ARABIC_WORD_RE

LOGGER = logging.getLogger(__name__)


def instantiate_model(
    model_path: str,
    num_gpus: int = 1,
    max_model_len: int = 1024,
    model_role: str = "extractor",
):
    if LLM_BACKEND == "gemini_api":
        backend = GeminiBackend(model_name=model_path, api_key=GEMINI_API_KEY or None)
        return backend, backend.get_tokenizer()

    if LLM_BACKEND != "custom":
        raise ValueError(
            f"Unsupported LLM_BACKEND='{LLM_BACKEND}'. Expected one of: custom, gemini_api."
        )

    backend = CustomBackend(
        model_path=model_path,
        num_gpus=num_gpus,
        max_model_len=max_model_len,
    )
    return backend, backend.get_tokenizer()


def generate_json_output(
    model: LLMBackend,
    tokenizer,
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


def extract_keywords(model: LLMBackend, tokenizer, question: str) -> List[str]:
    schema = {
        "type": "object",
        "properties": {
            "keywords": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 10,
                "maxItems": 10,
            }
        },
        "required": ["keywords"],
        "additionalProperties": False,
    }
    system_prompt = """You are an Islamic research assistant.
Return exactly 10 Arabic keywords in JSON.
Rules:
- Keep only Arabic keywords.
- Prefer short base-form terms useful for retrieval.
- Do not answer the question.
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    try:
        data = generate_json_output(model, tokenizer, messages, schema, max_tokens=256, temperature=0.0, top_p=1.0)
        kws = data.get("keywords", [])
    except Exception:
        raw = model.generate_text(messages, max_tokens=128, temperature=0.0, top_p=1.0)
        kws = [line.strip("-* \t\r\n") for line in raw.splitlines() if line.strip()]

    filtered = []
    for kw in kws:
        kw = re.sub(r"^\d+[\).\-\s]*", "", kw).strip()
        if kw and ARABIC_WORD_RE.search(kw):
            filtered.append(kw)
    return filtered[:10]


def translate_question_to_arabic(model: LLMBackend, tokenizer, question: str) -> str:
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
            tokenizer,
            messages,
            schema,
            max_tokens=220,
            temperature=0.0,
            top_p=1.0,
        )
        translated = str(data.get("question_ar", "")).strip()
        return translated or question
    except Exception:
        raw = model.generate_text(messages, max_tokens=220, temperature=0.0, top_p=1.0).strip()
        return raw or question


def generate_pedagogical_answer(
    model: LLMBackend,
    tokenizer,
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
        tokenizer,
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
                tokenizer,
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
    tokenizer,
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
            tokenizer,
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


def translate_pages_to_french(model: LLMBackend, tokenizer, top_pages: List[Dict]) -> List[str]:
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
                tokenizer,
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
