import json
import logging
import re
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from src.config import (
    REASONER_OUTPUT_MAX_TOKENS,
    REASONER_TEMPERATURE,
    REASONER_TOP_P,
    TRANSLATION_MAX_TOKENS,
)
from src.text_utils import ARABIC_WORD_RE

LOGGER = logging.getLogger(__name__)


def instantiate_model(model_path: str, num_gpus: int = 1, max_model_len: int = 1024):
    if torch.cuda.device_count() == 0:
        tensor_parallel_size = 1
    else:
        tensor_parallel_size = min(num_gpus, torch.cuda.device_count())

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=model_path,
        local_files_only=True,
        trust_remote_code=True,
    )

    model = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        max_num_batched_tokens=min(4096, max_model_len),
        max_num_seqs=1,
        gpu_memory_utilization=0.90,
        swap_space=0,
        enforce_eager=False,
        dtype="bfloat16",
        disable_custom_all_reduce=True,
    )
    return model, tokenizer


def generate_json_output(
    model,
    tokenizer,
    messages,
    schema,
    max_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    current_max_tokens = max(64, int(max_tokens))
    attempt_idx = 0
    while True:
        attempt_idx += 1
        sampling = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=current_max_tokens,
            structured_outputs=StructuredOutputsParams(
                json=schema,
                disable_additional_properties=True,
            ),
        )
        output = model.generate(prompt, sampling)[0].outputs[0].text.strip()
        try:
            return json.loads(output)
        except json.JSONDecodeError as exc:
            LOGGER.error(
                "JSON parsing failed in generate_json_output (attempt %s, max_tokens=%s): %s at line=%s col=%s pos=%s",
                attempt_idx,
                current_max_tokens,
                exc.msg,
                exc.lineno,
                exc.colno,
                exc.pos,
            )
            LOGGER.error("---- RAW_MODEL_OUTPUT_ATTEMPT_%s_START ----", attempt_idx)
            LOGGER.error(output)
            LOGGER.error("---- RAW_MODEL_OUTPUT_ATTEMPT_%s_END ----", attempt_idx)
            current_max_tokens = max(current_max_tokens + 1, int(current_max_tokens * 1.5))


def extract_keywords(model, tokenizer, question: str) -> List[str]:
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
        fallback_sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=128)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        raw = model.generate(prompt, fallback_sampling)[0].outputs[0].text
        kws = [line.strip("-* \t\r\n") for line in raw.splitlines() if line.strip()]

    filtered = []
    for kw in kws:
        kw = re.sub(r"^\d+[\).\-\s]*", "", kw).strip()
        if kw and ARABIC_WORD_RE.search(kw):
            filtered.append(kw)
    return filtered[:10]


def generate_pedagogical_answer(
    model,
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
    model,
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


def translate_pages_to_french(model, tokenizer, top_pages: List[Dict]) -> List[str]:
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
    structural_tag_spec = {
        "structures": [
            {
                "begin": "<traduction>",
                "schema": schema,
                "end": "</traduction>",
            }
        ],
        "triggers": ["<traduction>"],
    }

    translations: List[str] = []
    for p in top_pages:
        user_text = (
            "Translate the following page into French. "
            "Respond only with the <traduction> tag and its content.\n\n"
            f"{p['text']}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        src_tokens = len(tokenizer.encode(p["text"], add_special_tokens=False))
        max_tokens = min(TRANSLATION_MAX_TOKENS, max(700, int(src_tokens * 1.2) + 200))
        sampling = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_tokens,
            structured_outputs=StructuredOutputsParams(
                structural_tag=json.dumps(structural_tag_spec, ensure_ascii=False),
                disable_additional_properties=True,
            ),
        )
        try:
            out = model.generate(prompt, sampling)[0].outputs[0].text.strip()
            m = re.search(r"<traduction>(.*?)</traduction>", out, flags=re.DOTALL)
            payload = m.group(1).strip() if m else out
            data = json.loads(payload)
            tr = (data.get("translation_fr") or "").strip()
            translations.append(tr if tr else "[Traduction indisponible]")
        except Exception:
            fallback_sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens)
            raw = model.generate(prompt, fallback_sampling)[0].outputs[0].text.strip()
            translations.append(raw if raw else "[Traduction indisponible]")

    return translations
