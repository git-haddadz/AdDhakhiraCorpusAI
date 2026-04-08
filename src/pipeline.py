import os
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer

from src.config import (
    EMBEDDING_MODEL,
    JSON_INPUT_PATH,
    MAX_MODEL_LEN_EXTRACTOR,
    MIN_MODEL_LEN_REASONER,
    MODEL_EXTRACTOR_PATH,
    MODEL_REASONER_PATH,
    NUM_GPUS_EXTRACTOR,
    NUM_GPUS_REASONER,
    REASONER_CONTEXT_SAFETY_TOKENS,
    TOP_K_CHUNKS,
    TOP_K_PAGES,
)
from src.data_loader import load_chunks
from src.llm_ops import (
    assess_answer_consistency,
    extract_keywords,
    generate_pedagogical_answer,
    instantiate_model,
    translate_pages_to_french,
)
from src.reporting import print_final
from src.retrieval import HybridRetriever, top_pages_from_chunks
from src.text_utils import truncate_text_by_tokens


def build_reasoning_context(
    pages: List[Dict], tokenizer: AutoTokenizer, max_tokens: int
) -> Tuple[str, Dict[str, Dict[str, str]]]:
    lines = []
    source_map: Dict[str, Dict[str, str]] = {}
    for p in pages:
        section_bits = []
        if p.get("source_id"):
            section_bits.append(f"source={p.get('source_id')}")
        if p.get("section_path"):
            section_bits.append(f"section={p.get('section_path')}")
        extra = f" {' | '.join(section_bits)}" if section_bits else ""
        header = (
            f"[page_ref] page_number={p['page_number']} "
            f"page_id={p['page_id']} part={p['part_index']}{extra}"
        )
        lines.append(f"{header}\n{p['text']}")
        page_ref_key = f"{p.get('page_number')}|{p.get('page_id')}"
        source_map[page_ref_key] = {
            "page_number": str(p.get("page_number")),
            "page_id": str(p.get("page_id")),
            "author": str(p.get("author", "Auteur inconnu")),
            "title": str(p.get("book_title", "Livre inconnu")),
            "section_path": str(p.get("section_path") or ""),
            "source_id": str(p.get("source_id") or ""),
        }
    joined = "\n\n".join(lines)
    return truncate_text_by_tokens(joined, tokenizer, max_tokens), source_map


def compute_page_token_counts(top_pages: List[Dict], tokenizer: AutoTokenizer) -> List[Dict]:
    stats = []
    for p in top_pages:
        token_count = len(tokenizer.encode(p["text"], add_special_tokens=False))
        stats.append(
            {
                "page_number": p.get("page_number"),
                "page_id": p.get("page_id"),
                "part_index": p.get("part_index"),
                "tokens": token_count,
            }
        )
    return stats


def build_final_report(question: str, translate_to_french: bool, diagnostic_coherence: bool = False) -> str:
    extractor_model, extractor_tokenizer = instantiate_model(
        model_path=MODEL_EXTRACTOR_PATH,
        num_gpus=NUM_GPUS_EXTRACTOR,
        max_model_len=MAX_MODEL_LEN_EXTRACTOR,
    )

    keywords = extract_keywords(extractor_model, extractor_tokenizer, question)
    if len(keywords) < 4:
        raise ValueError(f"Too few valid Arabic keywords extracted: {keywords}")

    chunks, pages_by_key = load_chunks(JSON_INPUT_PATH)
    retriever = HybridRetriever(chunks, embedding_model_name=EMBEDDING_MODEL)

    top_chunks = retriever.search(question, keywords, top_k=TOP_K_CHUNKS)
    top_pages = top_pages_from_chunks(top_chunks, pages_by_key, top_k_pages=TOP_K_PAGES)
    if not top_pages:
        fallback = {
            "status": "not_enough_context",
            "reponse_courte": "Cela n'est pas mentionné dans les extraits fournis.",
            "points": [],
            "limites": "Aucun extrait pertinent n'a été récupéré.",
        }
        report = print_final(question, keywords, [], fallback)
        return report

    if os.path.normpath(MODEL_EXTRACTOR_PATH) == os.path.normpath(MODEL_REASONER_PATH):
        sizing_tokenizer = extractor_tokenizer
    else:
        sizing_tokenizer = AutoTokenizer.from_pretrained(
            MODEL_REASONER_PATH,
            cache_dir=MODEL_REASONER_PATH,
            local_files_only=True,
            trust_remote_code=True,
        )
    page_token_counts = compute_page_token_counts(top_pages, sizing_tokenizer)
    total_page_tokens = sum(p["tokens"] for p in page_token_counts)
    token_by_page = {
        (str(p["page_number"]), str(p["page_id"]), str(p["part_index"])): p["tokens"]
        for p in page_token_counts
    }
    for p in top_pages:
        key = (str(p.get("page_number")), str(p.get("page_id")), str(p.get("part_index")))
        p["page_tokens"] = token_by_page.get(key)
    reasoner_model_len = max(MIN_MODEL_LEN_REASONER, total_page_tokens + REASONER_CONTEXT_SAFETY_TOKENS)

    del extractor_model
    torch.cuda.empty_cache()
    reasoner_model, reasoner_tokenizer = instantiate_model(
        model_path=MODEL_REASONER_PATH,
        num_gpus=NUM_GPUS_REASONER,
        max_model_len=reasoner_model_len,
    )

    context, source_page_map = build_reasoning_context(
        top_pages,
        reasoner_tokenizer,
        max_tokens=total_page_tokens + 256,
    )
    def _merge_verdicts(primary: Dict, adversarial: Dict) -> Dict[str, object]:
        primary_verdict = primary.get("verdict", "insufficient")
        adversarial_verdict = adversarial.get("verdict", "insufficient")
        if "contradicted" in (primary_verdict, adversarial_verdict):
            merged_verdict = "contradicted"
        elif "insufficient" in (primary_verdict, adversarial_verdict):
            merged_verdict = "insufficient"
        else:
            merged_verdict = "supported"
        merged_issues = []
        for issue in (primary.get("issues") or []) + (adversarial.get("issues") or []):
            if issue not in merged_issues:
                merged_issues.append(issue)
        return {
            "verdict": merged_verdict,
            "issues": merged_issues,
            "primary_verdict": primary_verdict,
            "adversarial_verdict": adversarial_verdict,
        }

    def _evaluate_answer(candidate_answer: Dict) -> Dict[str, object]:
        primary = assess_answer_consistency(
            reasoner_model,
            reasoner_tokenizer,
            question,
            context,
            candidate_answer,
        )
        adversarial = assess_answer_consistency(
            reasoner_model,
            reasoner_tokenizer,
            question,
            context,
            candidate_answer,
            extra_verifier_rules=(
                "Adversarial mode: actively try to falsify the answer.\n"
                "Do not return supported unless all key claims are explicitly grounded and no "
                "question fact weakens them."
            ),
        )
        return _merge_verdicts(primary, adversarial)

    def _safe_generate_answer(extra_rules: str = None) -> Tuple[Dict, bool]:
        try:
            generated = generate_pedagogical_answer(
                reasoner_model,
                reasoner_tokenizer,
                question,
                context,
                extra_system_rules=extra_rules,
            )
            return generated, False
        except Exception:
            fallback_answer = {
                "status": "not_enough_context",
                "reponse_courte": (
                    "La génération structurée a échoué malgré plusieurs tentatives. "
                    "Voici les éléments textuels récupérés."
                ),
                "points": [],
                "limites": (
                    "Le modèle a produit une sortie JSON incomplète (troncature), "
                    "malgré une augmentation progressive du budget de tokens."
                ),
            }
            return fallback_answer, True

    def _build_evidence_aligned_fallback(
        points: List[Dict],
        failure_reason: str,
    ) -> Dict:
        if points:
            first_point = points[0]
            title = str(first_point.get("titre", "")).strip()
            explanation = str(first_point.get("explication_fr", "")).strip()
            if explanation:
                concise = explanation
            elif title:
                concise = title
            else:
                concise = "Les extraits disponibles permettent de dégager une règle exploitable."
            return {
                "status": "enough_context",
                "reponse_courte": concise,
                "points": points,
                "limites": (
                    "Le contrôle de cohérence a détecté une formulation trop forte dans une version intermédiaire. "
                    f"Version finale alignée sur les preuves textuelles. Détail: {failure_reason}"
                ),
            }
        return {
            "status": "not_enough_context",
            "reponse_courte": (
                "Les extraits ne permettent pas de formuler une conclusion catégorique sans risque de contradiction."
            ),
            "points": [],
            "limites": failure_reason,
        }

    answer, generation_failed = _safe_generate_answer()
    preserved_points = list(answer.get("points", [])) if isinstance(answer.get("points"), list) else []
    if generation_failed:
        consistency = {
            "verdict": "insufficient",
            "issues": ["Structured generation failed before consistency checks."],
            "primary_verdict": "insufficient",
            "adversarial_verdict": "insufficient",
        }
    else:
        consistency = _evaluate_answer(answer)
    coherence_diag: Dict[str, object] = {
        "initial_verdict": consistency.get("verdict"),
        "initial_issues": consistency.get("issues", []),
        "initial_primary_verdict": consistency.get("primary_verdict"),
        "initial_adversarial_verdict": consistency.get("adversarial_verdict"),
        "retry_verdict": None,
        "retry_issues": [],
        "retry_primary_verdict": None,
        "retry_adversarial_verdict": None,
        "final_pass_verdict": None,
        "final_pass_issues": [],
        "final_pass_primary_verdict": None,
        "final_pass_adversarial_verdict": None,
        "generation_failed": generation_failed,
        "fallback_used": False,
    }
    if consistency.get("verdict") != "supported":
        issues = consistency.get("issues") or []
        extra_rules = (
            "The previous draft was not fully supported by the excerpts/question facts.\n"
            "Correct all overclaims and contradictions and ground every key claim in direct evidence.\n"
            "If a required condition is not explicitly verified in the question, respond conditionally.\n"
            f"Verifier issues: {issues}"
        )
        answer, retry_generation_failed = _safe_generate_answer(extra_rules)
        if retry_generation_failed:
            coherence_diag["retry_verdict"] = "insufficient"
            coherence_diag["retry_issues"] = ["Structured generation failed during retry."]
            coherence_diag["retry_primary_verdict"] = "insufficient"
            coherence_diag["retry_adversarial_verdict"] = "insufficient"
            coherence_diag["fallback_used"] = True
            answer = _build_evidence_aligned_fallback(
                preserved_points,
                "La régénération structurée a échoué (sortie JSON tronquée).",
            )
            report = print_final(
                question,
                keywords,
                top_pages,
                answer,
                source_page_map=source_page_map,
                consistency_diagnostic=coherence_diag if diagnostic_coherence else None,
            )
            return report
        retry_points = answer.get("points", [])
        if isinstance(retry_points, list) and retry_points:
            preserved_points = list(retry_points)
        consistency_retry = _evaluate_answer(answer)
        coherence_diag["retry_verdict"] = consistency_retry.get("verdict")
        coherence_diag["retry_issues"] = consistency_retry.get("issues", [])
        coherence_diag["retry_primary_verdict"] = consistency_retry.get("primary_verdict")
        coherence_diag["retry_adversarial_verdict"] = consistency_retry.get("adversarial_verdict")
        if consistency_retry.get("verdict") != "supported":
            # Final correction pass before fallback: force alignment with question facts + excerpts.
            all_issues = []
            for issue in (coherence_diag.get("initial_issues") or []) + (coherence_diag.get("retry_issues") or []):
                if issue not in all_issues:
                    all_issues.append(issue)
            final_pass_rules = (
                "Final correction pass.\n"
                "Resolve all listed issues explicitly.\n"
                "When question facts already determine a condition, give a direct conclusion (not a hypothetical one).\n"
                "Do not keep mutually inconsistent statements between short answer and limits.\n"
                f"Issues to resolve: {all_issues}"
            )
            answer, final_generation_failed = _safe_generate_answer(final_pass_rules)
            if final_generation_failed:
                coherence_diag["final_pass_verdict"] = "insufficient"
                coherence_diag["final_pass_primary_verdict"] = "insufficient"
                coherence_diag["final_pass_adversarial_verdict"] = "insufficient"
                coherence_diag["final_pass_issues"] = ["Structured generation failed during final pass."]
                coherence_diag["fallback_used"] = True
                answer = _build_evidence_aligned_fallback(
                    preserved_points,
                    "La passe finale a échoué (sortie JSON tronquée).",
                )
                report = print_final(
                    question,
                    keywords,
                    top_pages,
                    answer,
                    source_page_map=source_page_map,
                    consistency_diagnostic=coherence_diag if diagnostic_coherence else None,
                )
                return report
            final_points = answer.get("points", [])
            if isinstance(final_points, list) and final_points:
                preserved_points = list(final_points)
            final_consistency = _evaluate_answer(answer)
            coherence_diag["final_pass_verdict"] = final_consistency.get("verdict")
            coherence_diag["final_pass_issues"] = final_consistency.get("issues", [])
            coherence_diag["final_pass_primary_verdict"] = final_consistency.get("primary_verdict")
            coherence_diag["final_pass_adversarial_verdict"] = final_consistency.get("adversarial_verdict")
            if final_consistency.get("verdict") != "supported":
                coherence_diag["fallback_used"] = True
                answer = _build_evidence_aligned_fallback(
                    preserved_points,
                    "Le contrôle de cohérence a détecté une contradiction persistante entre la réponse générée et les extraits.",
                )
    if translate_to_french:
        translations = translate_pages_to_french(reasoner_model, reasoner_tokenizer, top_pages)
        for page, tr in zip(top_pages, translations):
            page["translation_fr"] = tr
    report = print_final(
        question,
        keywords,
        top_pages,
        answer,
        source_page_map=source_page_map,
        consistency_diagnostic=coherence_diag if diagnostic_coherence else None,
    )
    return report
