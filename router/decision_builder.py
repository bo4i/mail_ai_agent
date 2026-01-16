from __future__ import annotations

import re
from typing import Any

from router.models import (
    CandidateDepartment,
    DepartmentsCatalog,
    NormalizedLetter,
    RoutingDecision,
    RulesContext,
)


def _derive_text_source(letter: NormalizedLetter) -> str:
    ocr_pages = [page for page in letter.pages if page.confidence_flags.get("ocr_used")]
    if not ocr_pages:
        return "native"
    if len(ocr_pages) == len(letter.pages):
        return "ocr"
    return "mixed"


def _build_page_map(letter: NormalizedLetter) -> list[dict[str, Any]]:
    page_map: list[dict[str, Any]] = []
    cursor = 0
    for page in letter.pages:
        page_text = page.clean_text_for_llm or ""
        char_start = cursor
        char_end = cursor + len(page_text)
        page_map.append(
            {
                "page": page.page,
                "text_span_id": f"p{page.page}",
                "char_start": char_start,
                "char_end": char_end,
            }
        )
        cursor = char_end + 2
    return page_map


def _ensure_summary(letter: NormalizedLetter) -> str:
    if letter.subject:
        return letter.subject
    cleaned = re.sub(r"\s+", " ", letter.clean_text_for_llm).strip()
    return cleaned[:200] if cleaned else "No summary available"


def _build_suggestions(
    candidates: list[CandidateDepartment],
    rules_context: RulesContext,
    max_score: float,
) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []
    for candidate in candidates:
        confidence = 0.0 if max_score <= 0 else min(1.0, candidate.score / max_score)
        rules_triggered = rules_context.rules_triggered.get(candidate.department_id, [])
        keywords = (
            candidate.keyword_hits.get("high_precision", [])
            + candidate.keyword_hits.get("medium_precision", [])
        )
        why = "Совпали ключевые слова" if keywords else "Ключевых совпадений нет"
        if rules_triggered:
            why += "; сработали правила триажа"

        suggestions.append(
            {
                "department_id": candidate.department_id,
                "department_name": candidate.department_name,
                "confidence": confidence,
                "priority": "primary",
                "why": why,
                "matched_signals": {
                    "keywords": keywords,
                    "rules_triggered": rules_triggered,
                    "semantic_score": 0.0,
                },
                "evidence": [],
                "next_actions": [],
            }
        )

    return suggestions


def build_decision(
    letter: NormalizedLetter,
    catalog: DepartmentsCatalog,
    candidates: list[CandidateDepartment],
    rules_context: RulesContext,
    routing_decision: RoutingDecision,
    *,
    processing_time_ms: int,
) -> dict[str, Any]:
    text_source = _derive_text_source(letter)
    max_score = max((candidate.score for candidate in candidates), default=0.0)
    summary = _ensure_summary(letter)
    organization_entities: list[dict[str, str]] = []
    if letter.issuer:
        organization_entities.append({"name": letter.issuer, "role": "sender"})
    if letter.addressee:
        organization_entities.append({"name": letter.addressee, "role": "mentioned"})

    review_reasons = list(rules_context.review_reasons)
    if routing_decision.fallback_reason:
        review_reasons.append(routing_decision.fallback_reason)

    decision = {
        "schema_version": "1.0",
        "request_id": letter.request_id,
        "created_at": letter.created_at,
        "input": {
            "source_channel": letter.source_channel,
            "file": {"filename": letter.filename, "pages": len(letter.pages)},
            "metadata": letter.metadata,
        },
        "extraction": {
            "text_source": text_source,
            "language": "ru",
            "page_map": _build_page_map(letter),
            "quality": {
                "ocr_confidence": 0.7 if text_source != "native" else 0.95,
                "has_tables": False,
                "has_stamps_signatures": "unknown",
                "warnings": [
                    "OCR used" if text_source != "native" else ""
                ],
            },
        },
        "understanding": {
            "doc_type": "unknown",
            "summary": summary,
            "topics": letter.topics,
            "urgency": {
                "level": "normal",
                "signals": [],
            },
            "entities": {
                "organizations": organization_entities,
                "people": [],
                "numbers": {
                    "contract_numbers": [],
                    "invoice_numbers": [],
                    "letter_numbers": [],
                    "law_refs": [],
                },
                "dates": [],
                "amounts": [],
                "locations": [],
            },
        },
        "routing": {
            "mode": "auto_route_allowed",
            "suggestions": _build_suggestions(candidates, rules_context, max_score),
            "final_recommendation": {
                "department_ids": routing_decision.department_ids or [candidates[0].department_id],
                "confidence": routing_decision.confidence,
                "comment": routing_decision.comment,
            },
            "needs_human_review": bool(review_reasons) or routing_decision.confidence < 0.4,
            "review_reasons": review_reasons,
        },
        "compliance": {
            "sensitive_flags": [],
            "safe_to_log_text": "yes",
            "masking": {"enabled": False, "masked_fields": []},
        },
        "diagnostics": {
            "processing_time_ms": processing_time_ms,
            "model": {"name": "heuristic-router", "version": "dev"},
            "trace": {
                "rules_version": "dev",
                "catalog_version": catalog.catalog_version,
            },
            "errors": [],
            "warnings": [],
        },
    }

    decision["extraction"]["quality"]["warnings"] = [
        warning for warning in decision["extraction"]["quality"]["warnings"] if warning
    ]
    return decision