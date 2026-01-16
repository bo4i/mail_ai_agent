from __future__ import annotations

from router.models import CandidateDepartment, DepartmentsCatalog

HIGH_PRECISION_WEIGHT = 3.0
MEDIUM_PRECISION_WEIGHT = 1.0
OUT_OF_SCOPE_PENALTY = 2.0


def _find_matches(text: str, keywords: list[str]) -> list[str]:
    lowered = text.lower()
    hits = []
    for keyword in keywords:
        if keyword.lower() in lowered:
            hits.append(keyword)
    return hits


def retrieve_candidates(
    clean_text_for_llm: str,
    catalog: DepartmentsCatalog,
) -> list[CandidateDepartment]:
    candidates: list[CandidateDepartment] = []
    for department in catalog.departments:
        keywords = department.routing_keywords
        high_hits = _find_matches(clean_text_for_llm, keywords.get("high_precision", []))
        medium_hits = _find_matches(clean_text_for_llm, keywords.get("medium_precision", []))
        out_of_scope_hits = _find_matches(clean_text_for_llm, department.raw.get("out_of_scope", []))

        score = (
            len(high_hits) * HIGH_PRECISION_WEIGHT
            + len(medium_hits) * MEDIUM_PRECISION_WEIGHT
            - len(out_of_scope_hits) * OUT_OF_SCOPE_PENALTY
        )

        candidates.append(
            CandidateDepartment(
                department_id=department.department_id,
                department_name=department.department_name,
                keyword_hits={
                    "high_precision": high_hits,
                    "medium_precision": medium_hits,
                    "out_of_scope": out_of_scope_hits,
                },
                score=score,
            )
        )

    return sorted(candidates, key=lambda item: item.score, reverse=True)
