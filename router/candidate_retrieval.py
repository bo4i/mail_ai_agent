from __future__ import annotations

from router.models import CandidateDepartment, DepartmentsCatalog, KeywordSpec, RulesContext
from router.text_processing import normalize_text

HIGH_PRECISION_WEIGHT = 3.0
MEDIUM_PRECISION_WEIGHT = 1.0
OUT_OF_SCOPE_PENALTY = 2.0
RULE_TRIGGERED_BOOST = 2.0
HIGH_PRIORITY_EXTRA_BOOST = 2.0

HIGH_PRECISION_MIN_COVERAGE = 0.66
MEDIUM_PRECISION_MIN_COVERAGE = 0.4
OUT_OF_SCOPE_MIN_COVERAGE = 0.5


def _keyword_coverage(keyword: KeywordSpec, lemma_set: set[str]) -> float:
    if not keyword.lemmas:
        return 0.0
    unique_lemmas = list(dict.fromkeys(keyword.lemmas))
    matched = sum(1 for lemma in unique_lemmas if lemma in lemma_set)
    return matched / len(unique_lemmas)


def _format_hit(keyword: KeywordSpec, coverage: float) -> str:
    if coverage >= 0.999:
        return keyword.text
    return f"{keyword.text} ({coverage:.2f})"


def _collect_hits(
    keywords: list[KeywordSpec],
    lemma_set: set[str],
    *,
    min_coverage: float,
    weight: float,
) -> tuple[list[str], float]:
    hits: list[str] = []
    score = 0.0
    for keyword in keywords:
        coverage = _keyword_coverage(keyword, lemma_set)
        if coverage >= min_coverage:
            score += weight * coverage
            hits.append(_format_hit(keyword, coverage))
    return hits, score


def _collect_out_of_scope(
    keywords: list[KeywordSpec],
    lemma_set: set[str],
    *,
    min_coverage: float,
) -> tuple[list[str], float]:
    hits: list[str] = []
    penalty = 0.0
    for keyword in keywords:
        coverage = _keyword_coverage(keyword, lemma_set)
        if coverage >= min_coverage:
            penalty += OUT_OF_SCOPE_PENALTY
            hits.append(_format_hit(keyword, coverage))
    return hits, penalty


def retrieve_candidates(
    clean_text_for_llm: str,
    catalog: DepartmentsCatalog,
) -> list[CandidateDepartment]:
    normalized = normalize_text(clean_text_for_llm)
    lemma_set = normalized.lemma_set
    candidates: list[CandidateDepartment] = []
    for department in catalog.departments:
        keyword_index = department.keyword_index
        high_hits, high_score = _collect_hits(
            keyword_index.get("high_precision", []),
            lemma_set,
            min_coverage=HIGH_PRECISION_MIN_COVERAGE,
            weight=HIGH_PRECISION_WEIGHT,
        )
        medium_hits, medium_score = _collect_hits(
            keyword_index.get("medium_precision", []),
            lemma_set,
            min_coverage=MEDIUM_PRECISION_MIN_COVERAGE,
            weight=MEDIUM_PRECISION_WEIGHT,
        )
        out_of_scope_hits, out_penalty = _collect_out_of_scope(
            keyword_index.get("out_of_scope", []),
            lemma_set,
            min_coverage=OUT_OF_SCOPE_MIN_COVERAGE,
        )
        score = high_score + medium_score - out_penalty

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


def apply_rules_boosts(
    candidates: list[CandidateDepartment],
    rules_context: RulesContext,
) -> list[CandidateDepartment]:
    if not rules_context.rules_triggered and not rules_context.priority_boosts:
        return candidates
    boosted: list[CandidateDepartment] = []
    for candidate in candidates:
        score = candidate.score
        if candidate.department_id in rules_context.rules_triggered:
            score += RULE_TRIGGERED_BOOST
        if rules_context.priority_boosts.get(candidate.department_id, 0) > 0:
            score += HIGH_PRIORITY_EXTRA_BOOST
        boosted.append(
            CandidateDepartment(
                department_id=candidate.department_id,
                department_name=candidate.department_name,
                keyword_hits=candidate.keyword_hits,
                score=score,
            )
        )
    return sorted(boosted, key=lambda item: item.score, reverse=True)
