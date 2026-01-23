from __future__ import annotations

from router.models import CandidateDepartment, DepartmentsCatalog, KeywordSpec, RulesContext
from router.text_processing import normalize_text

HIGH_PRECISION_WEIGHT = 3.0
MEDIUM_PRECISION_WEIGHT = 1.0
STRUCTURAL_WEIGHT = 0.5

OUT_OF_SCOPE_PENALTY = 2.0
NEGATIVE_CONTEXT_PENALTY = 1.0
RULE_TRIGGERED_BOOST = 2.0
HIGH_PRIORITY_EXTRA_BOOST = 2.0

# Реальный текст редко совпадает 100% по фразе → 1.0 слишком строго.
HIGH_PRECISION_MIN_COVERAGE = 0.66
MEDIUM_PRECISION_MIN_COVERAGE = 0.66
STRUCTURAL_MIN_COVERAGE = 1.0  # структурные термины обычно одиночные леммы
OUT_OF_SCOPE_MIN_COVERAGE = 0.5
NEGATIVE_CONTEXT_MIN_COVERAGE = 1.0
PROXIMITY_WINDOW = 7
NEGATIVE_CONTEXT_MIN_MATCHES = 2
NEGATIVE_CONTEXT_DEPARTMENT_IDS = {"FIN_CIVIL_SERVICE_ADMIN"}


def _anchors_in_proximity(
    keyword_lemma: str,
    anchors: list[str],
    lemma_list: list[str],
    *,
    window: int,
) -> bool:
    if not anchors or not lemma_list:
        return False
    anchor_set = set(anchors)
    for idx, lemma in enumerate(lemma_list):
        if lemma != keyword_lemma:
            continue
        start = max(0, idx - window)
        end = min(len(lemma_list), idx + window + 1)
        if any(anchor in anchor_set for anchor in lemma_list[start:end]):
            return True
    return False


def _keyword_coverage(
    keyword: KeywordSpec,
    lemma_set: set[str],
    lemma_list: list[str] | None = None,
) -> float:
    # 1) Проверка якорей
    if keyword.anchors:
        if lemma_list and len(keyword.lemmas) == 1:
            if not _anchors_in_proximity(
                keyword.lemmas[0], keyword.anchors, lemma_list, window=PROXIMITY_WINDOW
            ):
                return 0.0
        elif not all(anchor in lemma_set for anchor in keyword.anchors):
            return 0.0

    # 2) Coverage по леммам
    if not keyword.lemmas:
        return 0.0
    matched = sum(1 for lemma in keyword.lemmas if lemma in lemma_set)
    return matched / len(keyword.lemmas)


def _format_hit(keyword: KeywordSpec, coverage: float) -> str:
    if coverage >= 0.999:
        return keyword.text
    return f"{keyword.text} ({coverage:.2f})"


def _collect_hits(
    keywords: list[KeywordSpec],
    lemma_set: set[str],
    lemma_list: list[str],
    *,
    min_coverage: float,
    weight: float,
) -> tuple[list[str], float]:
    hits: list[str] = []
    score = 0.0
    for keyword in keywords:
        coverage = _keyword_coverage(keyword, lemma_set, lemma_list)
        if coverage >= min_coverage:
            score += weight * coverage
            hits.append(_format_hit(keyword, coverage))
    return hits, score


def _collect_out_of_scope(
    keywords: list[KeywordSpec],
    lemma_set: set[str],
    lemma_list: list[str],
    *,
    min_coverage: float,
) -> tuple[list[str], float]:
    hits: list[str] = []
    penalty = 0.0
    for keyword in keywords:
        coverage = _keyword_coverage(keyword, lemma_set, lemma_list)
        if coverage >= min_coverage:
            penalty += OUT_OF_SCOPE_PENALTY * coverage
            hits.append(_format_hit(keyword, coverage))
    return hits, penalty


def _collect_negative_context(
    keywords: list[KeywordSpec],
    lemma_set: set[str],
    lemma_list: list[str],
    *,
    min_coverage: float,
) -> list[str]:
    hits: list[str] = []
    for keyword in keywords:
        coverage = _keyword_coverage(keyword, lemma_set, lemma_list)
        if coverage >= min_coverage:
            hits.append(_format_hit(keyword, coverage))
    return hits


def retrieve_candidates(clean_text_for_llm: str, catalog: DepartmentsCatalog) -> list[CandidateDepartment]:
    normalized = normalize_text(clean_text_for_llm)
    lemma_set = normalized.lemma_set
    lemma_list = normalized.lemma_list

    candidates: list[CandidateDepartment] = []
    for department in catalog.departments:
        keyword_index = department.keyword_index

        high_hits, high_score = _collect_hits(
            keyword_index.get("high_precision", []),
            lemma_set,
            lemma_list,
            min_coverage=HIGH_PRECISION_MIN_COVERAGE,
            weight=HIGH_PRECISION_WEIGHT,
        )
        medium_hits, medium_score = _collect_hits(
            keyword_index.get("medium_precision", []),
            lemma_set,
            lemma_list,
            min_coverage=MEDIUM_PRECISION_MIN_COVERAGE,
            weight=MEDIUM_PRECISION_WEIGHT,
        )
        structural_hits, structural_score = _collect_hits(
            keyword_index.get("structural_terms", []),
            lemma_set,
            lemma_list,
            min_coverage=STRUCTURAL_MIN_COVERAGE,
            weight=STRUCTURAL_WEIGHT,
        )
        out_of_scope_hits, out_penalty = _collect_out_of_scope(
            keyword_index.get("out_of_scope", []),
            lemma_set,
            lemma_list,
            min_coverage=OUT_OF_SCOPE_MIN_COVERAGE,
        )
        negative_hits: list[str] = []
        negative_penalty = 0.0
        if department.department_id in NEGATIVE_CONTEXT_DEPARTMENT_IDS:
            negative_hits = _collect_negative_context(
                keyword_index.get("negative_context", []),
                lemma_set,
                lemma_list,
                min_coverage=NEGATIVE_CONTEXT_MIN_COVERAGE,
            )
            if not high_hits and len(negative_hits) >= NEGATIVE_CONTEXT_MIN_MATCHES:
                negative_penalty = NEGATIVE_CONTEXT_PENALTY * len(negative_hits)

        score = high_score + medium_score + structural_score - out_penalty - negative_penalty
        candidates.append(
            CandidateDepartment(
                department_id=department.department_id,
                department_name=department.department_name,
                keyword_hits={
                    "high_precision": high_hits,
                    "medium_precision": medium_hits,
                    "structural_terms": structural_hits,
                    "out_of_scope": out_of_scope_hits,
                    "negative_context": negative_hits,
                },
                score=score,
                score_breakdown={
                    "high_precision": high_score,
                    "medium_precision": medium_score,
                    "structural_terms": structural_score,
                    "out_of_scope_penalty": out_penalty,
                    "negative_context_penalty": negative_penalty,
                },
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
                score_breakdown=dict(candidate.score_breakdown),
            )
        )

    return sorted(boosted, key=lambda item: item.score, reverse=True)
