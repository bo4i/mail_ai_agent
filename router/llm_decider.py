from __future__ import annotations

from router.models import CandidateDepartment, RoutingDecision, RulesContext

MIN_SCORE_THRESHOLD = 1.0
LOW_SCORE_CONFIDENCE = 0.2
HIGH_PRECISION_CONFIDENCE_CAP = 0.6
MEDIUM_DOMINANCE_THRESHOLD = 0.6
MEDIUM_DOMINANCE_PENALTY = 0.7


def _gap_confidence(top_score: float, second_score: float) -> float:
    if top_score <= 0:
        return 0.0
    gap = top_score - second_score
    return max(0.0, min(1.0, gap / max(top_score, 1e-6)))


def _has_full_high_precision(candidate: CandidateDepartment) -> bool:
    return bool(candidate.keyword_hits.get("high_precision"))


def _is_medium_dominant(candidate: CandidateDepartment) -> bool:
    breakdown = candidate.score_breakdown or {}
    high_score = max(breakdown.get("high_precision", 0.0), 0.0)
    medium_score = max(breakdown.get("medium_precision", 0.0), 0.0)
    structural_score = max(breakdown.get("structural_terms", 0.0), 0.0)
    total_positive = high_score + medium_score + structural_score
    if total_positive <= 0:
        return False
    return medium_score / total_positive >= MEDIUM_DOMINANCE_THRESHOLD


def decide_routing(
    candidates: list[CandidateDepartment],
    rules_context: RulesContext,
    *,
    mode: str = "heuristic_only",
) -> RoutingDecision:
    if not candidates:
        return RoutingDecision(
            department_ids=[],
            confidence=0.0,
            mode=mode,
            comment="No candidates available",
            used_llm=False,
        )

    sorted_candidates = sorted(candidates, key=lambda item: item.score, reverse=True)
    top_candidate = sorted_candidates[0]
    second_score = sorted_candidates[1].score if len(sorted_candidates) > 1 else 0.0
    confidence = _gap_confidence(top_candidate.score, second_score)
    if not _has_full_high_precision(top_candidate):
        confidence = min(confidence, HIGH_PRECISION_CONFIDENCE_CAP)
    if _is_medium_dominant(top_candidate):
        confidence *= MEDIUM_DOMINANCE_PENALTY
    if top_candidate.score < MIN_SCORE_THRESHOLD:
        confidence = min(confidence, LOW_SCORE_CONFIDENCE)
        low_score_note = " Low score; needs human review."
    else:
        low_score_note = ""

    if mode == "llm_assisted":
        return RoutingDecision(
            department_ids=[top_candidate.department_id],
            confidence=confidence,
            mode=mode,
            comment="LLM-assisted mode stubbed; used heuristic fallback." + low_score_note,
            used_llm=False,
            fallback_reason="llm_not_configured",
        )

    return RoutingDecision(
        department_ids=[top_candidate.department_id],
        confidence=confidence,
        mode=mode,
        comment="Heuristic selection based on keyword scores." + low_score_note,
        used_llm=False,
    )
