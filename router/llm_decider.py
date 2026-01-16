from __future__ import annotations

from router.models import CandidateDepartment, RoutingDecision, RulesContext

MIN_SCORE_THRESHOLD = 1.0
LOW_SCORE_CONFIDENCE = 0.2


def _gap_confidence(top_score: float, second_score: float) -> float:
    if top_score <= 0:
        return 0.0
    gap = top_score - second_score
    return max(0.0, min(1.0, gap / max(top_score, 1e-6)))


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
