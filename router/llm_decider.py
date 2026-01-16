from __future__ import annotations

from router.models import CandidateDepartment, RoutingDecision, RulesContext


def _normalize_confidence(score: float, max_score: float) -> float:
    if max_score <= 0:
        return 0.0
    return max(0.0, min(1.0, score / max_score))


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

    top_candidate = max(candidates, key=lambda item: item.score)
    max_score = top_candidate.score
    confidence = _normalize_confidence(top_candidate.score, max_score)

    if mode == "llm_assisted":
        return RoutingDecision(
            department_ids=[top_candidate.department_id],
            confidence=confidence,
            mode=mode,
            comment="LLM-assisted mode stubbed; used heuristic fallback.",
            used_llm=False,
            fallback_reason="llm_not_configured",
        )

    return RoutingDecision(
        department_ids=[top_candidate.department_id],
        confidence=confidence,
        mode=mode,
        comment="Heuristic selection based on keyword scores.",
        used_llm=False,
    )
