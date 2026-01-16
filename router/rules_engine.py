from __future__ import annotations

from router.models import DepartmentsCatalog, RulesContext


def _matches_rule(text: str, triggers: list[str]) -> bool:
    lowered = text.lower()
    return any(trigger.lower() in lowered for trigger in triggers)


def _extract_review_reason(rule_text: str) -> str | None:
    lowered = rule_text.lower()
    if "соисполнителя" in lowered or "соисполнитель" in lowered:
        return rule_text
    return None


def _priority_boost(rule_text: str) -> int:
    lowered = rule_text.lower()
    if "высок" in lowered and "приоритет" in lowered:
        return 2
    return 0


def apply_triage_rules(clean_text_for_llm: str, catalog: DepartmentsCatalog) -> RulesContext:
    rules_triggered: dict[str, list[str]] = {}
    priority_boosts: dict[str, int] = {}
    review_reasons: list[str] = []

    for department in catalog.departments:
        triggered_rules: list[str] = []
        for rule in department.triage_rules:
            triggers = rule.get("if_any", [])
            if not triggers:
                continue
            if _matches_rule(clean_text_for_llm, triggers):
                rule_text = str(rule.get("then", "rule matched"))
                triggered_rules.append(rule_text)
                boost = _priority_boost(rule_text)
                if boost:
                    priority_boosts[department.department_id] = (
                        priority_boosts.get(department.department_id, 0) + boost
                    )
                review_reason = _extract_review_reason(rule_text)
                if review_reason and review_reason not in review_reasons:
                    review_reasons.append(review_reason)

        if triggered_rules:
            rules_triggered[department.department_id] = triggered_rules

    return RulesContext(
        rules_triggered=rules_triggered,
        priority_boosts=priority_boosts,
        review_reasons=review_reasons,
    )