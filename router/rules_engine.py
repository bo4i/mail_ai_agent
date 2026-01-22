from __future__ import annotations

from router.models import DepartmentsCatalog, RulesContext
from router.text_processing import normalize_text

PRIMARY_ANCHORS = {"суд", "прокуратура", "иск"}
SECONDARY_ANCHORS = {"жалоба", "проверка", "представление", "исковой"}


def _matches_rule(lemma_set: set[str], triggers: list[str]) -> bool:
    trigger_lemmas = {lemma for trigger in triggers for lemma in normalize_text(trigger).lemma_list}
    if not trigger_lemmas:
        return False
    if PRIMARY_ANCHORS.intersection(trigger_lemmas):
        if not PRIMARY_ANCHORS.intersection(lemma_set):
            return False
        if not SECONDARY_ANCHORS.intersection(lemma_set):
            return False
    return bool(trigger_lemmas.intersection(lemma_set))


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

    lemma_set = normalize_text(clean_text_for_llm).lemma_set
    for department in catalog.departments:
        triggered_rules: list[str] = []
        for rule in department.triage_rules:
            triggers = rule.get("if_any", [])
            if not triggers:
                continue
            if _matches_rule(lemma_set, triggers):
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
