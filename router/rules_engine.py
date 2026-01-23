from __future__ import annotations

from typing import Any

from router.candidate_retrieval import _keyword_coverage
from router.models import DepartmentsCatalog, KeywordSpec, RulesContext
from router.text_processing import lemmatize_tokens, normalize_text, tokenize

TRIAGE_MIN_COVERAGE = 0.5
TRIAGE_SINGLE_WORD_MIN_COVERAGE = 1.0


def _normalize_triggers(triggers: list[str | dict[str, Any]]) -> list[KeywordSpec]:
    specs: list[KeywordSpec] = []
    for item in triggers:
        if isinstance(item, str):
            text = item
            anchors: list[str] = []
        else:
            text = str(item.get("text", ""))
            anchors = list(item.get("anchors", []))

        tokens = tokenize(text)
        lemmas = lemmatize_tokens(tokens)

        anchor_lemmas: list[str] = []
        for anchor in anchors:
            anchor_lemmas.extend(lemmatize_tokens(tokenize(str(anchor))))

        specs.append(KeywordSpec(text=text, lemmas=lemmas, anchors=anchor_lemmas))
    return specs


def _matches_rule(
    lemma_set: set[str],
    lemma_list: list[str],
    triggers: list[str | dict[str, Any]],
) -> bool:
    specs = _normalize_triggers(triggers)
    for spec in specs:
        if len(spec.lemmas) == 1:
            if not spec.anchors:
                continue
            min_coverage = TRIAGE_SINGLE_WORD_MIN_COVERAGE
        else:
            min_coverage = TRIAGE_MIN_COVERAGE
        if _keyword_coverage(spec, lemma_set, lemma_list) >= min_coverage:
            return True
    return False


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

    normalized = normalize_text(clean_text_for_llm)
    lemma_set = normalized.lemma_set
    lemma_list = normalized.lemma_list

    for department in catalog.departments:
        triggered_rules: list[str] = []

        for rule in department.triage_rules:
            triggers = rule.get("if_any", [])
            if not triggers:
                continue

            if _matches_rule(lemma_set, lemma_list, triggers):
                rule_text = str(rule.get("then", "rule matched"))
                triggered_rules.append(rule_text)

                boost = _priority_boost(rule_text)
                if boost:
                    priority_boosts[department.department_id] = priority_boosts.get(
                        department.department_id, 0
                    ) + boost

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
