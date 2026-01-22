from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from router.models import Department, DepartmentsCatalog, KeywordSpec
from router.text_processing import lemmatize_tokens, tokenize, top_terms

MAX_STRUCTURAL_TERMS = 8
STRUCTURAL_STOPWORDS = {
    "организация",
    "обеспечение",
    "осуществление",
    "подготовка",
    "разработка",
    "контроль",
    "мониторинг",
    "работа",
    "вопрос",
    "вопросы",
    "порядок",
    "планирование",
    "проведение",
    "поддержка",
    "формирование",
    "совершенствование",
    "функция",
    "функции",
}


def _normalize_keywords(keywords: list[str | dict]) -> list[KeywordSpec]:
    specs = []
    for item in keywords:
        if isinstance(item, str):
            text = item
            anchors = []
        else:
            text = item.get("text", "")
            anchors = item.get("anchors", [])

        tokens = tokenize(text)
        lemmas = lemmatize_tokens(tokens)

        anchor_lemmas = []
        for anchor in anchors:
            anchor_lemmas.extend(lemmatize_tokens(tokenize(anchor)))

        specs.append(
            KeywordSpec(
                text=text,
                lemmas=lemmas,
                anchors=anchor_lemmas,
            )
        )
    return specs


def _build_keyword_index(
    routing_keywords: dict[str, list[str]],
    out_of_scope: list[str],
) -> dict[str, list[KeywordSpec]]:
    return {
        "high_precision": _normalize_keywords(routing_keywords.get("high_precision", [])),
        "medium_precision": _normalize_keywords(routing_keywords.get("medium_precision", [])),
        "out_of_scope": _normalize_keywords(out_of_scope),
    }


def _extract_structural_keywords(
    raw: dict[str, Any],
    *,
    existing_lemmas: set[str],
    max_terms: int,
) -> list[str]:
    texts: list[str] = []
    for field in ("mission_short", "responsibilities", "typical_incoming_requests"):
        value = raw.get(field)
        if isinstance(value, list):
            texts.extend(str(item) for item in value if item)
        elif isinstance(value, str):
            texts.append(value)
    if not texts:
        return []
    candidates = top_terms(texts, max_terms=max_terms, stopwords=STRUCTURAL_STOPWORDS)
    return [term for term in candidates if term not in existing_lemmas]


def _ensure_list(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    msg = "Catalog payload must be an object or list of objects"
    raise ValueError(msg)


def load_departments_catalog(path: Path) -> DepartmentsCatalog:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = _ensure_list(payload)
    department_ids: set[str] = set()
    departments: list[Department] = []

    for item in items:
        department_id = item.get("department_id")
        if not department_id:
            raise ValueError("department_id is required for each department")
        if department_id in department_ids:
            raise ValueError(f"department_id must be unique: {department_id}")
        department_ids.add(department_id)

        routing_keywords = item.get("routing_keywords")
        triage_rules = item.get("triage_rules")
        if not routing_keywords:
            raise ValueError(f"routing_keywords missing for {department_id}")
        if triage_rules is None:
            raise ValueError(f"triage_rules missing for {department_id}")

        routing_keywords = {
            key: list(value)
            for key, value in routing_keywords.items()
            if isinstance(value, list)
        }
        out_of_scope = list(item.get("out_of_scope", []))
        keyword_index = _build_keyword_index(routing_keywords, out_of_scope)
        existing_lemmas = {
            lemma
            for specs in keyword_index.values()
            for spec in specs
            for lemma in spec.lemmas
        }
        structural_terms = _extract_structural_keywords(
            item,
            existing_lemmas=existing_lemmas,
            max_terms=MAX_STRUCTURAL_TERMS,
        )
        if structural_terms:
            routing_keywords.setdefault("medium_precision", []).extend(structural_terms)
            keyword_index = _build_keyword_index(routing_keywords, out_of_scope)

        departments.append(
            Department(
                department_id=department_id,
                department_name=item.get("department_name", department_id),
                routing_keywords=routing_keywords,
                triage_rules=triage_rules,
                raw=item,
                keyword_index=keyword_index,
            )
        )

    catalog_version = payload.get("catalog_version") if isinstance(payload, dict) else None
    return DepartmentsCatalog(
        departments=departments,
        catalog_version=catalog_version or "dev",
    )
