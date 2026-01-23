from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class NormalizedPage:
    page: int
    source: str
    clean_text_for_llm: str
    confidence_flags: dict[str, bool]


@dataclass
class NormalizedLetter:
    request_id: str
    created_at: str
    source_channel: str
    filename: str
    pages: list[NormalizedPage]
    subject: str | None
    issuer: str | None
    addressee: str | None
    topics: list[str]
    attachments: list[str]
    clean_text_for_llm: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KeywordSpec:
    text: str
    lemmas: list[str]
    anchors: list[str]


@dataclass
class Department:
    department_id: str
    department_name: str
    routing_keywords: dict[str, list[Any]]
    triage_rules: list[dict[str, Any]]
    raw: dict[str, Any]
    keyword_index: dict[str, list[KeywordSpec]] = field(default_factory=dict)


@dataclass
class DepartmentsCatalog:
    departments: list[Department]
    catalog_version: str = "dev"


@dataclass
class CandidateDepartment:
    department_id: str
    department_name: str
    keyword_hits: dict[str, list[str]]
    score: float
    score_breakdown: dict[str, float] = field(default_factory=dict)


@dataclass
class RulesContext:
    rules_triggered: dict[str, list[str]]
    priority_boosts: dict[str, int]
    review_reasons: list[str]


@dataclass
class RoutingDecision:
    department_ids: list[str]
    confidence: float
    mode: str
    comment: str
    used_llm: bool
    fallback_reason: str | None = None
