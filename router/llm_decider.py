from __future__ import annotations

import json
import math
from typing import Any

from router.models import CandidateDepartment, DepartmentsCatalog, RoutingDecision, RulesContext
from router.ollama_client import OllamaClient, OllamaConfig

MIN_SCORE_THRESHOLD = 1.0
LOW_SCORE_CONFIDENCE = 0.2
HIGH_PRECISION_CONFIDENCE_CAP = 0.6
TRIAGE_CONFIDENCE_CAP = 0.55
MEDIUM_DOMINANCE_THRESHOLD = 0.6
MEDIUM_DOMINANCE_PENALTY = 0.7


def _gap_confidence(top_score: float, second_score: float) -> float:
    if top_score <= 0:
        return 0.0
    gap = top_score - second_score
    return max(0.0, min(1.0, gap / max(top_score, 1e-6)))


def _sigmoid_confidence(score: float) -> float:
    return 1.0 / (1.0 + math.exp(-score))


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


def _needs_llm(sorted_candidates: list[CandidateDepartment], confidence: float) -> bool:
    """
    Когда LLM реально нужен:
    - низкий score (слабые сигналы)
    - низкий gap confidence
    - у top1 нет high_precision (часто "попали" только по triage)
    """
    top = sorted_candidates[0]
    if top.score < MIN_SCORE_THRESHOLD:
        return True
    if confidence < 0.35:
        return True
    if not _has_full_high_precision(top):
        return True
    # если вообще нет keyword hits (все пусто) — точно нужен LLM
    if not (top.keyword_hits.get("high_precision") or top.keyword_hits.get("medium_precision")):
        return True
    return False


def _extract_json_object(text: str) -> str:
    """
    Ollama-модели иногда возвращают лишний текст вокруг JSON.
    Берём самый внешний {...}.
    """
    if not text:
        return ""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""
    return text[start : end + 1].strip()


def _build_department_card(dept_raw: dict[str, Any]) -> dict[str, Any]:
    """
    Сжимаем отдел до карточки (чтобы не пихать весь JSON).
    """
    rk = dept_raw.get("routing_keywords") or {}
    high = list((rk.get("high_precision") or []))[:10]
    medium = list((rk.get("medium_precision") or []))[:10]
    typical = list((dept_raw.get("typical_incoming_requests") or []))[:8]
    return {
        "department_id": dept_raw.get("department_id"),
        "department_name": dept_raw.get("department_name"),
        "mission_short": dept_raw.get("mission_short") or "",
        "typical_incoming_requests": typical,
        "routing_keywords": {"high_precision": high, "medium_precision": medium},
        "out_of_scope": list((dept_raw.get("out_of_scope") or []))[:6],
    }


def _build_llm_prompt(
    *,
    letter_text: str,
    candidates: list[CandidateDepartment],
    catalog: DepartmentsCatalog,
    top_k: int,
    rules_context: RulesContext,
) -> tuple[str, str, list[str]]:
    """
    Возвращает (system_prompt, user_prompt, allowed_department_ids)
    """
    top = candidates[:top_k]
    allowed_ids = [c.department_id for c in top]

    dept_by_id = {d.department_id: d for d in catalog.departments}
    cards: list[dict[str, Any]] = []
    for c in top:
        dept = dept_by_id.get(c.department_id)
        if not dept:
            continue
        cards.append(_build_department_card(dept.raw))

    candidates_payload = []
    for c in top:
        candidates_payload.append(
            {
                "department_id": c.department_id,
                "department_name": c.department_name,
                "heuristic_score": c.score,
                "keyword_hits": {
                    "high_precision": c.keyword_hits.get("high_precision", []),
                    "medium_precision": c.keyword_hits.get("medium_precision", []),
                },
                "score_breakdown": c.score_breakdown,
            }
        )

    system = (
        "Ты — агент маршрутизации входящих писем по отделам.\n"
        "Твоя задача — выбрать ОДИН primary_department_id из списка candidates либо OUT_OF_SCOPE.\n"
        "Запрещено выбирать department_id, которого нет в candidates.\n"
        "Отвечай СТРОГО валидным JSON без markdown и без лишнего текста.\n"
        "Если данных недостаточно — needs_human_review=true и задай questions.\n"
        "Если основания общие (например, «представить сведения/информацию» без предмета контроля),\n"
        "выбирай OUT_OF_SCOPE и needs_human_review=true.\n"
        "Если триаж указывает на соисполнителя, можно заполнить secondary_department_ids.\n"
    )

    matched_snippets = _extract_match_snippets(letter_text, top, limit=5)

    user = {
        "task": "route_letter",
        "allowed_department_ids": allowed_ids,
        "note": "Выбирай только из allowed_department_ids или OUT_OF_SCOPE.",
        "letter": {"text": letter_text[:12000]},
        "match_snippets": matched_snippets,
        "triage_review_reasons": rules_context.review_reasons,
        "candidates": candidates_payload,
        "department_cards": cards,
        "response_schema_example": {
            "primary_department_id": allowed_ids[0] if allowed_ids else "OUT_OF_SCOPE",
            "secondary_department_ids": [],
            "confidence": 0.0,
            "rationale": [
                {
                    "claim": "краткая причина выбора",
                    "evidence_from_letter": "цитата/фрагмент письма",
                    "catalog_support": "ссылка на mission/typical_incoming_requests/keywords из карточки",
                }
            ],
            "needs_human_review": False,
            "questions": [],
        },
    }
    user_text = json.dumps(user, ensure_ascii=False, indent=2)
    return system, user_text, allowed_ids


def _strip_hit_label(hit: str) -> str:
    if " (" in hit and hit.endswith(")"):
        return hit.split(" (", 1)[0].strip()
    return hit.strip()


def _extract_match_snippets(
    letter_text: str,
    candidates: list[CandidateDepartment],
    *,
    limit: int,
) -> list[str]:
    lowered = letter_text.lower()
    keywords: list[str] = []
    for candidate in candidates:
        for hit_list in candidate.keyword_hits.values():
            for hit in hit_list:
                text = _strip_hit_label(hit)
                if text and text not in keywords:
                    keywords.append(text)

    snippets: list[str] = []
    for keyword in keywords:
        idx = lowered.find(keyword.lower())
        if idx == -1:
            continue
        start = max(0, idx - 80)
        end = min(len(letter_text), idx + len(keyword) + 80)
        snippet = letter_text[start:end].strip()
        if snippet and snippet not in snippets:
            snippets.append(snippet)
        if len(snippets) >= limit:
            break
    return snippets


def _validate_llm_payload(payload: dict[str, Any], allowed_ids: list[str]) -> tuple[bool, str]:
    """
    Лёгкая валидация до общей jsonschema (чтобы быстро чинить).
    """
    primary = payload.get("primary_department_id")
    if not primary:
        return False, "primary_department_id is required"
    if primary != "OUT_OF_SCOPE" and primary not in allowed_ids:
        return False, f"primary_department_id must be one of {allowed_ids} or OUT_OF_SCOPE"
    sec = payload.get("secondary_department_ids", [])
    if not isinstance(sec, list):
        return False, "secondary_department_ids must be a list"
    for d in sec:
        if d not in allowed_ids:
            return False, f"secondary_department_ids must contain only allowed ids: {allowed_ids}"
    conf = payload.get("confidence")
    if conf is None or not isinstance(conf, (int, float)):
        return False, "confidence must be a number"
    needs = payload.get("needs_human_review")
    if needs is None or not isinstance(needs, bool):
        return False, "needs_human_review must be boolean"
    if not isinstance(payload.get("rationale", []), list):
        return False, "rationale must be a list"
    if not isinstance(payload.get("questions", []), list):
        return False, "questions must be a list"
    return True, ""


def decide_routing(
    candidates: list[CandidateDepartment],
    rules_context: RulesContext,
    *,
    mode: str = "heuristic_only",
    letter_text: str | None = None,
    catalog: DepartmentsCatalog | None = None,
    ollama_url: str = "http://localhost:11434",
    ollama_model: str = "gpt-oss",
    ollama_temperature: float = 0.2,
    llm_top_k: int = 5,
    llm_force: bool = False,
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

    gap_conf = _gap_confidence(top_candidate.score, second_score)
    abs_conf = _sigmoid_confidence(max(top_candidate.score, 0.0))
    confidence = 0.6 * gap_conf + 0.4 * abs_conf
    if not _has_full_high_precision(top_candidate):
        confidence = min(confidence, HIGH_PRECISION_CONFIDENCE_CAP)
        if top_candidate.department_id in rules_context.rules_triggered:
            confidence = min(confidence, TRIAGE_CONFIDENCE_CAP)
    if _is_medium_dominant(top_candidate):
        confidence *= MEDIUM_DOMINANCE_PENALTY

    low_score_note = ""
    if top_candidate.score < MIN_SCORE_THRESHOLD:
        confidence = min(confidence, LOW_SCORE_CONFIDENCE)
        low_score_note = " Low score; needs human review."

    if mode != "llm_assisted":
        return RoutingDecision(
            department_ids=[top_candidate.department_id],
            confidence=confidence,
            mode=mode,
            comment="Heuristic selection based on keyword scores." + low_score_note,
            used_llm=False,
        )

    if not llm_force and not _needs_llm(sorted_candidates, confidence):
        return RoutingDecision(
            department_ids=[top_candidate.department_id],
            confidence=confidence,
            mode=mode,
            comment="LLM-assisted mode: heuristic confident enough; LLM skipped." + low_score_note,
            used_llm=False,
            fallback_reason="llm_skipped_confident",
        )

    if not letter_text or not catalog:
        return RoutingDecision(
            department_ids=[top_candidate.department_id],
            confidence=confidence,
            mode=mode,
            comment="LLM-assisted mode: missing letter_text/catalog; used heuristic fallback."
            + low_score_note,
            used_llm=False,
            fallback_reason="llm_missing_inputs",
        )

    llm_top_k = max(1, min(int(llm_top_k), len(sorted_candidates)))
    system, user_text, allowed_ids = _build_llm_prompt(
        letter_text=letter_text,
        candidates=sorted_candidates,
        catalog=catalog,
        top_k=llm_top_k,
        rules_context=rules_context,
    )

    client = OllamaClient(
        OllamaConfig(
            base_url=ollama_url,
            model=ollama_model,
            temperature=ollama_temperature,
        )
    )

    raw = client.chat(system=system, user=user_text)

    for _ in range(3):
        candidate_json = _extract_json_object(raw)
        if candidate_json:
            try:
                payload = json.loads(candidate_json)
                ok, err = _validate_llm_payload(payload, allowed_ids)
                if ok:
                    primary = payload["primary_department_id"]
                    secondary = payload.get("secondary_department_ids", [])
                    llm_conf = float(payload.get("confidence", 0.0))
                    needs_human = bool(payload.get("needs_human_review", False))

                    rationale = payload.get("rationale", [])
                    first_reason = ""
                    if isinstance(rationale, list) and rationale:
                        r0 = rationale[0] if isinstance(rationale[0], dict) else {}
                        first_reason = (r0.get("claim") or "").strip()
                    comment = "LLM decision via Ollama."
                    if first_reason:
                        comment += f" Причина: {first_reason}"
                    if needs_human:
                        comment += " Требуется проверка человеком."

                    if primary == "OUT_OF_SCOPE":
                        return RoutingDecision(
                            department_ids=[top_candidate.department_id],
                            confidence=min(llm_conf, 0.35),
                            mode=mode,
                            comment="LLM: письмо вне компетенции каталога (OUT_OF_SCOPE). Нужна ручная маршрутизация.",
                            used_llm=True,
                            fallback_reason="llm_out_of_scope",
                        )

                    return RoutingDecision(
                        department_ids=[primary] + [d for d in secondary if d != primary],
                        confidence=max(0.0, min(1.0, llm_conf)),
                        mode=mode,
                        comment=comment,
                        used_llm=True,
                    )
            except Exception as exc:
                err_text = str(exc)
        else:
            err_text = "No JSON object found in model output"

        raw = client.chat(
            system=system,
            user=(
                "Исправь ответ. Нужен СТРОГО валидный JSON по схеме из запроса.\n"
                f"Ошибка: {err_text}\n"
                "Верни только JSON, без текста.\n"
                "Твой предыдущий ответ:\n"
                + raw
            ),
        )

    return RoutingDecision(
        department_ids=[top_candidate.department_id],
        confidence=confidence,
        mode=mode,
        comment="LLM-assisted mode: LLM failed to produce valid JSON; used heuristic fallback."
        + low_score_note,
        used_llm=True,
        fallback_reason="llm_invalid_output",
    )
