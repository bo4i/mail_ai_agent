from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from router.models import NormalizedLetter, NormalizedPage


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_raw_pages(payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        msg = "Expected payload to be a list of pages from 1.json"
        raise ValueError(msg)
    return payload


def normalized_letter_from_pages(
    pages_payload: list[dict[str, Any]],
    *,
    source_channel: str = "manual_upload",
    filename: str = "unknown",
    request_id: str | None = None,
    created_at: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> NormalizedLetter:
    request_id = request_id or f"REQ-{uuid4()}"
    created_at = created_at or _iso_now()
    metadata = metadata or {}

    subjects: list[str] = []
    issuers: list[str] = []
    addressees: list[str] = []
    topics: list[str] = []
    attachments: list[str] = []
    clean_text_blocks: list[str] = []
    pages: list[NormalizedPage] = []

    for item in pages_payload:
        normalized = item.get("normalized", {})
        subject = normalized.get("subject")
        issuer = normalized.get("issuer")
        addressee = normalized.get("addressee")
        if subject:
            subjects.append(subject)
        if issuer:
            issuers.append(issuer)
        if addressee:
            addressees.append(addressee)
        for topic in normalized.get("topics", []) or []:
            if topic not in topics:
                topics.append(topic)
        for attachment in normalized.get("attachments", []) or []:
            if attachment not in attachments:
                attachments.append(attachment)
        page_text = normalized.get("clean_text_for_llm", "")
        if page_text:
            clean_text_blocks.append(page_text)
        confidence_flags = normalized.get("confidence_flags", {})
        if "ocr_used" not in confidence_flags:
            confidence_flags = dict(confidence_flags)
            confidence_flags["ocr_used"] = item.get("source") == "ocr"
        pages.append(
            NormalizedPage(
                page=int(item.get("page") or 0),
                source=str(item.get("source") or "unknown"),
                clean_text_for_llm=page_text,
                confidence_flags=confidence_flags,
            )
        )

    subject = subjects[0] if subjects else None
    issuer = issuers[0] if issuers else None
    addressee = addressees[0] if addressees else None
    clean_text_for_llm = "\n\n".join(block for block in clean_text_blocks if block)

    return NormalizedLetter(
        request_id=request_id,
        created_at=created_at,
        source_channel=source_channel,
        filename=filename,
        pages=pages,
        subject=subject,
        issuer=issuer,
        addressee=addressee,
        topics=topics,
        attachments=attachments,
        clean_text_for_llm=clean_text_for_llm,
        metadata=metadata,
    )


def normalized_letter_from_json(
    json_path: Path,
    *,
    source_channel: str = "manual_upload",
    filename: str | None = None,
    request_id: str | None = None,
    created_at: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> NormalizedLetter:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    pages = load_raw_pages(payload)
    return normalized_letter_from_pages(
        pages,
        source_channel=source_channel,
        filename=filename or json_path.name,
        request_id=request_id,
        created_at=created_at,
        metadata=metadata,
    )


def normalized_letter_to_dict(letter: NormalizedLetter) -> dict[str, Any]:
    payload = asdict(letter)
    payload["pages"] = [asdict(page) for page in letter.pages]
    return payload