#!/usr/bin/env python3
"""Normalize extracted mail text into structured fields."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


@dataclass
class NormalizationResult:
    doc_type: str | None
    issuer: str | None
    addressee: str | None
    subject: str | None
    action_required: bool
    deadline: str | None
    attachments: list[str]
    topics: list[str]
    signatory: str | None
    contacts: list[str]
    clean_text_for_llm: str
    confidence_flags: dict[str, bool]

    def to_dict(self) -> dict:
        return {
            "doc_type": self.doc_type,
            "issuer": self.issuer,
            "addressee": self.addressee,
            "subject": self.subject,
            "action_required": self.action_required,
            "deadline": self.deadline,
            "attachments": self.attachments,
            "topics": self.topics,
            "signatory": self.signatory,
            "contacts": self.contacts,
            "clean_text_for_llm": self.clean_text_for_llm,
            "confidence_flags": self.confidence_flags,
        }


PLACEHOLDER_PATTERNS = (
    re.compile(r"\[REGNUMDATESTAMP\]"),
    re.compile(r"\[SIGNERSTAMP\d*\]"),
)

SIGNATURE_BLOCK_RE = re.compile(
    r"ДОКУМЕНТ\s+ПОДПИСАН[\s\S]{0,500}?(?:ЭЛЕКТРОННОЙ\s+ПОДПИСЬЮ|СЕРТИФИКАТ)",
    re.IGNORECASE,
)

HEADER_RE = re.compile(r"^(.*?)\n(?:\[REGNUMDATESTAMP\]|На\s+№\s+от)", re.DOTALL | re.MULTILINE)
ADDRESSEE_RE = re.compile(r"^(Руководителям|Органам|В\s+адрес)\s+(.+)$", re.MULTILINE)
SUBJECT_RE = re.compile(r"^О\s+.+$", re.MULTILINE)
REG_NUMBER_RE = re.compile(r"№\s*([А-ЯA-Z0-9\-–/]+)")
REG_DATE_RE = re.compile(r"от\s+(\d{1,2}\s+[а-яё]+\s+\d{4}\s+года)", re.IGNORECASE)
ATTACHMENT_RE = re.compile(r"Приложени[ея]:?\s*(.+)", re.IGNORECASE)
DEADLINE_RE = re.compile(r"в\s+срок\s+до\s+(\d{2}\.\d{2}\.\d{4})", re.IGNORECASE)
DEADLINE_PLEASE_RE = re.compile(r"просим.*?до\s+(\d{2}\.\d{2}\.\d{4})", re.IGNORECASE | re.DOTALL)
SIGNATORY_RE = re.compile(
    r"^(Министр|Заместитель министра|Директор)[\s\S]{0,100}?\n"
    r"([А-ЯЁA-Z]\.[А-ЯЁA-Z]\.\s*[А-ЯЁа-яё]+)",
    re.MULTILINE,
)
CONTACTS_RE = re.compile(r"(Исполнитель:|Тел:|Телефон)\s*([^\n]+)", re.IGNORECASE)


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


def _hard_cleanup(text: str) -> str:
    for pattern in PLACEHOLDER_PATTERNS:
        text = pattern.sub("", text)
    text = SIGNATURE_BLOCK_RE.sub("", text)
    return _normalize_whitespace(text).strip()


def _extract_header(text: str) -> str | None:
    match = HEADER_RE.search(text)
    if not match:
        return None
    return match.group(1).strip()


def _extract_issuer(header_raw: str | None) -> str | None:
    if not header_raw:
        return None
    lines = [line.strip() for line in header_raw.splitlines() if line.strip()]
    if not lines:
        return None
    stop_rx = re.compile(
        r"^(Юр\.адрес|Адрес|ИНН|КПП|ОГРН|тел|телефон|факс|e-mail|сайт|http|www|@|пл)",
        re.IGNORECASE,
    )
    collected: list[str] = []
    opened_quote = False
    for line in lines:
        if stop_rx.match(line) and not opened_quote:
            break
        collected.append(line)
        if "«" in line:
            opened_quote = True
        if "»" in line:
            opened_quote = False
    issuer = " ".join(collected).strip()
    if issuer:
        word_count = len(issuer.split())
    else:
        word_count = 0
    if word_count <= 4:
        for line in lines[len(collected) :]:
            if stop_rx.match(line):
                break
            if line.isupper():
                collected.append(line)
            else:
                break
        issuer = " ".join(collected).strip()
    return issuer or None


def _extract_addressee(text: str) -> str | None:
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        match = ADDRESSEE_RE.match(line)
        if not match:
            continue
        block = [line.strip()]
        for next_line in lines[idx + 1 :]:
            if not next_line.strip():
                break
            block.append(next_line.strip())
        return " ".join(block)
    return None


def _extract_subject(text: str) -> str | None:
    match = SUBJECT_RE.search(text)
    return match.group(0).strip() if match else None


def _extract_reg_number(text: str) -> str | None:
    match = REG_NUMBER_RE.search(text)
    return match.group(1).strip() if match else None


def _extract_reg_date(text: str) -> str | None:
    match = REG_DATE_RE.search(text)
    return match.group(1).strip() if match else None


def _extract_attachments(text: str) -> list[str]:
    attachments: list[str] = []
    for match in ATTACHMENT_RE.findall(text):
        chunks = [chunk.strip() for chunk in re.split(r"[;,]", match) if chunk.strip()]
        attachments.extend(chunks)
    return attachments


def _extract_deadline(text: str) -> str | None:
    match = DEADLINE_RE.search(text)
    if match:
        return match.group(1)
    match = DEADLINE_PLEASE_RE.search(text)
    if match:
        return match.group(1)
    return None


def _extract_signatory(text: str) -> str | None:
    match = SIGNATORY_RE.search(text)
    return match.group(2).strip() if match else None


def _extract_contacts(text: str) -> list[str]:
    return [match[1].strip() for match in CONTACTS_RE.findall(text)]


def _detect_doc_type(text: str, subject: str | None, deadline: str | None) -> str | None:
    lowered = text.lower()
    has_deadline = deadline is not None
    if "приложение xlsx" in lowered and has_deadline:
        return "reporting"
    if "просим заполнить" in lowered or "направить форму" in lowered:
        return "action_required"
    if ("о запросе информации" in lowered or "просим направить" in lowered) and has_deadline:
        return "request"
    if "направляем для ознакомления" in lowered:
        return "informational"
    if subject and subject.lower().startswith("о запросе информации") and has_deadline:
        return "request"
    return None


def _detect_topics(text: str) -> list[str]:
    lowered = text.lower()
    topics: list[str] = []
    if re.search(r"\bии\b", lowered) or "искусственного интеллекта" in lowered:
        topics.append("AI")
    if re.search(r"отч[её]т", lowered) or "отчетность" in lowered:
        topics.append("reporting")
    if "финансирование" in lowered or "субсидия" in lowered:
        topics.append("finance")
    if "видеонаблюдение" in lowered or "компьютерное зрение" in lowered:
        topics.append("CV")
    return topics


def normalize_text(text: str, *, ocr_used: bool = False) -> NormalizationResult:
    cleaned = _hard_cleanup(text)
    header_raw = _extract_header(cleaned)
    issuer = _extract_issuer(header_raw)
    addressee = _extract_addressee(cleaned)
    subject = _extract_subject(cleaned)
    reg_number = _extract_reg_number(cleaned)
    _ = _extract_reg_date(cleaned)
    attachments = _extract_attachments(cleaned)
    deadline = _extract_deadline(cleaned)
    signatory = _extract_signatory(cleaned)
    contacts = _extract_contacts(cleaned)
    doc_type = _detect_doc_type(cleaned, subject, deadline)
    topics = _detect_topics(cleaned)
    action_required = deadline is not None
    confidence_flags = {
        "ocr_used": ocr_used,
        "missing_reg_number": reg_number is None,
    }
    return NormalizationResult(
        doc_type=doc_type,
        issuer=issuer,
        addressee=addressee,
        subject=subject,
        action_required=action_required,
        deadline=deadline,
        attachments=attachments,
        topics=topics,
        signatory=signatory,
        contacts=contacts,
        clean_text_for_llm=cleaned,
        confidence_flags=confidence_flags,
    )


def normalize_texts(texts: Iterable[str], *, ocr_used: bool = False) -> list[NormalizationResult]:
    return [normalize_text(text, ocr_used=ocr_used) for text in texts]
