from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from router.models import Department, DepartmentsCatalog


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

        departments.append(
            Department(
                department_id=department_id,
                department_name=item.get("department_name", department_id),
                routing_keywords=routing_keywords,
                triage_rules=triage_rules,
                raw=item,
            )
        )

    catalog_version = payload.get("catalog_version") if isinstance(payload, dict) else None
    return DepartmentsCatalog(
        departments=departments,
        catalog_version=catalog_version or "dev",
    )
