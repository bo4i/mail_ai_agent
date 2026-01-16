from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from router.models import DepartmentsCatalog


def validate_schema(decision: dict, schema_path: Path) -> list[str]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    errors = [error.message for error in validator.iter_errors(decision)]
    return errors


def validate_departments(decision: dict, catalog: DepartmentsCatalog) -> list[str]:
    valid_ids = {dept.department_id for dept in catalog.departments}
    missing: list[str] = []
    for suggestion in decision.get("routing", {}).get("suggestions", []):
        department_id = suggestion.get("department_id")
        if department_id and department_id not in valid_ids:
            missing.append(department_id)
    for department_id in decision.get("routing", {}).get("final_recommendation", {}).get(
        "department_ids", []
    ):
        if department_id not in valid_ids:
            missing.append(department_id)
    return missing


def validate_decision(
    decision: dict,
    catalog: DepartmentsCatalog,
    schema_path: Path,
) -> None:
    schema_errors = validate_schema(decision, schema_path)
    if schema_errors:
        details = "\n".join(schema_errors)
        raise ValueError(f"Schema validation failed:\n{details}")

    missing_departments = validate_departments(decision, catalog)
    if missing_departments:
        missing_list = ", ".join(sorted(set(missing_departments)))
        raise ValueError(f"Unknown department_id in decision: {missing_list}")