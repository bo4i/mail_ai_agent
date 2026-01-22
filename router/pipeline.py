from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pdf_extractor import extract_pdf
from text_normalizer import normalize_texts

from router.candidate_retrieval import apply_rules_boosts, retrieve_candidates
from router.catalog_loader import load_departments_catalog
from router.decision_builder import build_decision
from router.input_adapter import normalized_letter_from_pages
from router.llm_decider import decide_routing
from router.rules_engine import apply_triage_rules
from router.validator import validate_decision


def route_document(
    pdf_path: Path,
    *,
    catalog_path: Path,
    schema_path: Path,
    output_path: Path | None = None,
    mode: str = "heuristic_only",
    ollama_url: str = "http://localhost:11434",
    ollama_model: str = "gpt-oss",
    ollama_temperature: float = 0.2,
    llm_topk: int = 5,
    llm_force: bool = False,
) -> dict:
    started = time.perf_counter()
    pages = extract_pdf(pdf_path=pdf_path, min_text_chars=50, dpi=300, lang="rus+eng")
    normalized_pages = []
    for page in pages:
        normalized = normalize_texts([page.text], ocr_used=page.source == "ocr")[0]
        normalized_pages.append(
            {
                "page": page.page_number,
                "source": page.source,
                "normalized": normalized.to_dict(),
            }
        )

    letter = normalized_letter_from_pages(
        normalized_pages,
        filename=pdf_path.name,
    )
    catalog = load_departments_catalog(catalog_path)
    rules_context = apply_triage_rules(letter.clean_text_for_llm, catalog)
    candidates = retrieve_candidates(letter.clean_text_for_llm, catalog)
    candidates = apply_rules_boosts(candidates, rules_context)
    routing_decision = decide_routing(
        candidates,
        rules_context,
        mode=mode,
        letter_text=letter.clean_text_for_llm,
        catalog=catalog,
        ollama_url=ollama_url,
        ollama_model=ollama_model,
        ollama_temperature=ollama_temperature,
        llm_top_k=llm_topk,
        llm_force=llm_force,
    )

    processing_time_ms = int((time.perf_counter() - started) * 1000)
    decision = build_decision(
        letter,
        catalog,
        candidates,
        rules_context,
        routing_decision,
        processing_time_ms=processing_time_ms,
    )
    validate_decision(decision, catalog, schema_path)

    if output_path is None:
        output_path = pdf_path.with_suffix(".routing_decision.json")
    output_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Route PDF document to department decision")
    parser.add_argument("pdf", type=Path, help="Path to input PDF")
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("depatments_catalog.json"),
        help="Path to departments catalog JSON",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("routing_decision.schema.json"),
        help="Path to routing decision schema JSON",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["heuristic_only", "llm_assisted"],
        default="heuristic_only",
        help="Routing mode",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama base URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default="gpt-oss",
        help="Ollama model name (default: gpt-oss)",
    )
    parser.add_argument(
        "--ollama-temperature",
        type=float,
        default=0.2,
        help="LLM temperature (default: 0.2)",
    )
    parser.add_argument(
        "--llm-topk",
        type=int,
        default=5,
        help="How many top heuristic candidates to pass into LLM (default: 5)",
    )
    parser.add_argument(
        "--llm-force",
        action="store_true",
        help="Force LLM call even if heuristic confidence is high",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: <pdf>.routing_decision.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    route_document(
        args.pdf,
        catalog_path=args.catalog,
        schema_path=args.schema,
        output_path=args.output,
        mode=args.mode,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        ollama_temperature=args.ollama_temperature,
        llm_topk=args.llm_topk,
        llm_force=args.llm_force,
    )


if __name__ == "__main__":
    main()
