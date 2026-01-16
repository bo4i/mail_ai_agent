#!/usr/bin/env python3
"""Run PDF extraction + text normalization pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pdf_extractor import extract_pdf
from text_normalizer import normalize_texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract PDF text and normalize output.")
    parser.add_argument("pdf", type=Path, help="Path to input PDF")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON file")
    parser.add_argument("--min-text-chars", type=int, default=50, help="Minimum text length to skip OCR")
    parser.add_argument("--dpi", type=int, default=300, help="Render DPI for OCR")
    parser.add_argument(
        "--lang",
        type=str,
        default="rus+eng",
        help="Tesseract language (default: rus+eng)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = extract_pdf(
        pdf_path=args.pdf,
        min_text_chars=args.min_text_chars,
        dpi=args.dpi,
        lang=args.lang,
    )
    payload = []
    for page in results:
        normalized = normalize_texts([page.text], ocr_used=page.source == "ocr")[0]
        payload.append(
            {
                "page": page.page_number,
                "source": page.source,
                "normalized": normalized.to_dict(),
            }
        )
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()