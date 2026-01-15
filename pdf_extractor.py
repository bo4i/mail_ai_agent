#!/usr/bin/env python3
"""PDF text extractor with OCR fallback."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import fitz  # PyMuPDF
import numpy as np
import pytesseract
from PIL import Image
import cv2


@dataclass
class PageResult:
    page_number: int
    text: str
    source: str


def extract_text_from_page(page: fitz.Page) -> str:
    # Native text extraction via PyMuPDF.
    return page.get_text("text")


def render_page_to_image(page: fitz.Page, dpi: int) -> Image.Image:
    # Render a page at the requested DPI for OCR.
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img


def deskew_and_threshold(image: Image.Image) -> Image.Image:
    # Convert to grayscale and apply Otsu thresholding for OCR readiness.
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Estimate skew angle from foreground pixels and rotate to deskew.
    coords = np.column_stack(np.where(binary < 255))
    if coords.size == 0:
        return Image.fromarray(binary)

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = binary.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(binary, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(rotated)


def ocr_image(image: Image.Image, lang: str) -> str:
    # Run Tesseract OCR with the selected language model.
    return pytesseract.image_to_string(image, lang=lang)


def extract_pdf(
    pdf_path: Path,
    min_text_chars: int,
    dpi: int,
    lang: str,
) -> list[PageResult]:
    results: list[PageResult] = []
    with fitz.open(pdf_path) as doc:
        for index, page in enumerate(doc, start=1):
            # Try native text extraction first.
            text = extract_text_from_page(page)
            source = "text"
            # Fallback to OCR if the text content is too small.
            if len(text.strip()) < min_text_chars:
                image = render_page_to_image(page, dpi)
                processed = deskew_and_threshold(image)
                text = ocr_image(processed, lang=lang)
                source = "ocr"
            results.append(PageResult(page_number=index, text=text, source=source))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PDF extractor with PyMuPDF + OCR fallback")
    parser.add_argument("pdf", type=Path, help="Path to input PDF")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON file (default: stdout)")
    parser.add_argument("--min-text-chars", type=int, default=50, help="Minimum text length to skip OCR")
    parser.add_argument("--dpi", type=int, default=300, help="Render DPI for OCR")
    parser.add_argument("--lang", type=str, default="rus", help="Tesseract language (default: rus)")
    return parser.parse_args()


def page_results_to_json(results: Iterable[PageResult]) -> dict:
    # Map pages to their extracted text and source.
    return {
        "pages": [
            {"page": result.page_number, "text": result.text, "source": result.source}
            for result in results
        ]
    }


def main() -> None:
    args = parse_args()
    # Execute extraction and emit JSON output.
    results = extract_pdf(
        pdf_path=args.pdf,
        min_text_chars=args.min_text_chars,
        dpi=args.dpi,
        lang=args.lang,
    )
    payload = page_results_to_json(results)
    output = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
