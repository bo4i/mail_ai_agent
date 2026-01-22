# PDF Extractor

Извлечение текста из PDF с fallback на OCR (PyMuPDF + Tesseract RU).

## Установка

```bash
pip install -r requirements.txt
```

## Использование

```bash
python pdf_extractor.py input.pdf --output result.json --min-text-chars 50 --dpi 300 --lang rus
```

Выходные данные содержат mapping страниц:

```json
{
  "pages": [
    {"page": 1, "text": "...", "source": "text"},
    {"page": 2, "text": "...", "source": "ocr"}
  ]
}
```
