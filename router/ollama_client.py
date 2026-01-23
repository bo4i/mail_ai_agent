from __future__ import annotations

import http.client
import json
import time
import urllib.request
from urllib.error import URLError
from dataclasses import dataclass
from typing import Any


@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "gpt-oss"
    temperature: float = 0.2
    timeout_s: float = 90.0
    max_retries: int = 2
    retry_backoff_s: float = 1.0


class OllamaError(RuntimeError):
    """Ollama request failed after retries."""


class OllamaClient:
    """Minimal Ollama /api/chat client (stdlib-only)."""

    def __init__(self, cfg: OllamaConfig):
        self.cfg = cfg

    def chat(self, *, system: str, user: str) -> str:
        url = f"{self.cfg.base_url.rstrip('/')}/api/chat"
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": float(self.cfg.temperature)},
        }

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url=url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        last_err: Exception | None = None
        attempts = max(0, int(self.cfg.max_retries)) + 1
        for attempt in range(1, attempts + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
                    resp_text = resp.read().decode("utf-8", errors="replace")
                obj = json.loads(resp_text)
                return (obj.get("message") or {}).get("content", "") or ""
            except (TimeoutError, URLError, http.client.HTTPException, ValueError) as exc:
                last_err = exc
                if attempt >= attempts:
                    break
                sleep_s = max(0.0, float(self.cfg.retry_backoff_s)) * attempt
                if sleep_s:
                    time.sleep(sleep_s)

        raise OllamaError(f"Ollama chat request failed after {attempts} attempts") from last_err
