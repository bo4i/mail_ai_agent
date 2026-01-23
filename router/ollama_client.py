from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "gpt-oss"
    temperature: float = 0.2
    timeout_s: float = 90.0


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

        with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
            resp_text = resp.read().decode("utf-8", errors="replace")

        obj = json.loads(resp_text)
        return (obj.get("message") or {}).get("content", "") or ""
