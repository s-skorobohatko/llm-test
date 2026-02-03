from __future__ import annotations
from typing import Any, Dict, List, Optional
import requests


class OllamaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def embed(self, model: str, text: str) -> List[float]:
        url = f"{self.base_url}/api/embeddings"
        resp = requests.post(url, json={"model": model, "prompt": text}, timeout=600)
        resp.raise_for_status()
        return resp.json()["embedding"]

    def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        num_ctx: int,
        num_predict: int,
        temperature: float = 0.1,
        stop: Optional[list[str]] = None,
        timeout_sec: int = 1800,
    ) -> str:
        url = f"{self.base_url}/api/chat"
        opts: Dict[str, Any] = {
            "num_ctx": int(num_ctx),
            "num_predict": int(num_predict),
            "temperature": float(temperature),
        }
        if stop:
            opts["stop"] = stop

        payload = {"model": model, "messages": messages, "stream": False, "options": opts}
        resp = requests.post(url, json=payload, timeout=(10, int(timeout_sec)))
        resp.raise_for_status()
        return resp.json()["message"]["content"]
