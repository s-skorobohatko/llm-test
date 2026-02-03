from __future__ import annotations
from typing import Any, Dict, List, Optional
import requests


class OllamaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def list_models(self) -> List[str]:
        """
        Returns a list of model names known to Ollama locally.
        """
        url = f"{self.base_url}/api/tags"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        models = resp.json().get("models") or []
        return [m.get("name", "") for m in models if m.get("name")]

    def embed(self, model: str, text: str) -> List[float]:
        url = f"{self.base_url}/api/embeddings"
        payload = {"model": model, "prompt": text}
        resp = requests.post(url, json=payload, timeout=600)

        if resp.status_code >= 400:
            # Ollama often returns useful "error" in JSON body
            detail = ""
            try:
                j = resp.json()
                detail = j.get("error") or str(j)
            except Exception:
                detail = (resp.text or "").strip()
            raise RuntimeError(f"Ollama embeddings error {resp.status_code}: {detail}")

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

        if resp.status_code >= 400:
            detail = ""
            try:
                j = resp.json()
                detail = j.get("error") or str(j)
            except Exception:
                detail = (resp.text or "").strip()
            raise RuntimeError(f"Ollama chat error {resp.status_code}: {detail}")

        return resp.json()["message"]["content"]
