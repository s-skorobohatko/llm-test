import requests


class OllamaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def embed(self, model: str, text: str):
        url = f"{self.base_url}/api/embeddings"
        payload = {"model": model, "prompt": text}
        resp = requests.post(url, json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        return data["embedding"]

    def chat(self, model: str, prompt: str = None, messages=None, system: str = None, options: dict = None):
        """
        Compatibility chat:
        - If messages is provided, uses /api/chat with messages.
        - Else uses /api/generate with prompt (and optional system).

        Returns: response text string
        """
        options = options or {}

        if messages is not None:
            url = f"{self.base_url}/api/chat"
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": options,
            }
            resp = requests.post(url, json=payload, timeout=600)
            resp.raise_for_status()
            data = resp.json()
            # Ollama chat response shape: {"message":{"role":"assistant","content":"..."}, ...}
            return data.get("message", {}).get("content", "")

        # fallback: generate endpoint
        if prompt is None:
            raise TypeError("chat() requires either prompt=... or messages=[...]")

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if system:
            payload["system"] = system

        resp = requests.post(url, json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        # Ollama generate response shape: {"response":"...", ...}
        return data.get("response", "")
