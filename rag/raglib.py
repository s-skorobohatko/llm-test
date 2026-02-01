import json
import requests


class OllamaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def embed(self, model: str, text: str):
        url = f"{self.base_url}/api/embeddings"
        payload = {"model": model, "prompt": text}
        resp = requests.post(url, json=payload, timeout=600)
        resp.raise_for_status()
        return resp.json()["embedding"]

    def chat(
        self,
        model: str,
        prompt: str = None,
        messages=None,
        system: str = None,
        options: dict = None,
        stream: bool = False,
        stream_print: bool = True,
    ) -> str:
        """
        If messages is provided -> /api/chat
        else -> /api/generate

        stream=True prints tokens live (if stream_print=True) and returns full text.
        """
        options = options or {}

        # --- CHAT endpoint (messages) ---
        if messages is not None:
            url = f"{self.base_url}/api/chat"
            payload = {
                "model": model,
                "messages": messages,
                "stream": bool(stream),
                "options": options,
            }

            if not stream:
                resp = requests.post(url, json=payload, timeout=3600)
                resp.raise_for_status()
                data = resp.json()
                return data.get("message", {}).get("content", "")

            # streaming mode
            resp = requests.post(url, json=payload, stream=True, timeout=3600)
            resp.raise_for_status()

            out_chunks = []
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                data = json.loads(line)

                # Each chunk has {"message":{"role":"assistant","content":"..."}, "done":false}
                msg = data.get("message") or {}
                piece = msg.get("content") or ""
                if piece:
                    out_chunks.append(piece)
                    if stream_print:
                        print(piece, end="", flush=True)

                if data.get("done") is True:
                    break

            return "".join(out_chunks)

        # --- GENERATE endpoint (prompt) ---
        if prompt is None:
            raise TypeError("chat() requires either prompt=... or messages=[...]")

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": bool(stream),
            "options": options,
        }
        if system:
            payload["system"] = system

        if not stream:
            resp = requests.post(url, json=payload, timeout=3600)
            resp.raise_for_status()
            return resp.json().get("response", "")

        resp = requests.post(url, json=payload, stream=True, timeout=3600)
        resp.raise_for_status()

        out_chunks = []
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            data = json.loads(line)

            # generate stream chunks have {"response":"...", "done":false}
            piece = data.get("response") or ""
            if piece:
                out_chunks.append(piece)
                if stream_print:
                    print(piece, end="", flush=True)

            if data.get("done") is True:
                break

        return "".join(out_chunks)
