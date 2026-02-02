import hashlib
import json
import os
import fnmatch
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Tuple

import requests


# ----------------------------
# Text / file utilities (used by ingest.py)
# ----------------------------

def load_text_file(path: str) -> str:
    """
    Read a text file as UTF-8 (with replacement), return full contents as string.
    """
    with open(path, "rb") as f:
        raw = f.read()
    return raw.decode("utf-8", errors="replace")


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()


def iter_files(root: str, glob_pattern: str) -> Iterator[str]:
    """
    Yield file paths under root matching a glob like **/*.md or **/*.{pp,epp,md}.
    Supports:
      - ** recursion
      - brace sets: *.{pp,epp,md}
    """
    # Expand brace sets: *.{a,b,c}
    patterns = []
    if "{" in glob_pattern and "}" in glob_pattern:
        pre = glob_pattern.split("{", 1)[0]
        rest = glob_pattern.split("{", 1)[1]
        inner, post = rest.split("}", 1)
        alts = [x.strip() for x in inner.split(",")]
        for a in alts:
            patterns.append(pre + a + post)
    else:
        patterns.append(glob_pattern)

    # Walk all files and match using fnmatch
    for base, _, files in os.walk(root):
        for fn in files:
            full = os.path.join(base, fn)
            rel = os.path.relpath(full, root)
            rel_posix = rel.replace(os.sep, "/")
            for pat in patterns:
                # normalize patterns to posix
                pat_posix = pat.replace(os.sep, "/")
                if fnmatch.fnmatch(rel_posix, pat_posix):
                    yield full
                    break


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Chunk by characters (simple, fast) with overlap.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >=0 and < chunk_size")

    chunks = []
    i = 0
    n = len(text)
    step = chunk_size - chunk_overlap

    while i < n:
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
        i += step

    return chunks


# ----------------------------
# Ollama client (used by ask.py + ingest.py)
# ----------------------------

class OllamaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def embed(self, model: str, text: str) -> List[float]:
        url = f"{self.base_url}/api/embeddings"
        payload = {"model": model, "prompt": text}
        resp = requests.post(url, json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        return data["embedding"]

    def chat(
        self,
        model: str,
        prompt: str = None,
        messages=None,
        system: str = None,
        options: dict = None,
        stream: bool = False,
        stream_print: bool = True,
        timeout_sec: int = 3600,
    ) -> str:
        """
        If messages is provided -> /api/chat
        else -> /api/generate

        stream=True prints tokens live (if stream_print=True) and returns full text.
        """
        options = options or {}

        # --- CHAT endpoint ---
        if messages is not None:
            url = f"{self.base_url}/api/chat"
            payload = {
                "model": model,
                "messages": messages,
                "stream": bool(stream),
                "options": options,
            }

            if not stream:
                resp = requests.post(url, json=payload, timeout=timeout_sec)
                resp.raise_for_status()
                data = resp.json()
                return data.get("message", {}).get("content", "")

            resp = requests.post(url, json=payload, stream=True, timeout=(10, timeout_sec))
            resp.raise_for_status()

            out_chunks = []
            for raw in resp.iter_lines(chunk_size=1, delimiter=b"\n"):
                if not raw:
                    continue
                try:
                    line = raw.decode("utf-8", errors="replace")
                    data = json.loads(line)
                except Exception:
                    continue

                msg = data.get("message") or {}
                piece = msg.get("content") or ""
                if piece:
                    out_chunks.append(piece)
                    if stream_print:
                        print(piece, end="", flush=True)

                if data.get("done") is True:
                    break

            return "".join(out_chunks)

        # --- GENERATE endpoint ---
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
            resp = requests.post(url, json=payload, timeout=timeout_sec)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")

        resp = requests.post(url, json=payload, stream=True, timeout=(10, timeout_sec))
        resp.raise_for_status()

        out_chunks = []
        for raw in resp.iter_lines(chunk_size=1, delimiter=b"\n"):
            if not raw:
                continue
            try:
                line = raw.decode("utf-8", errors="replace")
                data = json.loads(line)
            except Exception:
                continue

            piece = data.get("response") or ""
            if piece:
                out_chunks.append(piece)
                if stream_print:
                    print(piece, end="", flush=True)

            if data.get("done") is True:
                break

        return "".join(out_chunks)
