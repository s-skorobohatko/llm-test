import hashlib
import os
import fnmatch
import subprocess
import uuid
from typing import Iterator, List, Optional

import requests


# ============================================================
# Text / file utilities (used by ingest.py)
# ============================================================

def load_text_file(path: str) -> str:
    """Read a text file as UTF-8 (with replacement)."""
    with open(path, "rb") as f:
        raw = f.read()
    return raw.decode("utf-8", errors="replace")


def sha256_str(s: str) -> str:
    """Stable hash for string content."""
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()


sha256_text = sha256_str


def uuid5_str(s: str, namespace: uuid.UUID = uuid.NAMESPACE_URL) -> str:
    """Deterministic UUID (v5) for a given string."""
    return str(uuid.uuid5(namespace, s))


def _expand_brace_glob(glob_pattern: str) -> List[str]:
    if "{" in glob_pattern and "}" in glob_pattern:
        pre = glob_pattern.split("{", 1)[0]
        rest = glob_pattern.split("{", 1)[1]
        inner, post = rest.split("}", 1)
        alts = [x.strip() for x in inner.split(",")]
        return [pre + a + post for a in alts]
    return [glob_pattern]


def iter_files(root: str, glob_pattern: str) -> Iterator[str]:
    patterns = _expand_brace_glob(glob_pattern)

    SKIP_DIRS = {
        ".git", ".svn", ".hg", ".idea", ".vscode", "__pycache__", ".venv",
        "vendor", "node_modules",
    }

    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]

        for fn in files:
            if fn.startswith("."):
                continue
            full = os.path.join(base, fn)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            for pat in patterns:
                if fnmatch.fnmatch(rel, pat.replace(os.sep, "/")):
                    yield full
                    break


def list_files_from_dir(root: str, glob_pattern: str) -> List[str]:
    return sorted(iter_files(root, glob_pattern))


def list_files_multi_glob(root: str, glob_patterns: List[str]) -> List[str]:
    seen = set()
    for pat in glob_patterns:
        for p in iter_files(root, pat):
            seen.add(p)
    return sorted(seen)


def chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int = 0,
    *,
    overlap: Optional[int] = None,
) -> List[str]:
    if overlap is not None:
        chunk_overlap = overlap

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("overlap must be >=0 and < chunk_size")

    chunks: List[str] = []
    step = chunk_size - chunk_overlap
    i = 0
    n = len(text)

    while i < n:
        part = text[i:i + chunk_size]
        if part.strip():
            chunks.append(part)
        i += step

    return chunks


# ============================================================
# Git utilities (used by ingest.py)
# ============================================================

def git_sync(repo_url: str, dest: str, depth: int = 1) -> None:
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        cmd = ["git", "clone", "--depth", str(depth), repo_url, dest]
    else:
        cmd = ["git", "-C", dest, "pull", "--ff-only"]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"git_sync failed for {repo_url}\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )


# ============================================================
# Ollama client (non-stream only)
# ============================================================

class OllamaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def embed(self, model: str, text: str) -> List[float]:
        url = f"{self.base_url}/api/embeddings"
        payload = {"model": model, "prompt": text}
        resp = requests.post(url, json=payload, timeout=600)
        resp.raise_for_status()
        return resp.json()["embedding"]

    def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        temperature: float | None = None,
        stop: list[str] | None = None,
        timeout_sec: int = 1800,
        **_ignored,
    ) -> str:
        """
        Non-streaming chat. Predictable: request completes and returns one JSON response.
        No num_predict (no client-side hard stop).
        """
        url = f"{self.base_url}/api/chat"
        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": False,
        }

        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        if stop:
            options["stop"] = stop
        if options:
            payload["options"] = options

        resp = requests.post(url, json=payload, timeout=(10, int(timeout_sec)))
        resp.raise_for_status()
        return resp.json()["message"]["content"]
