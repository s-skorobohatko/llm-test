import hashlib
import json
import os
import fnmatch
import subprocess
import uuid
from typing import Iterator, List, Optional
import time 

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


# Backward-compatible alias
sha256_text = sha256_str


def uuid5_str(s: str, namespace: uuid.UUID = uuid.NAMESPACE_URL) -> str:
    """
    Deterministic UUID (v5) for a given string.
    Qdrant accepts UUID strings as point IDs.
    """
    return str(uuid.uuid5(namespace, s))


def _expand_brace_glob(glob_pattern: str) -> List[str]:
    """
    Expand brace sets like **/*.{pp,epp,md} into multiple patterns.
    If no braces, returns [glob_pattern].
    """
    if "{" in glob_pattern and "}" in glob_pattern:
        pre = glob_pattern.split("{", 1)[0]
        rest = glob_pattern.split("{", 1)[1]
        inner, post = rest.split("}", 1)
        alts = [x.strip() for x in inner.split(",")]
        return [pre + a + post for a in alts]
    return [glob_pattern]


def iter_files(root: str, glob_pattern: str) -> Iterator[str]:
    """
    Yield full file paths under root matching a glob.
    Supports ** recursion and brace sets.

    Always skips:
      - .git/ and other junk dirs
      - hidden dirs and hidden files
    """
    patterns = _expand_brace_glob(glob_pattern)

    SKIP_DIRS = {
        ".git", ".svn", ".hg", ".idea", ".vscode", "__pycache__", ".venv",
        "vendor", "node_modules",
    }

    for base, dirs, files in os.walk(root):
        # prune dirs in-place (important)
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
    """ingest.py expects this name."""
    return sorted(iter_files(root, glob_pattern))


def list_files_multi_glob(root: str, glob_patterns: List[str]) -> List[str]:
    """Return sorted, deduplicated list of files matching ANY glob in list."""
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
    """
    Chunk text by characters with overlap.

    Backward compatible:
      - chunk_text(text, chunk_size, overlap=200)
      - chunk_text(text, chunk_size, chunk_overlap=200)
    """
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
    """
    Clone or fast-forward pull a git repository.

    - If dest does not exist -> git clone --depth <depth>
    - If dest exists -> git pull --ff-only
    """
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
# Ollama client (used by ingest.py / ask.py)
# ============================================================

class OllamaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    # ---------- embeddings ----------
    def embed(self, model: str, text: str) -> List[float]:
        url = f"{self.base_url}/api/embeddings"
        payload = {"model": model, "prompt": text}
        resp = requests.post(url, json=payload, timeout=600)
        resp.raise_for_status()
        return resp.json()["embedding"]

    # ---------- chat ----------
    def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        stream: bool = False,
        temperature: float | None = None,
        stream_print: bool = False,
        timeout_sec: int = 7200,              # allow long CPU runs
        first_token_timeout_sec: int = 900,   # 15 min default for CPU prefill
        num_predict: int = 1024,
        **_ignored,
    ):
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }

        options = {"num_predict": int(num_predict)}
        if temperature is not None:
            options["temperature"] = temperature
        payload["options"] = options

        connect_timeout = 10
        read_timeout = int(timeout_sec)

        # ---------------- non-streaming ----------------
        def _non_stream() -> str:
            p = dict(payload)
            p["stream"] = False
            resp = requests.post(url, json=p, timeout=(connect_timeout, read_timeout))
            resp.raise_for_status()
            return resp.json()["message"]["content"]

        if not stream:
            return _non_stream()

        # ---------------- streaming with fallback ----------------
        out_parts: list[str] = []
        got_any_token = False
        start_time = time.time()

        try:
            with requests.post(url, json=payload, stream=True, timeout=(connect_timeout, read_timeout)) as resp:
                resp.raise_for_status()

                for line in resp.iter_lines():
                    now = time.time()

                    # If streaming produces nothing for too long, fallback
                    if not got_any_token and (now - start_time) > first_token_timeout_sec:
                        # stop streaming attempt; fallback to non-stream
                        return _non_stream() if not stream_print else (
                            print(_non_stream(), end="", flush=True) or ""
                        )

                    if not line:
                        continue

                    data = json.loads(line.decode("utf-8"))
                    if data.get("done") is True:
                        break

                    msg = data.get("message") or {}
                    token = msg.get("content", "")
                    if not token:
                        continue

                    got_any_token = True
                    if stream_print:
                        print(token, end="", flush=True)
                    else:
                        out_parts.append(token)

        except requests.exceptions.ReadTimeout:
            # also fallback on read timeout
            if stream_print:
                print(_non_stream(), end="", flush=True)
                return ""
            return _non_stream()

        if stream_print:
            return ""
        return "".join(out_parts)