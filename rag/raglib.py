import hashlib
import json
import os
import fnmatch
import subprocess
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


# Backward-compatible alias (some code uses sha256_text)
sha256_text = sha256_str


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
    Supports:
      - ** recursion
      - brace sets: *.{a,b,c}

    Always skips:
      - .git/
      - common junk dirs
    """
    patterns = _expand_brace_glob(glob_pattern)

    SKIP_DIRS = {".git", ".svn", ".hg", ".idea", ".vscode", "__pycache__", ".venv", "vendor", "node_modules"}

    for base, dirs, files in os.walk(root):
        # prune unwanted dirs in-place (important for performance)
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]

        for fn in files:
            # skip hidden files too
            if fn.startswith("."):
                continue

            full = os.path.join(base, fn)
            rel = os.path.relpath(full, root).replace(os.sep, "/")

            for pat in patterns:
                pat_posix = pat.replace(os.sep, "/")
                if fnmatch.fnmatch(rel, pat_posix):
                    yield full
                    break


def list_files_from_dir(root: str, glob_pattern: str) -> List[str]:
    """
    ingest.py expects this name.
    Returns sorted list of matching full paths.
    """
    return sorted(iter_files(root, glob_pattern))

def list_files_multi_glob(root: str, glob_patterns: List[str]) -> List[str]:
    """
    Return sorted, deduplicated list of files matching ANY of the glob patterns.
    Used by ingest.py.
    """
    seen = set()
    for pat in glob_patterns:
        for path in iter_files(root, pat):
            seen.add(path)
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
      - ingest.py may call chunk_text(..., overlap=200)
      - newer code may call chunk_text(..., chunk_overlap=200)
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

    while i < len(text):
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
# Ollama client (used by ask.py + ingest.py)
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
        prompt: Optional[str] = None,
        messages=None,
        system: Optional[str] = None,
        options: Optional[dict] = None,
        stream: bool = False,
        stream_print: bool = True,
        timeout_sec: int = 3600,
    ) -> str:
        """
        - messages -> /api/chat
        - prompt   -> /api/generate

        stream=True prints tokens live and returns full text.
        """
        options = options or {}

        # ---------------- /api/chat ----------------
        if messages is not None:
            url = f"{self.base_url}/api/chat"
            payload = {
                "model": model,
                "messages": messages,
                "stream": bool(stream),
                "options": options,
            }

            if not stream:
                r = requests.post(url, json=payload, timeout=timeout_sec)
                r.raise_for_status()
                return r.json().get("message", {}).get("content", "")

            r = requests.post(url, json=payload, stream=True, timeout=(10, timeout_sec))
            r.raise_for_status()

            out: List[str] = []
            for raw in r.iter_lines(chunk_size=1, delimiter=b"\n"):
                if not raw:
                    continue
                try:
                    data = json.loads(raw.decode("utf-8", errors="replace"))
                except Exception:
                    continue

                msg = data.get("message") or {}
                piece = msg.get("content") or ""
                if piece:
                    out.append(piece)
                    if stream_print:
                        print(piece, end="", flush=True)

                if data.get("done"):
                    break

            return "".join(out)

        # ---------------- /api/generate ----------------
        if prompt is None:
            raise TypeError("chat() requires prompt or messages")

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
            r = requests.post(url, json=payload, timeout=timeout_sec)
            r.raise_for_status()
            return r.json().get("response", "")

        r = requests.post(url, json=payload, stream=True, timeout=(10, timeout_sec))
        r.raise_for_status()

        out: List[str] = []
        for raw in r.iter_lines(chunk_size=1, delimiter=b"\n"):
            if not raw:
                continue
            try:
                data = json.loads(raw.decode("utf-8", errors="replace"))
            except Exception:
                continue

            piece = data.get("response") or ""
            if piece:
                out.append(piece)
                if stream_print:
                    print(piece, end="", flush=True)

            if data.get("done"):
                break

        return "".join(out)
