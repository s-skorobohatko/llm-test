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
        stream: bool = False,
        temperature: float | None = None,
        stop: list[str] | None = None,
        stream_print: bool = False,
        timeout_sec: int = 7200,            # total wall-clock max
        first_token_timeout_sec: int = 900, # max wait for first token
        idle_timeout_sec: int = 120,        # max seconds with no bytes (prevents “stuck”)
        **_ignored,
    ):
        url = f"{self.base_url}/api/chat"
        payload = {"model": model, "messages": messages, "stream": stream}

        # IMPORTANT: no num_predict here
        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        if stop:
            options["stop"] = stop
        if options:
            payload["options"] = options

        connect_timeout = 10

        # Key trick: make read timeout short so we can treat ReadTimeout as “idle”
        read_timeout = int(max(10, idle_timeout_sec))

        def _non_stream() -> str:
            p = dict(payload)
            p["stream"] = False
            resp = requests.post(url, json=p, timeout=(connect_timeout, int(timeout_sec)))
            resp.raise_for_status()
            return resp.json()["message"]["content"]

        if not stream:
            return _non_stream()

        out_parts: list[str] = []
        got_any = False
        start = time.time()
        stop_markers = stop or []

        try:
            with requests.post(url, json=payload, stream=True, timeout=(connect_timeout, read_timeout)) as resp:
                resp.raise_for_status()

                for line in resp.iter_lines():
                    now = time.time()

                    # Total wall clock guard (not token-based)
                    if (now - start) > timeout_sec:
                        break

                    if not got_any and (now - start) > first_token_timeout_sec:
                        # model never started — fall back once
                        text = _non_stream()
                        if stream_print:
                            print(text, end="", flush=True)
                        return text

                    if not line:
                        continue

                    data = json.loads(line.decode("utf-8"))
                    if data.get("done") is True:
                        break

                    token = (data.get("message") or {}).get("content", "")
                    if not token:
                        continue

                    got_any = True

                    # IMPORTANT: always buffer, even if printing
                    out_parts.append(token)
                    if stream_print:
                        print(token, end="", flush=True)

                    # ---- Client-side stop (the key fix) ----
                    if stop_markers:
                        cur = "".join(out_parts)
                        for m in stop_markers:
                            if m and m in cur:
                                trimmed = cur.split(m, 1)[0].rstrip()
                                if stream_print:
                                    print("\n", flush=True)
                                return trimmed

        except requests.exceptions.ReadTimeout:
            # No bytes within idle_timeout_sec → return partial output instead of hanging
            cur = "".join(out_parts).rstrip()
            if stream_print:
                print("\n", flush=True)
            if cur:
                # If marker was produced but not caught (rare), trim anyway
                for m in stop_markers:
                    if m and m in cur:
                        cur = cur.split(m, 1)[0].rstrip()
                        break
                return cur
            return _non_stream()

        final = "".join(out_parts).rstrip()
        # final safety trim
        for m in stop_markers:
            if m and m in final:
                final = final.split(m, 1)[0].rstrip()
                break
        return final