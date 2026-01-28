import os
import re
import glob
import time
import sqlite3
import hashlib
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import requests


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def norm_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = "\n".join(line.rstrip() for line in s.split("\n"))
    return s.strip()


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """
    Chunker (by paragraphs) with overlap in characters.
    Good enough for markdown + code; keeps logical blocks together.
    """
    text = norm_text(text)
    if not text:
        return []

    parts = re.split(r"\n\s*\n", text)
    chunks: List[str] = []
    cur = ""

    def flush(buf: str):
        buf = buf.strip()
        if buf:
            chunks.append(buf)

    for p in parts:
        p = p.strip()
        if not p:
            continue

        candidate = (cur + "\n\n" + p).strip() if cur else p
        if len(candidate) <= chunk_size:
            cur = candidate
            continue

        if cur:
            flush(cur)
            cur = ""

        if len(p) > chunk_size:
            start = 0
            while start < len(p):
                end = min(start + chunk_size, len(p))
                chunks.append(p[start:end].strip())
                if end == len(p):
                    break
                start = max(end - overlap, 0)
        else:
            cur = p

    if cur:
        flush(cur)

    # Apply overlap between chunks
    if overlap > 0 and len(chunks) > 1:
        out = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = out[-1]
            tail = prev[-overlap:] if len(prev) > overlap else prev
            out.append((tail + "\n" + chunks[i]).strip())
        chunks = out

    return chunks


@dataclass
class DocChunk:
    source: str
    path: str
    chunk_index: int
    content: str
    content_hash: str


class OllamaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def embed(self, model: str, text: str) -> List[float]:
        # Ollama embeddings endpoint
        url = f"{self.base_url}/api/embeddings"
        resp = requests.post(url, json={"model": model, "prompt": text}, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["embedding"]

    def chat(
        self,
        model: str,
        system: str,
        user: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        url = f"{self.base_url}/api/chat"
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
        if options:
            payload["options"] = options

        resp = requests.post(url, json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]


def ensure_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            path TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            embedding BLOB NOT NULL,
            dim INTEGER NOT NULL,
            created_at INTEGER NOT NULL,
            UNIQUE(path, chunk_index, content_hash)
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);")
    return conn


def to_blob(vec: np.ndarray) -> bytes:
    vec = vec.astype(np.float32)
    return vec.tobytes()


def from_blob(blob: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32, count=dim)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def list_files_from_dir(path: str, pattern: str) -> List[str]:
    full = os.path.join(path, pattern)
    return [p for p in glob.glob(full, recursive=True) if os.path.isfile(p)]

def list_files_multi_glob(root: str, patterns: List[str]) -> List[str]:
    out = []
    for pat in patterns:
        out.extend(list_files_from_dir(root, pat))
    # uniq
    seen = set()
    uniq = []
    for p in out:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def git_sync(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.isdir(dest) and os.path.isdir(os.path.join(dest, ".git")):
        subprocess.check_call(["git", "-C", dest, "pull", "--ff-only"])
    else:
        subprocess.check_call(["git", "clone", url, dest])


def load_text_file(path: str, max_bytes: int = 2_000_000) -> str:
    # Avoid indexing huge files/binaries accidentally
    with open(path, "rb") as f:
        data = f.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"File too large (>{max_bytes} bytes): {path}")
    return data.decode("utf-8", errors="replace")


def ingest_sources(cfg: dict) -> List[Tuple[str, str]]:
    """
    Returns list of (source_name, file_path).
    Source naming is fully controlled by config.yaml via src['name'].
    """
    files: List[Tuple[str, str]] = []
    for src in cfg.get("sources", []):
        stype = src["type"]
        name = src.get("name") or "unknown"

        if stype == "git":
            url = src["url"]
            dest = src["dest"]
            git_sync(url, dest)
            pattern = src.get("glob", "**/*")
            for fp in list_files_from_dir(dest, pattern):
                files.append((name, fp))

        elif stype == "dir":
            d = src["path"]
            pattern = src.get("glob", "**/*")
            for fp in list_files_from_dir(d, pattern):
                files.append((name, fp))

        else:
            raise ValueError(f"Unknown source type: {stype}")

    return files
