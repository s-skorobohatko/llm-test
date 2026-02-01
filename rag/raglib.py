import os
import re
import glob
import hashlib
import subprocess
from typing import List, Tuple, Optional, Dict, Any

import requests


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def norm_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = "\n".join(line.rstrip() for line in s.split("\n"))
    return s.strip()


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
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

    if overlap > 0 and len(chunks) > 1:
        out = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = out[-1]
            tail = prev[-overlap:] if len(prev) > overlap else prev
            out.append((tail + "\n" + chunks[i]).strip())
        chunks = out

    return chunks


class OllamaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def embed(self, model: str, text: str) -> List[float]:
        url = f"{self.base_url}/api/embeddings"
        resp = requests.post(url, json={"model": model, "prompt": text}, timeout=180)
        resp.raise_for_status()
        return resp.json()["embedding"]

    def chat(
        self,
        model: str,
        system: str,
        user: str,
        options: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        timeout: int = 3600,
    ) -> str:
        import json

        url = f"{self.base_url}/api/chat"
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": stream,
        }
        if options:
            payload["options"] = options

        if not stream:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()["message"]["content"]

        out = []
        with requests.post(url, json=payload, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                obj = json.loads(line)
                msg = obj.get("message", {}).get("content", "")
                if msg:
                    print(msg, end="", flush=True)
                    out.append(msg)
                if obj.get("done") is True:
                    break
        print()
        return "".join(out)


def list_files_from_dir(path: str, pattern: str) -> List[str]:
    full = os.path.join(path, pattern)
    return [p for p in glob.glob(full, recursive=True) if os.path.isfile(p)]


def list_files_multi_glob(root: str, patterns: List[str]) -> List[str]:
    out: List[str] = []
    for pat in patterns:
        out.extend(list_files_from_dir(root, pat))
    seen = set()
    uniq: List[str] = []
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
        subprocess.check_call(["git", "clone", "--depth", "1", url, dest])


def load_text_file(path: str, max_bytes: int = 2_000_000) -> str:
    with open(path, "rb") as f:
        data = f.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"File too large (>{max_bytes} bytes): {path}")
    return data.decode("utf-8", errors="replace")


def ingest_sources(cfg: dict) -> List[Tuple[str, str]]:
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

        elif stype == "forge_discover":
            # handled in ingest.py
            continue

        else:
            raise ValueError(f"Unknown source type: {stype}")

    return files
