#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, List

import yaml
from qdrant_client.http import models as qm

from src.ollama_client import OllamaClient
from src.puppet_chunker import semantic_chunks_pp
from src.qdrant_store import QdrantStore


def load_cfg(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()


def uuid5(s: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, s))


def git_sync(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.isdir(dest) and os.path.isdir(os.path.join(dest, ".git")):
        subprocess.check_call(["git", "-C", dest, "pull", "--ff-only"])
    else:
        subprocess.check_call(["git", "clone", "--depth", "1", url, dest])


def iter_files(root: str, pattern: str | List[str]) -> List[str]:
    rootp = Path(root).resolve()
    patterns: List[str] = pattern[:] if isinstance(pattern, list) else [pattern]

    out: List[str] = []
    seen: set[str] = set()

    for pat in patterns:
        pats = [pat]
        if pat.startswith("**/"):
            pats.append(pat[3:])  # also try "*.md" for root files

        for one in pats:
            for p in rootp.glob(one):
                if p.is_file():
                    s = str(p)
                    if s not in seen:
                        out.append(s)
                        seen.add(s)

    return sorted(out)


def safe_text(path: str, max_bytes: int) -> str | None:
    try:
        st = os.stat(path)
        if st.st_size > max_bytes:
            return None
        with open(path, "rb") as f:
            raw = f.read()
        if b"\x00" in raw[:4096]:
            return None
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return None


def normalize_source(src: Dict[str, Any]) -> Dict[str, Any]:
    stype = (src.get("type") or "").strip()

    if not stype:
        if src.get("url") and src.get("dest"):
            stype = "git"
        elif src.get("path"):
            stype = "dir"

    if stype not in ("git", "dir"):
        raise RuntimeError(
            "Invalid source entry in config.yaml: each source must have "
            "type: git|dir (or provide keys to infer it).\n"
            f"Bad entry: {src}"
        )

    out = dict(src)
    out["type"] = stype
    out.setdefault("name", "unknown")
    out.setdefault("glob", "**/*")

    if out["type"] == "git" and (not out.get("url") or not out.get("dest")):
        raise RuntimeError(f"Git source missing url/dest: {src}")

    if out["type"] == "dir" and not out.get("path"):
        raise RuntimeError(f"Dir source missing path: {src}")

    return out


def cap_text_for_embedding(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[TRUNCATED FOR EMBEDDING]\n"


def chunk_markdown(text: str, *, chunk_size: int = 4000, overlap: int = 200) -> List[str]:
    """
    Simple, safe chunking for markdown to avoid embedding context overflow.
    Uses character windows (good enough for v1).
    """
    if chunk_size <= 0:
        return [text]
    if overlap < 0 or overlap >= chunk_size:
        overlap = 0

    out: List[str] = []
    step = chunk_size - overlap
    i = 0
    n = len(text)
    while i < n:
        part = text[i : i + chunk_size]
        if part.strip():
            out.append(part)
        i += step
    return out or [text]


def main() -> None:
    cfg = load_cfg()

    # minimal checks
    ollama_url = cfg.get("ollama_url")
    if not ollama_url:
        raise RuntimeError("Missing config key: ollama_url")

    models = cfg.get("models") or {}
    embed_model = models.get("embed")
    if not embed_model:
        raise RuntimeError("Missing config key: models.embed")

    vs = cfg.get("vector_store") or {}
    if not vs.get("url") or not vs.get("collection"):
        raise RuntimeError("Missing config keys: vector_store.url / vector_store.collection")

    ingestion = cfg.get("ingestion") or {}
    batch_size = int(ingestion.get("batch_size", 32))
    strategy = (ingestion.get("chunk_strategy") or "semantic").strip()

    limits = cfg.get("limits") or {}
    max_file_bytes = int(limits.get("max_file_bytes", 2_000_000))

    # LOWER this in config if you still see overflows
    max_embed_chars = int(limits.get("max_embed_chars", 6000))

    if "sources" not in cfg or not isinstance(cfg["sources"], list):
        raise RuntimeError("Missing config key: sources (must be a list)")
    sources = [normalize_source(s) for s in cfg.get("sources", [])]
    if not sources:
        raise RuntimeError("No sources enabled in config.yaml (sources list is empty).")

    ollama = OllamaClient(ollama_url)

    # preflight check embedding model exists locally (accept base name vs :latest)
    local_models = set(ollama.list_models())
    if embed_model not in local_models and f"{embed_model}:latest" not in local_models:
        raise RuntimeError(
            f"Embedding model '{embed_model}' not found in Ollama.\n"
            f"Local models: {sorted(local_models)}\n"
            f"Fix: run `ollama pull {embed_model}` (or use ':latest' tag in config)"
        )

    store = QdrantStore(vs["url"], vs["collection"])

    jobs: List[Dict[str, Any]] = []
    for src in sources:
        stype = src["type"]

        if stype == "git":
            git_sync(src["url"], src["dest"])
            root = src["dest"]
        else:
            root = src["path"]

        glob_pat = src.get("glob", "**/*")
        matched = iter_files(root, glob_pat)
        print(
            f"[ingest] source={src['name']} type={stype} root={root} glob={glob_pat} matched_files={len(matched)}",
            flush=True,
        )

        for fp in matched:
            jobs.append({"source": src["name"], "root": root, "path": fp})

    print(f"[ingest] total_jobs={len(jobs)}", flush=True)

    points: List[qm.PointStruct] = []
    collection_ready = False
    upserted = 0
    embed_errors = 0

    for j in jobs:
        text = safe_text(j["path"], max_bytes=max_file_bytes)
        if text is None:
            continue

        rel = os.path.relpath(j["path"], j["root"]).replace(os.sep, "/")

        # Build chunks based on file type
        if j["path"].endswith(".pp") and strategy == "semantic":
            chunks = semantic_chunks_pp(text, rel)
        elif j["path"].endswith(".md") or j["path"].endswith(".markdown"):
            md_parts = chunk_markdown(text, chunk_size=4000, overlap=200)
            chunks = [f"File: {rel}\nBlock: md_part\nPart: {i}\n---\n{p}" for i, p in enumerate(md_parts)]
        else:
            chunks = [f"File: {rel}\nBlock: raw\n---\n{text}"]

        for idx, ch in enumerate(chunks):
            ch2 = cap_text_for_embedding(ch, max_embed_chars)

            try:
                vec = ollama.embed(embed_model, ch2)
            except Exception as e:
                embed_errors += 1
                print(f"[ingest] WARN embed_failed file={rel} chunk={idx} err={e}", flush=True)
                continue

            if not collection_ready:
                store.ensure_collection(dim=len(vec))
                collection_ready = True

            pid = uuid5(f"{j['source']}|{j['path']}|{idx}|{sha256(ch2)}")
            payload = {
                "source": j["source"],
                "path": j["path"],
                "relpath": rel,
                "chunk_index": idx,
                "content": ch2,
            }
            points.append(qm.PointStruct(id=pid, vector=vec, payload=payload))

            if len(points) >= batch_size:
                store.upsert(points)
                upserted += len(points)
                points = []

    if points:
        store.upsert(points)
        upserted += len(points)

    print(f"[ingest] done upserted={upserted} embed_errors={embed_errors}", flush=True)


if __name__ == "__main__":
    main()
