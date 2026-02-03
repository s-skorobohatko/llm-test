#!/usr/bin/env python3
from __future__ import annotations
import os, fnmatch, subprocess, uuid, hashlib
from typing import List, Dict, Any
import yaml
from qdrant_client.http import models as qm

from src.ollama_client import OllamaClient
from src.qdrant_store import QdrantStore
from src.puppet_chunker import semantic_chunks_pp


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


def iter_files(root: str, pattern: str) -> List[str]:
    out = []
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in {".git", ".venv", "__pycache__"}]
        for fn in files:
            rel = os.path.relpath(os.path.join(base, fn), root).replace(os.sep, "/")
            if fnmatch.fnmatch(rel, pattern.replace(os.sep, "/")):
                out.append(os.path.join(base, fn))
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


def main():
    cfg = load_cfg()
    ollama = OllamaClient(cfg["ollama_url"])
    store = QdrantStore(cfg["vector_store"]["url"], cfg["vector_store"]["collection"])

    embed_model = cfg["models"]["embed"]
    batch_size = int(cfg["ingestion"]["batch_size"])
    max_bytes = int(cfg["limits"]["max_file_bytes"])
    strategy = cfg["ingestion"]["chunk_strategy"]

    jobs: List[Dict[str, Any]] = []
    for src in cfg["sources"]:
        if src["type"] == "git":
            git_sync(src["url"], src["dest"])
            root = src["dest"]
        else:
            root = src["path"]

        for fp in iter_files(root, src.get("glob", "**/*")):
            jobs.append({"source": src["name"], "root": root, "path": fp})

    points: List[qm.PointStruct] = []
    collection_ready = False
    upserted = 0

    for j in jobs:
        text = safe_text(j["path"], max_bytes=max_bytes)
        if text is None:
            continue

        rel = os.path.relpath(j["path"], j["root"]).replace(os.sep, "/")

        if j["path"].endswith(".pp") and strategy == "semantic":
            chunks = semantic_chunks_pp(text, rel)
        else:
            chunks = [f"File: {rel}\nBlock: raw\n---\n{text}"]

        for idx, ch in enumerate(chunks):
            vec = ollama.embed(embed_model, ch)
            if not collection_ready:
                store.ensure_collection(dim=len(vec))
                collection_ready = True

            pid = uuid5(f"{j['source']}|{j['path']}|{idx}|{sha256(ch)}")
            payload = {
                "source": j["source"],
                "path": j["path"],
                "relpath": rel,
                "chunk_index": idx,
                "content": ch,
            }
            points.append(qm.PointStruct(id=pid, vector=vec, payload=payload))

            if len(points) >= batch_size:
                store.upsert(points)
                upserted += len(points)
                points = []

    if points:
        store.upsert(points)
        upserted += len(points)

    print(f"[ingest] done upserted={upserted}")


if __name__ == "__main__":
    main()
