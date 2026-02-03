#!/usr/bin/env python3
from __future__ import annotations

import fnmatch
import hashlib
import os
import subprocess
import uuid
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


def iter_files(root: str, pattern: str) -> List[str]:
    """
    Walk root and return files matching the given glob-like pattern
    (relative path match via fnmatch).
    """
    out: List[str] = []
    root = os.path.abspath(root)

    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in {".git", ".venv", "__pycache__"}]

        for fn in files:
            full = os.path.join(base, fn)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            if fnmatch.fnmatch(rel, pattern.replace(os.sep, "/")):
                out.append(full)

    return sorted(out)


def safe_text(path: str, max_bytes: int) -> str | None:
    """
    Cheap heuristic to skip binaries / huge files.
    """
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
    """
    Accept slightly older/looser configs by inferring src.type when missing.

    Supported:
      - type: git (needs url + dest)
      - type: dir (needs path)
    """
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

    if not out.get("name"):
        out["name"] = "unknown"

    if out["type"] == "git":
        if not out.get("url") or not out.get("dest"):
            raise RuntimeError(f"Git source missing url/dest: {src}")

    if out["type"] == "dir":
        if not out.get("path"):
            raise RuntimeError(f"Dir source missing path: {src}")

    if not out.get("glob"):
        out["glob"] = "**/*"

    return out


def main() -> None:
    cfg = load_cfg()

    # Basic config checks (keep v1 simple, but fail with clear errors)
    if "ollama_url" not in cfg:
        raise RuntimeError("Missing config key: ollama_url")
    if "models" not in cfg or not isinstance(cfg["models"], dict):
        raise RuntimeError("Missing config key: models")
    if "embed" not in cfg["models"]:
        raise RuntimeError("Missing config key: models.embed")
    if "vector_store" not in cfg or not isinstance(cfg["vector_store"], dict):
        raise RuntimeError("Missing config key: vector_store")
    if "url" not in cfg["vector_store"] or "collection" not in cfg["vector_store"]:
        raise RuntimeError("Missing config keys: vector_store.url / vector_store.collection")
    if "ingestion" not in cfg or not isinstance(cfg["ingestion"], dict):
        raise RuntimeError("Missing config key: ingestion")
    if "batch_size" not in cfg["ingestion"]:
        raise RuntimeError("Missing config key: ingestion.batch_size")
    if "limits" not in cfg or not isinstance(cfg["limits"], dict):
        raise RuntimeError("Missing config key: limits")
    if "max_file_bytes" not in cfg["limits"]:
        raise RuntimeError("Missing config key: limits.max_file_bytes")
    if "sources" not in cfg or not isinstance(cfg["sources"], list):
        raise RuntimeError("Missing config key: sources (must be a list)")

    ollama = OllamaClient(cfg["ollama_url"])
    store = QdrantStore(cfg["vector_store"]["url"], cfg["vector_store"]["collection"])

    embed_model = cfg["models"]["embed"]
    batch_size = int(cfg["ingestion"]["batch_size"])
    max_bytes = int(cfg["limits"]["max_file_bytes"])
    strategy = (cfg["ingestion"].get("chunk_strategy") or "semantic").strip()

    # Normalize sources (fixes your KeyError: 'type')
    sources = [normalize_source(s) for s in cfg.get("sources", [])]

    jobs: List[Dict[str, Any]] = []
    for src in sources:
        stype = src["type"]

        if stype == "git":
            git_sync(src["url"], src["dest"])
            root = src["dest"]
        else:
            root = src["path"]

        glob_pat = src.get("glob", "**/*")
        for fp in iter_files(root, glob_pat):
            jobs.append({"source": src["name"], "root": root, "path": fp})

    points: List[qm.PointStruct] = []
    collection_ready = False
    upserted = 0

    for j in jobs:
        text = safe_text(j["path"], max_bytes=max_bytes)
        if text is None:
            continue

        rel = os.path.relpath(j["path"], j["root"]).replace(os.sep, "/")

        # Semantic chunking for Puppet manifests when configured
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
