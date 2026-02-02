#!/usr/bin/env python3
import argparse
import os
import time
import yaml
from typing import Optional

from raglib import (
    OllamaClient,
    load_text_file,
    chunk_text,
    sha256_str,
    uuid5_str,
    git_sync,
    list_files_from_dir,
    list_files_multi_glob,
)

from qdrant_store import QdrantStore


def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"{ts()} {msg}", flush=True)


def safe_is_text_file(path: str, max_bytes: int = 2_000_000) -> bool:
    """
    Cheap heuristic to avoid binary / huge files.
    """
    try:
        st = os.stat(path)
        if st.st_size > max_bytes:
            return False
        with open(path, "rb") as f:
            head = f.read(4096)
        if b"\x00" in head:
            return False
        return True
    except Exception:
        return False


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def require_keys(cfg: dict, keys: list[str], where: str) -> None:
    missing = [k for k in keys if k not in cfg or cfg[k] in (None, "")]
    if missing:
        raise SystemExit(f"Missing config keys in {where}: {', '.join(missing)}")


def collect_files_from_source(src: dict) -> list[str]:
    stype = src["type"]

    if stype == "git":
        git_sync(src["url"], src["dest"], depth=int(src.get("depth", 1)))
        return list_files_from_dir(src["dest"], src["glob"])

    if stype == "dir":
        root = src["path"]
        globp = src.get("glob", "**/*")
        return list_files_from_dir(root, globp)

    if stype == "forge_discover":
        # indexes whatever already exists in dest
        dest = src["dest"]
        include_globs = src.get("include_globs") or []
        if include_globs:
            return list_files_multi_glob(dest, include_globs)
        return list_files_from_dir(dest, "**/*")

    raise ValueError(f"Unknown source type: {stype}")


def source_module_root(src: dict) -> str:
    """
    Determine a stable module_root for payload scoping.

    This should match what you'll pass to ask.py --scope.
    """
    stype = src.get("type")

    if stype == "dir":
        return (src.get("path") or "").rstrip("/")

    if stype == "git":
        return (src.get("dest") or "").rstrip("/")

    if stype == "forge_discover":
        # This is the dest root; scoping by module_root for forge is less useful,
        # but still consistent if you want it.
        return (src.get("dest") or "").rstrip("/")

    return ""


def compute_relpath(module_root: str, path: str) -> str:
    """
    Compute relpath if path is under module_root. Otherwise return empty.
    Normalize to forward slashes for stable printing/prompts.
    """
    try:
        mr = os.path.abspath(module_root)
        p = os.path.abspath(path)
        if mr and os.path.commonpath([mr, p]) == mr:
            return os.path.relpath(p, mr).replace(os.sep, "/")
    except Exception:
        pass
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="./config.yaml")
    ap.add_argument("--batch", type=int, default=64, help="Upsert batch size (points)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    require_keys(cfg, ["ollama_url", "embed_model", "qdrant_url", "qdrant_collection", "sources"], args.config)

    ollama = OllamaClient(cfg["ollama_url"])
    store = QdrantStore(cfg["qdrant_url"], cfg["qdrant_collection"])

    chunk_size = int(cfg.get("chunk_size", 1200))
    chunk_overlap = int(cfg.get("chunk_overlap", 200))

    sources = cfg.get("sources", [])
    if not sources:
        raise SystemExit("No sources in config")

    # Collect files in CONFIG ORDER
    file_jobs: list[tuple[str, dict, str, str]] = []  # (source_name, src, module_root, file_path)
    for src in sources:
        name = src["name"]
        try:
            module_root = source_module_root(src)
            files = collect_files_from_source(src)
            for fp in files:
                file_jobs.append((name, src, module_root, fp))
        except Exception as e:
            log(f"[ERROR] source={name} err={e}")

    log(f"[ingest] Found {len(file_jobs)} files to index (config order)")

    # counts before
    total_before = store.count()
    log(f"[qdrant] total points before: {total_before}")
    for src in sources:
        name = src["name"]
        try:
            c = store.count(must=[{"key": "source", "match": {"value": name}}])
            log(f"[qdrant] before source={name} points={c}")
        except Exception as e:
            log(f"[qdrant] before source={name} err={e}")

    errors = 0
    upserted = 0
    t0 = time.time()

    batch_points = []
    collection_ready = False

    for i, (source_name, src, module_root, path) in enumerate(file_jobs, start=1):
        try:
            log(f"[ingest] {i}/{len(file_jobs)} {path}")

            if not safe_is_text_file(path):
                continue

            text = load_text_file(path)
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)

            relpath = compute_relpath(module_root, path)

            for chunk_index, chunk in enumerate(chunks):
                vec = ollama.embed(cfg["embed_model"], chunk)

                if not collection_ready:
                    store.ensure_collection(dim=len(vec))
                    collection_ready = True

                chunk_hash = sha256_str(chunk)
                point_id = uuid5_str(f"{source_name}|{path}|{chunk_index}|{chunk_hash}")

                payload = {
                    "source": source_name,
                    "module_root": module_root,
                    "path": path,
                    "relpath": relpath,
                    "chunk_id": f"{os.path.basename(path)}#chunk{chunk_index}",
                    "chunk_index": chunk_index,
                    "content": chunk,
                }

                pt = store.make_point(id=point_id, vector=vec, payload=payload)
                batch_points.append(pt)

                if len(batch_points) >= args.batch:
                    store.upsert_points(batch_points)
                    upserted += len(batch_points)
                    batch_points = []

        except Exception as e:
            errors += 1
            log(f"[ERROR] source={source_name} path={path} err={e}")

    if batch_points:
        store.upsert_points(batch_points)
        upserted += len(batch_points)

    elapsed = time.time() - t0

    total_after = store.count()
    log(f"[qdrant] total points after: {total_after} (delta={total_after - total_before})")
    for src in sources:
        name = src["name"]
        try:
            c = store.count(must=[{"key": "source", "match": {"value": name}}])
            log(f"[qdrant] after source={name} points={c}")
        except Exception as e:
            log(f"[qdrant] after source={name} err={e}")

    log(f"[ingest] Done. upserted_points={upserted} errors={errors} elapsed_sec={elapsed:.1f}")


if __name__ == "__main__":
    main()
