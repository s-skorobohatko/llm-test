import argparse
import os
import time
import yaml

from raglib import (
    OllamaClient,
    load_text_file,
    chunk_text,
    sha256_str,
    git_sync,
    list_files_from_dir,
    list_files_multi_glob,
)

from qdrant_store import QdrantStore


def ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str):
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
        # binary-ish if many nulls
        if b"\x00" in head:
            return False
        return True
    except Exception:
        return False


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
        # Your forge discover/sync is already implemented elsewhere in your repo.
        # Here we just keep compatibility: indexing happens from whatever exists in dest.
        dest = src["dest"]
        include_globs = src.get("include_globs") or []
        if include_globs:
            return list_files_multi_glob(dest, include_globs)
        # fallback
        return list_files_from_dir(dest, "**/*")

    raise ValueError(f"Unknown source type: {stype}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="./config.yaml")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    cfg = load_config(args.config)

    ollama = OllamaClient(cfg["ollama_url"])
    store = QdrantStore(cfg["qdrant_url"], cfg["qdrant_collection"])

    chunk_size = int(cfg.get("chunk_size", 1200))
    chunk_overlap = int(cfg.get("chunk_overlap", 200))

    sources = cfg.get("sources", [])
    if not sources:
        raise SystemExit("No sources in config")

    # Collect files in CONFIG ORDER
    file_jobs = []  # (source_name, root_hint, file_path)
    for src in sources:
        name = src["name"]
        try:
            files = collect_files_from_source(src)
            for fp in files:
                file_jobs.append((name, src, fp))
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

    # ingest loop
    errors = 0
    upserted = 0
    t0 = time.time()

    batch_points = []
    collection_ready = False

    for i, (source_name, src, path) in enumerate(file_jobs, start=1):
        try:
            log(f"[ingest] {i}/{len(file_jobs)} {path}")

            # skip non-text or too big
            if not safe_is_text_file(path):
                continue

            text = load_text_file(path)

            chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
            # embed + upsert each chunk
            for chunk_index, chunk in enumerate(chunks):
                # embed
                vec = ollama.embed(cfg["embed_model"], chunk)

                # ensure collection once we know dim
                if not collection_ready:
                    store.ensure_collection(dim=len(vec))
                    collection_ready = True

                # stable id per chunk
                chunk_hash = sha256_str(chunk)
                point_id = sha256_str(f"{source_name}|{path}|{chunk_index}|{chunk_hash}")

                payload = {
                    "source": source_name,
                    "path": path,
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

    # flush
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
і