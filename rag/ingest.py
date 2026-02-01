import os
import time
import yaml
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

from raglib import (
    OllamaClient,
    ingest_sources,
    load_text_file,
    chunk_text,
    sha256_str,
    list_files_multi_glob,
)
from forge import discover_and_sync_forge
from qdrant_store import QdrantStore


def log_line(fp, msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} {msg}"
    print(line)
    if fp:
        fp.write(line + "\n")
        fp.flush()


def qdrant_must_source(source: str) -> List[Dict[str, Any]]:
    # exact match on payload "source"
    return [{"key": "source", "match": {"value": source}}]


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # logging
    log_path = cfg.get("ingest_log_file", "./ingest.log")
    verbose = bool(cfg.get("ingest_verbose", False))
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    log_fp = open(log_path, "a", encoding="utf-8")

    try:
        client = OllamaClient(cfg["ollama_url"])
        embed_model = cfg["embed_model"]

        chunk_size = int(cfg.get("chunk_size", 1200))
        chunk_overlap = int(cfg.get("chunk_overlap", 200))

        if cfg.get("vector_store") != "qdrant":
            raise SystemExit("config.yaml vector_store must be 'qdrant'")

        qstore = QdrantStore(cfg["qdrant_url"], cfg["qdrant_collection"])

        # Ensure Qdrant collection exists (need dim)
        probe_vec = client.embed(embed_model, "probe")
        qstore.ensure_collection(dim=len(probe_vec))

        # Build file list: standard sources + forge synced dirs this run
        forge_index_files: List[Tuple[str, str]] = []
        for src in cfg.get("sources", []):
            if src.get("type") != "forge_discover":
                continue

            name = src.get("name") or "forge:unknown"
            api_base = src.get("api_base", "https://forgeapi.puppet.com")
            dest_root = src["dest"]
            state_file = src.get("state_file") or (dest_root.rstrip("/") + "/.state/forge_state.json")
            include_globs = src.get("include_globs", ["**/*.md", "**/*.pp", "**/*.epp", "metadata.json"])

            log_line(log_fp, f"[forge] discover/sync start name={name} dest={dest_root}")
            synced_dirs = discover_and_sync_forge(
                api_base=api_base,
                dest_root=dest_root,
                include_globs=include_globs,
                state_file=state_file,
                limit_per_page=int(src.get("limit_per_page", 50)),
                max_modules_seen=int(src.get("max_modules_seen", 500)),
                max_modules_synced=int(src.get("max_modules_synced", 50)),
                request_delay_sec=float(src.get("request_delay_sec", 0.5)),
                download_delay_sec=float(src.get("download_delay_sec", 1.0)),
                only_with_repo=bool(src.get("only_with_repo", True)),
                only_owner=src.get("only_owner"),
                allowlist=src.get("allowlist"),
                denylist=src.get("denylist"),
                index_unchanged=bool(src.get("index_unchanged", False)),
                check_interval_hours=int(src.get("check_interval_hours", 24)),
            )

            added_files = 0
            for d in synced_dirs:
                for fp in list_files_multi_glob(d, include_globs):
                    forge_index_files.append((name, fp))
                    added_files += 1
            log_line(log_fp, f"[forge] files to index this run: {added_files}")

        files: List[Tuple[str, str]] = []
        files.extend(ingest_sources(cfg))
        files.extend(forge_index_files)

        log_line(log_fp, f"[ingest] Found {len(files)} files to index")

        # Group files per source for per-source deltas
        files_by_source: Dict[str, List[str]] = defaultdict(list)
        for source_name, path in files:
            files_by_source[source_name].append(path)

        # Count points before per source
        before_counts: Dict[str, int] = {}
        total_before = qstore.count()
        log_line(log_fp, f"[qdrant] total points before: {total_before}")

        for source_name in sorted(files_by_source.keys()):
            try:
                c = qstore.count(must=qdrant_must_source(source_name))
            except Exception as e:
                c = -1
                log_line(log_fp, f"[qdrant] WARN count failed for source={source_name}: {e}")
            before_counts[source_name] = c
            log_line(log_fp, f"[qdrant] before source={source_name} points={c}")

        upserted_points = 0
        errors = 0
        per_source_files_ok = defaultdict(int)
        per_source_files_err = defaultdict(int)
        per_source_chunks = defaultdict(int)

        batch: List[Dict[str, Any]] = []
        BATCH_SIZE = int(cfg.get("ingest_batch_size", 128))
        t0 = time.time()

        for idx, (source_name, path) in enumerate(files, start=1):
            # progress print every 25 files
            if idx == 1 or idx % 25 == 0:
                log_line(log_fp, f"[ingest] {idx}/{len(files)} {path}")

            try:
                text = load_text_file(path)
                chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
                if not chunks:
                    continue

                per_source_files_ok[source_name] += 1
                per_source_chunks[source_name] += len(chunks)

                if verbose:
                    log_line(log_fp, f"[file] source={source_name} path={path} chunks={len(chunks)}")

                for i, ch in enumerate(chunks):
                    h = sha256_str(ch)
                    vec = client.embed(embed_model, ch)
                    batch.append(
                        qstore.make_point(
                            source=source_name,
                            path=path,
                            chunk_index=i,
                            content=ch,
                            content_hash=h,
                            vector=vec,
                        )
                    )

                    if len(batch) >= BATCH_SIZE:
                        qstore.upsert_points(batch)
                        upserted_points += len(batch)
                        log_line(log_fp, f"[qdrant] upsert batch size={len(batch)} total_upserted={upserted_points}")
                        batch = []

            except Exception as e:
                errors += 1
                per_source_files_err[source_name] += 1
                log_line(log_fp, f"[ERROR] source={source_name} path={path} err={e}")

        if batch:
            qstore.upsert_points(batch)
            upserted_points += len(batch)
            log_line(log_fp, f"[qdrant] upsert final batch size={len(batch)} total_upserted={upserted_points}")

        # Count points after per source (and compute delta)
        after_counts: Dict[str, int] = {}
        total_after = qstore.count()
        log_line(log_fp, f"[qdrant] total points after: {total_after} (delta={total_after - total_before})")

        for source_name in sorted(files_by_source.keys()):
            try:
                c = qstore.count(must=qdrant_must_source(source_name))
            except Exception as e:
                c = -1
                log_line(log_fp, f"[qdrant] WARN count failed for source={source_name}: {e}")
            after_counts[source_name] = c
            before = before_counts.get(source_name, -1)
            delta = (c - before) if (c >= 0 and before >= 0) else "n/a"
            log_line(log_fp, f"[qdrant] after source={source_name} points={c} delta={delta}")

        dt = time.time() - t0
        log_line(log_fp, f"[ingest] Done. upserted_points={upserted_points} errors={errors} elapsed_sec={dt:.1f}")

        # Per-source summary
        log_line(log_fp, "[summary] per-source files/chunks:")
        for source_name in sorted(files_by_source.keys()):
            ok = per_source_files_ok.get(source_name, 0)
            er = per_source_files_err.get(source_name, 0)
            ch = per_source_chunks.get(source_name, 0)
            log_line(log_fp, f"[summary] source={source_name} files_ok={ok} files_err={er} chunks={ch}")

        log_line(log_fp, f"[log] written to {log_path}")

    finally:
        try:
            log_fp.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
