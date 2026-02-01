#!/usr/bin/env python3
"""
ingest.py — ordered ingestion (CONFIG ORDER), with logging.

What it does:
- Processes sources in the exact order they appear in config.yaml:
  - dir sources (internal) first (if listed first)
  - git sources (vendor) next
  - forge_discover last (if listed last)
- Builds chunks, embeds with Ollama (embed_model), upserts into Qdrant.
- Writes a clear ingest log (default: ./ingest.log).
- Logs Qdrant point counts before/after per source (best “what was added” signal).

Notes:
- Python glob does NOT support brace expansion like **/*.{pp,md}. Use multiple sources/globs.
- If GitHub/Forge sync fails, ingestion continues for other sources and still indexes what it can.
"""

import os
import time
import yaml
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

from raglib import (
    OllamaClient,
    load_text_file,
    chunk_text,
    sha256_str,
    git_sync,
    list_files_from_dir,
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


def qdrant_must_source_exact(source: str) -> List[Dict[str, Any]]:
    # exact match on payload "source"
    return [{"key": "source", "match": {"value": source}}]


def safe_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


def safe_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def collect_files_in_config_order(cfg: Dict[str, Any], log_fp) -> List[Tuple[str, str]]:
    """
    Returns list of (source_name, file_path) in strict config order.
    For forge_discover, we sync first then list files only for dirs returned this run.
    """
    files: List[Tuple[str, str]] = []

    for src in cfg.get("sources", []):
        stype = src.get("type")
        name = src.get("name") or f"{stype}:unknown"

        if stype == "dir":
            root = src["path"]
            pattern = src.get("glob", "**/*")
            try:
                dir_files = list_files_from_dir(root, pattern)
                for fp in dir_files:
                    files.append((name, fp))
                log_line(log_fp, f"[dir] add name={name} path={root} pattern={pattern} files={len(dir_files)}")
            except Exception as e:
                log_line(log_fp, f"[dir] ERROR name={name} path={root}: {e}")

        elif stype == "git":
            url = src["url"]
            dest = src["dest"]
            pattern = src.get("glob", "**/*")
            try:
                log_line(log_fp, f"[git] sync start name={name} url={url} dest={dest}")
                git_sync(url, dest)
                git_files = list_files_from_dir(dest, pattern)
                for fp in git_files:
                    files.append((name, fp))
                log_line(log_fp, f"[git] sync done name={name} files={len(git_files)}")
            except Exception as e:
                log_line(log_fp, f"[git] ERROR name={name} url={url}: {e}")

        elif stype == "forge_discover":
            api_base = src.get("api_base", "https://forgeapi.puppet.com")
            dest_root = src["dest"]
            state_file = src.get("state_file") or (dest_root.rstrip("/") + "/.state/forge_state.json")
            include_globs = src.get("include_globs", ["**/*.md", "**/*.pp", "**/*.epp", "metadata.json"])

            limit_per_page = safe_int(src.get("limit_per_page", 50), 50)
            max_modules_seen = safe_int(src.get("max_modules_seen", 500), 500)
            max_modules_synced = safe_int(src.get("max_modules_synced", 50), 50)
            request_delay_sec = safe_float(src.get("request_delay_sec", 0.5), 0.5)
            download_delay_sec = safe_float(src.get("download_delay_sec", 1.0), 1.0)

            only_with_repo = bool(src.get("only_with_repo", True))
            only_owner = src.get("only_owner")
            allowlist = src.get("allowlist")
            denylist = src.get("denylist")

            index_unchanged = bool(src.get("index_unchanged", False))
            check_interval_hours = safe_int(src.get("check_interval_hours", 24), 24)

            log_line(log_fp, f"[forge] discover/sync start name={name} dest={dest_root}")
            try:
                synced_dirs = discover_and_sync_forge(
                    api_base=api_base,
                    dest_root=dest_root,
                    include_globs=include_globs,
                    state_file=state_file,
                    limit_per_page=limit_per_page,
                    max_modules_seen=max_modules_seen,
                    max_modules_synced=max_modules_synced,
                    request_delay_sec=request_delay_sec,
                    download_delay_sec=download_delay_sec,
                    only_with_repo=only_with_repo,
                    only_owner=only_owner,
                    allowlist=allowlist,
                    denylist=denylist,
                    index_unchanged=index_unchanged,
                    check_interval_hours=check_interval_hours,
                )
            except Exception as e:
                log_line(log_fp, f"[forge] ERROR name={name}: {e}")
                synced_dirs = []

            added = 0
            for d in synced_dirs:
                try:
                    forge_files = list_files_multi_glob(d, include_globs)
                    for fp in forge_files:
                        files.append((name, fp))
                        added += 1
                except Exception as e:
                    log_line(log_fp, f"[forge] WARN list files failed dir={d}: {e}")

            log_line(log_fp, f"[forge] files to index this run: {added}")

        else:
            raise ValueError(f"Unknown source type: {stype}")

    return files


def main() -> None:
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    log_path = cfg.get("ingest_log_file", "./ingest.log")
    verbose = bool(cfg.get("ingest_verbose", False))
    batch_size = safe_int(cfg.get("ingest_batch_size", 128), 128)

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    log_fp = open(log_path, "a", encoding="utf-8")

    try:
        client = OllamaClient(cfg["ollama_url"])
        embed_model = cfg["embed_model"]

        chunk_size = safe_int(cfg.get("chunk_size", 1200), 1200)
        chunk_overlap = safe_int(cfg.get("chunk_overlap", 200), 200)

        if cfg.get("vector_store") != "qdrant":
            raise SystemExit("config.yaml vector_store must be 'qdrant'")

        qstore = QdrantStore(cfg["qdrant_url"], cfg["qdrant_collection"])

        # Ensure Qdrant collection exists (need embedding dim)
        probe_vec = client.embed(embed_model, "probe")
        qstore.ensure_collection(dim=len(probe_vec))

        # 1) Collect files in strict config order (and sync forge only when we reach it)
        files = collect_files_in_config_order(cfg, log_fp)
        log_line(log_fp, f"[ingest] Found {len(files)} files to index (config order)")

        if not files:
            # Still print qdrant counts before/after for visibility
            total_before = qstore.count()
            log_line(log_fp, f"[qdrant] total points before: {total_before}")
            total_after = qstore.count()
            log_line(log_fp, f"[qdrant] total points after: {total_after} (delta={total_after - total_before})")
            log_line(log_fp, "[ingest] Done. upserted_points=0 errors=0 elapsed_sec=0.0")
            log_line(log_fp, f"[log] written to {log_path}")
            return

        # 2) Group per source for before/after deltas
        files_by_source: Dict[str, List[str]] = defaultdict(list)
        for source_name, path in files:
            files_by_source[source_name].append(path)

        # 3) Counts before
        before_counts: Dict[str, int] = {}
        total_before = qstore.count()
        log_line(log_fp, f"[qdrant] total points before: {total_before}")

        for source_name in sorted(files_by_source.keys()):
            try:
                c = qstore.count(must=qdrant_must_source_exact(source_name))
            except Exception as e:
                c = -1
                log_line(log_fp, f"[qdrant] WARN count failed source={source_name}: {e}")
            before_counts[source_name] = c
            log_line(log_fp, f"[qdrant] before source={source_name} points={c}")

        # 4) Embed + upsert
        upserted_points = 0
        errors = 0

        per_source_files_ok = defaultdict(int)
        per_source_files_err = defaultdict(int)
        per_source_chunks = defaultdict(int)

        batch: List[Dict[str, Any]] = []
        t0 = time.time()

        for idx, (source_name, path) in enumerate(files, start=1):
            # Don’t spam logs
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

                    if len(batch) >= batch_size:
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

        # 5) Counts after
        total_after = qstore.count()
        log_line(log_fp, f"[qdrant] total points after: {total_after} (delta={total_after - total_before})")

        for source_name in sorted(files_by_source.keys()):
            try:
                c = qstore.count(must=qdrant_must_source_exact(source_name))
            except Exception as e:
                c = -1
                log_line(log_fp, f"[qdrant] WARN count failed source={source_name}: {e}")
            before = before_counts.get(source_name, -1)
            delta = (c - before) if (c >= 0 and before >= 0) else "n/a"
            log_line(log_fp, f"[qdrant] after source={source_name} points={c} delta={delta}")

        dt = time.time() - t0
        log_line(log_fp, f"[ingest] Done. upserted_points={upserted_points} errors={errors} elapsed_sec={dt:.1f}")

        # 6) Per-source summary (files/chunks)
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
