import time
import yaml
import numpy as np

from raglib import (
    ensure_db,
    OllamaClient,
    ingest_sources,
    load_text_file,
    chunk_text,
    sha256_str,
    to_blob,
    list_files_multi_glob,
)
from forge import discover_and_sync_forge


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    conn = ensure_db(cfg["db_path"])
    client = OllamaClient(cfg["ollama_url"])

    embed_model = cfg["embed_model"]
    chunk_size = int(cfg.get("chunk_size", 1200))
    chunk_overlap = int(cfg.get("chunk_overlap", 200))

    # 1) Sync Forge-discovered modules (if configured)
    forge_index_files = []
    for src in cfg.get("sources", []):
        if src.get("type") != "forge_discover":
            continue

        name = src.get("name") or "forge:unknown"
        api_base = src.get("api_base", "https://forgeapi.puppet.com")
        dest_root = src["dest"]
        include_globs = src.get("include_globs", ["**/*.md", "**/*.pp", "**/*.epp", "metadata.json"])

        limit_per_page = int(src.get("limit_per_page", 50))
        max_modules = int(src.get("max_modules", 200))
        request_delay_sec = float(src.get("request_delay_sec", 0.2))
        only_with_repo = bool(src.get("only_with_repo", True))
        only_owner = src.get("only_owner")

        print(f"[forge] discovering/syncing into {dest_root} ...")
        synced_dirs = discover_and_sync_forge(
            conn=conn,
            api_base=api_base,
            dest_root=dest_root,
            include_globs=include_globs,
            limit_per_page=limit_per_page,
            max_modules=max_modules,
            request_delay_sec=request_delay_sec,
            only_with_repo=only_with_repo,
            only_owner=only_owner,
        )

        # Convert synced dirs to file list for indexing, tagged with the forge source name
        for d in synced_dirs:
            for fp in list_files_multi_glob(d, include_globs):
                forge_index_files.append((name, fp))

        print(f"[forge] ready to index files: {len(forge_index_files)}")

    # 2) Collect files from standard sources (git/dir)
    #    Note: forge_discover sources are handled above and should not be included here.
    standard_sources_cfg = dict(cfg)
    standard_sources_cfg["sources"] = [s for s in cfg.get("sources", []) if s.get("type") != "forge_discover"]
    files = ingest_sources(standard_sources_cfg)

    # Add forge-derived files
    files.extend(forge_index_files)

    print(f"Found {len(files)} files to index (including Forge)")

    inserted = 0
    skipped = 0
    errors = 0

    for source_name, path in files:
        try:
            text = load_text_file(path)
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
            if not chunks:
                continue

            for i, ch in enumerate(chunks):
                h = sha256_str(ch)

                exists = conn.execute(
                    "SELECT 1 FROM chunks WHERE path=? AND chunk_index=? AND content_hash=?",
                    (path, i, h),
                ).fetchone()
                if exists:
                    skipped += 1
                    continue

                emb = client.embed(embed_model, ch)
                vec = np.array(emb, dtype=np.float32)

                conn.execute(
                    """
                    INSERT INTO chunks (source, path, chunk_index, content, content_hash, embedding, dim, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        source_name,
                        path,
                        i,
                        ch,
                        h,
                        to_blob(vec),
                        int(vec.shape[0]),
                        int(time.time()),
                    ),
                )
                inserted += 1

            conn.commit()

        except Exception as e:
            errors += 1
            print(f"[ERROR] {path}: {e}")

    print(f"Done. inserted={inserted} skipped={skipped} errors={errors}")
    conn.close()


if __name__ == "__main__":
    main()
