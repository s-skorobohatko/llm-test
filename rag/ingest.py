import yaml

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


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    client = OllamaClient(cfg["ollama_url"])
    embed_model = cfg["embed_model"]

    chunk_size = int(cfg.get("chunk_size", 1200))
    chunk_overlap = int(cfg.get("chunk_overlap", 200))

    if cfg.get("vector_store") != "qdrant":
        raise SystemExit("config.yaml vector_store must be 'qdrant'")

    qstore = QdrantStore(cfg["qdrant_url"], cfg["qdrant_collection"])

    # Forge: discover/sync + files to index this run
    forge_index_files = []
    for src in cfg.get("sources", []):
        if src.get("type") != "forge_discover":
            continue

        name = src.get("name") or "forge:unknown"
        api_base = src.get("api_base", "https://forgeapi.puppet.com")
        dest_root = src["dest"]
        state_file = src.get("state_file") or (dest_root.rstrip("/") + "/.state/forge_state.json")

        include_globs = src.get(
            "include_globs",
            ["**/*.md", "**/*.pp", "**/*.epp", "metadata.json"],
        )

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

        for d in synced_dirs:
            for fp in list_files_multi_glob(d, include_globs):
                forge_index_files.append((name, fp))

        print(f"[forge] files to index this run: {len(forge_index_files)}")

    # Standard sources (dir/git)
    files = ingest_sources(cfg)
    files.extend(forge_index_files)

    print(f"Found {len(files)} files to index")

    # Ensure Qdrant collection exists (need dim)
    probe_vec = client.embed(embed_model, "probe")
    qstore.ensure_collection(dim=len(probe_vec))

    upserted = 0
    errors = 0

    batch = []
    BATCH_SIZE = 128

    for idx, (source_name, path) in enumerate(files, start=1):
        if idx == 1 or idx % 25 == 0:
            print(f"[ingest] {idx}/{len(files)} {path}")

        try:
            text = load_text_file(path)
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
            if not chunks:
                continue

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
                    upserted += len(batch)
                    batch = []

        except Exception as e:
            errors += 1
            print(f"[ERROR] {path}: {e}")

    if batch:
        qstore.upsert_points(batch)
        upserted += len(batch)

    print(f"Done. upserted_points={upserted} errors={errors}")


if __name__ == "__main__":
    main()
