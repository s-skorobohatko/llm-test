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
)


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    conn = ensure_db(cfg["db_path"])
    client = OllamaClient(cfg["ollama_url"])

    embed_model = cfg["embed_model"]
    chunk_size = int(cfg.get("chunk_size", 1200))
    chunk_overlap = int(cfg.get("chunk_overlap", 200))

    files = ingest_sources(cfg)
    print(f"Found {len(files)} files to index")

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
            print(f"Indexed: {path} ({len(chunks)} chunks)")

        except Exception as e:
            errors += 1
            print(f"[ERROR] {path}: {e}")

    print(f"Done. inserted={inserted} skipped={skipped} errors={errors}")
    conn.close()


if __name__ == "__main__":
    main()
