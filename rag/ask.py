import argparse
import yaml
import numpy as np

from raglib import ensure_db, OllamaClient, from_blob, cosine_sim


SYSTEM_RAG = """You are a senior infrastructure engineer and Puppet specialist (Puppet 8+).

You will receive CONTEXT snippets retrieved from documentation and repositories.
Rules:
- Use CONTEXT as primary evidence; cite snippets like [1], [2].
- Prefer sources named "internal:*" over "vendor:*" when they conflict.
- If context is insufficient, say what is missing and make minimal assumptions.
- Output Puppet code that is idempotent and Hiera-friendly.
"""


def build_prompt(question: str, contexts: list[dict]) -> str:
    blocks = []
    for idx, c in enumerate(contexts, start=1):
        blocks.append(
            f"[{idx}] SOURCE: {c['source']}\nPATH: {c['path']}#chunk{c['chunk_index']}\n---\n{c['content']}\n"
        )

    ctx = "\n\n".join(blocks) if blocks else "(no context retrieved)"

    return f"""CONTEXT (retrieved):
{ctx}

TASK:
{question}

Instructions:
- Cite which context snippets you used, like [1], [2].
- If you generate a module, output a directory tree + full files with paths.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="+", help="Question to ask")
    ap.add_argument("--topk", type=int, default=None)
    ap.add_argument("--minsim", type=float, default=None)
    args = ap.parse_args()

    question = " ".join(args.question)

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    topk = args.topk if args.topk is not None else int(cfg.get("top_k", 6))
    minsim = args.minsim if args.minsim is not None else float(cfg.get("min_sim", 0.18))

    conn = ensure_db(cfg["db_path"])
    client = OllamaClient(cfg["ollama_url"])

    q_vec = np.array(client.embed(cfg["embed_model"], question), dtype=np.float32)

    rows = conn.execute(
        "SELECT source, path, chunk_index, content, embedding, dim FROM chunks"
    ).fetchall()

    scored = []
    for source, path, chunk_index, content, blob, dim in rows:
        vec = from_blob(blob, dim)
        sim = cosine_sim(q_vec, vec)
        if sim >= minsim:
            scored.append((sim, source, path, chunk_index, content))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:topk]

    contexts = [
        {
            "sim": sim,
            "source": source,
            "path": path,
            "chunk_index": chunk_index,
            "content": content,
        }
        for (sim, source, path, chunk_index, content) in top
    ]

    prompt = build_prompt(question, contexts)

    answer = client.chat(
        model=cfg["chat_model"],
        system=SYSTEM_RAG,
        user=prompt,
        options=None,  # uses your model defaults
    )

    if contexts:
        print("\n=== Retrieved context ===")
        for i, c in enumerate(contexts, start=1):
            print(f"[{i}] sim={c['sim']:.3f} {c['source']} {c['path']}#chunk{c['chunk_index']}")
        print("\n=== Answer ===\n")
    else:
        print("\n=== Retrieved context ===\n(none)\n\n=== Answer ===\n")

    print(answer)


if __name__ == "__main__":
    main()
