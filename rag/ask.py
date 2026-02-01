import argparse
import yaml

from raglib import OllamaClient
from qdrant_store import QdrantStore


SYSTEM_RAG = """You are a senior Puppet engineer (Puppet 8+). You write and refactor production-grade Puppet code and modules.

Rules:
- Prefer data-driven design: class parameters + Hiera (lookup()).
- Remove legacy patterns: params.pp, validate_* functions, unnecessary inheritance.
- Ensure idempotency and correct resource ordering.
- Use modern Puppet types and EPP templates.
- Cite retrieved snippets like [1], [2], etc when you rely on them.
- Keep answers practical and production-focused.

Output formatting:
- Put code in fenced blocks.
"""


def build_prompt(question: str, contexts: list) -> str:
    if not contexts:
        return question

    lines = ["You have these retrieved snippets. Use them when relevant:\n"]
    for idx, (score, payload) in enumerate(contexts, start=1):
        path = payload.get("path", "")
        source = payload.get("source", "")
        chunk_index = payload.get("chunk_index", 0)
        content = payload.get("content", "")
        lines.append(f"[{idx}] score={score:.3f} {source} {path}#chunk{chunk_index}\n{content}\n")

    lines.append("User question:\n" + question)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--topk", type=int, default=None)
    ap.add_argument("--minsim", type=float, default=None)
    args = ap.parse_args()

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    client = OllamaClient(cfg["ollama_url"])
    embed_model = cfg["embed_model"]
    chat_model = cfg["chat_model"]

    top_k = int(args.topk if args.topk is not None else cfg.get("top_k", 6))
    min_sim = float(args.minsim if args.minsim is not None else cfg.get("min_sim", 0.18))

    qstore = QdrantStore(cfg["qdrant_url"], cfg["qdrant_collection"])

    q_vec = client.embed(embed_model, args.question)
    contexts = qstore.search(query_vector=q_vec, top_k=top_k, min_sim=min_sim)

    print("=== Retrieved context ===")
    if not contexts:
        print("(none)")
    else:
        for i, (score, payload) in enumerate(contexts, start=1):
            print(f"[{i}] sim={score:.3f} {payload.get('source')} {payload.get('path')}#chunk{payload.get('chunk_index')}")

    print("\n=== Answer ===")
    prompt = build_prompt(args.question, contexts)

    # stream=True prints tokens as they come
    client.chat(
        model=chat_model,
        system=SYSTEM_RAG,
        user=prompt,
        options={"num_predict": 800},
        stream=True,
        timeout=3600,
    )


if __name__ == "__main__":
    main()
