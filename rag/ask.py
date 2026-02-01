#!/usr/bin/env python3
import argparse
import sys
import time
import yaml

from raglib import OllamaClient
from qdrant_store import QdrantStore


def format_context(items, title: str, limit: int | None = None) -> str:
    blocks = []
    for i, (score, payload) in enumerate(items, start=1):
        if limit is not None and i > limit:
            break
        content = payload.get("content", "")
        src = payload.get("source", "")
        path = payload.get("path", "")
        chunk_idx = payload.get("chunk_index", "")
        blocks.append(f"[{i}] source={src} path={path} chunk={chunk_idx} sim={score:.3f}\n{content}")
    joined = "\n\n---\n\n".join(blocks).strip()
    return f"{title}\n{joined}\n"


def must_source(value: str):
    return {"key": "source", "match": {"value": value}}


def must_path_contains(text: str):
    return {"key": "path", "match": {"text": text}}


def main():
    ap = argparse.ArgumentParser(description="Ask with RAG (Qdrant + Ollama), module-grounded + reference-guided.")
    ap.add_argument("question", nargs="?", help="Question to ask (or use stdin if omitted)")
    ap.add_argument("--config", default="config.yaml", help="Path to config.yaml")

    ap.add_argument("--topk", type=int, default=None, help="Top-k for MODULE retrieval (scoped)")
    ap.add_argument("--minsim", type=float, default=None, help="Minimum similarity threshold")

    # Grounding: restrict edits to a real module
    ap.add_argument("--scope", default=None, help="Module root path; retrieve ONLY chunks under this path")

    # Reference guidance sources (repeatable)
    ap.add_argument(
        "--ref_source",
        action="append",
        default=[],
        help="Reference source name (repeatable). Example: vendor:puppetlabs:best-practices",
    )
    ap.add_argument("--ref_topk", type=int, default=12, help="Top-k for REFERENCE retrieval (per run)")

    ap.add_argument("--show_context", type=int, default=30, help="How many retrieved items to print per section")
    args = ap.parse_args()

    # Read question
    if args.question is None:
        q = sys.stdin.read().strip()
        if not q:
            print("No question provided (arg or stdin).", file=sys.stderr)
            sys.exit(2)
        question = q
    else:
        question = args.question.strip()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ollama_url = cfg["ollama_url"]
    embed_model = cfg["embed_model"]
    chat_model = cfg["chat_model"]

    top_k_module = args.topk if args.topk is not None else int(cfg.get("top_k", 6))
    min_sim = args.minsim if args.minsim is not None else float(cfg.get("min_sim", 0.18))

    if cfg.get("vector_store") != "qdrant":
        raise SystemExit("config.yaml vector_store must be 'qdrant'")

    qdrant_url = cfg["qdrant_url"]
    collection = cfg["qdrant_collection"]

    client = OllamaClient(ollama_url)
    store = QdrantStore(qdrant_url, collection)

    t0 = time.time()

    # 1) Embed query
    q_vec = client.embed(embed_model, question)

    # 2) MODULE retrieval (ground truth for what exists)
    if not args.scope:
        print("ERROR: --scope is required for refactor runs (to prevent hallucinated files).", file=sys.stderr)
        print("If you want pure advisory Q&A, run without refactor prompt or add --ref_source only.", file=sys.stderr)
        sys.exit(2)

    module_hits = store.search(
        query_vector=q_vec,
        top_k=max(top_k_module, 60),   # when scoped, it’s safe to fetch more
        min_sim=min_sim if min_sim <= 0.10 else 0.06,  # keep it permissive for scoped retrieval
        must=[must_path_contains(args.scope)],
    )

    # 3) REFERENCE retrieval (Puppet knowledge you trust)
    # If user didn't provide --ref_source, use safe defaults from config or a good baseline.
    ref_sources = args.ref_source[:]  # copy
    if not ref_sources:
        # Default: best practices + (optional) a curated forge slice if you want later
        ref_sources = ["vendor:puppetlabs:best-practices"]

    ref_hits_all = []
    for src in ref_sources:
        ref_hits_all.extend(
            store.search(
                query_vector=q_vec,
                top_k=max(4, args.ref_topk // max(1, len(ref_sources))),
                min_sim=0.12,  # keep reference context higher precision
                must=[must_source(src)],
            )
        )

    # sort combined refs by similarity desc and keep top N
    ref_hits_all.sort(key=lambda x: x[0], reverse=True)
    ref_hits = ref_hits_all[: args.ref_topk]

    # 4) Print retrieval diagnostics
    print("=== Retrieved context (MODULE) ===")
    for i, (score, payload) in enumerate(module_hits[: args.show_context], start=1):
        print(f"[{i}] sim={score:.3f} {payload.get('source','?')} {payload.get('path','?')}#chunk{payload.get('chunk_index','?')}")

    print("\n=== Retrieved context (REFERENCE) ===")
    for i, (score, payload) in enumerate(ref_hits[: args.show_context], start=1):
        print(f"[{i}] sim={score:.3f} {payload.get('source','?')} {payload.get('path','?')}#chunk{payload.get('chunk_index','?')}")

    # 5) Build prompt with strict separation + “RAG-first” instruction
    module_ctx = format_context(module_hits, "MODULE_CONTEXT (authoritative; only these files exist):", None)
    ref_ctx = format_context(ref_hits, "REFERENCE_CONTEXT (authoritative Puppet guidance; justify refactor decisions using this):", None)

    user_prompt = f"""You must refactor ONLY the module under: {args.scope}

RAG-FIRST RULES:
- Use MODULE_CONTEXT to know what files/classes/params actually exist.
- Use REFERENCE_CONTEXT to decide HOW to refactor (Puppet 8 best practices, patterns, examples).
- If you cannot justify a change using REFERENCE_CONTEXT or obvious correctness/idempotency, keep the change minimal.
- Do NOT invent file paths. Only change existing MODULE files or create clearly NEW files when necessary.

{ref_ctx}

{module_ctx}

TASK:
{question}
"""

    answer = client.chat(
        model=chat_model,
        messages=[{"role": "user", "content": user_prompt}],
        # options={"num_predict": 3000},
    )

    dt = time.time() - t0
    print("\n=== Answer ===")
    print(answer.strip())
    print(f"\n[ask] elapsed_sec={dt:.1f}", file=sys.stderr)


if __name__ == "__main__":
    main()
