#!/usr/bin/env python3
"""
ask.py — RAG (Qdrant + Ollama), module-grounded + reference-guided

Changes:
- Prompts loaded from separate text files (prompts/common_rules.txt, prompts/plan.txt, prompts/diff.txt)
- Safe diff formatting (DIFF FILE blocks) to avoid markdown/rendering issues
- Still requires --scope
- No num_predict handling anywhere
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import yaml

from raglib import OllamaClient
from qdrant_store import QdrantStore

Hit = Tuple[float, Dict[str, Any]]


# ----------------------------
# Filter builders (QdrantStore)
# ----------------------------

def must_source(value: str) -> Dict[str, Any]:
    return {"key": "source", "match": {"value": value}}


def must_path_contains(text: str) -> Dict[str, Any]:
    return {"key": "path", "match": {"text": text}}


def must_module_root(value: str) -> Dict[str, Any]:
    return {"key": "module_root", "match": {"value": value}}


# ----------------------------
# Utilities
# ----------------------------

def require_keys(cfg: Dict[str, Any], keys: List[str], where: str) -> None:
    missing = [k for k in keys if k not in cfg or cfg[k] in (None, "")]
    if missing:
        raise SystemExit(f"Missing config keys in {where}: {', '.join(missing)}")


def read_question(positional: Optional[str]) -> str:
    if positional is None:
        q = sys.stdin.read().strip()
        if not q:
            print("No question provided (arg or stdin).", file=sys.stderr)
            sys.exit(2)
        return q
    q = positional.strip()
    if not q:
        print("Empty question provided.", file=sys.stderr)
        sys.exit(2)
    return q


def format_context(items: List[Hit], title: str, limit: Optional[int] = None) -> str:
    blocks: List[str] = []
    for i, (score, payload) in enumerate(items, start=1):
        if limit is not None and i > limit:
            break

        content = payload.get("content", "") or ""
        src = payload.get("source", "") or ""
        module_root = payload.get("module_root", "") or ""
        path = payload.get("path", "") or ""
        relpath = payload.get("relpath", "") or ""
        chunk_idx = payload.get("chunk_index", "")

        blocks.append(
            f"[{i}] source={src} module_root={module_root} relpath={relpath} "
            f"path={path} chunk={chunk_idx} sim={score:.3f}\n{content}"
        )

    joined = "\n\n---\n\n".join(blocks).strip()
    return f"{title}\n{joined}\n"


def clamp_scoped_minsim(min_sim: float) -> float:
    # When scoped, allow more permissive similarity to avoid missing relevant module chunks.
    return min_sim if min_sim <= 0.10 else 0.06


def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def prompt_path(prompt_dir: str, name: str) -> str:
    return os.path.join(prompt_dir, name)


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Ask with RAG (Qdrant + Ollama), module-grounded + reference-guided."
    )
    ap.add_argument("question", nargs="?", help="Question to ask (or use stdin if omitted)")
    ap.add_argument("--config", default="config.yaml", help="Path to config.yaml")

    # Retrieval tuning
    ap.add_argument("--topk", type=int, default=None, help="Top-k for MODULE retrieval (scoped)")
    ap.add_argument("--minsim", type=float, default=None, help="Minimum similarity threshold")

    # Grounding: restrict edits to a real module
    ap.add_argument("--scope", default=None, help="Module root path (module_root). Required.")

    # Reference guidance sources (repeatable)
    ap.add_argument(
        "--ref_source",
        action="append",
        default=[],
        help="Reference source name (repeatable). Example: vendor:puppetlabs:best-practices",
    )
    ap.add_argument("--ref_topk", type=int, default=12, help="Top-k for REFERENCE retrieval (per run)")

    # Diagnostics / output
    ap.add_argument("--show_context", type=int, default=30, help="How many retrieved items to print per section")
    ap.add_argument("--json", action="store_true", help="Output JSON (answer + retrieval + timings)")

    # Quality modes
    ap.add_argument("--two_pass", action="store_true", help="Run plan pass first, then diff pass")
    ap.add_argument(
        "--mode",
        choices=["diff", "plan", "both"],
        default="diff",
        help="What to generate: diff | plan | both (default: diff)",
    )

    # Prompt directory
    ap.add_argument("--prompt_dir", default=None, help="Directory with prompt templates (default: from config or ./prompts)")

    args = ap.parse_args()
    question = read_question(args.question)

    if not args.scope:
        print("ERROR: --scope is required (prevents hallucinated file paths).", file=sys.stderr)
        sys.exit(2)

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    require_keys(
        cfg,
        ["ollama_url", "embed_model", "chat_model", "qdrant_url", "qdrant_collection"],
        args.config,
    )

    # Validate vector_store if present
    if cfg.get("vector_store") and cfg["vector_store"] != "qdrant":
        raise SystemExit("config.yaml vector_store must be 'qdrant' (or omit vector_store)")

    ollama_url = cfg["ollama_url"]
    embed_model = cfg["embed_model"]
    chat_model = cfg["chat_model"]
    qdrant_url = cfg["qdrant_url"]
    collection = cfg["qdrant_collection"]

    top_k_module = args.topk if args.topk is not None else int(cfg.get("top_k", 6))
    min_sim = args.minsim if args.minsim is not None else float(cfg.get("min_sim", 0.18))

    default_ref_sources = list(cfg.get("default_ref_sources") or ["vendor:puppetlabs:best-practices"])
    ref_sources = args.ref_source[:] if args.ref_source else default_ref_sources

    prompt_dir = args.prompt_dir or cfg.get("prompt_dir") or "./prompts"

    # Load prompt templates from files
    common_rules_tpl = load_prompt(prompt_path(prompt_dir, "common_rules.txt"))
    plan_tpl = load_prompt(prompt_path(prompt_dir, "plan.txt"))
    diff_tpl = load_prompt(prompt_path(prompt_dir, "diff.txt"))

    client = OllamaClient(ollama_url)
    store = QdrantStore(qdrant_url, collection)

    t0 = time.time()

    # 1) Embed query
    q_vec = client.embed(embed_model, question)

    # 2) MODULE retrieval (authoritative for what exists)
    scoped_min_sim = clamp_scoped_minsim(min_sim)

    module_hits = store.search(
        query_vector=q_vec,
        top_k=max(top_k_module, 60),
        min_sim=scoped_min_sim,
        must=[must_module_root(args.scope)],
    )
    if not module_hits:
        module_hits = store.search(
            query_vector=q_vec,
            top_k=max(top_k_module, 60),
            min_sim=scoped_min_sim,
            must=[must_path_contains(args.scope)],
        )

    # 3) REFERENCE retrieval (authoritative guidance)
    ref_hits_all: List[Hit] = []
    per_src = max(4, args.ref_topk // max(1, len(ref_sources)))

    for src in ref_sources:
        ref_hits_all.extend(
            store.search(
                query_vector=q_vec,
                top_k=per_src,
                min_sim=0.12,
                must=[must_source(src)],
            )
        )

    ref_hits_all.sort(key=lambda x: x[0], reverse=True)
    ref_hits = ref_hits_all[: args.ref_topk]

    # 4) Diagnostics (unless JSON)
    if not args.json:
        print("=== Retrieved context (MODULE) ===")
        for i, (score, payload) in enumerate(module_hits[: args.show_context], start=1):
            p = payload.get("relpath") or payload.get("path") or "?"
            print(
                f"[{i}] sim={score:.3f} {payload.get('source','?')} "
                f"{p}#chunk{payload.get('chunk_index','?')}"
            )

        print("\n=== Retrieved context (REFERENCE) ===")
        for i, (score, payload) in enumerate(ref_hits[: args.show_context], start=1):
            print(
                f"[{i}] sim={score:.3f} {payload.get('source','?')} "
                f"{payload.get('path','?')}#chunk{payload.get('chunk_index','?')}"
            )

    # 5) Build prompt contexts
    module_ctx = format_context(module_hits, "MODULE_CONTEXT (authoritative; only these files exist):", None)
    ref_ctx = format_context(ref_hits, "REFERENCE_CONTEXT (authoritative Puppet guidance):", None)

    # 6) Render prompts from templates (NO triple quotes in code)
    common_rules = common_rules_tpl.format(
        scope=args.scope,
        ref_ctx=ref_ctx,
        module_ctx=module_ctx,
    )

    plan_prompt = plan_tpl.format(
        common_rules=common_rules,
        question=question,
    )

    diff_prompt = diff_tpl.format(
        common_rules=common_rules,
        question=question,
    )

    # 7) Generate (streaming)
    if not args.json:
        print("\n=== Answer (streaming) ===")

    plan_text = ""
    diff_text = ""

    # ---- Plan pass ----
    if args.two_pass or args.mode in ("plan", "both"):
        plan_text = client.chat(
            model=chat_model,
            messages=[{"role": "user", "content": plan_prompt}],
            stream=True,
            stream_print=(not args.json),
        )
        if not args.json:
            print("\n")  # newline after streaming

        if args.mode == "both" and not args.json:
            print("=== Plan complete; generating diffs ===\n")

    # ---- Diff pass ----
    if args.mode == "plan":
        final_answer = plan_text.strip()
    else:
        if args.two_pass and plan_text.strip():
            diff_prompt2 = diff_prompt + "\n\nCONSTRAINT: Follow this plan:\n" + plan_text
            diff_text = client.chat(
                model=chat_model,
                messages=[{"role": "user", "content": diff_prompt2}],
                stream=True,
                stream_print=(not args.json),
            )
        else:
            diff_text = client.chat(
                model=chat_model,
                messages=[{"role": "user", "content": diff_prompt}],
                stream=True,
                stream_print=(not args.json),
            )

        if not args.json:
            print("\n")
        final_answer = diff_text.strip()

    dt = time.time() - t0

    # 8) Output
    result = {
        "answer": final_answer,
        "elapsed_sec": round(dt, 3),
        "scope": args.scope,
        "run": {
            "two_pass": bool(args.two_pass),
            "mode": args.mode,
            "topk_module": int(top_k_module),
            "minsim": float(min_sim),
            "ref_sources": ref_sources,
            "ref_topk": int(args.ref_topk),
            "prompt_dir": prompt_dir,
        },
        "retrieval": {
            "module": [
                {
                    "score": float(score),
                    "source": payload.get("source", ""),
                    "module_root": payload.get("module_root", ""),
                    "path": payload.get("path", ""),
                    "relpath": payload.get("relpath", ""),
                    "chunk_index": payload.get("chunk_index", ""),
                }
                for (score, payload) in module_hits[: args.show_context]
            ],
            "reference": [
                {
                    "score": float(score),
                    "source": payload.get("source", ""),
                    "path": payload.get("path", ""),
                    "chunk_index": payload.get("chunk_index", ""),
                }
                for (score, payload) in ref_hits[: args.show_context]
            ],
        },
    }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("\n=== Answer ===")
        print(final_answer)
        print(f"\n[ask] elapsed_sec={dt:.1f}", file=sys.stderr)


if __name__ == "__main__":
    main()
