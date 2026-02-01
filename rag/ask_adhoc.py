#!/usr/bin/env python3
import argparse
import os
import re
import sys
import tarfile
import zipfile
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import yaml

from raglib import OllamaClient, chunk_text, load_text_file, sha256_str


DEFAULT_INCLUDE = [
    "**/*.pp",
    "**/*.epp",
    "**/*.md",
    "**/*.markdown",
    "**/*.yml",
    "**/*.yaml",
    "metadata.json",
    "README*",
    "REFERENCE.md",
]


SYSTEM_PUPPET = """You are a senior Puppet engineer (Puppet 8+). You write and refactor production-grade Puppet code and modules.

Rules:
- Prefer data-driven design: class parameters + Hiera (lookup()).
- Remove legacy patterns: params.pp, validate_* functions, unnecessary inheritance.
- Ensure idempotency and correct resource ordering.
- Use modern Puppet types and EPP templates.
- State assumptions briefly.
- Cite retrieved snippets like [1], [2] when you rely on them.

Output formatting:
- Put code in fenced blocks.
- If output is long, continue rather than truncating.
"""


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def extract_archive(archive_path: str, dest_dir: str) -> str:
    ap = archive_path.lower()
    if ap.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(dest_dir)
    elif ap.endswith(".tar.gz") or ap.endswith(".tgz") or ap.endswith(".tar"):
        mode = "r:gz" if (ap.endswith(".tar.gz") or ap.endswith(".tgz")) else "r:"
        with tarfile.open(archive_path, mode) as t:
            # basic safety: prevent path traversal
            for m in t.getmembers():
                p = os.path.abspath(os.path.join(dest_dir, m.name))
                if not p.startswith(os.path.abspath(dest_dir) + os.sep) and p != os.path.abspath(dest_dir):
                    raise RuntimeError(f"Unsafe path in tar: {m.name}")
            t.extractall(dest_dir)
    else:
        raise ValueError("Unsupported archive. Use .zip, .tar.gz, .tgz, or .tar")
    return dest_dir


def glob_files(root: str, patterns: List[str]) -> List[str]:
    out: List[str] = []
    rootp = Path(root)
    for pat in patterns:
        out.extend([str(p) for p in rootp.glob(pat) if p.is_file()])
    # dedupe
    seen = set()
    uniq = []
    for p in out:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def build_prompt(question: str, contexts: List[Tuple[float, Dict[str, Any]]]) -> str:
    if not contexts:
        return question

    lines = ["You have these retrieved snippets. Use them when relevant:\n"]
    for idx, (score, meta) in enumerate(contexts, start=1):
        rel = meta["relpath"]
        chunk_index = meta["chunk_index"]
        content = meta["content"]
        lines.append(f"[{idx}] score={score:.3f} {rel}#chunk{chunk_index}\n{content}\n")
    lines.append("User question:\n" + question)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="Ad-hoc Puppet module RAG: point to a module folder or archive; no DB needed."
    )
    ap.add_argument("question", help="Your question to the model")
    ap.add_argument("--path", required=True, help="Path to module directory OR archive (.zip/.tar.gz/.tgz/.tar)")
    ap.add_argument("--include", action="append", help="Extra glob to include (repeatable)")
    ap.add_argument("--topk", type=int, default=12, help="How many chunks to include in context")
    ap.add_argument("--minsim", type=float, default=0.10, help="Minimum similarity threshold")
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--chunk-overlap", type=int, default=None)
    ap.add_argument("--max-files", type=int, default=300, help="Safety cap for number of files read")
    ap.add_argument("--max-chunks", type=int, default=3000, help="Safety cap for number of chunks embedded")
    ap.add_argument("--num-predict", type=int, default=900, help="Output token cap")
    args = ap.parse_args()

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    client = OllamaClient(cfg["ollama_url"])
    embed_model = cfg["embed_model"]
    chat_model = cfg["chat_model"]

    chunk_size = int(args.chunk_size if args.chunk_size is not None else cfg.get("chunk_size", 1200))
    chunk_overlap = int(args.chunk_overlap if args.chunk_overlap is not None else cfg.get("chunk_overlap", 200))

    src_path = os.path.abspath(args.path)

    with tempfile.TemporaryDirectory(prefix="adhoc_module_") as td:
        if os.path.isdir(src_path):
            module_root = src_path
        else:
            # archive -> extract
            extract_archive(src_path, td)
            # pick the first directory that looks like a module root
            # (either extracted root itself, or first subdir)
            candidates = [td] + [str(p) for p in Path(td).iterdir() if p.is_dir()]
            module_root = candidates[0]

        patterns = list(DEFAULT_INCLUDE)
        if args.include:
            patterns.extend(args.include)

        files = glob_files(module_root, patterns)
        if not files:
            print("No files matched include patterns. Try --include '**/*' or check the path.", file=sys.stderr)
            sys.exit(2)

        if len(files) > args.max_files:
            files = files[: args.max_files]

        # Embed the question
        q_vec = np.array(client.embed(embed_model, args.question), dtype=np.float32)

        # Build chunk list and embed chunks (in-memory)
        candidates: List[Tuple[float, Dict[str, Any]]] = []
        total_chunks = 0

        for fp in files:
            try:
                text = load_text_file(fp)
            except Exception:
                continue

            rel = os.path.relpath(fp, module_root)
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
            if not chunks:
                continue

            for i, ch in enumerate(chunks):
                total_chunks += 1
                if total_chunks > args.max_chunks:
                    break

                vec = np.array(client.embed(embed_model, ch), dtype=np.float32)
                sim = cosine_sim(q_vec, vec)
                if sim >= args.minsim:
                    candidates.append(
                        (
                            sim,
                            {
                                "relpath": rel,
                                "chunk_index": i,
                                "content": ch,
                                "content_hash": sha256_str(ch),
                            },
                        )
                    )

            if total_chunks > args.max_chunks:
                break

        # Pick top-K
        candidates.sort(key=lambda x: x[0], reverse=True)
        contexts = candidates[: args.topk]

        print("=== Retrieved context (ad-hoc) ===")
        if not contexts:
            print("(none)  (Try lowering --minsim or increasing --max-chunks)")
        else:
            for idx, (score, meta) in enumerate(contexts, start=1):
                print(f"[{idx}] sim={score:.3f} {meta['relpath']}#chunk{meta['chunk_index']}")

        prompt = build_prompt(args.question, contexts)

        print("\n=== Answer ===")
        client.chat(
            model=chat_model,
            system=SYSTEM_PUPPET,
            user=prompt,
            options={"num_predict": args.num_predict},
            stream=True,
            timeout=3600,
        )


if __name__ == "__main__":
    main()
