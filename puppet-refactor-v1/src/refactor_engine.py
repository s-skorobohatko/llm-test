from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set, Tuple

from src.module_scan import scan_module, render_module_context
from src.ollama_client import OllamaClient
from src.prompts import COMMON, DIFF_PROMPT, PLAN_PROMPT, SYSTEM
from src.retrieval import format_hits, retrieve


@dataclass
class RefactorOutput:
    plan: str
    diff: str
    report_md: str
    dropped_new_files: List[str]


def load_cfg(path: str) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def extract_diff_paths(diff_text: str) -> List[str]:
    out: List[str] = []
    for line in diff_text.splitlines():
        if line.startswith("DIFF FILE:"):
            p = line[len("DIFF FILE:") :].strip()
            p = p.replace("(NEW FILE)", "").strip()
            out.append(p)
    return out


def filter_diff_blocks(
    diff_text: str,
    *,
    existing_files: Set[str],
    allow_new_files: Set[str],
) -> Tuple[str, List[str]]:
    """
    Keep diff blocks for:
      - files that exist
      - NEW FILE blocks only if allow_new_files contains that path

    Returns (filtered_diff, dropped_paths)
    """
    lines = diff_text.splitlines()
    kept: List[str] = []
    dropped: List[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("DIFF FILE:"):
            header = line[len("DIFF FILE:") :].strip()
            is_new = "(NEW FILE)" in header
            path = header.replace("(NEW FILE)", "").strip()

            # capture the entire block until END DIFF (inclusive)
            block = [line]
            i += 1
            while i < len(lines):
                block.append(lines[i])
                if lines[i].strip() == "END DIFF":
                    i += 1
                    break
                i += 1

            if path in existing_files:
                kept.extend(block)
            elif is_new and path in allow_new_files:
                kept.extend(block)
            else:
                dropped.append(path)
            continue

        # ignore any stray non-diff text
        i += 1

    filtered = "\n".join(kept).strip()
    if filtered:
        filtered += "\n"
    return filtered, dropped


def refactor_module(module_path: str, task: str, *, cfg_path: str = "config.yaml") -> RefactorOutput:
    cfg = load_cfg(cfg_path)

    ollama = OllamaClient(cfg["ollama_url"])
    model = cfg["models"]["chat"]

    limits = cfg.get("limits") or {}
    num_ctx = int(limits.get("num_ctx", 8192))
    np_plan = int(limits.get("num_predict_plan", 500))
    np_diff = int(limits.get("num_predict_diff", 1100))
    max_blob = int(limits.get("max_files_blob_chars", 60000))

    policy = cfg.get("policy") or {}
    allow_new_files = set(policy.get("allow_new_files") or [])

    # Scan module
    ctx = scan_module(
        module_path,
        max_files=int(limits.get("max_files_read", 200)),
        max_file_bytes=int(limits.get("max_file_bytes", 2_000_000)),
    )
    module_ctx = render_module_context(ctx, max_blob_chars=max_blob)

    # Retrieval
    retrieval_query = f"{task}\n\nModule files:\n" + "\n".join(ctx.files[:120])
    hits = retrieve(retrieval_query, cfg_path=cfg_path)
    ref_ctx = format_hits(hits, max_chars=int(limits.get("retrieve_max_chars", 9000))) or "(none)"

    common = COMMON.format(
        module_path=module_path,
        ref_ctx=ref_ctx,
        module_ctx=module_ctx,
        allow_new_files=", ".join(sorted(allow_new_files)) if allow_new_files else "(none)",
    )

    # Pass 1: Plan
    plan_prompt = PLAN_PROMPT.format(common=common, task=task)
    plan = ollama.chat(
        model=model,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": plan_prompt}],
        num_ctx=num_ctx,
        num_predict=np_plan,
        temperature=0.1,
        timeout_sec=1800,
        stop=["DIFF FILE:", "END DIFF"],
    ).strip()

    # Pass 2: Diff
    diff_prompt = DIFF_PROMPT.format(common=common, task=task, plan=plan)
    diff_raw = ollama.chat(
        model=model,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": diff_prompt}],
        num_ctx=num_ctx,
        num_predict=np_diff,
        temperature=0.1,
        timeout_sec=1800,
    ).strip()

    # ✅ Never crash: filter invalid blocks instead
    existing = set(ctx.files)
    diff, dropped = filter_diff_blocks(
        diff_raw,
        existing_files=existing,
        allow_new_files=allow_new_files,
    )

    report = []
    report.append("# Puppet Refactor v1 Report\n")
    report.append(f"## Module\n`{module_path}`\n")
    report.append("## Task\n" + task + "\n")
    report.append("## Retrieval\n```\n" + ref_ctx + "\n```\n")
    report.append("## Plan\n" + plan + "\n")
    if dropped:
        report.append("## Dropped diffs (not allowed / non-existent)\n")
        for p in dropped:
            report.append(f"- {p}")
        report.append("")
    report.append("## Diff\n" + (diff or "(no allowed diffs produced)") + "\n")

    return RefactorOutput(plan=plan, diff=diff, report_md="\n".join(report), dropped_new_files=dropped)
