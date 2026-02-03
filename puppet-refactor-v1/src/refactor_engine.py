from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.module_scan import scan_module, render_module_context
from src.ollama_client import OllamaClient
from src.prompts import COMMON, DIFF_PROMPT, PLAN_PROMPT, SYSTEM
from src.retrieval import format_hits, retrieve


@dataclass
class RefactorOutput:
    plan: str
    diff: str
    report_md: str


def extract_diff_paths(diff_text: str) -> List[str]:
    out: List[str] = []
    for line in diff_text.splitlines():
        if line.startswith("DIFF FILE:"):
            p = line[len("DIFF FILE:") :].strip()
            # allow "(NEW FILE)" suffix in future, strip it
            p = p.replace("(NEW FILE)", "").strip()
            out.append(p)
    return out


def load_cfg(path: str) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def refactor_module(module_path: str, task: str, *, cfg_path: str = "config.yaml") -> RefactorOutput:
    cfg = load_cfg(cfg_path)

    ollama = OllamaClient(cfg["ollama_url"])
    model = cfg["models"]["chat"]

    limits = cfg.get("limits") or {}
    num_ctx = int(limits.get("num_ctx", 8192))
    np_plan = int(limits.get("num_predict_plan", 500))
    np_diff = int(limits.get("num_predict_diff", 1100))  # slightly higher to avoid truncating diffs
    max_blob = int(limits.get("max_files_blob_chars", 60000))

    # Scan module (authoritative list + content)
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

    common = COMMON.format(module_path=module_path, ref_ctx=ref_ctx, module_ctx=module_ctx)

    # Pass 1: Plan (short)
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

    # Pass 2: Diff (diff-only)
    diff_prompt = DIFF_PROMPT.format(common=common, task=task, plan=plan)
    diff = ollama.chat(
        model=model,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": diff_prompt}],
        num_ctx=num_ctx,
        num_predict=np_diff,
        temperature=0.1,
        timeout_sec=1800,
        stop=None,
    ).strip()

    # Enforce no hallucinated paths
    allowed = set(ctx.files)
    diff_paths = extract_diff_paths(diff)
    bad = [p for p in diff_paths if p not in allowed]
    if bad:
        raise RuntimeError(f"Model produced DIFF for non-existent files: {bad}")

    report = []
    report.append("# Puppet Refactor v1 Report\n")
    report.append(f"## Module\n`{module_path}`\n")
    report.append("## Task\n" + task + "\n")
    report.append("## Retrieval\n```\n" + ref_ctx + "\n```\n")
    report.append("## Plan\n" + plan + "\n")
    report.append("## Diff\n" + diff + "\n")

    return RefactorOutput(plan=plan, diff=diff, report_md="\n".join(report))
