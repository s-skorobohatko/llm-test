from __future__ import annotations
import re
import yaml
from dataclasses import dataclass
from typing import List, Dict

from src.ollama_client import OllamaClient
from src.module_scan import scan_module, render_module_context
from src.retrieval import retrieve, format_hits
from src.prompts import SYSTEM, COMMON, PLAN_PROMPT, DIFF_PROMPT


@dataclass
class RefactorOutput:
    plan: str
    diff: str
    report_md: str


_DIFF_FILE_RE = re.compile(r"(?m)^DIFF FILE:\s+(.+?)\s*$")


def load_cfg(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def extract_diff_paths(diff_text: str) -> List[str]:
    return [m.group(1).strip() for m in _DIFF_FILE_RE.finditer(diff_text)]


def refactor_module(module_path: str, task: str, *, cfg_path: str = "config.yaml") -> RefactorOutput:
    cfg = load_cfg(cfg_path)
    ollama = OllamaClient(cfg["ollama_url"])

    model = cfg["models"]["chat"]
    num_ctx = int(cfg["limits"]["num_ctx"])
    np_plan = int(cfg["limits"]["num_predict_plan"])
    np_diff = int(cfg["limits"]["num_predict_diff"])
    ref_max = int(cfg["limits"]["retrieve_max_chars"])

    # 1) scan module (authoritative)
    ctx = scan_module(
        module_path,
        max_files=int(cfg["limits"]["max_files_read"]),
        max_file_bytes=int(cfg["limits"]["max_file_bytes"]),
    )
    module_ctx = render_module_context(ctx, max_blob_chars=int(cfg["limits"]["max_files_blob_chars"]))

    # 2) retrieval (task + module file list helps)
    retrieval_query = f"Task:\n{task}\n\nModule files:\n" + "\n".join(ctx.files[:80])
    hits = retrieve(retrieval_query, cfg_path=cfg_path)
    ref_ctx = format_hits(hits, max_chars=ref_max) or "(none)"

    # 3) plan pass
    plan_prompt = PLAN_PROMPT.format(common=COMMON, task=task, ref_ctx=ref_ctx, module_ctx=module_ctx)
    plan = ollama.chat(
        model=model,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": plan_prompt}],
        num_ctx=num_ctx,
        num_predict=np_plan,
        temperature=0.1,
        timeout_sec=1800,
    ).strip()

    # 4) diff pass
    diff_prompt = DIFF_PROMPT.format(common=COMMON, task=task, plan=plan, ref_ctx=ref_ctx, module_ctx=module_ctx)
    diff = ollama.chat(
        model=model,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": diff_prompt}],
        num_ctx=num_ctx,
        num_predict=np_diff,
        temperature=0.1,
        timeout_sec=1800,
    ).strip()

    # 5) server-side enforcement: no hallucinated paths
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
