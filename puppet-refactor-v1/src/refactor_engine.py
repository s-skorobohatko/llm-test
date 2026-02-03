from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Set, Tuple

from src.logger import Logger
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


def _looks_complete(diff_text: str) -> bool:
    t = diff_text.strip()
    if not t:
        return True
    if "DIFF FILE:" not in t:
        return True
    return t.endswith("END DIFF")


def _contains_unified_hunks(text: str) -> bool:
    return any(line.startswith("@@ ") for line in text.splitlines())


def _parse_plan_file_list(plan_text: str, allowed: Set[str], allow_new: Set[str]) -> List[str]:
    """
    Plan is 'one file per line'. Return only valid/allowed paths.
    """
    out: List[str] = []
    for raw in (plan_text or "").splitlines():
        p = raw.strip().lstrip("-").strip()
        if not p:
            continue
        if p in allowed or p in allow_new:
            out.append(p)
    # de-dupe preserving order
    seen = set()
    uniq: List[str] = []
    for p in out:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def extract_diff_paths(diff_text: str) -> List[str]:
    out: List[str] = []
    for line in diff_text.splitlines():
        if line.startswith("DIFF FILE:"):
            p = line[len("DIFF FILE:") :].strip()
            p = p.replace("(NEW FILE)", "").strip()
            out.append(p)
    # de-dupe preserving order
    seen = set()
    uniq: List[str] = []
    for p in out:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def filter_diff_blocks(
    diff_text: str,
    *,
    existing_files: Set[str],
    allow_new_files: Set[str],
) -> Tuple[str, List[str]]:
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

        i += 1

    filtered = "\n".join(kept).strip()
    if filtered:
        filtered += "\n"
    return filtered, dropped


def _chat_with_continuations(
    ollama: OllamaClient,
    *,
    log: Logger,
    model: str,
    system: str,
    user_prompt: str,
    num_ctx: int,
    num_predict: int,
    timeout_sec: int,
    max_rounds: int,
) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}]
    out_parts: List[str] = []

    for r in range(1, max_rounds + 1):
        log.phase("diff", f"round {r}/{max_rounds} (num_predict={num_predict}, num_ctx={num_ctx})")
        t0 = time.time()
        chunk = ollama.chat(
            model=model,
            messages=messages,
            num_ctx=num_ctx,
            num_predict=num_predict,
            temperature=0.1,
            timeout_sec=timeout_sec,
        ).strip()
        dt = time.time() - t0

        log.metric("diff_round_sec", f"{dt:.1f}")
        log.metric("diff_round_chars", len(chunk))

        if chunk:
            out_parts.append(chunk)

        merged = "\n".join(out_parts).strip()
        if _looks_complete(merged):
            log.phase("diff", "complete")
            return merged

        messages.append({"role": "assistant", "content": chunk})
        messages.append(
            {
                "role": "user",
                "content": "CONTINUE. Output the remaining DIFF blocks only. Do not repeat any previous DIFF FILE blocks.",
            }
        )

    log.phase("diff", "max rounds reached (may be incomplete)")
    return "\n".join(out_parts).strip()


def _repair_diff_format(
    ollama: OllamaClient,
    *,
    log: Logger,
    model: str,
    num_ctx: int,
    num_predict: int,
    timeout_sec: int,
    system: str,
    diff_prompt: str,
    bad_output: str,
) -> str:
    log.phase("diff", "repairing format (found unified hunks @@)")
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": diff_prompt},
        {"role": "assistant", "content": bad_output},
        {
            "role": "user",
            "content": (
                "FORMAT ERROR: You output unified diff hunks (lines starting with @@). "
                "Re-output using ONLY the required format:\n"
                "DIFF FILE: <path>\n--------------------------------\nNEW:\n<full updated file content>\n--------------------------------\nEND DIFF\n\n"
                "Do NOT include @@ hunks. Do NOT include any text outside DIFF blocks."
            ),
        },
    ]
    return ollama.chat(
        model=model,
        messages=messages,
        num_ctx=num_ctx,
        num_predict=num_predict,
        temperature=0.1,
        timeout_sec=timeout_sec,
    ).strip()


def refactor_module(
    module_path: str,
    task: str,
    *,
    cfg_path: str = "config.yaml",
    log_enabled: bool = False,
) -> RefactorOutput:
    log = Logger(enabled=log_enabled)

    t_all = time.time()
    cfg = load_cfg(cfg_path)

    ollama = OllamaClient(cfg["ollama_url"])
    model = cfg["models"]["chat"]

    limits = cfg.get("limits") or {}
    num_ctx = int(limits.get("num_ctx", 8192))
    np_plan = int(limits.get("num_predict_plan", 250))
    np_diff = int(limits.get("num_predict_diff", 2200))
    max_blob = int(limits.get("max_files_blob_chars", 30000))
    retrieve_max = int(limits.get("retrieve_max_chars", 9000))
    diff_max_rounds = int(limits.get("diff_max_rounds", 6))
    timeout_sec = int(limits.get("timeout_sec", 1800))

    policy = cfg.get("policy") or {}
    allow_new_files = set(policy.get("allow_new_files") or [])

    # scan
    log.phase("scan", module_path)
    t0 = time.time()
    ctx = scan_module(
        module_path,
        max_files=int(limits.get("max_files_read", 250)),
        max_file_bytes=int(limits.get("max_file_bytes", 2_000_000)),
    )
    module_ctx = render_module_context(ctx, max_blob_chars=max_blob)
    scan_sec = time.time() - t0
    log.metric("scan_sec", f"{scan_sec:.1f}")
    log.metric("files", len(ctx.files))
    log.metric("module_ctx_chars", len(module_ctx))

    existing = set(ctx.files)

    # retrieval
    log.phase("retrieve")
    t1 = time.time()
    retrieval_query = f"{task}\n\nModule files:\n" + "\n".join(ctx.files[:160])
    hits = retrieve(retrieval_query, cfg_path=cfg_path)
    ref_ctx = format_hits(hits, max_chars=retrieve_max) or "(none)"
    retrieve_sec = time.time() - t1
    log.metric("retrieve_sec", f"{retrieve_sec:.1f}")
    log.metric("retrieve_hits", len(hits))
    log.metric("ref_ctx_chars", len(ref_ctx))

    common = COMMON.format(
        module_path=module_path,
        ref_ctx=ref_ctx,
        module_ctx=module_ctx,
        allow_new_files=", ".join(sorted(allow_new_files)) if allow_new_files else "(none)",
    )

    # plan (short: file list only)
    log.phase("plan")
    t2 = time.time()
    plan_prompt = PLAN_PROMPT.format(common=common, task=task)
    plan = ollama.chat(
        model=model,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": plan_prompt}],
        num_ctx=num_ctx,
        num_predict=np_plan,
        temperature=0.1,
        timeout_sec=timeout_sec,
        stop=["END DIFF", "DIFF FILE:"],
    ).strip()
    plan_sec = time.time() - t2
    log.metric("plan_sec", f"{plan_sec:.1f}")
    log.metric("plan_chars", len(plan))

    planned_files = _parse_plan_file_list(plan, existing, allow_new_files)
    if log_enabled:
        log.phase("plan_files", f"{len(planned_files)} selected")
        for p in planned_files[:25]:
            log.info(f"plan_file={p}")
        if len(planned_files) > 25:
            log.info(f"plan_file=... (+{len(planned_files)-25} more)")

    # diff
    diff_prompt = DIFF_PROMPT.format(common=common, task=task, plan="\n".join(planned_files))
    diff_raw = _chat_with_continuations(
        ollama,
        log=log,
        model=model,
        system=SYSTEM,
        user_prompt=diff_prompt,
        num_ctx=num_ctx,
        num_predict=np_diff,
        timeout_sec=timeout_sec,
        max_rounds=diff_max_rounds,
    )

    if _contains_unified_hunks(diff_raw):
        diff_raw = _repair_diff_format(
            ollama,
            log=log,
            model=model,
            num_ctx=num_ctx,
            num_predict=np_diff,
            timeout_sec=timeout_sec,
            system=SYSTEM,
            diff_prompt=diff_prompt,
            bad_output=diff_raw,
        )

    # filter + extract produced files
    log.phase("filter")
    diff, dropped = filter_diff_blocks(diff_raw, existing_files=existing, allow_new_files=allow_new_files)
    produced_files = extract_diff_paths(diff)

    # summary anchors (stderr only)
    if log_enabled:
        log.phase("summary")
        log.metric("planned_files", len(planned_files))
        log.metric("produced_files", len(produced_files))
        if dropped:
            log.metric("dropped_blocks", len(dropped))

        missing = [p for p in planned_files if p not in set(produced_files)]
        extra = [p for p in produced_files if p not in set(planned_files)]

        if missing:
            log.info("WARN planned_but_not_produced=" + ", ".join(missing[:15]) + ("" if len(missing) <= 15 else f" (+{len(missing)-15})"))
        if extra:
            log.info("WARN produced_not_in_plan=" + ", ".join(extra[:15]) + ("" if len(extra) <= 15 else f" (+{len(extra)-15})"))

        log.metric("total_sec", f"{time.time()-t_all:.1f}")

    # report (optional for --mode report)
    report = []
    report.append("# Puppet Refactor v1 Report\n")
    report.append(f"## Module\n`{module_path}`\n")
    report.append("## Task\n" + task + "\n")
    report.append("## Retrieval\n```\n" + ref_ctx + "\n```\n")
    report.append("## Planned files\n```\n" + "\n".join(planned_files) + "\n```\n")
    if dropped:
        report.append("## Dropped diffs (not allowed / non-existent)\n")
        for p in dropped:
            report.append(f"- {p}")
        report.append("")
    report.append("## Diff\n" + (diff or "(no allowed diffs produced)") + "\n")

    return RefactorOutput(plan=plan, diff=diff, report_md="\n".join(report), dropped_new_files=dropped)
