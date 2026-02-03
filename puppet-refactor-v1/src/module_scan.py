from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class ModuleContext:
    root: str
    files: List[str]              # relative paths
    file_contents: Dict[str, str] # relpath -> content


def render_module_context(ctx: ModuleContext, *, max_blob_chars: int) -> str:
    """
    CPU-friendly module context:
    - Always include full file list
    - Include content ONLY for manifests/ and templates/ (and only .pp/.epp/.erb)
    - Hard cap total chars
    """
    out: List[str] = []
    out.append("MODULE_FILES (authoritative):")
    for f in ctx.files:
        out.append(f"- {f}")

    out.append("\nMODULE_CONTENT (authoritative; truncated):")
    used = 0

    def want(rel: str) -> bool:
        if rel.startswith("manifests/") and rel.endswith(".pp"):
            return True
        if rel.startswith("templates/") and (rel.endswith(".epp") or rel.endswith(".erb")):
            return True
        return False

    # prioritize manifests first, then templates
    priority = [f for f in ctx.files if f.startswith("manifests/")]
    rest = [f for f in ctx.files if f.startswith("templates/")]

    for rel in priority + rest:
        if not want(rel):
            continue
        content = ctx.file_contents.get(rel)
        if content is None:
            continue

        block = f"\n=== FILE: {rel} ===\n{content}\n"
        if used + len(block) > max_blob_chars:
            break
        out.append(block)
        used += len(block)

    return "\n".join(out).strip()
