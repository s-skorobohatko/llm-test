from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class ModuleContext:
    module_path: str
    files: List[str]                 # relpaths
    file_contents: Dict[str, str]    # relpath -> content


def _safe_read(path: str, max_bytes: int) -> str | None:
    try:
        st = os.stat(path)
        if st.st_size > max_bytes:
            return None
        with open(path, "rb") as f:
            raw = f.read()
        if b"\x00" in raw[:4096]:
            return None
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return None


def scan_module(
    module_path: str,
    *,
    max_files: int,
    max_file_bytes: int,
) -> ModuleContext:
    files: List[str] = []
    file_contents: Dict[str, str] = {}

    for base, dirs, fs in os.walk(module_path):
        dirs[:] = [d for d in dirs if d not in {".git", ".venv", "__pycache__"}]
        for fn in fs:
            if fn.startswith("."):
                continue
            if not (fn.endswith(".pp") or fn.endswith(".epp") or fn.endswith(".md") or fn.endswith(".yaml") or fn.endswith(".yml") or fn.endswith(".json")):
                continue
            abspath = os.path.join(base, fn)
            rel = os.path.relpath(abspath, module_path).replace(os.sep, "/")
            files.append(rel)
            if len(files) >= max_files:
                break
        if len(files) >= max_files:
            break

    files = sorted(set(files))
    for rel in files:
        abspath = os.path.join(module_path, rel)
        txt = _safe_read(abspath, max_file_bytes)
        if txt is not None:
            file_contents[rel] = txt

    return ModuleContext(module_path=module_path, files=files, file_contents=file_contents)


def render_module_context(ctx: ModuleContext, *, max_blob_chars: int) -> str:
    """
    Provide both: file list + content blocks (limited).
    """
    out = []
    out.append("MODULE_FILES (authoritative):")
    for f in ctx.files:
        out.append(f"- {f}")

    out.append("\nMODULE_CONTENT (authoritative; truncated by size cap):")
    used = 0
    for rel in ctx.files:
        content = ctx.file_contents.get(rel)
        if content is None:
            continue
        block = f"\n=== FILE: {rel} ===\n{content}\n"
        if used + len(block) > max_blob_chars:
            break
        out.append(block)
        used += len(block)

    return "\n".join(out).strip()
