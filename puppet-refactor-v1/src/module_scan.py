from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ModuleContext:
    root: str
    files: List[str]               # relative paths (posix style)
    file_contents: Dict[str, str]  # relpath -> content


def _is_probably_text(path: Path, *, max_bytes: int) -> bool:
    try:
        st = path.stat()
        if st.st_size > max_bytes:
            return False
        with path.open("rb") as f:
            head = f.read(4096)
        if b"\x00" in head:
            return False
        return True
    except Exception:
        return False


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    return raw.decode("utf-8", errors="replace")


def _rel(root: Path, p: Path) -> str:
    return p.relative_to(root).as_posix()


def scan_module(
    module_root: str,
    *,
    max_files: int = 250,
    max_file_bytes: int = 2_000_000,
    include_globs: Optional[List[str]] = None,
) -> ModuleContext:
    """
    Scan a Puppet module directory and load file contents (text only).

    Defaults:
    - includes .pp, .epp, .erb, .md, metadata.json, hiera.yaml/yml
    - skips common junk dirs (.git, vendor, etc.)
    """
    root = Path(module_root).resolve()
    if not root.is_dir():
        raise ValueError(f"module_root is not a directory: {module_root}")

    if include_globs is None:
        include_globs = [
            "manifests/**/*.pp",
            "templates/**/*.epp",
            "templates/**/*.erb",
            "README*",
            "metadata.json",
            "hiera.yaml",
            "hiera.yml",
            "data/**/*.yaml",
            "data/**/*.yml",
        ]

    skip_dirs = {
        ".git", ".svn", ".hg", ".idea", ".vscode", "__pycache__", ".venv",
        "vendor", "node_modules",
    }

    matched: List[Path] = []
    # We walk manually so we can skip dirs efficiently.
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]
        basep = Path(base)

        for fn in files:
            if fn.startswith("."):
                continue
            fp = basep / fn
            rel = _rel(root, fp)

            # glob matching: check against patterns
            ok = False
            for pat in include_globs:
                # Path.match matches from beginning and supports **, but needs posix paths
                if Path(rel).match(pat):
                    ok = True
                    break
            if ok:
                matched.append(fp)

            if len(matched) >= max_files:
                break
        if len(matched) >= max_files:
            break

    # Stable ordering
    matched = sorted(set(matched), key=lambda p: _rel(root, p))

    files_rel: List[str] = []
    contents: Dict[str, str] = {}

    for p in matched:
        rel = _rel(root, p)
        files_rel.append(rel)

        if not _is_probably_text(p, max_bytes=max_file_bytes):
            continue

        try:
            contents[rel] = _read_text(p)
        except Exception:
            # keep file listed, content missing
            continue

    return ModuleContext(root=str(root), files=files_rel, file_contents=contents)


def render_module_context(ctx: ModuleContext, *, max_blob_chars: int = 30000) -> str:
    """
    CPU-friendly context:
    - Always include full file list
    - Include content ONLY for manifests/*.pp and templates/*.(epp|erb)
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
