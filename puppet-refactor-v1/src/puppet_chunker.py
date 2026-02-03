from __future__ import annotations
import re
from typing import List

CLASS_RE = re.compile(r"(?m)^\s*class\s+([a-zA-Z0-9_:]+)\b")
DEFINE_RE = re.compile(r"(?m)^\s*define\s+([a-zA-Z0-9_:]+)\b")
RESOURCE_RE = re.compile(r"(?m)^\s*([A-Z][A-Za-z0-9_:]*)\s*\{")


def _find_block_end(text: str, start: int) -> int:
    n = len(text)
    brace_start = text.find("{", start)
    if brace_start == -1:
        return min(n, start + 2000)

    depth = 0
    i = brace_start
    in_sq = in_dq = False
    esc = False

    while i < n:
        ch = text[i]
        if esc:
            esc = False
            i += 1
            continue
        if ch == "\\":
            esc = True
            i += 1
            continue

        if ch == "'" and not in_dq:
            in_sq = not in_sq
            i += 1
            continue
        if ch == '"' and not in_sq:
            in_dq = not in_dq
            i += 1
            continue

        if in_sq or in_dq:
            i += 1
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return n


def semantic_chunks_pp(text: str, filename: str) -> List[str]:
    cands = []
    for m in CLASS_RE.finditer(text):
        cands.append(("class", m.group(1), m.start()))
    for m in DEFINE_RE.finditer(text):
        cands.append(("define", m.group(1), m.start()))
    for m in RESOURCE_RE.finditer(text):
        cands.append(("resource", m.group(1), m.start()))
    cands.sort(key=lambda x: x[2])

    if not cands:
        return [f"File: {filename}\nBlock: file\n---\n{text}"]

    out: List[str] = []
    used = []
    for kind, name, start in cands:
        end = _find_block_end(text, start)
        if any(s <= start and end <= e for s, e in used):
            continue
        used.append((start, end))
        body = text[start:end].strip("\n")
        if not body.strip():
            continue
        header = f"File: {filename}\nBlock: {kind}\nName: {name}\n---\n"
        out.append(header + body)
    return out
