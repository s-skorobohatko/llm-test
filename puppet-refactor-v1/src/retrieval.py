from __future__ import annotations
from typing import List, Dict, Any, Tuple
import yaml

from src.ollama_client import OllamaClient
from src.qdrant_store import QdrantStore

Hit = Tuple[float, Dict[str, Any]]


def load_cfg(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def retrieve(query: str, *, cfg_path: str = "config.yaml") -> List[Hit]:
    cfg = load_cfg(cfg_path)
    ollama = OllamaClient(cfg["ollama_url"])
    store = QdrantStore(cfg["vector_store"]["url"], cfg["vector_store"]["collection"])

    q_vec = ollama.embed(cfg["models"]["embed"], query)
    hits = store.search(q_vec, top_k=10, min_sim=0.10)
    hits.sort(key=lambda x: x[0], reverse=True)
    return hits[: int(cfg["limits"]["retrieve_top_k"])]


def format_hits(hits: List[Hit], max_chars: int) -> str:
    out = []
    used = 0
    for i, (score, p) in enumerate(hits, start=1):
        content = (p.get("content") or "").strip()
        src = p.get("source", "?")
        path = p.get("relpath") or p.get("path") or "?"
        block = f"[REF {i}] sim={score:.3f} src={src} file={path}\n{content}\n"
        if used + len(block) > max_chars:
            break
        out.append(block)
        used += len(block)
    return "\n".join(out).strip()
