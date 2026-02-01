import time
import uuid
import requests
from typing import Any, Dict, List, Optional, Tuple


def _point_id(path: str, chunk_index: int, content_hash: str) -> str:
    s = f"{path}#{chunk_index}#{content_hash}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, s))


class QdrantStore:
    def __init__(self, url: str, collection: str):
        self.url = url.rstrip("/")
        self.collection = collection

    def _c(self, suffix: str) -> str:
        return f"{self.url}/collections/{self.collection}{suffix}"

    def collection_exists(self) -> bool:
        r = requests.get(f"{self.url}/collections/{self.collection}", timeout=30)
        return r.status_code == 200

    def create_collection(self, dim: int) -> None:
        payload = {"vectors": {"size": dim, "distance": "Cosine"}}
        r = requests.put(f"{self.url}/collections/{self.collection}", json=payload, timeout=60)
        r.raise_for_status()

    def ensure_collection(self, dim: int) -> None:
        if not self.collection_exists():
            self.create_collection(dim)

    def upsert_points(self, points: List[Dict[str, Any]]) -> None:
        r = requests.put(self._c("/points"), json={"points": points}, timeout=180)
        r.raise_for_status()

    def search(
        self,
        query_vector: List[float],
        top_k: int,
        min_sim: float,
        must: Optional[List[Dict[str, Any]]] = None,
        should: Optional[List[Dict[str, Any]]] = None,
        must_not: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        q: Dict[str, Any] = {
            "vector": query_vector,
            "limit": top_k,
            "with_payload": True,
        }
        filt: Dict[str, Any] = {}
        if must:
            filt["must"] = must
        if should:
            filt["should"] = should
        if must_not:
            filt["must_not"] = must_not
        if filt:
            q["filter"] = filt

        r = requests.post(self._c("/points/search"), json=q, timeout=120)
        r.raise_for_status()
        data = r.json()

        out: List[Tuple[float, Dict[str, Any]]] = []
        for item in data.get("result", []):
            score = float(item.get("score", 0.0))
            if score < min_sim:
                continue
            payload = item.get("payload") or {}
            out.append((score, payload))
        return out

    def make_point(
        self,
        source: str,
        path: str,
        chunk_index: int,
        content: str,
        content_hash: str,
        vector: List[float],
    ) -> Dict[str, Any]:
        return {
            "id": _point_id(path, chunk_index, content_hash),
            "vector": vector,
            "payload": {
                "source": source,
                "path": path,
                "chunk_index": chunk_index,
                "content": content,
                "content_hash": content_hash,
                "created_at": int(time.time()),
            },
        }
