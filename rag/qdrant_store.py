from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


def _field_condition(cond: Dict[str, Any]) -> qm.FieldCondition:
    """
    Convert a simple dict to Qdrant FieldCondition.
    Supports:
      {"key":"source","match":{"value":"internal:module:iptables"}}  (exact)
      {"key":"path","match":{"text":"/opt/llm/llm-test/iptables"}}  (substring)
    """
    key = cond["key"]
    m = cond["match"]
    if "value" in m:
        return qm.FieldCondition(key=key, match=qm.MatchValue(value=m["value"]))
    if "text" in m:
        return qm.FieldCondition(key=key, match=qm.MatchText(text=m["text"]))
    raise ValueError(f"Unsupported match in condition: {cond}")


class QdrantStore:
    def __init__(self, url: str, collection: str):
        self.client = QdrantClient(url=url)
        self.collection = collection

    def ensure_collection(self, dim: int) -> None:
        """
        Create collection if missing, using cosine distance.
        """
        cols = self.client.get_collections().collections
        if any(c.name == self.collection for c in cols):
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )

    def make_point(
        self,
        source: str,
        path: str,
        chunk_index: int,
        content: str,
        content_hash: str,
        vector: List[float],
    ) -> qm.PointStruct:
        """
        Create a Qdrant point with payload for RAG.
        """
        point_id = content_hash  # stable id, overwritten on upsert
        payload = {
            "source": source,
            "path": path,
            "chunk_index": chunk_index,
            "content": content,
            "content_hash": content_hash,
        }
        return qm.PointStruct(id=point_id, vector=vector, payload=payload)

    def upsert_points(self, points: List[qm.PointStruct]) -> None:
        self.client.upsert(collection_name=self.collection, points=points)

    def count(self, must: Optional[List[Dict[str, Any]]] = None) -> int:
        """
        Count points, optionally filtered by must conditions.
        """
        flt = None
        if must:
            flt = qm.Filter(must=[_field_condition(c) for c in must])
        res = self.client.count(collection_name=self.collection, count_filter=flt, exact=True)
        return int(res.count)

    def search(
        self,
        query_vector: List[float],
        top_k: int,
        min_sim: float,
        must: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Vector search with optional must filter.
        Returns list of (score, payload).
        """
        flt = None
        if must:
            flt = qm.Filter(must=[_field_condition(c) for c in must])

        hits = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
            score_threshold=min_sim,
            query_filter=flt,
        )

        out: List[Tuple[float, Dict[str, Any]]] = []
        for h in hits:
            out.append((float(h.score), dict(h.payload or {})))
        return out
