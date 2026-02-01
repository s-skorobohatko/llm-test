from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


def _field_condition(cond: Dict[str, Any]) -> qm.FieldCondition:
    """
    Convert a simple dict to Qdrant FieldCondition.
    Supports:
      {"key":"source","match":{"value":"vendor:puppetlabs:best-practices"}}  (exact)
      {"key":"path","match":{"text":"/opt/llm/llm-test/iptables"}}           (substring)
    """
    key = cond["key"]
    m = cond["match"]
    if "value" in m:
        return qm.FieldCondition(key=key, match=qm.MatchValue(value=m["value"]))
    if "text" in m:
        return qm.FieldCondition(key=key, match=qm.MatchText(text=m["text"]))
    raise ValueError(f"Unsupported match in condition: {cond}")


def _make_filter(must: Optional[List[Dict[str, Any]]]) -> Optional[qm.Filter]:
    if not must:
        return None
    return qm.Filter(must=[_field_condition(c) for c in must])


class QdrantStore:
    def __init__(self, url: str, collection: str):
        self.client = QdrantClient(url=url)
        self.collection = collection

    def ensure_collection(self, dim: int) -> None:
        cols = self.client.get_collections().collections
        if any(c.name == self.collection for c in cols):
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )

    def upsert_points(self, points: List[qm.PointStruct]) -> None:
        self.client.upsert(collection_name=self.collection, points=points)

    def count(self, must: Optional[List[Dict[str, Any]]] = None) -> int:
        flt = _make_filter(must)
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
        Return list of (score, payload).
        Compatible with multiple qdrant-client API variants.
        """
        flt = _make_filter(must)

        # Variant A: classic client.search(...)
        if hasattr(self.client, "search"):
            hits = self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
                score_threshold=min_sim,
                query_filter=flt,
            )
            return [(float(h.score), dict(h.payload or {})) for h in hits]

        # Variant B: newer client.query_points(...)
        # Some versions use "query" or "vector" naming; we handle both by trying.
        if hasattr(self.client, "query_points"):
            try:
                res = self.client.query_points(
                    collection_name=self.collection,
                    query=query_vector,            # some versions
                    limit=top_k,
                    with_payload=True,
                    score_threshold=min_sim,
                    query_filter=flt,
                )
            except TypeError:
                # alternate signature
                res = self.client.query_points(
                    collection_name=self.collection,
                    vector=query_vector,           # some versions
                    limit=top_k,
                    with_payload=True,
                    score_threshold=min_sim,
                    query_filter=flt,
                )

            # res can be object with .points or list-like depending on version
            points = getattr(res, "points", res)
            out = []
            for p in points:
                score = getattr(p, "score", None)
                payload = getattr(p, "payload", None)
                if score is None and isinstance(p, dict):
                    score = p.get("score")
                    payload = p.get("payload")
                out.append((float(score), dict(payload or {})))
            return out

        # Variant C: client.search_points(...)
        if hasattr(self.client, "search_points"):
            try:
                res = self.client.search_points(
                    collection_name=self.collection,
                    vector=query_vector,
                    limit=top_k,
                    with_payload=True,
                    score_threshold=min_sim,
                    query_filter=flt,
                )
            except TypeError:
                res = self.client.search_points(
                    collection_name=self.collection,
                    query_vector=query_vector,
                    limit=top_k,
                    with_payload=True,
                    score_threshold=min_sim,
                    query_filter=flt,
                )

            points = getattr(res, "points", res)
            out = []
            for p in points:
                score = getattr(p, "score", None)
                payload = getattr(p, "payload", None)
                out.append((float(score), dict(payload or {})))
            return out

        # Variant D: very old low-level HTTP API
        req = qm.SearchRequest(
            vector=query_vector,
            limit=top_k,
            with_payload=True,
            score_threshold=min_sim,
            filter=flt,
        )
        hits = self.client.http.search(
            collection_name=self.collection,
            search_request=req,
        )
        return [(float(h.score), dict(h.payload or {})) for h in hits]
