from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

Hit = Tuple[float, Dict[str, Any]]


def _make_filter(must_source: Optional[str]) -> Optional[qm.Filter]:
    if not must_source:
        return None
    return qm.Filter(
        must=[qm.FieldCondition(key="source", match=qm.MatchValue(value=must_source))]
    )


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

    def upsert(self, points: List[qm.PointStruct]) -> None:
        self.client.upsert(collection_name=self.collection, points=points)

    def search(
        self,
        query_vector: List[float],
        *,
        top_k: int,
        min_sim: float = 0.10,
        must_source: Optional[str] = None,
    ) -> List[Hit]:
        """
        Return list of (score, payload). Works across qdrant-client versions.

        Newer versions often support: client.search(...)
        Others use: client.query_points(...)
        """
        flt = _make_filter(must_source)

        # 1) Try .search() (some versions)
        if hasattr(self.client, "search"):
            hits = self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                limit=int(top_k),
                with_payload=True,
                score_threshold=float(min_sim),
                query_filter=flt,
            )
            return [(float(h.score), dict(h.payload or {})) for h in hits]

        # 2) Try .query_points() (newer API in other versions)
        if hasattr(self.client, "query_points"):
            try:
                res = self.client.query_points(
                    collection_name=self.collection,
                    query=query_vector,
                    limit=int(top_k),
                    with_payload=True,
                    score_threshold=float(min_sim),
                    query_filter=flt,
                )
            except TypeError:
                # some versions use vector= instead of query=
                res = self.client.query_points(
                    collection_name=self.collection,
                    vector=query_vector,
                    limit=int(top_k),
                    with_payload=True,
                    score_threshold=float(min_sim),
                    query_filter=flt,
                )

            points = getattr(res, "points", res)
            out: List[Hit] = []
            for p in points:
                score = getattr(p, "score", None)
                payload = getattr(p, "payload", None)
                if score is None and isinstance(p, dict):
                    score = p.get("score")
                    payload = p.get("payload")
                out.append((float(score), dict(payload or {})))
            return out

        # 3) Last resort: HTTP API wrapper
        req = qm.SearchRequest(
            vector=query_vector,
            limit=int(top_k),
            with_payload=True,
            score_threshold=float(min_sim),
            filter=flt,
        )
        hits = self.client.http.search(collection_name=self.collection, search_request=req)
        return [(float(h.score), dict(h.payload or {})) for h in hits]
