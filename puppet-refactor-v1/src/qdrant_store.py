from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

Hit = Tuple[float, Dict[str, Any]]


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
        flt = None
        if must_source:
            flt = qm.Filter(must=[qm.FieldCondition(key="source", match=qm.MatchValue(value=must_source))])

        hits = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
            score_threshold=min_sim,
            query_filter=flt,
        )
        return [(float(h.score), dict(h.payload or {})) for h in hits]
