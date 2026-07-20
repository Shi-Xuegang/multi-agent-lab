"""Qdrant 向量记忆存储。"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from src.embeddings.base import BaseEmbedder
from src.memory.chunk_utils import (
    ChunkingStrategy,
    FixedLengthChunker,
    RecursiveChunker,
    SemanticChunker,
    chunk_search_results,
)
from src.models.schemas import SearchResult

_DISTANCE_MAP: dict[str, Distance] = {
    "cosine": Distance.COSINE,
    "euclid": Distance.EUCLID,
    "dot": Distance.DOT,
}


def build_chunker(strategy: str = "semantic") -> ChunkingStrategy | None:
    """按名称构建分块策略；none 表示不分块。"""
    if strategy == "none":
        return None
    mapping: dict[str, ChunkingStrategy] = {
        "fixed": FixedLengthChunker(chunk_size=512, overlap=64),
        "semantic": SemanticChunker(max_chunk_size=1024),
        "recursive": RecursiveChunker(max_chunk_size=1024),
    }
    return mapping.get(strategy, mapping["semantic"])


class QdrantMemoryStore:
    """将搜索文档向量化后存入 Qdrant，支持语义检索与 URL 去重。"""

    def __init__(
        self,
        url: str,
        collection_name: str,
        embedder: BaseEmbedder,
        api_key: str = "",
        vector_size: int = 1024,
        distance: str = "cosine",
        timeout: int = 30,
        chunk_strategy: str = "semantic",
    ) -> None:
        client_kwargs: dict[str, Any] = {
            "url": url,
            "timeout": timeout,
        }
        if api_key:
            client_kwargs["api_key"] = api_key

        self._client = QdrantClient(**client_kwargs)
        self._collection = collection_name
        self._embedder = embedder
        self._vector_size = vector_size
        self._distance = _DISTANCE_MAP.get(distance.lower(), Distance.COSINE)
        self._chunk_strategy = chunk_strategy
        self._chunker = build_chunker(chunk_strategy)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """确保 collection 存在，不存在则按配置创建。"""
        collections = [c.name for c in self._client.get_collections().collections]
        if self._collection not in collections:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=self._distance,
                ),
            )

    @staticmethod
    def _url_hash(url: str) -> str:
        """URL 哈希，用于去重。"""
        return hashlib.md5(url.encode()).hexdigest()

    def url_exists(self, url: str) -> bool:
        """检查 URL 是否已入库。"""
        if not url:
            return False
        url_hash = self._url_hash(url)
        try:
            points, _ = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="url_hash",
                            match=MatchValue(value=url_hash),
                        )
                    ]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
            return len(points) > 0
        except Exception:
            return False

    def store_search_results(
        self,
        query: str,
        results: list[SearchResult],
        session_id: str = "default",
        chunk: bool | None = None,
    ) -> int:
        """
        将搜索结果存入 Qdrant。

        - 相同 URL 不重复入库
        - 可选按策略分块后入库
        返回实际入库条数。
        """
        if not results:
            return 0

        # 过滤已存在 URL
        fresh: list[SearchResult] = []
        for result in results:
            if not result.content.strip():
                continue
            if result.url and self.url_exists(result.url):
                continue
            fresh.append(result)

        if not fresh:
            return 0

        do_chunk = self._chunker is not None if chunk is None else chunk
        if do_chunk and self._chunker is not None:
            to_store = chunk_search_results(fresh, self._chunker)
        else:
            to_store = fresh

        points: list[PointStruct] = []
        texts_to_embed: list[str] = []
        payloads: list[dict[str, Any]] = []

        for result in to_store:
            if not result.content.strip():
                continue
            text = f"{result.title}\n{result.content}"
            texts_to_embed.append(text)
            payloads.append(
                {
                    "query": query,
                    "title": result.title,
                    "url": result.url,
                    "content": result.content,
                    "score": result.score,
                    "url_hash": self._url_hash(result.url),
                    "session_id": session_id,
                    "stored_at": datetime.now(timezone.utc).isoformat(),
                }
            )

        if not texts_to_embed:
            return 0

        vectors = self._embedder.embed_batch(texts_to_embed)

        for vector, payload in zip(vectors, payloads, strict=True):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=payload,
                )
            )

        self._client.upsert(collection_name=self._collection, points=points)
        return len(points)

    def search_similar(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.7,
    ) -> list[dict[str, Any]]:
        """语义检索与 query 相似的已存储文档。"""
        query_vector = self._embedder.embed_text(query)

        # qdrant-client >=1.12 起逐步废弃 search，1.18 已移除；统一走 query_points
        if hasattr(self._client, "query_points"):
            response = self._client.query_points(
                collection_name=self._collection,
                query=query_vector,
                limit=limit,
                score_threshold=score_threshold,
            )
            hits = response.points
        else:
            hits = self._client.search(
                collection_name=self._collection,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
            )

        return [
            {
                "title": hit.payload.get("title", ""),
                "url": hit.payload.get("url", ""),
                "content": hit.payload.get("content", ""),
                "score": hit.score,
                "original_query": hit.payload.get("query", ""),
            }
            for hit in hits
            if hit.payload
        ]
