"""Qdrant 向量记忆存储。
入库 + 语义检索 + URL 去重，支持分块与非分块两种模式。

RAG / 缓存的底座，负责存储、检索、去重。"""

from __future__ import annotations

import hashlib  # 做 URL 去重哈希
import uuid  # 生成唯一 ID
from datetime import datetime, timezone  # 时间戳、时区处理
from typing import Any

from qdrant_client import QdrantClient  # Qdrant 客户端
from qdrant_client.models import (
    Distance,  # 距离度量枚举
    FieldCondition,  # 字段条件
    Filter,  # 过滤条件
    MatchValue,  # 匹配值
    PointStruct,  # 点结构
    VectorParams,  # 向量参数
)

from src.embeddings.base import BaseEmbedder  # 基础嵌入器接口
from src.memory.chunk_utils import (
    ChunkingStrategy,  # 分块策略基类
    FixedLengthChunker,  # 固定长度分块器
    RecursiveChunker,  # 递归分块器
    SemanticChunker,  # 语义分块器
    chunk_search_results,  # 分块搜索结果工具函数
)
from src.models.schemas import SearchResult  # 搜索结果模式

_DISTANCE_MAP: dict[str, Distance] = {  # 距离度量映射
    "cosine": Distance.COSINE,  # 余弦距离
    "euclid": Distance.EUCLID,  # 欧氏距离
    "dot": Distance.DOT,  # 点积距离
}  # 默认使用余弦距离


def build_chunker(strategy: str = "semantic") -> ChunkingStrategy | None:
    """按名称构建分块策略；none 表示不分块。"""
    # 解耦设计：向量存储层无需关心分块实现，直接调用工具函数完成文档预处理。
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
    def __init__( # 初始化
        self,
        url: str,
        collection_name: str,
        embedder: BaseEmbedder,   # （文本 → 向量）
        api_key: str = "",   # Qdrant API 密钥
        vector_size: int = 1024,   # 向量维度
        distance: str = "cosine",   # 距离度量
        timeout: int = 30,   # 超时时间
        chunk_strategy: str = "semantic",   # 分块策略
    ) -> None:
        client_kwargs: dict[str, Any] = {   # 客户端配置参数
            "url": url,
            "timeout": timeout,
        }
        if api_key: # 区分有无密钥
            client_kwargs["api_key"] = api_key

        self._client = QdrantClient(**client_kwargs)
        self._collection = collection_name   # 集合名称
        self._embedder = embedder   # （文本 → 向量）
        self._vector_size = vector_size   # 向量维度
        self._distance = _DISTANCE_MAP.get(distance.lower(), Distance.COSINE)  # 距离度量
        self._chunk_strategy = chunk_strategy   # 分块策略
        self._chunker = build_chunker(chunk_strategy)   # 构建分块器
        self._ensure_collection()   # 确保集合存在

    def _ensure_collection(self) -> None:  # 懒创建
        """确保 collection 存在，不存在则按配置创建。"""
        collections = [c.name for c in self._client.get_collections().collections]
        if self._collection not in collections:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams( # 向量配置
                    size=self._vector_size,
                    distance=self._distance,
                ),
            )

    @staticmethod   # 无实例依赖，仅处理纯字符串哈希，标记静态方法节省内存
    def _url_hash(url: str) -> str:
        """URL 哈希，用于去重。"""
        return hashlib.md5(url.encode()).hexdigest() # md5 哈希压缩 URL 用于过滤查重，减少索引存储占用

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

        # 过滤已存在 URL与空内容
        fresh: list[SearchResult] = []
        for result in results:
            if not result.content.strip():
                continue
            if result.url and self.url_exists(result.url):
                continue
            fresh.append(result)

        if not fresh:
            return 0
        # 判断是否分块 动态控制分块开关，支持运行时强制开启 / 关闭分块
        do_chunk = self._chunker is not None if chunk is None else chunk
        if do_chunk and self._chunker is not None:
            to_store = chunk_search_results(fresh, self._chunker)
        else:
            to_store = fresh
        # 批量构造文本、载荷  作为嵌入文本，提升向量语义质量
        points: list[PointStruct] = []
        texts_to_embed: list[str] = []
        # 批量构造载荷 payload：存储原始查询、会话 ID、入库时间、网页元数据，用于过滤检索
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
                    "stored_at": datetime.now(timezone.utc).isoformat(), # 标准 UTC 时间入库，避免本地时区混乱，多服务器部署时间统一
                }
            )

        if not texts_to_embed:
            return 0
        # 批量嵌入embed_batch，相比逐条嵌入大幅提升速度
        vectors = self._embedder.embed_batch(texts_to_embed)
        # 构造向量点
        # strict=True 严格 zip 并行遍历：vectors 与 payloads 长度必须完全匹配
        for vector, payload in zip(vectors, payloads, strict=True):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=payload,
                )
            )
        # upsert 批量写入 Qdrant
        self._client.upsert(collection_name=self._collection, points=points)
        return len(points) # 返回实际入库条数

    def search_similar( # 语义检索
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.7,
    ) -> list[dict[str, Any]]:
        """语义检索与 query 相似的已存储文档。"""
        query_vector = self._embedder.embed_text(query) #  把查询变成向量

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
            for hit in hits # 隐藏底层 PointStruct 结构体
            if hit.payload 
        ]
