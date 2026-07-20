"""Qdrant 连接辅助函数。"""

from __future__ import annotations

from qdrant_client import QdrantClient


def resolve_qdrant_vector_size(
    url: str,
    collection_name: str,
    default_size: int,
    api_key: str = "",
    timeout: int = 30,
) -> int:
    """
    解析向量维度：若 collection 已存在则读取其配置，否则使用默认值。

    避免 .env 中 QDRANT_VECTOR_SIZE 与已有 collection / 嵌入模型不一致。
    """
    try:
        client_kwargs: dict = {"url": url, "timeout": timeout}
        if api_key:
            client_kwargs["api_key"] = api_key

        client = QdrantClient(**client_kwargs)
        collections = [c.name for c in client.get_collections().collections]
        if collection_name in collections:
            info = client.get_collection(collection_name)
            return info.config.params.vectors.size
    except Exception as exc:
        # 连接失败时回退默认维度，但保留可观测信息
        print(f"[qdrant] 无法读取 collection 维度，使用默认 {default_size}: {exc}")

    return default_size
