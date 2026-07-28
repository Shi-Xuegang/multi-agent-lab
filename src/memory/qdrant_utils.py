"""Qdrant 连接辅助函数。
解决向量维度不匹配致命错误。

若.env 配置向量维度与已有集合向量维度不一致，写入向量直接报错；
该函数启动时自动读取已存在集合的真实向量维度，不存在集合才使用配置默认值。
"""

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

    本地新建集合使用 env 默认维度；
复用历史集合自动读取历史维度，支持项目迭代、切换 Embedding 模型场景
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
