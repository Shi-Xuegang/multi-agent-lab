"""嵌入模型工厂。"""

from __future__ import annotations

from src.config import Settings
from src.embeddings.base import BaseEmbedder
from src.embeddings.dashscope import DashScopeEmbedder
from src.embeddings.openai_compatible import OpenAICompatibleEmbedder


def create_embedder(settings: Settings, dimension: int | None = None) -> BaseEmbedder:
    """
    根据 .env 中的 EMBED_* 配置创建嵌入模型。

    优先级：
    1. 配置了 EMBED_BASE_URL → OpenAI 兼容接口（阿里云 MaaS）
    2. EMBED_MODEL_TYPE=dashscope → DashScope SDK
    """
    vector_size = dimension or settings.qdrant_vector_size

    # 优先级1：配置兼容接口地址，使用OpenAI通用客户端
    if settings.embed_base_url:
        if not settings.embed_api_key:
            raise EnvironmentError("已配置 EMBED_BASE_URL，但缺少 EMBED_API_KEY")
        return OpenAICompatibleEmbedder(
            api_key=settings.embed_api_key,
            base_url=settings.embed_base_url,
            model=settings.embed_model_name,
            dimension=vector_size,
        )

    # 优先级2：指定dashscope厂商，使用专属SDK
    if settings.embed_model_type == "dashscope":
        if not settings.embed_api_key:
            raise EnvironmentError("EMBED_MODEL_TYPE=dashscope 时需要配置 EMBED_API_KEY")
        return DashScopeEmbedder(
            api_key=settings.embed_api_key,
            model=settings.embed_model_name,
            dimension=vector_size,
        )

    raise EnvironmentError(
        f"不支持的 EMBED_MODEL_TYPE: {settings.embed_model_type}，"
        f"请设置为 dashscope 或配置 EMBED_BASE_URL。"
    )
