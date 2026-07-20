"""嵌入向量模块。"""

from src.embeddings.base import BaseEmbedder
from src.embeddings.dashscope import DashScopeEmbedder
from src.embeddings.factory import create_embedder
from src.embeddings.openai_compatible import OpenAICompatibleEmbedder

__all__ = [
    "BaseEmbedder",
    "DashScopeEmbedder",
    "OpenAICompatibleEmbedder",
    "create_embedder",
]
