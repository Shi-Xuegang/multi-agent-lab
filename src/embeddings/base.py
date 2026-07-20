"""嵌入向量抽象接口。"""

from __future__ import annotations
from abc import ABC, abstractmethod

# 所有向量生成器必须继承并实现规定方法，实现多嵌入模型可无缝互换
class BaseEmbedder(ABC):
    """嵌入模型统一接口。"""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """向量维度，需与 QDRANT_VECTOR_SIZE 一致。"""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """对单条文本生成嵌入向量。"""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量生成嵌入向量。"""
