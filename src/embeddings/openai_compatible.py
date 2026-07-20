"""OpenAI 兼容 Embedding API（阿里云 MaaS compatible-mode 等）。"""

from __future__ import annotations
from openai import OpenAI
from src.embeddings.base import BaseEmbedder

class OpenAICompatibleEmbedder(BaseEmbedder):
    """通过 OpenAI 兼容接口调用嵌入模型。"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        dimension: int,
    ) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        kwargs: dict = {
            "model": self._model,
            "input": texts,
        }
        # 部分模型支持指定输出维度（如 text-embedding-v3）
        if self._dimension:
            kwargs["dimensions"] = self._dimension
        response = self._client.embeddings.create(**kwargs)
        # 按 index 排序，保证与输入顺序一致
        data = sorted(response.data, key=lambda item: item.index)
        vectors = [item.embedding for item in data]
        self._sync_dimension(vectors)
        return vectors

    def _sync_dimension(self, vectors: list[list[float]]) -> None:
        """同步实际向量维度，若 API 返回值与配置不同则自动适配。"""
        if not vectors:
            return
        actual = len(vectors[0])
        # 如果接口返回向量维度与配置不一致，自动覆盖内部维度属性
        if actual != self._dimension:
            self._dimension = actual