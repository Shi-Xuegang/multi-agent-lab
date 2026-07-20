"""DashScope SDK 文本嵌入服务。"""

from __future__ import annotations
import dashscope
from dashscope import TextEmbedding
from src.embeddings.base import BaseEmbedder

class DashScopeEmbedder(BaseEmbedder):
    def __init__(self, api_key: str, model: str = "text-embedding-v3", dimension: int = 1024) -> None:
        dashscope.api_key = api_key
        self._model = model
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # 空输入兜底
        if not texts:
            return []
        # 调用阿里云向量接口
        response = TextEmbedding.call(
            model=self._model,
            input=texts,
            text_type="document",
            dimension=self._dimension,
        )
        # 请求失败抛出运行时异常
        if response.status_code != 200:
            raise RuntimeError(f"DashScope 嵌入失败: {response.code} - {response.message}")
        
        # 根据text_index重排序，保证向量顺序与输入文本完全对应
        embeddings = sorted(response.output["embeddings"], key=lambda item: item["text_index"])
        return [item["embedding"] for item in embeddings]  # 列表推导式批量提取向量