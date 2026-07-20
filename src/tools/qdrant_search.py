"""Qdrant 语义检索工具。"""

from __future__ import annotations

from typing import Any

from src.memory.qdrant_store import QdrantMemoryStore
from src.models.schemas import SearchResult, ToolExecutionResult
from src.tools.base import Tool


class QdrantSearchTool(Tool):
    """从 Qdrant 中检索已存储文档（RAG 缓存）。"""

    def __init__(self, memory_store: QdrantMemoryStore) -> None:
        self._memory = memory_store

    @property
    def name(self) -> str:
        return "qdrant_search"

    @property
    def description(self) -> str:
        return "通过语义检索已存储在 Qdrant 中的文档，进行去重缓存。"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词，用于语义检索已存文档",
                },
                "limit": {
                    "type": "integer",
                    "description": "返回相似文档的数量上限",
                    "default": 5,
                },
                "score_threshold": {
                    "type": "number",
                    "description": "相似度阈值，低于此值的文档将被忽略",
                    "default": 0.7,
                },
            },
            "required": ["query"],
        }

    def execute(self, **kwargs: Any) -> ToolExecutionResult:
        try:
            query = kwargs.get("query", "")
            limit = kwargs.get("limit", 5)
            score_threshold = kwargs.get("score_threshold", 0.7)

            if not str(query).strip():
                return ToolExecutionResult(
                    tool_name=self.name,
                    success=False,
                    error="搜索关键词 query 不能为空",
                )

            results = self._memory.search_similar(
                query=query,
                limit=limit,
                score_threshold=score_threshold,
            )
            search_results = [
                SearchResult(
                    title=result["title"],
                    url=result["url"],
                    content=result["content"],
                    score=result["score"],
                )
                for result in results
            ]
            return ToolExecutionResult(
                tool_name=self.name,
                success=True,
                data={
                    "query": query,
                    "results": [r.model_dump() for r in search_results],
                    "result_count": len(search_results),
                },
            )
        except Exception as exc:
            return ToolExecutionResult(
                tool_name=self.name,
                success=False,
                error=f"Qdrant 检索失败: {exc}",
            )
