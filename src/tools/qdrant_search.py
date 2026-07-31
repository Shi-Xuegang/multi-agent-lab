"""Qdrant 语义检索工具。
本地私有知识库检索，优先查向量库；
外部新知识交给互联网搜索工具，构成「本地知识库优先 + 外网补充」的 RAG 双检索策略
"""

from __future__ import annotations

from typing import Any

from src.memory.qdrant_store import QdrantMemoryStore
from src.models.schemas import SearchResult, ToolExecutionResult
from src.tools.base import Tool


class QdrantSearchTool(Tool):
    """从 Qdrant 中检索已存储文档（RAG 缓存）。"""

    # 不直接在工具内部初始化 Qdrant 客户端，外部传入向量存储实例。
    # 优势：解耦、方便单元测试、全局复用同一个向量库连接。
    def __init__(self, memory_store: QdrantMemoryStore) -> None:
        self._memory = memory_store   # 创建 QdrantMemoryStore 实例，依赖注入

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

            if not str(query).strip(): # 参数合法性校验（query 非空）；非法参数直接返回失败结构化结果
                return ToolExecutionResult(
                    tool_name=self.name,
                    success=False,
                    error="搜索关键词 query 不能为空",
                )

            results = self._memory.search_similar(  # 请求本地检索
                query=query,
                limit=limit,
                score_threshold=score_threshold,
            )
            search_results = [  # 搜索结果列表
                SearchResult(
                    title=result["title"],
                    url=result["url"],
                    content=result["content"],
                    score=result["score"],
                )
                for result in results
            ]
            return ToolExecutionResult(  # 返回成功结构化结果
                tool_name=self.name,
                success=True,
                data={
                    "query": query,
                    "results": [r.model_dump() for r in search_results], # 使用 model_dump() 转为字典，存入 data
                    "result_count": len(search_results),
                },
            )
        except Exception as exc:
            return ToolExecutionResult(
                tool_name=self.name,
                success=False,
                error=f"Qdrant 检索失败: {exc}",
            )
