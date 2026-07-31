"""Tavily 互联网搜索工具。
第三方 SDK 封装模式：把外部 API 客户端包装在工具内部，隔离外部依赖
"""

from __future__ import annotations

from typing import Any

from tavily import TavilyClient

from src.models.schemas import SearchResult, ToolExecutionResult
from src.tools.base import Tool


class TavilySearchTool(Tool):
    """通过 Tavily API 搜索互联网信息。"""

    def __init__(self, api_key: str, max_results: int = 5) -> None:
        self._client = TavilyClient(api_key=api_key)   # 创建 TavilyClient 实例，封装
        self._max_results = max_results

    @property
    def name(self) -> str:
        return "tavily_search"

    @property
    def description(self) -> str:
        return (
            "搜索互联网获取最新信息。"
            "适用于需要实时数据、新闻、研究报告、技术文档等场景。"
            "输入搜索关键词，返回相关网页标题、摘要和链接。"
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词或问题，建议使用英文或中文精确描述",
                },
                "search_depth": {
                    "type": "string",
                    "enum": ["basic", "advanced"],
                    "description": "搜索深度：basic 快速搜索，advanced 深度搜索",
                },
            },
            "required": ["query"],
        }

    def execute(self, **kwargs: Any) -> ToolExecutionResult:
        query: str = kwargs.get("query", "")
        search_depth: str = kwargs.get("search_depth", "basic")

        if not query.strip():
            return ToolExecutionResult(
                tool_name=self.name,
                success=False,
                error="搜索关键词 query 不能为空",
            )

        try:
            response = self._client.search(  # 请求外网搜索
                query=query,
                search_depth=search_depth,
                max_results=self._max_results,
                include_answer=False,   # 只要网页结果列表，不要 Tavily 额外生成的综合答案
            )

            results: list[SearchResult] = [] # 搜索结果列表
            for item in response.get("results", []):
                results.append(
                    SearchResult(
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        content=item.get("content", ""),
                        score=float(item.get("score", 0.0)),
                    )
                )

            return ToolExecutionResult(  # 返回成功结构化结果
                tool_name=self.name,
                success=True,
                data={
                    "query": query,
                    "results": [r.model_dump() for r in results], # 使用 model_dump() 转为字典，存入 data
                    "result_count": len(results),
                },
            )
        except Exception as exc:  # noqa: BLE001
            return ToolExecutionResult(
                tool_name=self.name,
                success=False,
                error=f"Tavily 搜索失败: {exc}",
            )
