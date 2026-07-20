"""Day2 验收：Tavily 搜索连通性。"""

from __future__ import annotations

import _bootstrap  # noqa: F401

from src.config import get_settings
from src.tools.tavily_search import TavilySearchTool


def main() -> None:
    settings = get_settings()
    if not settings.tavily_api_key:
        raise SystemExit("缺少 TAVILY_API_KEY")

    tool = TavilySearchTool(
        api_key=settings.tavily_api_key,
        max_results=3,
    )
    result = tool.execute(query="全球光伏装机量 2024", search_depth="basic")
    if not result.success:
        raise SystemExit(f"Tavily 失败: {result.error}")

    data = result.data or {}
    print(f"query  = {data.get('query')}")
    print(f"count  = {data.get('result_count')}")
    for i, item in enumerate(data.get("results", [])[:3], 1):
        print(f"[{i}] {item.get('title')} | {item.get('url')}")
    print("OK: Tavily 连通")


if __name__ == "__main__":
    main()
