"""Day3 验收：不经过 Agent 类，跑通一轮 tool calling。"""

from __future__ import annotations

import json

import _bootstrap  # noqa: F401

from src.config import get_settings
from src.llm.siliconflow import SiliconFlowClient
from src.tools.registry import ToolRegistry
from src.tools.tavily_search import TavilySearchTool


def main() -> None:
    settings = get_settings()
    if not settings.effective_llm_api_key:
        raise SystemExit("缺少 LLM API Key")
    if not settings.tavily_api_key:
        raise SystemExit("缺少 TAVILY_API_KEY")

    llm = SiliconFlowClient.from_settings(settings)
    registry = ToolRegistry()
    registry.register(
        TavilySearchTool(
            api_key=settings.tavily_api_key,
            max_results=settings.tavily_max_results,
        )
    )

    messages = [
        {
            "role": "system",
            "content": "你是研究助手。需要最新信息时必须调用 tavily_search。",
        },
        {
            "role": "user",
            "content": "用一句话说明 2024 年全球光伏装机的大致趋势，并给出信息来源。",
        },
    ]

    first = llm.chat(messages, tools=registry.to_openai_tools())
    tool_calls = first.get("tool_calls") or []
    print(f"finish_reason tool_calls={len(tool_calls)}")

    if not tool_calls:
        print("模型未调用工具，直接回复：")
        print(first.get("content"))
        return

    messages.append(
        {
            "role": "assistant",
            "content": first.get("content"),
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["arguments"], ensure_ascii=False),
                    },
                }
                for tc in tool_calls
            ],
        }
    )

    for tc in tool_calls:
        print(f"call -> {tc['name']}({tc['arguments']})")
        result = registry.execute(tc["name"], tc["arguments"])
        print(f"      success={result.success}")
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps(result.model_dump(), ensure_ascii=False),
            }
        )

    final = llm.chat(messages)
    print("--- final ---")
    print((final.get("content") or "").strip())
    print("OK: tool calling 一轮完成")


if __name__ == "__main__":
    main()
