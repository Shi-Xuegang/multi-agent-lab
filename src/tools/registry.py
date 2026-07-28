"""工具注册表，管理所有可用工具。

充当 Agent 和各个 Tool 之间的中间层，负责工具的注册、查找和执行。

"""

from __future__ import annotations

from src.models.schemas import ToolExecutionResult
from src.tools.base import Tool


class ToolRegistry:
    """工具注册与查找中心。"""

    def __init__(self) -> None:
        # key = 工具name，value = 工具实例，全局唯一索引
        self._tools: dict[str, Tool] = {}  # 字典充当内存注册表，Dict[str, Tool] 实现简易注册器模式

    def register(self, tool: Tool) -> None:
        """注册一个工具实例。"""
        if tool.name in self._tools: # 工具名已存在，抛出异常
            raise ValueError(f"工具 '{tool.name}' 已注册")
        self._tools[tool.name] = tool # 工具名不存在，注册工具实例

    def get(self, name: str) -> Tool | None:
        """按名称获取工具。"""
        return self._tools.get(name) # 工具名存在，返回工具实例；工具名不存在，返回 None

    def list_tools(self) -> list[Tool]:
        """返回所有已注册工具。"""
        return list(self._tools.values()) 

    def to_openai_tools(self) -> list[dict]:
        """批量转换为 OpenAI function calling 格式。"""
        return [tool.to_openai_tool() for tool in self._tools.values()]

    def execute(self, name: str, arguments: dict) -> ToolExecutionResult:
        """查找并执行指定工具。"""
        tool = self.get(name)
        if tool is None:
            return ToolExecutionResult(
                tool_name=name,
                success=False,
                error=f"未找到工具: {name}",
            )
        try:
            return tool.execute(**arguments)
        except Exception as exc:  # noqa: BLE001 — 工具层统一捕获并回传
            return ToolExecutionResult(
                tool_name=name,
                success=False,
                error=str(exc),
            )