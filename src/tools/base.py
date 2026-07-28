"""通用 Tool 基类，定义工具接口规范。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.models.schemas import ToolExecutionResult


class Tool(ABC):
    """
    所有 Agent 工具的抽象基类。

    后续扩展（代码执行器、数据查询器等）只需实现同一接口并注册即可。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """工具唯一标识，供 LLM function calling 使用。"""

    @property
    @abstractmethod
    def description(self) -> str:
        """工具功能描述，帮助模型决定是否调用。"""

    @property
    @abstractmethod
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema 格式的参数定义。"""

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolExecutionResult:
        """执行工具逻辑并返回结构化结果。"""

    def to_openai_tool(self) -> dict[str, Any]:
        """转换为 OpenAI 兼容的 function calling 格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }

    def __repr__(self) -> str:
        """友好打印，调试时快速查看工具名称和描述。"""
        return f"<Tool name={self.name!r}>"
