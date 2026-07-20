"""工具注册表测试。"""

from src.models.schemas import ToolExecutionResult
from src.tools.base import Tool
from src.tools.registry import ToolRegistry


class _EchoTool(Tool):
    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "echo tool"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

    def execute(self, **kwargs):
        return ToolExecutionResult(
            tool_name=self.name,
            success=True,
            data={"echo": kwargs.get("text")},
        )


def test_register_and_execute():
    registry = ToolRegistry()
    registry.register(_EchoTool())
    result = registry.execute("echo", {"text": "hi"})
    assert result.success
    assert result.data["echo"] == "hi"


def test_missing_tool():
    registry = ToolRegistry()
    result = registry.execute("nope", {})
    assert not result.success


def test_openai_schema():
    registry = ToolRegistry()
    registry.register(_EchoTool())
    tools = registry.to_openai_tools()
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "echo"
