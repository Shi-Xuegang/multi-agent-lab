"""SiliconFlow LLM 客户端，兼容 OpenAI API。

提供与 SiliconFlow API 交互的客户端，支持对话、JSON 输出等功能。

包括：
- 对话功能：发送对话请求，获取对话响应
- JSON 输出功能：发送 JSON 请求，获取 JSON 响应
- 工具调用功能：支持工具调用，获取工具调用响应
- 知识图谱抽取功能：支持知识图谱抽取，获取知识图谱抽取响应
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from openai import OpenAI

if TYPE_CHECKING:  # 静态检查专用导入：运行时跳过导入，规避循环导入
    from src.config import Settings

# 与 .env 中 SILICONFLOW_MODEL_ID / LLM_MODEL_ID 保持一致
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash"


class SiliconFlowClient:
    """封装 SiliconFlow Chat Completions API，所有 LLM 推理统一走此类。"""

    def __init__(
        self,
        api_key: str, # API 密钥
        base_url: str = DEFAULT_BASE_URL, # 默认 API 地址
        model: str = DEFAULT_MODEL_ID, # 默认模型 ID
    ) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url) # 创建 OpenAI 客户端，传入 API 密钥和 API 地址
        self._model = model # 当前使用的模型 ID
        self._base_url = base_url # 当前使用的 API 地址

    @classmethod   # 类方法，可以直接通过类名调用，无需创建实例
    def from_settings(cls, settings: Settings) -> SiliconFlowClient:
        """从 .env 配置创建客户端（推荐方式）。
        从 .env 配置文件中获取 API 密钥、API 地址和模型 ID，创建 SiliconFlowClient 实例。
        """
        return cls(
            api_key=settings.effective_llm_api_key,
            base_url=settings.effective_llm_base_url,
            model=settings.effective_llm_model_id,
        )

    @property
    def model(self) -> str:
        """当前使用的模型 ID。"""
        return self._model

    @property
    def base_url(self) -> str:
        """当前使用的 API 地址。"""
        return self._base_url

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,  # 工具 Function Schema 列表，None 代表不启用工具调用
        temperature: float = 0.3,  # 生成随机性，默认 0.3（研究场景低温度，保证严谨）
    ) -> dict[str, Any]:
        """
        发送对话请求。
        返回包含 content、tool_calls、raw_message 的字典。
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools: # 如果传入 tools，自动追加 tools 参数 + tool_choice="auto"，交给模型自主判断是否调用工具
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs) # 发起网络请求
        message = response.choices[0].message  # 提取模型返回第一条消息 

        tool_calls = None
        if message.tool_calls: 
            tool_calls = []
            for tc in message.tool_calls:
                raw_args = tc.function.arguments or "{}"  # 取出 function 参数字符串，空则填充{}
                try:
                    arguments = json.loads(raw_args)
                except json.JSONDecodeError:
                    arguments = {}
                if not isinstance(arguments, dict): # 强制校验参数必须是字典
                    arguments = {}
                tool_calls.append(
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": arguments,
                    }
                )

        return {
            "content": message.content, 
            "tool_calls": tool_calls, 
            "raw_message": message, # 原生 OpenAI 消息对象（用于调试）
            "finish_reason": response.choices[0].finish_reason,  # 结束原因（stop/tool_calls）
        }

    def chat_json(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        """请求模型输出 JSON，并解析为字典。"""
        enhanced_messages = messages + [
            {
                "role": "user",
                "content": "请严格以 JSON 格式输出，不要包含 markdown 代码块标记。",
            }
        ]

        result = self.chat(enhanced_messages, temperature=temperature)  # 复用自身 chat() 方法完成 LLM 推理
        content = result["content"] or ""

        # 清理可能的 markdown 代码块包裹
        cleaned = content.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            cleaned = "\n".join(lines)

        try:
            return json.loads(cleaned)  # 解析字符串为字典
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM 返回的内容不是有效 JSON: {content}") from exc
