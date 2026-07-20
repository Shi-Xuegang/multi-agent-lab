"""Day1 验收：LLM 连通性。"""

from __future__ import annotations

import _bootstrap  # noqa: F401

from src.config import get_settings
from src.llm.siliconflow import SiliconFlowClient


def main() -> None:
    settings = get_settings()
    if not settings.effective_llm_api_key:
        raise SystemExit("缺少 SILICONFLOW_API_KEY / LLM_API_KEY")

    client = SiliconFlowClient.from_settings(settings)
    print(f"model = {client.model}")
    print(f"base  = {client.base_url}")

    result = client.chat(
        [{"role": "user", "content": "用一句话介绍你自己。"}],
        temperature=0.2,
    )
    print("reply =", (result.get("content") or "").strip())
    print("OK: LLM 连通")


if __name__ == "__main__":
    main()
