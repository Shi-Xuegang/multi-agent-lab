"""Agent 基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """所有研究角色 Agent 的抽象基类。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent 名称。"""

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """执行角色职责。"""
