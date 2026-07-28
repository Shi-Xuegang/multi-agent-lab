"""用户偏好管理模块

用于存储和读取用户的历史研究偏好、术语偏好、输出格式要求等。
这些偏好会作为上下文信息在生成报告时被LLM读取。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field 
# BaseModel 结构化数据校验、序列化；
# Field、default_factory 处理可变默认参数

from src.models.schemas import ToolExecutionResult


class UserPreferences(BaseModel):
    """用户偏好设置"""
    # 用户基本信息
    username: str = Field(default="", description="用户名")

    # 技术偏好
    preferred_domains: List[str] = Field(default_factory=list, description="偏好的技术领域（如：AI, Web, 数据库）")
    preferred_languages: List[str] = Field(default_factory=list, description="偏好的编程语言或技术栈")

    # 输出格式偏好
    output_format: str = Field(default="plain", description="报告输出格式：plain(纯文本), markdown, html")
    citation_style: str = Field(default="numeric", description="引用样式：numeric(数字), author-year, ieee")

    # 研究深度偏好
    research_depth: str = Field(default="balanced", description="研究深度：basic(基础), balanced(平衡), detailed(详细)")

    # 隐私与安全偏好
    private_research: bool = Field(default=False, description="是否为私密研究，需要额外保护")

    # 时间偏好
    deadline: Optional[datetime] = Field(default=None, description="研究截止时间")

    # 用户针对特定主题的偏好
    topic_preferences: Dict[str, Dict[str, Union[str, int, bool]]] = Field(default_factory=dict, description="特定主题的偏好设置")

    # 历史研究主题
    research_history: List[str] = Field(default_factory=list, description="用户曾经研究过的主题列表")


class UserPreferenceManager:
    """用户偏好管理器

    负责读取、保存和更新用户偏好。
    """

    def __init__(self, storage_path: str = "user_preferences.json"):
        """初始化偏好管理器

        Args:
            storage_path: 存储偏好数据的文件路径
        """
        self.storage_path = storage_path
        self._preferences = self._load_preferences()

    def get_preferences(self) -> UserPreferences:
        """获取当前用户偏好"""
        return self._preferences

    def update_preferences(self, updates: Dict) -> ToolExecutionResult:
        """更新用户偏好设置

        Args:
            updates: 偏好更新字典

        Returns:
            处理结果
        """
        try:
            # 更新偏好对象
            for key, value in updates.items():
                if hasattr(self._preferences, key):
                    setattr(self._preferences, key, value)
                else:
                    # 对于嵌套字典类型的字段 (如topic_preferences)
                    if key == 'topic_preferences':
                        self._preferences.topic_preferences.update(value)
                    elif key == 'research_history':
                        # 合并历史主题
                        existing = set(self._preferences.research_history)
                        new_items = set(value)
                        self._preferences.research_history = list(existing | new_items)
            self._save_preferences()
            return ToolExecutionResult(
                tool_name="user_preference_update",
                success=True,
                data={"message": "偏好更新成功"}
            )
        except Exception as e:
            return ToolExecutionResult(
                tool_name="user_preference_update",
                success=False,
                error=f"更新偏好失败: {str(e)}"
            )

    def get_context_prompt(self) -> str:
        """获取用于LLM的上下文提示，包含用户偏好信息

        Returns:
            用户偏好的字符串格式，可用于构建提示词
        """
        prefs = self._preferences
        return (
            f"用户偏好信息：\n"
            f"- 偏好领域：{', '.join(prefs.preferred_domains) if prefs.preferred_domains else '无'}\n"
            f"- 偏好语言：{', '.join(prefs.preferred_languages) if prefs.preferred_languages else '无'}\n"
            f"- 输出格式：{prefs.output_format}\n"
            f"- 引用样式：{prefs.citation_style}\n"
            f"- 研究深度：{prefs.research_depth}\n"
            f"- 私密研究：{'是' if prefs.private_research else '否'}\n"
            f"- 历史研究：{', '.join(prefs.research_history[-3:]) if prefs.research_history else '无'}"
        )

    def _load_preferences(self) -> UserPreferences:
        """从文件加载偏好设置"""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return UserPreferences(**data)
        except Exception as e:
            print(f"加载用户偏好失败: {e}")
        # 如果失败，返回默认偏好
        return UserPreferences(username="default_user", research_history=[])

    def add_research_topic(self, topic: str) -> None:
        """将研究主题追加到历史记录。"""
        topic = (topic or "").strip()
        if not topic:
            return
        history = list(self._preferences.research_history)
        if topic in history:
            history.remove(topic)
        history.append(topic)
        self._preferences.research_history = history[-50:] # 每次完成研究自动追加主题，去重、限制最多 50 条，最新主题置顶
        self._save_preferences()

    def _save_preferences(self) -> None:
        """保存偏好设置到文件"""
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(
                    self._preferences.model_dump(mode="json"),
                    f,
                    ensure_ascii=False, # JSON 保留中文，便于人工查看配置文件
                    indent=2,
                )
        except Exception as e:
            print(f"保存用户偏好失败: {e}")


# 单例实例化
user_preference_manager = UserPreferenceManager()