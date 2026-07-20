"""图存储抽象基类"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from src.models.schemas import EntityRelation, ResearchTask, TaskDag, TaskStatus


class GraphStore(ABC):
    """任务 DAG + 证据链 + 知识图谱的统一接口。"""

    @property
    @abstractmethod
    def backend(self) -> str:
        """存储后端标识。用于日志打印、调试排查、前端展示当前存储引"""

    @abstractmethod
    def create_session(self, session_id: str, question: str) -> None:
        """创建研究会话根节点。每个研究问题对应一个独立会话，作为图数据的根节点，实现多任务数据隔离"""

# 任务 DAG 管理接口
    @abstractmethod
    def save_task_dag(self, dag: TaskDag) -> None:
        """持久化任务 DAG。Planner 拆分完任务后，一次性写入完整任务依赖图"""

    @abstractmethod
    def get_tasks(self, session_id: str) -> list[ResearchTask]:
        """获取会话下全部任务。编排器调度时拉取全量任务，用于拓扑排序、状态判断"""

    @abstractmethod
    def update_task_status(
        self,
        session_id: str,
        task_id: str,
        status: TaskStatus,
        error: str | None = None,
    ) -> None:
        """更新任务状态。"""

    @abstractmethod
    def save_task_result(
        self,
        session_id: str,
        task_id: str,
        findings: list[str],
        sources: list[dict[str, Any]],
    ) -> None:
        """保存子任务研究发现与来源。"""

    @abstractmethod
    def add_supplement_tasks(
        self, session_id: str, tasks: list[ResearchTask]
    ) -> None:
        """追加补充研究任务。Verifier 自反思发现信息缺口时，动态追加补充任务，支持 DAG 动态扩展"""

# 证据链接口
    @abstractmethod
    def link_source(
        self,
        session_id: str,
        task_id: str,
        url: str,
        title: str,
        snippet: str = "",
    ) -> None:
        """记录任务 → 搜索来源。建立任务与搜索来源的关联关系，是「结论可溯源」能力的底层支撑"""

# 知识图谱接口
    @abstractmethod
    def upsert_relations(
        self, session_id: str, relations: list[EntityRelation]
    ) -> None:
        """批量写入实体 - 关系三元组，幂等更新"""

    @abstractmethod
    def get_relations(self, session_id: str) -> list[EntityRelation]:
        """读取全量三元组，用于生成 Mermaid 图谱、后续推理。"""

    def close(self) -> None:
        """关闭连接（可选）。"""
