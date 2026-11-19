"""内存图存储，作为 Neo4j 不可用时的回退。
纯 Python 字典模拟的进程内图存储，无任何外部依赖，作为 Neo4j 不可用时的降级方案；
数据随进程销毁，适合本地调试、单用户 CLI 场景。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.graph.base import GraphStore
from src.models.schemas import EntityRelation, ResearchTask, TaskDag, TaskStatus


class MemoryGraphStore(GraphStore):
    """进程内图存储。"""

    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}

    @property
    def backend(self) -> str:
        return "memory"

    def _session(self, session_id: str) -> dict[str, Any]:
        # 懒加载：会话不存在则自动创建
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "question": "",
                "tasks": {},
                "relations": [],
            }
        return self._sessions[session_id]

    def create_session(self, session_id: str, question: str) -> None:
        sess = self._session(session_id)
        sess["question"] = question

    def save_task_dag(self, dag: TaskDag) -> None:
        sess = self._session(dag.session_id)
        sess["question"] = dag.root_question
        for task in dag.tasks:
            sess["tasks"][task.id] = task.model_copy(deep=True)

    def get_tasks(self, session_id: str) -> list[ResearchTask]:
        sess = self._session(session_id)
        return [t.model_copy(deep=True) for t in sess["tasks"].values()]
        # model_copy(deep=True) 深拷贝模型对象；核心目的是防止外部代码意外修改存储层内部状态

    def update_task_status(
        self,
        session_id: str,
        task_id: str,
        status: TaskStatus,
        error: str | None = None,
    ) -> None:
        task = self._session(session_id)["tasks"].get(task_id)
        if not task:
            return
        task.status = status
        if error is not None:
            task.error = error
        if status == TaskStatus.FAILED:
            task.retries += 1

    def save_task_result(
        self,
        session_id: str,
        task_id: str,
        findings: list[str],
        sources: list[dict[str, Any]],
    ) -> None:
        task = self._session(session_id)["tasks"].get(task_id)
        if not task:
            return
        task.findings = list(findings)
        task.sources = deepcopy(sources)
        task.status = TaskStatus.DONE
        task.error = None

    def add_supplement_tasks(
        self, session_id: str, tasks: list[ResearchTask]
    ) -> None:
        sess = self._session(session_id)
        for task in tasks:
            task.is_supplement = True
            sess["tasks"][task.id] = task.model_copy(deep=True)

    def link_source(
        self,
        session_id: str,
        task_id: str,
        url: str,
        title: str,
        snippet: str = "",
    ) -> None:
        task = self._session(session_id)["tasks"].get(task_id)
        if not task:
            return
        source = {"url": url, "title": title, "snippet": snippet}
        if source not in task.sources:
            task.sources.append(source)

    def upsert_relations(
        self, session_id: str, relations: list[EntityRelation]
    ) -> None:
        sess = self._session(session_id)
        existing = {
            (r.subject, r.predicate, r.object) for r in sess["relations"]
        }
        for rel in relations:
            key = (rel.subject, rel.predicate, rel.object)
            if key not in existing:
                sess["relations"].append(rel.model_copy(deep=True))
                existing.add(key)

    def get_relations(self, session_id: str) -> list[EntityRelation]:
        return [
            r.model_copy(deep=True)
            for r in self._session(session_id)["relations"]
        ]
