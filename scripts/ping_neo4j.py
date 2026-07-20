"""Day15 验收：Neo4j / 内存图回退。"""

from __future__ import annotations

import _bootstrap  # noqa: F401

from src.config import get_settings
from src.graph.factory import create_graph_store
from src.models.schemas import ResearchTask, TaskDag, TaskStatus


def main() -> None:
    settings = get_settings()
    store = create_graph_store(settings)
    print(f"backend = {store.backend}")

    session_id = "scripts_ping_graph"
    dag = TaskDag(
        root_question="图存储连通性测试",
        session_id=session_id,
        tasks=[
            ResearchTask(id="t1", question="背景", keywords=["ping"]),
            ResearchTask(id="t2", question="结论", depends_on=["t1"]),
        ],
    )
    store.save_task_dag(dag)
    store.update_task_status(session_id, "t1", TaskStatus.IN_PROGRESS)
    store.save_task_result(
        session_id,
        "t1",
        findings=["图存储写入成功"],
        sources=[{"url": "https://example.com/graph", "title": "graph ping"}],
    )
    tasks = store.get_tasks(session_id)
    for t in tasks:
        print(f"- {t.id} status={t.status.value} findings={t.findings}")
    store.close()
    print("OK: GraphStore 可用")


if __name__ == "__main__":
    main()
