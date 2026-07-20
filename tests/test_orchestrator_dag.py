"""Planner DAG 与内存图调度相关测试。"""

from src.agents.planner import PlannerAgent
from src.graph.memory_store import MemoryGraphStore
from src.models.schemas import ResearchTask, TaskDag, TaskStatus


class _FakeLLM:
    def chat_json(self, messages):
        return {
            "tasks": [
                {
                    "id": "t1",
                    "question": "背景",
                    "keywords": ["背景"],
                    "depends_on": [],
                    "priority": 1,
                },
                {
                    "id": "t2",
                    "question": "趋势",
                    "keywords": ["趋势"],
                    "depends_on": ["t1"],
                    "priority": 2,
                },
                {
                    "id": "t3",
                    "question": "影响",
                    "keywords": ["影响"],
                    "depends_on": ["t2", "t1"],
                    "priority": 3,
                },
            ]
        }


def test_planner_builds_dag():
    graph = MemoryGraphStore()
    planner = PlannerAgent(llm=_FakeLLM(), graph=graph)  # type: ignore[arg-type]
    dag = planner.run(question="测试问题", session_id="s1", user_context="")
    assert len(dag.tasks) == 3
    assert dag.tasks[1].depends_on == ["t1"]
    stored = graph.get_tasks("s1")
    assert len(stored) == 3


def test_memory_graph_task_lifecycle():
    graph = MemoryGraphStore()
    dag = TaskDag(
        root_question="q",
        session_id="s2",
        tasks=[
            ResearchTask(id="a", question="a"),
            ResearchTask(id="b", question="b", depends_on=["a"]),
        ],
    )
    graph.save_task_dag(dag)
    graph.update_task_status("s2", "a", TaskStatus.IN_PROGRESS)
    graph.save_task_result(
        "s2",
        "a",
        findings=["f1"],
        sources=[{"url": "http://x", "title": "x"}],
    )
    tasks = {t.id: t for t in graph.get_tasks("s2")}
    assert tasks["a"].status == TaskStatus.DONE
    assert tasks["a"].findings == ["f1"]


def test_planner_breaks_cycles():
    class _CycleLLM:
        def chat_json(self, messages):
            return {
                "tasks": [
                    {
                        "id": "t1",
                        "question": "a",
                        "keywords": [],
                        "depends_on": ["t2"],
                        "priority": 1,
                    },
                    {
                        "id": "t2",
                        "question": "b",
                        "keywords": [],
                        "depends_on": ["t1"],
                        "priority": 2,
                    },
                ]
            }

    graph = MemoryGraphStore()
    planner = PlannerAgent(llm=_CycleLLM(), graph=graph)  # type: ignore[arg-type]
    dag = planner.run(question="cycle", session_id="s3")
    # 破环后应可拓扑：不应再互相依赖成环
    by_id = {t.id: t for t in dag.tasks}
    assert not (
        "t2" in by_id["t1"].depends_on and "t1" in by_id["t2"].depends_on
    )


def test_evaluate_metrics_smoke():
    from src.evaluation.metrics import evaluate_report
    from src.models.schemas import Citation, ResearchReport

    report = ResearchReport(
        question="q",
        summary="摘要内容" * 5,
        key_findings=["发现 A [1]", "发现 B [2]"],
        citations=[
            Citation(index=1, title="a", url="http://a.com"),
            Citation(index=2, title="b", url="http://b.com"),
        ],
        verify_rounds=2,
    )
    scores = evaluate_report(
        report,
        evidence_urls={"http://a.com", "http://b.com"},
    )
    assert scores.overall > 0
    assert scores.accuracy == 100.0
