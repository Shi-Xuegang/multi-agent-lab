"""中央 Orchestrator：任务状态机 + 并行研究 + 验证循环。"""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.agents.planner import PlannerAgent
from src.agents.researcher import ResearcherAgent
from src.agents.synthesizer import SynthesizerAgent
from src.agents.verifier import VerifierAgent
from src.graph.base import GraphStore
from src.memory.user_preferences import user_preference_manager
from src.models.schemas import ResearchReport, ResearchTask, TaskStatus
from src.visualization.kg_export import export_kg_html, relations_to_mermaid
from src.visualization.mermaid import build_process_mermaid


class ResearchOrchestrator:
    """多 Agent 深度研究编排器。"""

    def __init__(
        self,
        planner: PlannerAgent,
        researcher: ResearcherAgent,
        synthesizer: SynthesizerAgent,
        verifier: VerifierAgent,
        graph: GraphStore,
        max_verify_rounds: int = 2,
        concurrency: int = 3,
        max_task_retries: int = 2,
        enable_visualization: bool = True,
        output_dir: str | Path = "outputs",
    ) -> None:
        self._planner = planner
        self._researcher = researcher
        self._synthesizer = synthesizer
        self._verifier = verifier
        self._graph = graph
        self._max_verify_rounds = max_verify_rounds
        self._concurrency = max(1, concurrency)
        self._max_task_retries = max_task_retries
        self._enable_visualization = enable_visualization
        self._output_dir = Path(output_dir)
        self._last_tasks: list[ResearchTask] = []
        self._events: list[str] = []

    @property
    def events(self) -> list[str]:
        return list(self._events)

    @property
    def last_tasks(self) -> list[ResearchTask]:
        return list(self._last_tasks)

    def research(self, question: str, session_id: str | None = None) -> ResearchReport:
        """执行完整多 Agent 研究流水线。"""
        session_id = session_id or f"sess_{uuid.uuid4().hex[:10]}"
        self._events.clear()
        self._log(f"会话开始 session={session_id}")

        user_context = user_preference_manager.get_context_prompt()
        dag = self._planner.run(
            question=question,
            session_id=session_id,
            user_context=user_context,
        )
        self._log(f"Planner 生成 {len(dag.tasks)} 个子任务")

        self._execute_pending_tasks(session_id)
        report = self._synthesizer.run(
            question=question,
            session_id=session_id,
            verify_rounds=0,
        )
        self._log("Synthesizer 完成初稿")

        verify_rounds = 0
        for round_idx in range(1, self._max_verify_rounds + 1):
            tasks = self._graph.get_tasks(session_id)
            verdict = self._verifier.run(
                question=question,
                report=report,
                tasks=tasks,
            )
            self._log(
                f"Verifier 第 {round_idx} 轮: satisfied={verdict.satisfied}, "
                f"issues={len(verdict.issues)}, supplements={len(verdict.supplement_tasks)}"
            )
            if verdict.satisfied or not verdict.supplement_tasks:
                verify_rounds = round_idx
                break

            self._graph.add_supplement_tasks(session_id, verdict.supplement_tasks)
            self._execute_pending_tasks(session_id)
            report = self._synthesizer.run(
                question=question,
                session_id=session_id,
                verify_rounds=round_idx,
            )
            verify_rounds = round_idx

        tasks = self._graph.get_tasks(session_id)
        self._last_tasks = tasks
        relations = self._graph.get_relations(session_id)

        report.verify_rounds = verify_rounds
        report.mode = "multi"
        report.process_diagram = build_process_mermaid(
            question=question,
            tasks=tasks,
            sections=report.sections or report.key_findings,
        )
        report.kg_diagram = relations_to_mermaid(relations)

        if self._enable_visualization:
            self._export_artifacts(session_id, question, report)

        user_preference_manager.add_research_topic(question)
        report.metadata = {
            "session_id": session_id,
            "graph_backend": self._graph.backend,
            "task_count": len(tasks),
            "relation_count": len(relations),
            "verify_rounds": verify_rounds,
            "events": list(self._events),
        }
        self._log("研究流水线完成")
        return report

    def _execute_pending_tasks(self, session_id: str) -> None:
        """按 DAG 依赖并行执行可运行任务，直到没有 pending。"""
        while True:
            tasks = self._graph.get_tasks(session_id)
            status_map = {t.id: t.status for t in tasks}
            runnable = [
                t
                for t in tasks
                if t.status == TaskStatus.PENDING
                and all(
                    status_map.get(dep) == TaskStatus.DONE for dep in t.depends_on
                )
            ]

            # 依赖失败导致永久阻塞的任务标记失败
            blocked = [
                t
                for t in tasks
                if t.status == TaskStatus.PENDING
                and any(
                    status_map.get(dep) == TaskStatus.FAILED for dep in t.depends_on
                )
            ]
            for t in blocked:
                self._graph.update_task_status(
                    session_id,
                    t.id,
                    TaskStatus.FAILED,
                    error="依赖任务失败",
                )
                self._log(f"任务 {t.id} 因依赖失败而被跳过")

            if not runnable:
                # 还有 pending 但依赖未完成 → 死锁，跳出
                still_pending = [
                    t for t in self._graph.get_tasks(session_id)
                    if t.status == TaskStatus.PENDING
                ]
                if still_pending:
                    for t in still_pending:
                        self._graph.update_task_status(
                            session_id,
                            t.id,
                            TaskStatus.FAILED,
                            error="无法满足依赖，终止调度",
                        )
                break

            self._log(f"并行执行 {len(runnable)} 个任务: {[t.id for t in runnable]}")
            with ThreadPoolExecutor(max_workers=self._concurrency) as pool:
                futures = {
                    pool.submit(self._researcher.run, task=t, session_id=session_id): t
                    for t in runnable
                }
                for fut in as_completed(futures):
                    task = futures[fut]
                    try:
                        result = fut.result()
                        if result.status == TaskStatus.FAILED:
                            self._maybe_retry(session_id, result)
                        else:
                            self._log(f"任务 {task.id} 完成，发现 {len(result.findings)} 条")
                    except Exception as exc:
                        self._graph.update_task_status(
                            session_id, task.id, TaskStatus.FAILED, error=str(exc)
                        )
                        refreshed = self._graph.get_tasks(session_id)
                        failed_task = next(
                            (x for x in refreshed if x.id == task.id), task
                        )
                        self._maybe_retry(session_id, failed_task)

    def _maybe_retry(self, session_id: str, task: ResearchTask) -> None:
        """失败重试：重置为 pending（保留 retries 计数）。"""
        current = next(
            (t for t in self._graph.get_tasks(session_id) if t.id == task.id),
            task,
        )
        # failed 时已递增 retries；retries<=max 表示仍允许再试
        if current.retries <= self._max_task_retries:
            self._log(
                f"任务 {current.id} 失败，准备重试 "
                f"({current.retries}/{self._max_task_retries}): {current.error}"
            )
            self._graph.update_task_status(
                session_id, current.id, TaskStatus.PENDING, error=current.error
            )
        else:
            self._log(f"任务 {current.id} 超过最大重试次数，保持 failed")

    def _export_artifacts(
        self, session_id: str, question: str, report: ResearchReport
    ) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        process_path = self._output_dir / f"{session_id}_process.mmd"
        process_path.write_text(report.process_diagram, encoding="utf-8")
        kg_path = self._output_dir / f"{session_id}_kg.mmd"
        kg_path.write_text(report.kg_diagram, encoding="utf-8")
        html_path = export_kg_html(
            relations=self._graph.get_relations(session_id),
            output_path=self._output_dir / f"{session_id}_kg.html",
            title=f"KG: {question[:60]}",
        )
        report.metadata = {
            **(report.metadata or {}),
            "process_diagram_path": str(process_path),
            "kg_diagram_path": str(kg_path),
            "kg_html_path": str(html_path),
        }

    def _log(self, message: str) -> None:
        self._events.append(message)
        print(f"[orchestrator] {message}")
