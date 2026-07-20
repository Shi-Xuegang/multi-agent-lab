"""Neo4j 图存储实现。
将任务 DAG、证据链、知识图谱完整映射为原生图节点与关系，支持持久化、复杂图查询、路径分析，
是多 Agent 系统落地生产环境的核心存储底座。
"""

from __future__ import annotations

from typing import Any

from neo4j import GraphDatabase

from src.graph.base import GraphStore
from src.models.schemas import EntityRelation, ResearchTask, TaskDag, TaskStatus


class Neo4jGraphStore(GraphStore):
    """基于 Neo4j 的任务 DAG / 证据链 / 知识图谱存储。"""

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "",
        max_connection_lifetime: int = 3600,
        max_connection_pool_size: int = 50,
        connection_timeout: int = 60,
    ) -> None:
        self._database = database or None
        # 连接池核心参数配置：连接生命周期、最大池大小、连接超时
        self._driver = GraphDatabase.driver(
            uri,
            auth=(username, password),
            max_connection_lifetime=max_connection_lifetime,
            max_connection_pool_size=max_connection_pool_size,
            connection_timeout=connection_timeout,
        )
        # 连通性探测
        self._driver.verify_connectivity()  

    @property
    def backend(self) -> str:
        return "neo4j"

    def _run(self, query: str, **params: Any) -> list[dict[str, Any]]:
        with self._driver.session(database=self._database) as session:  # 自动获取与释放连接，避免连接泄漏
            result = session.run(query, **params)
            return [record.data() for record in result]

    def create_session(self, session_id: str, question: str) -> None:
        self._run(
            # 使用 MERGE 幂等创建 ResearchSession 节点，
            # 不存在则创建，存在则更新问题与时间戳，重复调用不会产生重复节点
            """
            MERGE (s:ResearchSession {id: $session_id})
            SET s.question = $question, s.updated_at = datetime()
            """,
            session_id=session_id,
            question=question,
        )

    def save_task_dag(self, dag: TaskDag) -> None:
        self.create_session(dag.session_id, dag.root_question)
        for task in dag.tasks:
            self._run(
                """
                MATCH (s:ResearchSession {id: $session_id})
                MERGE (t:ResearchTask {id: $task_id, session_id: $session_id})
                SET t.question = $question,
                    t.keywords = $keywords,
                    t.priority = $priority,
                    t.status = $status,
                    t.is_supplement = $is_supplement,
                    t.retries = $retries,
                    t.findings = $findings,
                    t.error = $error
                MERGE (s)-[:HAS_TASK]->(t)
                """,
                session_id=dag.session_id,
                task_id=task.id,
                question=task.question,
                keywords=task.keywords,
                priority=task.priority,
                status=task.status.value,
                is_supplement=task.is_supplement,
                retries=task.retries,
                findings=task.findings,
                error=task.error,
            )
            for dep in task.depends_on:
                self._run(
                    """
                    MATCH (a:ResearchTask {id: $task_id, session_id: $session_id})
                    MATCH (b:ResearchTask {id: $dep_id, session_id: $session_id})
                    MERGE (a)-[:DEPENDS_ON]->(b)
                    """,
                    session_id=dag.session_id,
                    task_id=task.id,
                    dep_id=dep,
                )

    def get_tasks(self, session_id: str) -> list[ResearchTask]:
        rows = self._run(
            """
            MATCH (t:ResearchTask {session_id: $session_id})
            OPTIONAL MATCH (t)-[:DEPENDS_ON]->(d:ResearchTask)
            OPTIONAL MATCH (t)-[:USES_SOURCE]->(src:Source)
            RETURN t.id AS id,
                   t.question AS question,
                   coalesce(t.keywords, []) AS keywords,
                   coalesce(collect(DISTINCT d.id), []) AS depends_on,
                   coalesce(t.priority, 1) AS priority,
                   coalesce(t.status, 'pending') AS status,
                   coalesce(t.retries, 0) AS retries,
                   coalesce(t.findings, []) AS findings,
                   coalesce(t.error, null) AS error,
                   coalesce(t.is_supplement, false) AS is_supplement,
                   collect(DISTINCT {
                       url: src.url,
                       title: src.title,
                       snippet: coalesce(src.snippet, '')
                   }) AS sources
            """,
            session_id=session_id,
        )
        tasks: list[ResearchTask] = []
        for row in rows:
            sources = [s for s in row.get("sources") or [] if s and s.get("url")]
            deps = [d for d in (row.get("depends_on") or []) if d]
            tasks.append(
                ResearchTask(
                    id=row["id"],
                    question=row.get("question") or "",
                    keywords=list(row.get("keywords") or []),
                    depends_on=deps,
                    priority=int(row.get("priority") or 1),
                    status=TaskStatus(row.get("status") or "pending"),
                    retries=int(row.get("retries") or 0),
                    findings=list(row.get("findings") or []),
                    sources=sources,
                    error=row.get("error"),
                    is_supplement=bool(row.get("is_supplement")),
                )
            )
        return tasks

    def update_task_status(
        self,
        session_id: str,
        task_id: str,
        status: TaskStatus,
        error: str | None = None,
    ) -> None:
        self._run(
            """
            MATCH (t:ResearchTask {id: $task_id, session_id: $session_id})
            SET t.status = $status,
                t.error = $error,
                t.retries = CASE
                    WHEN $status = 'failed' THEN coalesce(t.retries, 0) + 1
                    ELSE coalesce(t.retries, 0)
                END
            """,
            session_id=session_id,
            task_id=task_id,
            status=status.value,
            error=error,
        )

    def save_task_result(
        self,
        session_id: str,
        task_id: str,
        findings: list[str],
        sources: list[dict[str, Any]],
    ) -> None:
        self._run(
            """
            MATCH (t:ResearchTask {id: $task_id, session_id: $session_id})
            SET t.findings = $findings,
                t.status = 'done',
                t.error = null
            """,
            session_id=session_id,
            task_id=task_id,
            findings=findings,
        )
        for src in sources:
            self.link_source(
                session_id=session_id,
                task_id=task_id,
                url=str(src.get("url", "")),
                title=str(src.get("title", "")),
                snippet=str(src.get("snippet") or src.get("content") or ""),
            )

    def add_supplement_tasks(
        self, session_id: str, tasks: list[ResearchTask]
    ) -> None:
        dag = TaskDag(root_question="", session_id=session_id, tasks=tasks)
        # 不覆盖会话问题，仅写入任务
        for task in tasks:
            task.is_supplement = True
        # 临时：直接写入任务节点
        rows = self._run(
            "MATCH (s:ResearchSession {id: $session_id}) RETURN s.question AS q",
            session_id=session_id,
        )
        question = rows[0]["q"] if rows else ""
        dag.root_question = question or "supplement"
        self.save_task_dag(dag)

    def link_source(
        self,
        session_id: str,
        task_id: str,
        url: str,
        title: str,
        snippet: str = "",
    ) -> None:
        if not url:
            return
        self._run(
            """
            MATCH (t:ResearchTask {id: $task_id, session_id: $session_id})
            MERGE (src:Source {url: $url})
            SET src.title = $title, src.snippet = $snippet
            MERGE (t)-[:USES_SOURCE]->(src)
            """,
            session_id=session_id,
            task_id=task_id,
            url=url,
            title=title,
            snippet=snippet[:500],
        )

    def upsert_relations(
        self, session_id: str, relations: list[EntityRelation]
    ) -> None:
        for rel in relations:
            self._run(
                """
                MATCH (s:ResearchSession {id: $session_id})
                MERGE (a:Entity {name: $subject, session_id: $session_id})
                MERGE (b:Entity {name: $object, session_id: $session_id})
                MERGE (a)-[r:RELATED {predicate: $predicate}]->(b)
                SET r.evidence_url = $evidence_url,
                    r.confidence = $confidence
                MERGE (s)-[:HAS_ENTITY]->(a)
                MERGE (s)-[:HAS_ENTITY]->(b)
                """,
                session_id=session_id,
                subject=rel.subject,
                object=rel.object,
                predicate=rel.predicate,
                evidence_url=rel.evidence_url,
                confidence=rel.confidence,
            )

    def get_relations(self, session_id: str) -> list[EntityRelation]:
        rows = self._run(
            """
            MATCH (:Entity {session_id: $session_id})-[r:RELATED]->(:Entity {session_id: $session_id})
            MATCH (a:Entity {session_id: $session_id})-[r]->(b:Entity {session_id: $session_id})
            RETURN a.name AS subject,
                   r.predicate AS predicate,
                   b.name AS object,
                   coalesce(r.evidence_url, '') AS evidence_url,
                   coalesce(r.confidence, 0.5) AS confidence
            """,
            session_id=session_id,
        )
        return [
            EntityRelation(
                subject=row["subject"],
                predicate=row["predicate"],
                object=row["object"],
                evidence_url=row.get("evidence_url") or "",
                confidence=float(row.get("confidence") or 0.5),
            )
            for row in rows
            if row.get("subject") and row.get("object")
        ]

    def close(self) -> None:
        self._driver.close()
