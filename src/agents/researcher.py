"""Researcher Agent：执行子任务搜索与证据沉淀。

run：对外入口，调度整体流程；
_build_queries：构建查询语句；
_dedupe_evidence：对多条搜索证据去重；
_summarize_findings：总结证据，提取关键发现，LLM 输出 JSON 格式；

query 最多 3 个、缓存每条取 4 个、证据最多 12 条、总结最多 6 条，
全链路限制长度防止上下文超限
"""

from __future__ import annotations

from typing import Any

from src.agents.base import BaseAgent
from src.agents.kg_extractor import KnowledegExtractor
from src.graph.base import GraphStore
from src.llm.siliconflow import SiliconFlowClient
from src.memory.qdrant_store import QdrantMemoryStore
from src.models.schemas import ResearchTask, SearchResult, TaskStatus
from src.tools.tavily_search import TavilySearchTool


class ResearcherAgent(BaseAgent):
    """领取子任务并收集证据的研究 Agent。"""

    def __init__(
        self,
        llm: SiliconFlowClient,
        memory: QdrantMemoryStore,
        tavily: TavilySearchTool,
        graph: GraphStore,
        rag_score_threshold: float = 0.85,
        enable_kg: bool = True,   # 控制实体关系抽取，不需要知识图谱场景可关闭节省开销
    ) -> None:
     # _llm、_extractor，语义私有变量，仅内部访问
        self._llm = llm   # LLM 模型
        self._memory = memory   # 向量数据库
        self._tavily = tavily   # 搜索引擎
        self._graph = graph   # 图存储
        self._rag_score_threshold = rag_score_threshold   # 相似性得分阈值
        self._extractor = KnowledegExtractor(llm) if enable_kg else None   # 真值执行表达式 if 条件 else 假值表达式
        self._enable_kg = enable_kg

    @property
    def name(self) -> str:
        return "researcher"

    def run(
        self,
        task: ResearchTask,
        session_id: str,
        **_: Any,
    ) -> ResearchTask:
        """执行单个研究子任务。"""
        # 图存储状态更新
        self._graph.update_task_status(
            session_id, task.id, TaskStatus.IN_PROGRESS
        )

        evidence: list[dict[str, Any]] = []    # 证据列表，存储搜索结果
        queries = self._build_queries(task)   # 构建查询语句

        try:
            for query in queries:
                # 向量数据库相似性搜索
                cached = self._memory.search_similar(  # 相似性搜索
                    query=query,
                    limit=4,
                    score_threshold=self._rag_score_threshold,
                )
                if cached:  # 如果存在缓存结果
                    evidence.extend(cached) # 将可迭代对象全部元素并入列表
                    for item in cached:
                        self._graph.link_source(
                            session_id=session_id,  # 会话 ID
                            task_id=task.id,  # 任务 ID
                            url=item.get("url", ""),  # 链接 URL
                            title=item.get("title", ""),  # 标题
                            snippet=item.get("content", ""),  # 内容
                        )
                    continue
                # 联网搜索，搜索引擎搜索
                result = self._tavily.execute(query=query, search_depth="basic")  # 搜索引擎搜索
                if not result.success or not isinstance(result.data, dict):  # 如果搜索失败或结果不是字典
                    continue
                raw = result.data.get("results", [])
                search_results = [SearchResult(**r) for r in raw]  # 搜索结果转换为强类型SearchResult对象
                for sr in search_results:
                    evidence.append(sr.model_dump())  # 转换为字典，添加到证据列表
                    # 图存储链接源信息
                    self._graph.link_source(
                        session_id=session_id,
                        task_id=task.id,
                        url=sr.url,
                        title=sr.title,
                        snippet=sr.content,
                    )

                # 向量数据库存储搜索结果
                self._memory.store_search_results(
                    query=query,
                    results=search_results,
                    session_id=session_id,
                )

            evidence = self._dedupe_evidence(evidence)
            findings = self._summarize_findings(task.question, evidence)

            if self._enable_kg and self._extractor and evidence:
                relations = self._extractor.extract(task.question, evidence)
                if relations:
                    self._graph.upsert_relations(session_id, relations)

        # 带过滤条件的列表推导，只保留存在 url 的有效来源，
        # 内容切片截断 300 字符控制存储长度
            sources = [
                {
                    "url": e.get("url", ""),
                    "title": e.get("title", ""),
                    "snippet": str(e.get("content", ""))[:300],
                }
                for e in evidence
                if e.get("url")
            ]
            self._graph.save_task_result(
                session_id=session_id,
                task_id=task.id,
                findings=findings,
                sources=sources,
            )
            # 修改传入的 ResearchTask 对象属性，原地更新后返回
            task.findings = findings
            task.sources = sources
            task.status = TaskStatus.DONE
            task.error = None
            return task

        except Exception as exc:
            self._graph.update_task_status(
                session_id, task.id, TaskStatus.FAILED, error=str(exc)
            )
            task.status = TaskStatus.FAILED
            task.error = str(exc)
            return task

    # 从任务关键词 + 子问题生成搜索 query，自动去重、截断最多 3 条
    def _build_queries(self, task: ResearchTask) -> list[str]:
        queries: list[str] = []
        if task.keywords:
            queries.extend(str(k) for k in task.keywords if str(k).strip())
        queries.append(task.question)
        # 去重保序
        # set 记录已存在 query 保证 O (1) 查重，列表保存原始顺序（set 无序，不能直接去重）
        seen: set[str] = set()
        unique: list[str] = []  # 最终返回的唯一 query 列表
        for q in queries:
            q = q.strip()  # 去除前后空格
            if q and q not in seen:
                seen.add(q)  # 添加到已存在集合
                unique.append(q)
        return unique[:3]  # 截断最多 3 条

    # 对多条搜索证据去重，优先用 url 作为唯一键，无 url 则截取前 80 内容文本做标识
    def _dedupe_evidence(
        self, evidence: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        seen: set[str] = set()  # 已存在 url 集合，保证 O (1) 查重
        out: list[dict[str, Any]] = []  # 最终返回的唯一证据列表
        for item in evidence:
            key = item.get("url") or str(item.get("content", ""))[:80]  # 优先 url，否则内容前 80 字符
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out[:12]  # 截断最多 12 条

    # 拼接证据文本交给 LLM 提炼简短结论，解析失败则降级直接返回原文片段
    def _summarize_findings(
        self, question: str, evidence: list[dict[str, Any]]
    ) -> list[str]:
        if not evidence:  # 如果没有证据，返回兜底提示
            return ["未检索到充分证据。"]

        evidence_text = "\n".join(  # 将证据列表转换为文本格式，方便 LLM 处理 
            f"- [{e.get('title', '')}]({e.get('url', '')}): "
            f"{str(e.get('content', ''))[:220]}"
            for e in evidence[:8]
        )
        prompt = (
            f"子任务：{question}\n\n证据：\n{evidence_text}\n\n"
            "请输出 JSON：{\"findings\": [\"发现1\", \"发现2\", \"发现3\"]}。"
            "每条发现简洁、可核验，并尽可能标注来源标题。"
        )
        try:
            data = self._llm.chat_json([{"role": "user", "content": prompt}])  # 调用 LLM 生成 JSON 格式输出
            findings = [str(x) for x in data.get("findings", []) if str(x).strip()]
            return findings[:6] or ["证据已收集，但未能提炼关键发现。"]  # 截断最多 6 条发现，兜底提示
        except ValueError:
            return [
                f"{e.get('title', '来源')}: {str(e.get('content', ''))[:160]}"  # 截断内容前 160 字符，标注来源标题
                for e in evidence[:3]
            ]  # 兜底返回前 3 条证据的标题和内容摘要