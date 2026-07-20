"""Synthesizer Agent：综合证据与图谱关系生成结构化报告。

Synthesizer Agent 是基于子任务证据与知识图谱关系生成结构化报告的专家。
它负责将子任务的证据与知识图谱关系综合起来，生成一份结构化研究报告。

Synthesizer Agent 的输入包括：
- 总问题：用户的问题或查询
- 用户偏好：用户对报告的偏好，如语言、风格、格式等
- 子任务：子任务的证据与知识图谱关系
- 知识图谱关系：知识图谱关系

Synthesizer Agent 的输出包括：
- 结构化研究报告：结构化研究报告

Synthesizer Agent 的实现思路：
1. 根据用户偏好，生成结构化研究报告的模板
2. 根据子任务的证据与知识图谱关系，生成结构化研究报告
...
"""

from __future__ import annotations

from typing import Any

from src.agents.base import BaseAgent
from src.graph.base import GraphStore
from src.llm.siliconflow import SiliconFlowClient
from src.memory.user_preferences import user_preference_manager
from src.models.schemas import Citation, ResearchReport, ResearchTask


SYNTH_PROMPT = """你是研究报告综合专家。请基于子任务证据与知识图谱关系，生成结构化研究报告。

## 总问题
{question}

## 用户偏好
{user_context}

## 子任务与发现
{task_block}

## 知识图谱关系
{relations_block}

请严格输出 JSON：
{{
  "summary": "2-4句核心摘要",
  "sections": ["章节1论述", "章节2论述"],
  "key_findings": ["发现1 [1]", "发现2 [2]"],
  "citations": [
    {{"index": 1, "title": "标题", "url": "URL"}}
  ],
  "unresolved_questions": ["未充分回答的问题"]
}}

要求：
1. 覆盖各子任务核心结论
2. 引用必须来自下列真实来源，禁止编造 URL
3. 使用中文
"""


class SynthesizerAgent(BaseAgent):
    """报告综合 Agent。"""

    def __init__(self, llm: SiliconFlowClient, graph: GraphStore) -> None:
        # 依赖注入设计：外部传入 LLM、图存储，内部仅做挂载self._xxx
        self._llm = llm   # LLM 模型
        self._graph = graph   # 图存储

    # 属性装饰器，定义类实例的只读属性
    @property
    def name(self) -> str:
        return "synthesizer"

    def run(
        self,
        question: str,
        session_id: str,
        tasks: list[ResearchTask] | None = None,
        verify_rounds: int = 0,
        **_: Any,
    ) -> ResearchReport:
        tasks = tasks or self._graph.get_tasks(session_id)  # 从图存储获取任务  
        relations = self._graph.get_relations(session_id)  # 从图存储获取关系
        user_context = user_preference_manager.get_context_prompt()  # 获取用户偏好

        task_block = self._format_tasks(tasks)  # 格式化任务
        relations_block = self._format_relations(relations)  # 格式化关系
        prompt = SYNTH_PROMPT.format(  # 格式化提示词   批量替换 Prompt 占位符
            question=question,  # 总问题
            user_context=user_context,  # 用户偏好
            task_block=task_block,  # 任务
            relations_block=relations_block,  # 关系
        )

        try:
            data = self._llm.chat_json([{"role": "user", "content": prompt}])  # 调用 LLM 生成 JSON 格式输出
            citations = []  # 引用列表
            for c in data.get("citations", []):  # 遍历引用列表
                try:
                    citations.append(Citation(**c))  # 转换为强类型Citation对象
                except Exception:  # 如果转换失败，跳过
                    continue
            report = ResearchReport(  # 转换为强类型ResearchReport对象
                question=question,  # 总问题
                summary=str(data.get("summary", "")),  # 摘要
                sections=[str(s) for s in data.get("sections", [])],  # 章节
                key_findings=[str(x) for x in data.get("key_findings", [])],  # 关键发现
                citations=citations,  # 引用
                unresolved_questions=[str(x) for x in data.get("unresolved_questions", [])],  # 未充分回答的问题
                verify_rounds=verify_rounds,  # 验证轮次
                mode="multi",  # 模式
            )
        except ValueError:  # 如果解析失败，降级兜底
            report = self._fallback_report(question, tasks, verify_rounds) 

        if not report.citations:  # 如果引用为空，从任务来源回填
            report.citations = self._citations_from_tasks(tasks)  # 从任务来源回填引用

        return report  # 返回报告

    def _format_tasks(self, tasks: list[ResearchTask]) -> str:  # 格式化任务    
        parts: list[str] = []  # 任务列表
        for task in tasks:
            findings = "\n".join(f"  - {f}" for f in task.findings) or "  - （无）"
            sources = "\n".join(
                f"  - {s.get('title', '')} | {s.get('url', '')}"
                for s in task.sources[:5]
            ) or "  - （无）"
            parts.append(
                f"### {task.id} [{task.status.value}] {task.question}\n"
                f"发现:\n{findings}\n来源:\n{sources}"
            )
        return "\n\n".join(parts) if parts else "（无子任务结果）"

    def _format_relations(self, relations: list) -> str:  # 格式化关系
        if not relations:  # 如果关系为空，返回兜底提示
            return "（暂无图谱关系）"
        return "\n".join(
            f"- ({r.subject}) -[{r.predicate}]-> ({r.object})"  # 关系格式化
            for r in relations[:30]  # 截断最多 30 条关系
        )

    def _citations_from_tasks(self, tasks: list[ResearchTask]) -> list[Citation]:  # 从任务来源回填引用
        citations: list[Citation] = []  # 引用列表      
        seen: set[str] = set()  # 已存在 URL 集合
        idx = 1  # 索引
        for task in tasks:
            for src in task.sources:
                url = src.get("url", "")  # 获取 URL
                if not url or url in seen:  # 如果 URL 为空或已存在，跳过
                    continue
                seen.add(url)  # 添加到已存在 URL 集合
                citations.append(
                    Citation(  # 转换为强类型Citation对象
                        index=idx,
                        title=str(src.get("title") or url),  # 标题
                        url=url,
                    )  # 转换为强类型Citation对象
                )
                idx += 1  # 索引递增
                if idx > 20:
                    return citations  # 截断最多 20 条引用
        return citations  # 返回引用列表        

    def _fallback_report(  # 降级兜底报告
        self,
        question: str,
        tasks: list[ResearchTask],
        verify_rounds: int,
    ) -> ResearchReport:
        findings: list[str] = []  # 关键发现 列表
        for task in tasks:
            findings.extend(task.findings[:2])  # 将任务关键发现批量扩展到关键发现列表
        return ResearchReport(  # 转换为强类型ResearchReport对象
            question=question,  # 总问题
            summary=f"针对「{question}」已完成 {len(tasks)} 个子任务研究。",  # 摘要
            key_findings=findings[:8],  # 关键发现
            sections=[  # 章节
                f"{t.id}: {t.question}\n" + "\n".join(t.findings[:3])
                for t in tasks
            ],
            citations=self._citations_from_tasks(tasks),  # 引用    从任务来源回填引用
            unresolved_questions=[t.question for t in tasks if t.status.value == "failed"],  # 未充分回答的问题
            verify_rounds=verify_rounds,  # 验证轮次
            mode="multi",  # 模式
        )     