"""Verifier Agent：检查报告完整性并生成补充任务。

"""

from __future__ import annotations

import uuid
from typing import Any

from src.agents.base import BaseAgent
from src.llm.siliconflow import SiliconFlowClient
from src.models.schemas import (
    ResearchReport,
    ResearchTask,
    TaskStatus,
    VerificationIssue,
    VerificationResult,
)


VERIFY_PROMPT = """你是研究报告质检专家。请检查报告是否充分回答了全部子任务。

## 总问题
{question}

## 子任务
{tasks_block}

## 当前报告
摘要: {summary}
关键发现:
{findings}
未决问题:
{unresolved}
引用数: {citation_count}

请严格输出 JSON：
{{
  "satisfied": true,
  "notes": "总体评价",
  "issues": [
    {{
      "task_id": "t1",
      "issue_type": "insufficient_evidence|contradiction|low_quality_source|incomplete",
      "description": "问题描述",
      "severity": "low|medium|high"
    }}
  ],
  "supplement_tasks": [
    {{
      "id": "s1",
      "question": "需要补充研究的子问题",
      "keywords": ["关键词"],
      "depends_on": [],
      "priority": 1
    }}
  ]
}}

规则：
1. 若证据覆盖充分且无明显矛盾，satisfied=true，supplement_tasks 为空
2. 若某处证据过弱/矛盾/引用不足，satisfied=false，并给出 1-3 个补充任务
3. 补充任务应具体可搜索，避免重复原任务原文
"""


class VerifierAgent(BaseAgent):
    """自反思验证 Agent。"""

    def __init__(self, llm: SiliconFlowClient) -> None:
        self._llm = llm

    @property
    def name(self) -> str:
        return "verifier"

    def run(
        self,
        question: str,
        report: ResearchReport,
        tasks: list[ResearchTask],
        **_: Any,
    ) -> VerificationResult:
        tasks_block = "\n".join(
            f"- {t.id} [{t.status.value}] {t.question}\n"
            f"  发现数={len(t.findings)} 来源数={len(t.sources)}"
            for t in tasks
        )
        prompt = VERIFY_PROMPT.format(
            question=question,
            tasks_block=tasks_block or "（无）",
            summary=report.summary,
            findings="\n".join(f"- {f}" for f in report.key_findings) or "（无）",
            unresolved="\n".join(f"- {q}" for q in report.unresolved_questions)
            or "（无）",
            citation_count=len(report.citations),
        )

        try:
            data = self._llm.chat_json([{"role": "user", "content": prompt}])
        except ValueError:
            return self._heuristic(report, tasks)
        
        # 列表推导批量构造模型
        issues = [
            VerificationIssue(
                task_id=str(i.get("task_id", "")),
                issue_type=str(i.get("issue_type", "incomplete")),
                description=str(i.get("description", "")),
                severity=str(i.get("severity", "medium")),
            )  # dict.get(键, 默认值)安全字典取值，批量实例化 Pydantic 校验问题对象
            for i in data.get("issues", [])
        ]

        supplements: list[ResearchTask] = []  
        for item in data.get("supplement_tasks", []):
            sid = str(item.get("id") or f"s_{uuid.uuid4().hex[:6]}")
            supplements.append(
                ResearchTask(
                    id=sid,
                    question=str(item.get("question") or ""),
                    keywords=[str(k) for k in (item.get("keywords") or [])],  # 多层列表推导统一清洗 keywords/depends_on，空值兜底空列表
                    depends_on=[str(d) for d in (item.get("depends_on") or [])],
                    priority=int(item.get("priority") or 1),
                    status=TaskStatus.PENDING,
                    is_supplement=True,
                )
            )

        satisfied = bool(data.get("satisfied", False))
        # 有高严重度问题或补充任务时不允许满意
        if any(i.severity == "high" for i in issues) or supplements:
            satisfied = False
        if satisfied:
            supplements = []

        # 过滤空问题  最多保留 3 条补充任务，限制资源消耗
        supplements = [t for t in supplements if t.question.strip()][:3]

        return VerificationResult(
            satisfied=satisfied,
            issues=issues,
            supplement_tasks=supplements,
            notes=str(data.get("notes", "")),
        )

    def _heuristic(
        self, report: ResearchReport, tasks: list[ResearchTask]
    ) -> VerificationResult:
        """LLM 失败时的启发式质检。"""
        weak = [   # weak + failed列表拼接合并两类异常任务
            t
            for t in tasks
            if t.status == TaskStatus.DONE and (len(t.sources) < 1 or len(t.findings) < 1)
        ]
        failed = [t for t in tasks if t.status == TaskStatus.FAILED]   
        issues: list[VerificationIssue] = []
        supplements: list[ResearchTask] = []

        for t in weak + failed:
            issues.append(
                VerificationIssue(
                    task_id=t.id,
                    issue_type="insufficient_evidence",
                    description=f"子任务 {t.id} 证据不足或失败",
                    severity="high" if t.status == TaskStatus.FAILED else "medium",
                )
            )
            supplements.append(
                ResearchTask(
                    id=f"s_{t.id}_{uuid.uuid4().hex[:4]}",
                    question=f"补充研究：{t.question}",
                    keywords=t.keywords or [t.question],
                    depends_on=[],
                    priority=1,
                    is_supplement=True,
                )
            )

        if len(report.citations) == 0 and tasks:
            issues.append(
                VerificationIssue(
                    task_id="",
                    issue_type="incomplete",
                    description="报告缺少引用",
                    severity="medium",
                )
            )

        supplements = supplements[:3]
        return VerificationResult(
            satisfied=not issues,
            issues=issues,
            supplement_tasks=supplements,
            notes="启发式验证结果",
        )
