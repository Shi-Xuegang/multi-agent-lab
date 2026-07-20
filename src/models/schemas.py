"""核心数据结构定义。"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """单条搜索结果。"""

    title: str
    url: str
    content: str
    score: float = 0.0


class Citation(BaseModel):
    """报告引用。"""

    index: int
    title: str
    url: str


class ResearchReport(BaseModel):
    """结构化研究报告。"""

    question: str
    summary: str
    key_findings: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    raw_answer: str = ""
    sections: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    process_diagram: str = ""
    kg_diagram: str = ""
    verify_rounds: int = 0
    mode: str = "single"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolExecutionResult(BaseModel):
    """工具执行结果。"""

    tool_name: str
    success: bool
    data: Any = None
    error: str | None = None


class ToolCallRecord(BaseModel):
    """一次工具调用的完整记录。"""

    tool_name: str
    arguments: dict[str, Any]
    result: ToolExecutionResult
    timestamp: datetime = Field(default_factory=datetime.now)


class TaskStatus(str, Enum):
    """子任务状态。"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"


class ResearchTask(BaseModel):
    """研究子任务。"""

    id: str
    question: str
    keywords: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    priority: int = 1
    status: TaskStatus = TaskStatus.PENDING
    retries: int = 0
    findings: list[str] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None
    is_supplement: bool = False


class TaskDag(BaseModel):
    """子任务有向无环图。"""

    root_question: str
    tasks: list[ResearchTask] = Field(default_factory=list)
    session_id: str = "default"


class EntityRelation(BaseModel):
    """知识图谱三元组。"""

    subject: str
    predicate: str
    object: str
    evidence_url: str = ""
    confidence: float = 0.5


class VerificationIssue(BaseModel):
    """验证发现问题。"""

    task_id: str = ""
    issue_type: str
    description: str
    severity: str = "medium"


class VerificationResult(BaseModel):
    """Verifier 输出。"""

    satisfied: bool
    issues: list[VerificationIssue] = Field(default_factory=list)
    supplement_tasks: list[ResearchTask] = Field(default_factory=list)
    notes: str = ""


class EvaluationScores(BaseModel):
    """评估分数（0-100）。"""

    completeness: float = 0.0
    accuracy: float = 0.0
    citation_density: float = 0.0
    reflection_rounds: float = 0.0
    overall: float = 0.0
    details: dict[str, Any] = Field(default_factory=dict)
