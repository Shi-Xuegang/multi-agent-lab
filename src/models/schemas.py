"""核心数据结构定义（Pydantic models）。

本文件是整个研究系统的「数据合同」：
- Tool / 单 Agent 用搜索结果、工具记录、报告结构
- 多 Agent 用任务 DAG、验证结果、实体关系
- 评估阶段用分数结构

统一用 Pydantic BaseModel 的好处：
1. 字段类型自动校验，减少脏数据流入下游
2. model_dump() / model_dump(mode=\"json\") 方便序列化到 JSON、图存储、日志
3. 与 LLM 的 JSON 输出字段对齐，解析失败时可降级处理
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# BaseModel：结构化数据校验与序列化
# Field(default_factory=...)：可变默认值（list/dict）每次新建，避免共享同一对象

# ---------------------------------------------------------------------------
# 搜索与引用（Phase 1 / 单 Agent + Researcher）
# ---------------------------------------------------------------------------
class SearchResult(BaseModel):
    """单条搜索/缓存文档结果。

    来源可以是：
    - Tavily 联网搜索
    - Qdrant 语义检索命中的缓存文档（字段结构保持一致，便于统一处理）
    """

    title: str
    """网页或文档标题。"""

    url: str
    """来源 URL；也用于入库去重（url_hash）。"""

    content: str
    """正文摘要或 chunk 文本，供 LLM 阅读与 Embedding。"""

    score: float = 0.0
    """相关性分数：Tavily 相关分，或 Qdrant 余弦相似度等。"""


class Citation(BaseModel):
    """报告中的一条可核验引用。

    约定：url 应来自真实证据，不要让模型凭空编造。
    """

    index: int
    """引用编号，与正文中的 [1][2] 对应。"""

    title: str
    """来源标题，便于人类阅读。"""

    url: str
    """来源链接，评估阶段可用来做「引用是否真实存在于证据」校验。"""


# ---------------------------------------------------------------------------
# 研究报告（单 Agent / Synthesizer 最终产物）
# ---------------------------------------------------------------------------
class ResearchReport(BaseModel):
    """结构化研究报告。

    - single 模式：ResearchAgent 直接生成
    - multi 模式：Synthesizer（+ Verifier 循环后）生成，可附带过程图/知识图谱
    """
     
    question: str
    """用户原始研究问题。"""

    summary: str
    """2～4 句核心摘要。"""

    key_findings: list[str] = Field(default_factory=list)
    """关键发现列表；条目中宜带引用标记，如「发现内容 [1][2]」。"""

    citations: list[Citation] = Field(default_factory=list)
    """引用来源列表，index 与 key_findings 中的编号对应。"""

    raw_answer: str = ""
    """模型直接文本回答的原文备份（未走 JSON 结构化时常用）。"""

    sections: list[str] = Field(default_factory=list)
    """分节论述（多 Agent 综合报告更常用）。"""

    unresolved_questions: list[str] = Field(default_factory=list)
    """尚未充分回答的问题；Verifier 会重点参考。"""

    process_diagram: str = ""
    """研究过程 Mermaid 文本（子任务 DAG → 来源 → 章节）。"""

    kg_diagram: str = ""
    """知识图谱 Mermaid 文本（实体关系快照）。"""

    verify_rounds: int = 0
    """自反思实际执行轮次；评估指标 reflection_rounds 会用到。"""

    mode: str = "single"
    """运行模式标识：\"single\" | \"multi\" 等，便于日志与评估区分。"""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """附加统计，如 cache_hits、web_searches、session_id、graph_backend、编排事件等。"""


# ---------------------------------------------------------------------------
# 工具调用（Tool Calling 生命周期记录）
# ---------------------------------------------------------------------------
class ToolExecutionResult(BaseModel):
    """单次工具执行的返回结果。

    ToolRegistry / 各 Tool.execute() 统一返回此结构，
    再序列化后作为 role=tool 的消息回传给 LLM。
    """

    tool_name: str
    """工具名，如 tavily_search、qdrant_search。"""

    success: bool
    """是否执行成功。"""

    data: Any = None
    """成功时的载荷（常为 dict，含 results / from_cache 等）。"""

    error: str | None = None
    """失败时的错误信息；成功时为 None。"""


class ToolCallRecord(BaseModel):
    """一次完整的工具调用审计记录（参数 + 结果 + 时间）。

    用于 CLI 展示「工具调用历史」，以及报告生成时的 tool_calls 上下文。
    """

    tool_name: str
    """被调用的工具名。"""

    arguments: dict[str, Any]
    """模型生成的调用参数（已解析为 dict）。"""

    result: ToolExecutionResult
    """本地执行后的结果（可能是真实搜索，也可能是缓存命中伪装的成功结果）。"""

    timestamp: datetime = Field(default_factory=datetime.now)
    """调用发生时间。"""


# ---------------------------------------------------------------------------
# 多 Agent 任务图（Planner / Orchestrator / Researcher）
# ---------------------------------------------------------------------------


class TaskStatus(str, Enum):
    """子任务状态机。

    典型流转：
    pending → in_progress → done
                         ↘ failed（可重试则回到 pending）
    """

    PENDING = "pending"
    """等待依赖满足或等待调度。"""

    IN_PROGRESS = "in_progress"
    """Researcher 正在执行。"""

    DONE = "done"
    """已完成，findings/sources 已写入。"""

    FAILED = "failed"
    """失败；超过最大重试或依赖失败时保持此状态。"""


class ResearchTask(BaseModel):
    """研究子任务（DAG 中的一个节点）。

    由 Planner 生成；Orchestrator 按 depends_on 调度；
    Researcher 执行后回写 findings/sources；Verifier 可追加 is_supplement=True 的补充任务。
    """

    id: str
    """任务唯一标识，如 t1、s1（补充任务常用 s 前缀）。"""

    question: str
    """该子任务要研究的具体问题。"""

    keywords: list[str] = Field(default_factory=list)
    """建议搜索词，Researcher 会优先用这些 query 去检索/联网。"""

    depends_on: list[str] = Field(default_factory=list)
    """依赖的其它任务 id 列表；全部 done 后才可被调度。"""

    priority: int = 1
    """优先级，数字越小通常越优先（具体排序策略由调度器决定）。"""

    status: TaskStatus = TaskStatus.PENDING
    """当前状态，见 TaskStatus。"""

    retries: int = 0
    """已失败重试次数；与 max_task_retries 比较决定是否继续重试。"""

    findings: list[str] = Field(default_factory=list)
    """本子任务提炼出的关键发现（供 Synthesizer 综合）。"""

    sources: list[dict[str, Any]] = Field(default_factory=list)
    """来源列表，元素通常含 url / title / snippet。"""

    error: str | None = None
    """最近一次失败原因；成功完成时应清空或忽略。"""

    is_supplement: bool = False
    """是否为 Verifier 触发的补充研究任务。"""


class TaskDag(BaseModel):
    """一次研究会话下的子任务有向无环图（DAG）。

    Planner 的产出物；持久化到 MemoryGraphStore 或 Neo4j。
    """

    root_question: str
    """用户总问题。"""

    tasks: list[ResearchTask] = Field(default_factory=list)
    """全部子任务节点（含依赖边信息 depends_on）。"""

    session_id: str = "default"
    """会话 ID，用于图存储与 Qdrant 文档分组。"""


# ---------------------------------------------------------------------------
# 知识图谱（实体关系抽取）
# ---------------------------------------------------------------------------
class EntityRelation(BaseModel):
    """知识图谱三元组：(subject)-[predicate]->(object)。

    由 KG 抽取器从证据中抽出，写入图存储，供 Synthesizer / 可视化使用。
    """

    subject: str
    """头实体，如公司、技术、政策名。"""

    predicate: str
    """关系类型，如 竞争、供应、影响、依赖、合作 等。"""

    object: str
    """尾实体。"""

    evidence_url: str = ""
    """支撑该关系的来源 URL（可空）。"""

    confidence: float = 0.5
    """模型给出的置信度，建议落在 [0, 1]。"""


# ---------------------------------------------------------------------------
# 自反思验证（Verifier）
# ---------------------------------------------------------------------------
class VerificationIssue(BaseModel):
    """Verifier 发现的单个质量问题。"""

    task_id: str = ""
    """关联的子任务 id；与整篇报告相关时可为空。"""

    issue_type: str
    """问题类型，如 insufficient_evidence / contradiction / low_quality_source / incomplete。"""

    description: str
    """问题说明，供日志与后续人工查看。"""

    severity: str = "medium"
    """严重程度：low | medium | high；high 通常强制判定不合格。"""


class VerificationResult(BaseModel):
    """Verifier 一轮质检的完整输出。

    Orchestrator 根据本结果决定：
    - satisfied=True 且无补充任务 → 结束循环，输出最终报告
    - 否则把 supplement_tasks 写入图，再调度 Researcher，然后重新 Synthesize
    """

    satisfied: bool
    """是否达到「合格」标准（可交差）。"""

    issues: list[VerificationIssue] = Field(default_factory=list)
    """发现的问题列表。"""

    supplement_tasks: list[ResearchTask] = Field(default_factory=list)
    """需要追加的补充研究任务；非空时一般视为未合格。"""

    notes: str = ""
    """总体评价或说明文字。"""


# ---------------------------------------------------------------------------
# 评估体系（Evaluation）
# ---------------------------------------------------------------------------
class EvaluationScores(BaseModel):
    """自动化评估分数（各维度建议 0～100）。

    用于 single vs multi、有无自反思等对比实验。
    """

    completeness: float = 0.0
    """完整性：子任务覆盖、发现充实度、未决问题惩罚等。"""

    accuracy: float = 0.0
    """准确性启发式：引用 URL 是否出现在证据来源中。"""

    citation_density: float = 0.0
    """引用密度：正文引用标记与引用条数的综合分。"""

    reflection_rounds: float = 0.0
    """自反思轮次得分（由 verify_rounds 映射而来）。"""

    overall: float = 0.0
    """加权综合分。"""

    details: dict[str, Any] = Field(default_factory=dict)
    """明细，如 task_count、citation_count、error 信息等。"""
