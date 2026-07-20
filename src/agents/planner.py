"""Planner Agent：将复杂问题拆解为子任务 DAG。

LLM 解析失败降级兜底 → ID 重复自动加随机后缀 → 非法字符正则清洗
 → 无效依赖二次过滤 → DFS 自动破除循环依赖，
多层保障输出合法无环任务图。


run：对外入口，调度整体流程；
_normalize_tasks：数据清洗、类型转换；
_break_cycles：图结构校验、去环；
_fallback_tasks：故障降级兜底
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from src.agents.base import BaseAgent
from src.graph.base import GraphStore
from src.llm.siliconflow import SiliconFlowClient
from src.models.schemas import ResearchTask, TaskDag, TaskStatus


PLANNER_PROMPT = """你是研究规划专家。请将用户的复杂研究问题拆解为 3-6 个子任务 DAG。

## 用户问题
{question}

## 用户偏好
{user_context}

请严格输出 JSON：
{{
  "tasks": [
    {{
      "id": "t1",
      "question": "子问题",
      "keywords": ["搜索词1", "搜索词2"],
      "depends_on": [],
      "priority": 1
    }}
  ]
}}

要求：
1. id 使用 t1, t2, ... 简短标识
2. depends_on 只能引用已存在的任务 id，形成有向无环图
3. 覆盖问题的主要研究维度，避免重复
4. keywords 用于联网搜索，可中英混合
5. priority 数字越小越优先
"""


class PlannerAgent(BaseAgent):
    """规划拆解 Agent。"""

    def __init__(self, llm: SiliconFlowClient, graph: GraphStore) -> None:
        self._llm = llm
        self._graph = graph

    @property
    def name(self) -> str:
        return "planner"

    def run(
        self,
        question: str,
        session_id: str,
        user_context: str = "",
        **_: Any,                 #捕获多余无关关键字参数，下划线_表示该参数不会被使用，符合编码规范
    ) -> TaskDag:                 # 强类型返回结构化 DAG 模型
        """生成并持久化任务 DAG。"""
        prompt = PLANNER_PROMPT.format(
            question=question,
            user_context=user_context or "（无）",
        )
        try:
            data = self._llm.chat_json([{"role": "user", "content": prompt}])
            raw_tasks = data.get("tasks", [])
        except ValueError:
            #LLM JSON 解析失败抛ValueError时，不直接报错，调用内置兜底方法生成固定 3 步任务，
            # 保证流程不中断，线上鲁棒性。
            raw_tasks = self._fallback_tasks(question)

        tasks = self._normalize_tasks(raw_tasks, question)
        dag = TaskDag(
            root_question=question,
            tasks=tasks,
            session_id=session_id,
        )
        self._graph.create_session(session_id, question)
        self._graph.save_task_dag(dag)
        return dag

    # 清洗 LLM 返回的脏原始任务字典，统一 ID、依赖、关键词、优先级格式，
    # 转换为强类型ResearchTask对象，处理重复 ID、非法字符、无效依赖
    def _normalize_tasks(
        self, raw_tasks: list[dict[str, Any]], question: str
    ) -> list[ResearchTask]:
        if not raw_tasks:
            return self._fallback_tasks(question)

        tasks: list[ResearchTask] = []      # 显式变量类型标注；
        seen_ids: set[str] = set()     # 集合set用于 O (1) 查重任务 ID
        for i, item in enumerate(raw_tasks):
            task_id = str(item.get("id") or f"t{i + 1}")     # 默认生成 t1, t2, ...
            task_id = re.sub(r"[^a-zA-Z0-9_\-]", "", task_id) or f"t{i + 1}"     # 清洗非法字符
            if task_id in seen_ids:
                task_id = f"{task_id}_{uuid.uuid4().hex[:4]}"    #截取 4 位随机十六进制字符串，重复任务 ID 追加随机后缀，避免冲突
            seen_ids.add(task_id)

            deps = [
                str(d) for d in (item.get("depends_on") or [])  # 依赖项全部转为字符串，空兜底为空列表
                if str(d) in seen_ids or str(d) != task_id    # 二次过滤无效依赖：不重复自身，避免循环依赖
            ]
            # 稍后二次过滤无效依赖
            tasks.append(
                ResearchTask(
                    id=task_id,
                    question=str(item.get("question") or question),
                    keywords=[str(k) for k in (item.get("keywords") or [])][:5],  # 关键词列表截取前 5 个，兜底空列表
                    depends_on=deps,
                    priority=int(item.get("priority") or i + 1),
                    status=TaskStatus.PENDING,
                )
            )

        valid_ids = {t.id for t in tasks}
        for task in tasks:
            task.depends_on = [d for d in task.depends_on if d in valid_ids and d != task.id]  # 过滤无效依赖：只保留已存在、不重复自身

        return self._break_cycles(tasks)   # 传递清洗后任务，专门处理环形依赖

    def _break_cycles(self, tasks: list[ResearchTask]) -> list[ResearchTask]:
        """简单去环：保留拓扑可调度结构。"""  
         # 深度优先搜索DFS算法，检测并移除环形依赖，确保任务DAG无环，可调度执行
        by_id = {t.id: t for t in tasks}
        visiting: set[str] = set()  #当前递归栈中正在遍历的节点，检测环路核心；
        visited: set[str] = set()  #已访问过的节点，避免重复遍历；

        def dfs(node: str) -> None:
            if node in visited:
                return

            """节点存在于visiting集合 = 递归回头访问当前路径上的节点，
                说明存在循环依赖；直接删除该依赖打破环路"""
            if node in visiting:
                by_id[node].depends_on = [
                    d for d in by_id[node].depends_on if d not in visiting
                ]
                return
            visiting.add(node)
            for dep in list[str](by_id[node].depends_on):
                if dep in by_id:
                    dfs(dep)
            visiting.remove(node)
            visited.add(node)

        for tid in list[str](by_id):
            dfs(tid)
        return list[ResearchTask](by_id.values())

    def _fallback_tasks(self, question: str) -> list[ResearchTask]:
        """LLM 失败时的保守拆解。"""
        return [
            ResearchTask(
                id="t1",
                question=f"{question} — 背景与定义",
                keywords=[question, "overview"],
                depends_on=[],
                priority=1,
            ),
            ResearchTask(
                id="t2",
                question=f"{question} — 关键数据与趋势",
                keywords=[question, "trend", "data"],
                depends_on=["t1"],
                priority=2,
            ),
            ResearchTask(
                id="t3",
                question=f"{question} — 影响与结论",
                keywords=[question, "impact"],
                depends_on=["t2"],
                priority=3,
            ),
        ]
