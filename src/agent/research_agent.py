"""单 Agent 研究核心：RAG 缓存 + Tool calling 循环 + 总结 + 记忆存储。"""

from __future__ import annotations

import json
from typing import Any

from src.llm.siliconflow import SiliconFlowClient
from src.memory.qdrant_store import QdrantMemoryStore
from src.memory.user_preferences import user_preference_manager
from src.models.schemas import (
    Citation,
    ResearchReport,
    SearchResult,
    ToolCallRecord,
    ToolExecutionResult,
)
from src.tools.registry import ToolRegistry


SYSTEM_PROMPT = """你是一个专业的深度研究助手。你的任务是回答用户的研究问题。

## 工作流程
1. 分析用户问题，判断需要搜索哪些信息
2. 优先使用 qdrant_search 检索本地向量记忆，若已有高相关证据可直接使用
3. 本地证据不足时，再使用 tavily_search 搜索互联网
4. 如果需要多个角度的信息，可以多次调用搜索工具
5. 综合所有证据，给出准确、有引用的回答

## 规则
- 优先复用本地缓存，避免不必要的重复搜索
- 搜索关键词要精准，可以中英文混合
- 对于复杂问题，拆分为多个搜索查询
- 回答时必须标注信息来源
- 如果搜索结果不足以回答问题，明确说明不足之处
- 尊重下方用户偏好（输出风格、研究深度、领域偏好等）

## 输出要求
当你完成所有必要的搜索后，请直接给出最终回答，格式如下：
- 先给出 2-3 句话的核心摘要
- 然后分点列出关键发现
- 每个关键发现标注引用编号，如 [1][2]
- 最后列出引用来源列表

## 用户偏好上下文
{user_context}
"""


SUMMARIZE_PROMPT = """基于以下研究问题和收集到的证据，生成一份结构化研究报告。

## 研究问题
{question}

## 用户偏好
{user_context}

## 收集到的证据
{evidence}

## 工具调用记录
{tool_calls}

请以 JSON 格式输出，结构如下：
{{
  "summary": "2-3句话核心摘要",
  "key_findings": ["关键发现1", "关键发现2"],
  "citations": [
    {{"index": 1, "title": "来源标题", "url": "来源URL"}}
  ],
  "unresolved_questions": ["尚未充分回答的问题"]
}}

要求：
1. key_findings 中每条发现标注引用编号，如 "发现内容 [1][2]"
2. citations 中的 index 与 key_findings 中的编号对应
3. 只引用实际存在的来源，不要编造
4. 所有输出使用中文
"""


class ResearchAgent:
    """
    单 Agent 研究流水线。
    生命周期：接收问题 → RAG 预检 → Tool calling 循环 → 总结 → 存入 Qdrant
    """

    def __init__(
        self,
        llm: SiliconFlowClient,
        tools: ToolRegistry,
        memory: QdrantMemoryStore,
        max_iterations: int = 5,
        rag_score_threshold: float = 0.8,
    ) -> None:
        self._llm = llm # 大模型实例
        self._tools = tools
        self._memory = memory
        self._max_iterations = max_iterations
        self._rag_score_threshold = rag_score_threshold
        self._tool_call_history: list[ToolCallRecord] = []   # 保存本轮所有工具调用记录
        self._collected_evidence: list[dict[str, Any]] = []   # 保存本轮所有收集到的证据
        self._cache_hits: int = 0  # 保存本轮所有命中本地向量库的缓存次数
        self._web_searches: int = 0  # 保存本轮所有联网搜索次数

    def research(self, question: str, session_id: str = "default") -> ResearchReport:
        """执行完整的研究流程。"""
        self._tool_call_history.clear()   # 清空本轮所有工具调用记录
        self._collected_evidence.clear()   # 清空本轮所有收集到的证据
        self._cache_hits = 0   # 清空本轮所有命中本地向量库的缓存次数   
        self._web_searches = 0   # 清空本轮所有联网搜索次数

        user_context = user_preference_manager.get_context_prompt()   # 获取用户偏好上下文
        system_prompt = SYSTEM_PROMPT.format(user_context=user_context)   # 构建系统提示

        # 研究开始前：全局语义缓存预检
        cached = self._memory.search_similar(   # 检索本地向量库
            query=question,   # 查询问题
            limit=5,   # 限制最多返回5条相关文档
            score_threshold=self._rag_score_threshold,   # 设置本地向量库检索的相似度阈值
        )
        cache_note = ""   # 本轮所有命中本地向量库的缓存提示    
        if cached:
            self._cache_hits += len(cached)   # 更新本轮所有命中本地向量库的缓存次数
            for item in cached:
                self._collected_evidence.append(item)   # 更新本轮所有收集到的证据
            cache_note = (
                f"\n\n【系统提示】本地向量库已检索到 {len(cached)} 条高相关文档。"
                "请优先基于这些证据回答；仅在证据明显不足时再调用 tavily_search。"
            )

        messages: list[dict[str, Any]] = [ 
            {"role": "system", "content": system_prompt},   # 系统提示
            {"role": "user", "content": question + cache_note},   # 用户问题 + 本轮所有命中本地向量库的缓存提示
        ]
        openai_tools = self._tools.to_openai_tools()   # 将工具转换为 OpenAI 格式

        for _iteration in range(self._max_iterations):   # 迭代最大次数
            response = self._llm.chat(messages, tools=openai_tools)   # 调用大模型

            if not response["tool_calls"]:
                if response["content"]:   # 如果模型返回了文本内容
                    report = self._build_report_from_text(question, response["content"])
                    self._finalize(question, report)   # 结束本轮研究
                    return report
                break   # 如果没有工具调用，则结束本轮研究  

            messages.append( 
                {   # 添加助手消息
                    "role": "assistant",
                    "content": response["content"],
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(
                                    tc["arguments"], ensure_ascii=False
                                ),
                            },
                        }
                        for tc in response["tool_calls"] #单次返回多个并行工具调用，代码会循环遍历全部 tc 批量执行
                    ],
                }
            )

            for tc in response["tool_calls"]:   # 循环遍历全部 tc 批量执行
                tool_name = tc["name"]   # 获取工具名称
                arguments = tc["arguments"]   # 获取工具参数

                # 对联网搜索做“先查缓存”拦截
                if tool_name == "tavily_search":
                    query = str(arguments.get("query", "")).strip()   # 获取查询关键词
                    similar = self._memory.search_similar(   # 检索本地向量库
                        query=query or question,
                        limit=3,   # 限制最多返回3条相关文档
                        score_threshold=self._rag_score_threshold,   # 设置本地向量库检索的相似度阈值
                    )
                    if similar:   # 如果命中本地向量库
                        self._cache_hits += len(similar)   # 命中缓存：跳过真实联网，伪造tool返回结果
                        for item in similar:
                            self._collected_evidence.append(item)
                        cached_payload = {
                            "from_cache": True,
                            "query": query,
                            "results": similar,
                            "result_count": len(similar),
                            "message": "命中本地 Qdrant 缓存，已跳过 Tavily 联网搜索",
                        }
                        result = ToolExecutionResult(
                            tool_name=tool_name,
                            success=True,
                            data=cached_payload,
                        )
                        print(
                            f"【检索优化】query={query!r} 命中 {len(similar)} 条缓存，跳过 Tavily"
                        )
                    else: # 未命中缓存：真实联网搜索
                        result = self._tools.execute(tool_name, arguments)   # 执行工具
                        if result.success:   # 如果工具执行成功
                            self._web_searches += 1   # 更新本轮所有联网搜索次数
                            self._process_search_results(   # 将搜索结果存入本地向量库
                                query=query,   # 查询关键词
                                data=result.data,   # 搜索结果
                                session_id=session_id,   # 会话ID
                            )
                elif tool_name == "qdrant_search":  # qdrant_search 本地检索
                    result = self._tools.execute(tool_name, arguments)
                    if result.success and isinstance(result.data, dict):
                        for item in result.data.get("results", []):
                            self._collected_evidence.append(item)
                            self._cache_hits += 1
                else: # 其他工具：直接执行
                    result = self._tools.execute(tool_name, arguments) 

                record = ToolCallRecord(
                    tool_name=tool_name,
                    arguments=arguments,
                    result=result,
                )
                self._tool_call_history.append(record)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(result.model_dump(), ensure_ascii=False),  # 将结果转换为 JSON 格式
                    }
                )

        report = self._generate_structured_report(question, user_context)   # 生成结构化报告
        self._finalize(question, report)   # 结束本轮研究
        return report

    def _finalize(self, question: str, report: ResearchReport) -> None:
        """收尾：记录偏好历史与统计。"""
        user_preference_manager.add_research_topic(question)
        report.mode = "single"   # 设置报告模式为单agent模式
        report.metadata = {  # 设置报告元数据
            "cache_hits": self._cache_hits,   # 本轮所有命中本地向量库的缓存次数
            "web_searches": self._web_searches,   # 本轮所有联网搜索次数
            "evidence_count": len(self._collected_evidence),   # 本轮所有收集到的证据数量
        }

    def _process_search_results(   # 将搜索结果存入本地向量库
        self,
        query: str,   # 查询关键词
        data: Any,   # 搜索结果
        session_id: str,   # 会话ID
    ) -> None:
        """处理搜索结果：收集证据 + 分块存入向量库。"""
        if not isinstance(data, dict):   # 如果搜索结果不是字典，则结束
            return

        raw_results = data.get("results", [])   # 获取搜索结果
        search_results = [SearchResult(**r) for r in raw_results]   # 将搜索结果转换为 SearchResult 格式

        for sr in search_results:
            self._collected_evidence.append(sr.model_dump())   # 将搜索结果添加到本轮所有收集到的证据中

        stored = self._memory.store_search_results(
            query=query,
            results=search_results,
            session_id=session_id,
        )   # 将搜索结果存入本地向量库
        data["stored_to_qdrant"] = stored   # 设置搜索结果是否已存入本地向量库

    def _generate_structured_report(   # 生成结构化报告
        self, question: str, user_context: str   # 问题 + 用户偏好上下文
    ) -> ResearchReport:
        """基于收集的证据，让 LLM 生成结构化 JSON 报告。"""
        evidence_text = self._format_evidence()   # 格式化收集到的证据供 LLM 阅读
        tool_calls_text = self._format_tool_calls()   # 格式化工具调用记录
        prompt = SUMMARIZE_PROMPT.format(   # 构建总结提示
            question=question,   # 问题
            user_context=user_context,   # 用户偏好上下文
            evidence=evidence_text,   # 收集到的证据
            tool_calls=tool_calls_text,   # 工具调用记录
        )

        try:
            report_data = self._llm.chat_json(
                [{"role": "user", "content": prompt}]
            )
            return ResearchReport(
                question=question,
                summary=report_data.get("summary", ""),
                key_findings=report_data.get("key_findings", []),
                citations=[
                    Citation(**c) for c in report_data.get("citations", [])
                ],
                unresolved_questions=report_data.get("unresolved_questions", []),
            )
        except (ValueError, KeyError, TypeError):
            return ResearchReport(
                question=question,
                summary="报告生成过程中出现问题，以下为原始证据摘要。",
                key_findings=[
                    e.get("content", "")[:200] for e in self._collected_evidence[:5]
                ],
                citations=[
                    Citation(
                        index=i + 1,
                        title=e.get("title", ""),
                        url=e.get("url", ""),
                    )
                    for i, e in enumerate(self._collected_evidence[:10])
                ],
            )
        
        
        # 当模型中途判断证据充足、提前结束工具循环，输出自由文本时使用：
        # 自动遍历内存证据池生成引用列表，将自由文本塞入 key_findings，
        # 兼容两种输出模式，上层调用方无需区分 “直接回答” 和 “总结报告”，
        # 统一接收 ResearchReport 对象
    def _build_report_from_text(self, question: str, content: str) -> ResearchReport:
        """从模型的直接文本回答构建报告。"""
        citations = [
            Citation(index=i + 1, title=e.get("title", ""), url=e.get("url", ""))
            for i, e in enumerate(self._collected_evidence[:10])
        ]
        return ResearchReport(
            question=question,
            summary=content[:300],
            key_findings=[content],
            citations=citations,
            raw_answer=content,
        )

    # 证据压缩、去重、截断
    def _format_evidence(self) -> str:
        """格式化收集到的证据供 LLM 阅读。"""
        if not self._collected_evidence:
            return "（无搜索证据）"

        # 按 URL去重并截断，控制上下文窗口
        seen: set[str] = set()   # 保存已处理过的 URL
        parts: list[str] = []   # 保存处理后的证据
        idx = 1
        for ev in self._collected_evidence:   # 循环遍历所有收集到的证据
            url = ev.get("url", "")
            key = url or ev.get("content", "")[:80]   # 获取 URL 或内容的前80个字符作为唯一标识
            if key in seen:
                continue   # 如果已处理过，则跳过
            seen.add(key)   # 将 URL 或内容的前80个字符作为唯一标识添加到已处理过的 URL 集合中
            parts.append(   # 将证据添加到处理后的证据列表中
                f"[{idx}] {ev.get('title', '无标题')}\n"
                f"    URL: {url}\n"
                f"    内容: {ev.get('content', '')[:500]}"   # 截断内容为前500个字符
            )
            idx += 1   
            if idx > 20:   
                break  
        return "\n\n".join(parts)   

    def _format_tool_calls(self) -> str:   # 格式化工具调用记录
        """格式化工具调用记录。"""
        if not self._tool_call_history:   # 如果工具调用记录为空，则返回
            return "（无工具调用）"

        parts: list[str] = []   # 保存处理后的工具调用记录
        for record in self._tool_call_history:   # 循环遍历所有工具调用记录
            status = (   # 获取工具调用状态
                "成功" if record.result.success else f"失败: {record.result.error}"
            )
            parts.append(   # 将工具调用记录添加到处理后的工具调用记录列表中
                f"- {record.tool_name}"
                f"({json.dumps(record.arguments, ensure_ascii=False)}) → {status}"
            )
        return "\n".join(parts)   # 将处理后的工具调用记录列表拼接成字符串

    @property
    def tool_call_history(self) -> list[ToolCallRecord]:   # 获取本次研究的工具调用历史
        """获取本次研究的工具调用历史。"""
        return list(self._tool_call_history)

    @property
    def collected_evidence(self) -> list[dict[str, Any]]:
        """获取本次研究收集到的证据。"""
        return list(self._collected_evidence)
