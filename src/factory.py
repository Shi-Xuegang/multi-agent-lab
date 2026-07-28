"""应用工厂：组装所有组件。"""

from __future__ import annotations

from dataclasses import dataclass

from src.agent.research_agent import ResearchAgent
from src.agents.planner import PlannerAgent
from src.agents.researcher import ResearcherAgent
from src.agents.synthesizer import SynthesizerAgent
from src.agents.verifier import VerifierAgent
from src.config import Settings, get_settings
from src.embeddings.base import BaseEmbedder
from src.embeddings.factory import create_embedder
from src.graph.base import GraphStore
from src.graph.factory import create_graph_store
from src.llm.siliconflow import SiliconFlowClient
from src.memory.qdrant_store import QdrantMemoryStore
from src.memory.qdrant_utils import resolve_qdrant_vector_size
from src.orchestrator.orchestrator import ResearchOrchestrator
from src.tools.qdrant_search import QdrantSearchTool
from src.tools.registry import ToolRegistry
from src.tools.tavily_search import TavilySearchTool


@dataclass # 纯数据容器，自动生成构造函数、打印方法、等值判断
class AppComponents:
    """共享基础设施组件。"""

    settings: Settings           # 全局配置对象，统一入口
    llm: SiliconFlowClient       # 大模型客户端，所有 Agent 复用同一个连接池
    embedder: BaseEmbedder       # 嵌入模型实例，向量存储与检索共用
    memory: QdrantMemoryStore    # Qdrant 向量存储实例，全局复用同一个数据库连接
    tools: ToolRegistry          # 工具注册表，所有 Agent 共享同一套工具集
    tavily: TavilySearchTool     # 单独暴露 Tavily 工具实例，供需要直接调用的 Agent 使用
    graph: GraphStore            # 知识图谱存储实例，所有 Agent 共享同一套图谱数据


def _validate_settings(settings: Settings) -> None:
    """检查必要的 API Key 是否已配置。"""
    missing: list[str] = []

    if not settings.effective_llm_api_key:
        missing.append("SILICONFLOW_API_KEY 或 LLM_API_KEY")
    if not settings.tavily_api_key:
        missing.append("TAVILY_API_KEY")
    if not settings.embed_api_key:
        missing.append("EMBED_API_KEY")
    if not settings.qdrant_url:
        missing.append("QDRANT_URL")

    if missing:
        raise EnvironmentError(
            f"缺少必要的环境变量: {', '.join(missing)}。"
            f"请检查 .env 配置文件。"
        )


def create_components(settings: Settings | None = None) -> AppComponents:
    """创建可复用的基础设施组件。"""
    settings = settings or get_settings()
    _validate_settings(settings)

    llm = SiliconFlowClient.from_settings(settings)

    vector_size = resolve_qdrant_vector_size( # 先查询 Qdrant 中现有集合的向量维度
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection,
        default_size=settings.qdrant_vector_size,
        api_key=settings.qdrant_api_key,
        timeout=settings.qdrant_timeout,
    )

    # 调用嵌入工厂create_embedder，传入对齐后的维度参数，保证嵌入输出与向量库维度严格一致
    embedder = create_embedder(settings, dimension=vector_size)

    memory = QdrantMemoryStore(  # 向量存储实例化
    # 所有组件的依赖都从外部注入，组件内部不主动创建依赖
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.qdrant_collection,
        embedder=embedder,    
        vector_size=vector_size,
        distance=settings.qdrant_distance,
        timeout=settings.qdrant_timeout,
        chunk_strategy=settings.chunk_strategy,
    )

    tavily = TavilySearchTool( # 实例化 Tavily 搜索工具
        api_key=settings.tavily_api_key,
        max_results=settings.tavily_max_results,
    )

    registry = ToolRegistry() # 实例化工具注册表
    registry.register(tavily) # 注册 Tavily 搜索工具
    registry.register(QdrantSearchTool(memory)) # 注册 Qdrant 搜索工具  

    graph = create_graph_store(settings) # 创建知识图谱存储实例

    return AppComponents( # 所有基础设施封装进AppComponents数据类，供上层工厂函数复用
        settings=settings,
        llm=llm,
        embedder=embedder,
        memory=memory,
        tools=registry,
        tavily=tavily,
        graph=graph,
    )


def create_research_agent(settings: Settings | None = None) -> ResearchAgent:
    """创建完整配置的 ResearchAgent 实例（单 Agent 模式）。"""
    # 单 Agent 职责：独立完成思考、工具调用、RAG 检索、回答生成，适合简单问题快速响应
    components = create_components(settings)
    return ResearchAgent(
        llm=components.llm,  # 独立完成思考
        tools=components.tools,# 使用工具集
        memory=components.memory, # 使用向量存储
        max_iterations=components.settings.max_tool_iterations, # 使用最大迭代次数
        rag_score_threshold=components.settings.rag_score_threshold, # 使用 RAG 得分阈值
    )


def create_orchestrator(
    settings: Settings | None = None,
    max_verify_rounds: int | None = None,
    enable_kg: bool = True,
    enable_visualization: bool = True,
) -> ResearchOrchestrator:
    """创建多 Agent Orchestrator。"""
    components = create_components(settings)
    settings = components.settings

    # LLM + 图存储：任务拆解、生成 DAG 依赖关系
    planner = PlannerAgent(llm=components.llm, graph=components.graph) 
    
    # LLM + 记忆 + 搜索工具 + 图存储：  执行子任务检索、抽取知识、写入图谱
    researcher = ResearcherAgent(
        llm=components.llm, # 使用大模型
        memory=components.memory, # 使用向量存储
        tavily=components.tavily, # 使用 Tavily 搜索工具
        graph=components.graph, # 使用知识图谱
        rag_score_threshold=settings.rag_score_threshold, # 使用 RAG 得分阈值
        enable_kg=enable_kg, # 使用知识图谱
    )

    # LLM + 图存储： 汇总所有子任务结果，生成最终报告
    synthesizer = SynthesizerAgent(llm=components.llm, graph=components.graph) 
    
    # LLM： 多轮迭代验证、优化答案
    verifier = VerifierAgent(llm=components.llm)

    return ResearchOrchestrator(
        planner=planner,
        researcher=researcher,
        synthesizer=synthesizer,
        verifier=verifier,
        graph=components.graph,
        max_verify_rounds=(
            settings.max_verify_rounds
            if max_verify_rounds is None
            else max_verify_rounds
        ),
        concurrency=settings.researcher_concurrency,
        max_task_retries=settings.max_task_retries,
        enable_visualization=enable_visualization,
    )
