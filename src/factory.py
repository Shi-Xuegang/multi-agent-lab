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


@dataclass
class AppComponents:
    """共享基础设施组件。"""

    settings: Settings
    llm: SiliconFlowClient
    embedder: BaseEmbedder
    memory: QdrantMemoryStore
    tools: ToolRegistry
    tavily: TavilySearchTool
    graph: GraphStore


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

    vector_size = resolve_qdrant_vector_size(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection,
        default_size=settings.qdrant_vector_size,
        api_key=settings.qdrant_api_key,
        timeout=settings.qdrant_timeout,
    )

    embedder = create_embedder(settings, dimension=vector_size)

    memory = QdrantMemoryStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.qdrant_collection,
        embedder=embedder,
        vector_size=vector_size,
        distance=settings.qdrant_distance,
        timeout=settings.qdrant_timeout,
        chunk_strategy=settings.chunk_strategy,
    )

    tavily = TavilySearchTool(
        api_key=settings.tavily_api_key,
        max_results=settings.tavily_max_results,
    )

    registry = ToolRegistry()
    registry.register(tavily)
    registry.register(QdrantSearchTool(memory))

    graph = create_graph_store(settings)

    return AppComponents(
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
    components = create_components(settings)
    return ResearchAgent(
        llm=components.llm,
        tools=components.tools,
        memory=components.memory,
        max_iterations=components.settings.max_tool_iterations,
        rag_score_threshold=components.settings.rag_score_threshold,
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

    planner = PlannerAgent(llm=components.llm, graph=components.graph)
    researcher = ResearcherAgent(
        llm=components.llm,
        memory=components.memory,
        tavily=components.tavily,
        graph=components.graph,
        rag_score_threshold=settings.rag_score_threshold,
        enable_kg=enable_kg,
    )
    synthesizer = SynthesizerAgent(llm=components.llm, graph=components.graph)
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
