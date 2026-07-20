"""图存储工厂：优先 Neo4j，失败回退内存。"""

from __future__ import annotations

from src.config import Settings, get_settings
from src.graph.base import GraphStore
from src.graph.memory_store import MemoryGraphStore


def create_graph_store(settings: Settings | None = None) -> GraphStore:
    """创建图存储；Neo4j 不可用时自动回退 MemoryGraphStore。"""
    settings = settings or get_settings()

    if not settings.neo4j_uri or not settings.neo4j_username:
        print("[graph] 未配置 Neo4j，使用内存图存储")
        return MemoryGraphStore()

    try:
        from src.graph.neo4j_store import Neo4jGraphStore

        store = Neo4jGraphStore(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
            max_connection_lifetime=settings.neo4j_max_connection_lifetime,
            max_connection_pool_size=settings.neo4j_max_connection_pool_size,
            connection_timeout=settings.neo4j_connection_timeout,
        )
        print("[graph] 已连接 Neo4j")
        return store
    except Exception as exc:
        print(f"[graph] Neo4j 连接失败，回退内存图存储: {exc}")
        return MemoryGraphStore()
