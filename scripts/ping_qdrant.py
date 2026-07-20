"""Day6 验收：Qdrant 写入 + 检索。"""

from __future__ import annotations

import _bootstrap  # noqa: F401

from src.config import get_settings
from src.embeddings.factory import create_embedder
from src.memory.qdrant_store import QdrantMemoryStore
from src.memory.qdrant_utils import resolve_qdrant_vector_size
from src.models.schemas import SearchResult


def main() -> None:
    settings = get_settings()
    if not settings.qdrant_url:
        raise SystemExit("缺少 QDRANT_URL")
    if not settings.embed_api_key:
        raise SystemExit("缺少 EMBED_API_KEY")

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
        chunk_strategy="none",
    )

    sample = SearchResult(
        title="ping_qdrant 测试文档",
        url="https://example.com/ping-qdrant-smoke",
        content="这是多 Agent 深度研究系统的 Qdrant 连通性验收文档。",
        score=1.0,
    )
    stored = memory.store_search_results(
        query="qdrant ping",
        results=[sample],
        session_id="scripts_ping",
    )
    print(f"stored = {stored} (0 表示 URL 已存在，属正常)")

    hits = memory.search_similar("Qdrant 连通性验收", limit=3, score_threshold=0.3)
    print(f"hits   = {len(hits)}")
    for i, hit in enumerate(hits, 1):
        print(f"[{i}] score={hit['score']:.4f} | {hit['title']} | {hit['url']}")
    print("OK: Qdrant 连通")


if __name__ == "__main__":
    main()
