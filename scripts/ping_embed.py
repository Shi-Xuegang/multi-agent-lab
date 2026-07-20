"""Day6 验收：Embedding 连通性。"""

from __future__ import annotations

import _bootstrap  # noqa: F401

from src.config import get_settings
from src.embeddings.factory import create_embedder


def main() -> None:
    settings = get_settings()
    if not settings.embed_api_key:
        raise SystemExit("缺少 EMBED_API_KEY")

    embedder = create_embedder(settings, dimension=settings.qdrant_vector_size)
    vectors = embedder.embed_batch(["深度研究系统连通性测试"])
    vec = vectors[0]
    print(f"model     = {settings.embed_model_name}")
    print(f"base_url  = {settings.embed_base_url or '(dashscope sdk)'}")
    print(f"dimension = {len(vec)} (config={settings.qdrant_vector_size})")
    print(f"preview   = {vec[:5]}")
    print("OK: Embedding 连通")


if __name__ == "__main__":
    main()
