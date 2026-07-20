"""应用配置，从 .env 环境变量加载。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ENV_FILE = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """全局配置项，字段名与 .env 变量一一对应。"""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ========== Tavily ==========
    tavily_api_key: str = ""

    # ========== SiliconFlow ==========
    siliconflow_api_key: str = ""
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    siliconflow_model_id: str = "deepseek-ai/DeepSeek-V4-Flash"

    # ========== 通用 LLM（兼容回退） ==========
    llm_api_key: str = ""
    llm_base_url: str = "https://api.siliconflow.cn/v1"
    llm_model_id: str = "deepseek-ai/DeepSeek-V4-Flash"

    # ========== Qdrant ==========
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "hello_agents_vectors"
    qdrant_vector_size: int = 1024
    qdrant_distance: str = "cosine"
    qdrant_timeout: int = 30

    # ========== Neo4j ==========
    neo4j_uri: str = ""
    neo4j_username: str = ""
    neo4j_password: str = ""
    neo4j_database: str = ""
    neo4j_max_connection_lifetime: int = 3600
    neo4j_max_connection_pool_size: int = 50
    neo4j_connection_timeout: int = 60

    # ========== Embedding ==========
    embed_model_type: str = "dashscope"
    embed_model_name: str = "text-embedding-v3"
    embed_api_key: str = ""
    embed_base_url: str = ""

    # ========== Agent 行为 ==========
    max_tool_iterations: int = 5
    tavily_max_results: int = 5
    rag_score_threshold: float = 0.85
    chunk_strategy: str = "semantic"
    max_verify_rounds: int = 2
    researcher_concurrency: int = 3
    max_task_retries: int = 2

    @computed_field  # type: ignore[prop-decorator]
    @property
    def effective_llm_api_key(self) -> str:
        """优先 SiliconFlow，回退到通用 LLM 配置。"""
        return self.siliconflow_api_key or self.llm_api_key

    @computed_field  # type: ignore[prop-decorator]
    @property
    def effective_llm_base_url(self) -> str:
        return self.siliconflow_base_url or self.llm_base_url

    @computed_field  # type: ignore[prop-decorator]
    @property
    def effective_llm_model_id(self) -> str:
        return self.siliconflow_model_id or self.llm_model_id


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取配置单例。"""
    return Settings()
