# 多 Agent 深度研究系统

基于 Orchestrator 模式的 Multi-Agent 深度研究系统，覆盖完整流水线：

**复杂问题 → 规划拆解 → 搜索信息 → 分析证据 → 自我纠错 → 综合输出结构化报告**

技术栈：Tavily（搜索）、SiliconFlow/DeepSeek（推理）、Qdrant（向量记忆）、Neo4j（知识图谱，可回退内存）、DashScope/OpenAI 兼容嵌入。

## 架构概览

```
用户问题
   │
   ▼
Orchestrator
   ├── Planner      → 子任务 DAG（Neo4j / Memory）
   ├── Researcher ×N → Qdrant 缓存优先 → Tavily → 证据/实体关系
   ├── Synthesizer  → 结构化报告
   └── Verifier     → 不足则补充研究（最多 N 轮）
   │
   ▼
报告 + Mermaid 过程图 + 知识图谱 HTML
```

仍可通过 `--mode single` 使用单 Agent Tool-calling 路径。

## 目录结构

```
src/
├── agent/           # 单 Agent + Phase2 扩展工具
├── agents/          # Planner / Researcher / Synthesizer / Verifier
├── orchestrator/    # 中央调度
├── graph/           # Neo4j + 内存图存储
├── tools/           # Tool 抽象与 Tavily
├── memory/          # Qdrant / 分块 / 用户偏好
├── embeddings/      # 嵌入模型
├── llm/             # SiliconFlow 客户端
├── visualization/   # Mermaid / HTML 导出
├── evaluation/      # 评估指标与对比实验
├── cli/             # 命令行入口
├── config.py
└── factory.py
```

## 快速开始

### 1. 环境准备

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
```

### 2. 配置 .env

复制 `.env.example` 为 `.env` 并填入密钥。关键项：

| 变量 | 说明 |
|------|------|
| `SILICONFLOW_API_KEY` / `LLM_API_KEY` | LLM 推理 |
| `TAVILY_API_KEY` | 联网搜索 |
| `EMBED_API_KEY` + `EMBED_BASE_URL` | 嵌入（OpenAI 兼容优先） |
| `QDRANT_URL` + `QDRANT_API_KEY` | 向量库 |
| `QDRANT_VECTOR_SIZE` | 需与嵌入维度一致（如 1024） |
| `NEO4J_*` | 可选；连接失败自动回退内存图 |

### 3. 运行

```bash
# 默认：多 Agent 完整流水线
python -m src.cli.main "2024-2025 全球光伏供应链变化及对中国企业的影响"

# 单 Agent 模式
python -m src.cli.main --mode single "2024年全球光伏装机量趋势"

# 交互模式（复用同一 Orchestrator 实例）
python -m src.cli.main -i

# 控制自反思轮次
python -m src.cli.main --max-verify-rounds 2 "某信息稀疏问题"

# 对比评估
python -m src.cli.main --eval-quick "Transformer 架构的核心创新"
# 或
python -m src.evaluation.harness --quick "Transformer 架构的核心创新"
```

可视化产物默认写入 `outputs/`：

- `*_process.mmd` 研究过程图
- `*_kg.mmd` / `*_kg.html` 知识图谱快照

### 4. 连通性验收脚本（scripts/）

从项目根目录执行：

```bash
python scripts/ping_llm.py
python scripts/ping_tavily.py
python scripts/ping_embed.py
python scripts/ping_qdrant.py
python scripts/tool_calling_once.py
python scripts/ping_neo4j.py
```

### 5. 测试

```bash
pytest -q
```

## 阶段能力

| 阶段 | 能力 |
|------|------|
| 1 | Tool 框架 + Tavily + 单 Agent + Qdrant |
| 2 | RAG 先查缓存、分块入库、URL 去重、用户偏好 |
| 3 | Planner / Researcher / Synthesizer + Orchestrator 并行调度 |
| 3.5 | Verifier 自反思补充研究 |
| 4 | 实体关系抽取 + Mermaid/HTML 可视化 |
| 5 | 完整性/准确性/引用密度/反思轮次评估与对比实验 |

## 工具扩展

实现 `src.tools.base.Tool` 并在 `factory.create_components` 中注册即可接入。
