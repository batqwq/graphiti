# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此仓库中工作时提供指导。

---

## 项目概述

Graphiti 是一个 Python 框架，用于构建面向 AI 代理的**时序感知知识图谱**。它支持对知识图谱进行实时增量更新，无需批量重新计算，适用于动态环境。

### 核心特性

- **双时序数据模型** — 显式跟踪事件发生时间
- **混合检索** — 结合语义嵌入、关键词搜索 (BM25) 和图遍历
- **自定义实体** — 通过 Pydantic 模型定义实体
- **多数据库后端** — 支持 Neo4j 和 FalkorDB
- **可观测性** — 可选的 OpenTelemetry 分布式追踪

---

## 开发命令

### 主项目命令（在项目根目录执行）

```bash
# 安装依赖
uv sync --extra dev

# 格式化代码（ruff 导入排序 + 格式化）
make format

# 代码检查（ruff + pyright 类型检查）
make lint

# 运行测试
make test

# 运行全部检查（格式化、检查、测试）
make check
```

### Server 开发（在 `server/` 目录执行）

```bash
cd server/

# 安装 Server 依赖
uv sync --extra dev

# 以开发模式启动服务
uvicorn graph_service.main:app --reload

# 格式化、检查、测试
make format
make lint
make test
```

### MCP Server 开发（在 `mcp_server/` 目录执行）

```bash
cd mcp_server/

# 安装 MCP Server 依赖
uv sync

# 使用 Docker Compose 启动
docker-compose up
```

---

## 代码架构

### 核心库 `graphiti_core/`

| 模块 | 说明 |
|---|---|
| `graphiti.py` | 主入口，包含 `Graphiti` 类，协调所有功能 |
| `driver/` | 数据库驱动（Neo4j、FalkorDB） |
| `llm_client/` | LLM 客户端（OpenAI、Anthropic、Gemini、Groq） |
| `embedder/` | 各提供商的嵌入客户端 |
| `nodes.py` / `edges.py` | 图的核心数据结构（节点与边） |
| `search/` | 混合搜索，支持可配置策略 |
| `prompts/` | LLM 提示词（实体抽取、去重、摘要） |
| `utils/` | 维护操作、批量处理、日期时间工具 |

### REST API 服务 `server/`

| 模块 | 说明 |
|---|---|
| `graph_service/main.py` | FastAPI 服务入口 |
| `routers/` | API 端点（数据摄入与检索） |
| `dto/` | 数据传输对象（API 契约） |

### MCP 服务 `mcp_server/`

| 模块 | 说明 |
|---|---|
| `graphiti_mcp_server.py` | Model Context Protocol 服务端，供 AI 助手使用 |
| Docker 支持 | 容器化部署（含 Neo4j） |

---

## 测试

| 类别 | 位置/说明 |
|---|---|
| 单元测试 | `tests/` — 使用 pytest 的综合测试套件 |
| 集成测试 | 文件名以 `_int` 后缀标识，需要数据库连接 |
| 评估脚本 | `tests/evals/` — 端到端评估 |

### 常用测试命令

```bash
# 运行全部测试
make test
# 或使用 pytest
pytest

# 运行指定文件
pytest tests/test_specific_file.py

# 运行指定方法
pytest tests/test_file.py::test_method_name

# 仅运行集成测试
pytest tests/ -k "_int"

# 仅运行单元测试
pytest tests/ -k "not _int"
```

> **提示**：支持使用 `pytest-xdist` 进行并行测试执行。

---

## 配置

### 环境变量

| 变量 | 说明 |
|---|---|
| `OPENAI_API_KEY` | **必需** — 用于 LLM 推理和嵌入 |
| `ANTHROPIC_API_KEY` | 可选 — Anthropic 提供商 |
| `GOOGLE_API_KEY` | 可选 — Google Gemini 提供商 |
| `GROQ_API_KEY` | 可选 — Groq 提供商 |
| `VOYAGE_API_KEY` | 可选 — Voyage 嵌入提供商 |
| `USE_PARALLEL_RUNTIME` | 可选布尔值 — Neo4j 并行运行时（仅企业版） |

### 数据库配置

#### Neo4j

- 要求版本 **5.26+**，可通过 Neo4j Desktop 获取
- 数据库名默认为 `neo4j`（在 `Neo4jDriver` 中硬编码）
- 可通过向驱动构造函数传递 `database` 参数覆盖

#### FalkorDB

- 要求版本 **1.1.2+**，作为替代后端
- 数据库名默认为 `default_db`（在 `FalkorDriver` 中硬编码）
- 可通过向驱动构造函数传递 `database` 参数覆盖

---

## 开发规范

### 代码风格

- 使用 **Ruff** 进行格式化和代码检查（配置在 `pyproject.toml`）
- 行长度上限：**100 字符**
- 引号风格：**单引号**
- 强制使用 **Pyright** 类型检查
  - 主项目：`typeCheckingMode = "basic"`
  - Server：`typeCheckingMode = "standard"`

### LLM 提供商支持

代码库支持多种 LLM 提供商，但与支持**结构化输出**的服务（OpenAI、Gemini）配合最佳。其他提供商（尤其是较小模型）可能出现 Schema 验证问题。

#### 当前支持的模型（截至 2025 年 11 月）

**OpenAI 模型：**

| 系列 | 模型 | 说明 |
|---|---|---|
| GPT-5（推理模型，需 `temperature=0`） | `gpt-5-mini` | 快速推理模型 |
| | `gpt-5-nano` | 最小推理模型 |
| GPT-4.1（标准模型） | `gpt-4.1` | 全能力模型 |
| | `gpt-4.1-mini` | 适合大多数任务的高效模型 |
| | `gpt-4.1-nano` | 轻量模型 |
| 旧版（仍支持） | `gpt-4o` | 上一代旗舰 |
| | `gpt-4o-mini` | 上一代高效版 |

**Anthropic 模型：**

| 系列 | 模型 | 说明 |
|---|---|---|
| Claude 4.5 | `claude-sonnet-4-5-latest` | 旗舰模型，自动更新 |
| | `claude-sonnet-4-5-20250929` | 固定版本（2025年9月） |
| | `claude-haiku-4-5-latest` | 快速模型，自动更新 |
| Claude 3.7 | `claude-3-7-sonnet-latest` | 自动更新 |
| | `claude-3-7-sonnet-20250219` | 固定版本（2025年2月） |
| Claude 3.5 | `claude-3-5-sonnet-latest` | 自动更新 |
| | `claude-3-5-sonnet-20241022` | 固定版本（2024年10月） |
| | `claude-3-5-haiku-latest` | 快速模型 |

**Google Gemini 模型：**

| 系列 | 模型 | 说明 |
|---|---|---|
| Gemini 2.5 | `gemini-2.5-pro` | 旗舰推理与多模态 |
| | `gemini-2.5-flash` | 快速高效 |
| Gemini 2.0 | `gemini-2.0-flash` | 实验性快速模型 |
| Gemini 1.5（稳定版） | `gemini-1.5-pro` | 生产稳定版旗舰 |
| | `gemini-1.5-flash` | 生产稳定版高效 |

> **注意**：`gpt-5-mini`、`gpt-4.1`、`gpt-4.1-mini` 等模型名是有效的 OpenAI 模型标识符。GPT-5 系列为推理模型，需要 `temperature=0`（代码中已自动处理）。

---

## MCP Server 使用指南

使用 MCP Server 时，请遵循 `mcp_server/cursor_rules.md` 中的规范：

- 添加新信息前，**先搜索已有知识**
- 使用特定的实体类型过滤器（`Preference`、`Procedure`、`Requirement`）
- 使用 `add_memory` **立即存储**新信息
- 遵循已发现的流程，尊重已建立的偏好设置