# 仓库指南 (Repository Guidelines)

---

## 项目结构与模块组织

Graphiti 的核心库位于 `graphiti_core/`，划分为以下域模块：

| 模块 / 路径 | 说明 |
|---|---|
| `nodes.py` / `edges.py` / `models/` | 图的核心数据结构和模型 |
| `search/` | 检索管道 (Retrieval pipelines) |
| `graphiti_core/driver/` | 数据库驱动，支持 Neo4j, FalkorDB, Kuzu 和 Neptune |
| `cross_encoder/` | 重排序（通过 BGE, OpenAI 和 Gemini） |
| `telemetry/` | OpenTelemetry 分布式追踪 |
| `namespaces/` | 命名空间管理 |
| `migrations/` | 数据库迁移 |

其他重要目录：

| 目录 / 路径 | 说明 |
|---|---|
| `server/graph_service/` | 服务适配器和 API 接口层 |
| `mcp_server/` | MCP 集成（包含 `src/`, `tests/`, `config/`, `docker/` 子目录） |
| `images/` / `examples/` | 共享资源和示例代码 |
| `tests/` | 核心包的测试代码（配置位于 `conftest.py`, `pytest.ini` 等） |
| `spec/` / `signatures/` | 规范文件和类型签名 |
| 仓库根目录 | 工具清单和配置（如 `pyproject.toml`, `Makefile` 及部署 compose 文件） |

---

## 构建、测试与开发命令

```bash
# 安装开发环境 (uv sync --extra dev)
make install

# 运行 ruff 对导入进行排序并应用标准格式化
make format

# 针对 graphiti_core 运行 ruff 和 pyright 类型检查
make lint

# 仅运行单元测试（排除集成测试并禁用非 Neo4j 驱动）
make test

# 依次运行 format, lint, 和 test
make check
```

**其他常用命令：**

- `uv run pytest tests/path/test_file.py`：针对特定模块或选择部分测试。
- `docker-compose -f docker-compose.test.yml up`：为集成流程提供本地图/搜索依赖项。

---

## 编码风格与命名约定

- **基础格式**：使用 4 个空格缩进，行长度上限为 100 字符，优先使用**单引号**（如 `pyproject.toml` 中的配置）。
- **命名规范**：
  - 模块、文件和函数：使用 `snake_case` (蛇形命名法)。
  - `graphiti_core/models` 中的 Pydantic 模型：使用 `PascalCase` (帕斯卡命名法) 并且**必须包含显式的类型提示**。
- **架构原则**：将产生副作用（Side-effectful）的代码放在驱动或适配器中（如 `graphiti_core/driver`, `graphiti_core/cross_encoder`, `graphiti_core/utils`），其他地方依赖纯函数/辅助函数。
- **提交流程**：在提交代码之前，必须运行 `make format` 来规范化导入和文档字符串的格式。

---

## 测试指南

| 类别 | 规范说明 |
|---|---|
| **测试位置与命名** | 与 `tests/` 下的功能模块平行开发，文件命名为 `test_<feature>.py`，函数命名为 `test_<behavior>`。 |
| **集成测试** | 文件名使用 `_int` 后缀（例如 `test_edge_int.py`）。对于依赖数据库的场景，使用 `@pytest.mark.integration` 标记，以便 CI 进行控制；默认情况下 `make test` 会排除这些测试。 |
| **异步测试** | 通过 `pytest.ini` 中的 `asyncio_mode = auto` 自动运行。 |
| **回归测试** | 首先编写一个会失败的测试来复现回归问题，然后通过 `uv run pytest -k "pattern"` 验证修复效果。 |
| **本地集成测试** | 本地运行集成测试套件时，需先通过 `docker-compose.test.yml` 启动所需的后台服务。 |
| **MCP 测试** | `mcp_server/` 有其独立的测试套件，位于 `mcp_server/tests/`。 |

---

## 提交 (Commit) 与 PR (Pull Request) 指南

### Commit 规范
- **格式要求**：使用**祈使句、现在时**来总结改动（例如：`add async cache invalidation`）。
- **PR 编号**：可选择在历史记录中看到的那样加上 PR 编号（例如：`(#927)`）。
- **内容隔离**：将修复项（fixups）压缩合并（Squash），并保持不相关的更改相互隔离。

### Pull Request 规范
一个标准的 PR 应包含以下内容：
1. 简洁的描述。
2. 关联的追踪 Issue (Linked tracking issue)。
3. 关于架构 (Schema) 或 API 影响的注意事项。
4. 如果发生行为变化，需附上**截图或日志**。
5. **前置检查**：确保本地运行的 `make lint` 和 `make test` 均能通过。
6. **文档更新**：如果公共接口发生变化，务必更新对应的文档或示例。
