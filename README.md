# LangGraph Multi-Agent Task Scheduler

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-orange.svg)](https://github.com/langchain-ai/langgraph)
[![RAG](https://img.shields.io/badge/RAG-Chroma-green.svg)](https://www.trychroma.com/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

> **LangGraph 多 Agent 任务调度** · Chroma RAG · Supervisor 动态调度 · Function Calling · Gradio 网页 + FastAPI  
> 场景：电商 618 运营方案 — 用户中文描述需求，自动拆分、检索业务文档、逐条执行、质检重试、输出报告

基于 LangGraph 的多 Agent 任务调度服务。用户用日常中文描述需求，系统自动完成：**文档检索 → 任务拆分 → 多 Agent 执行 → 质量反思 → 人工确认 → 汇总报告**，支持网页一键使用，无需编程。

## 技术栈

| 类别 | 技术 |
|------|------|
| 编排框架 | LangGraph、LangChain |
| LLM | DeepSeek / OpenAI（OpenAI 兼容接口） |
| **向量库 / RAG** | **Chroma + langchain-chroma**，本地持久化 `data/chroma/` |
| 服务化 | FastAPI、Gradio |
| 数据模型 | Pydantic v2 |
| 持久化 | SQLite Checkpointer（流程断点续跑） |

## 项目结构

```
langgraph_multi_agent/
├── state.py            # 全局状态、RunMetrics、RetrievedDoc 等模型
├── tools.py            # 任务拆分、结果汇总（规则解析自然语言）
├── agents.py           # 6+ Agent 节点（文档加载、执行、研究、反思…）
├── supervisor.py       # Supervisor 动态路由
├── graph.py            # LangGraph 图（经典 / Supervisor 两种模式）
├── rag_store.py        # ★ Chroma 向量库：入库、检索
├── external_tools.py   # Function Calling 工具（文件、库存、搜索）
├── llm_utils.py        # LLM 统一调用 + token/耗时日志
├── metrics.py          # 首轮/最终通过率、重试提升统计
├── paths.py            # reports/ 与 data/ 路径常量
├── api.py              # FastAPI HTTP 接口
├── ui.py               # Gradio 网页（新手指南 + 书写提示 + 下载）
├── reports/            # ★ 用户最终报告（.md）
├── data/
│   ├── documents/      # 业务文档（RAG 源数据）
│   ├── chroma/         # ★ 向量库持久化目录
│   └── outputs/        # Agent 工具运行时文件
└── main.py             # CLI、内置测试
```

## 快速开始

### 安装

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 配置

```bash
copy .env.example .env
```

```env
DEEPSEEK_API_KEY=sk-your-key
DEEPSEEK_MODEL=deepseek-chat
AUTO_APPROVE_HUMAN=true
USE_SUPERVISOR=true
```

### 三种运行方式

| 方式 | 命令 | 适用 |
|------|------|------|
| **网页（推荐）** | `python ui.py` → http://localhost:7860 | 非技术用户、演示 |
| CLI | `python main.py` | 开发测试 |
| API | `uvicorn api:app --port 8000` | 集成调用 |

网页功能：新手指南、书写提示（实时 ✅/⚠️）、示例/填空模板、报告下载、自动保存到 `reports/`。

---

## 核心能力

### 1. RAG 向量库（Chroma）

- **入库**：`document_loader` 节点扫描 `data/documents/*.md`，分块写入 Chroma
- **检索**：每个子任务执行前，按任务标题+描述做 similarity search（Top-K）
- **持久化**：`data/chroma/`，重启不丢失
- **Embedding**：有 `OPENAI_API_KEY` 用 OpenAI Embeddings，否则 FakeEmbeddings（本地 demo）
- **API 上传**：`POST /documents/upload` 动态扩充知识库

内置示例文档：`618_activity_rules.md`、`618_history_2024.md`

### 2. 外部工具（Function Calling）

| 工具 | 作用 |
|------|------|
| `read_local_file` | 读取业务文档 |
| `write_local_file` | 写入 `data/outputs/` |
| `query_activity_inventory` | 模拟 618 库存查询 |
| `simple_web_search` | 模拟搜索引擎 |

### 3. Supervisor 动态调度

`USE_SUPERVISOR=true` 时，Supervisor 按任务关键词路由：

- 规则/文档/客服类 → `researcher`（更深 RAG，k=6）
- 其他 → `executor`（RAG + 工具，k=4）

### 4. 反思重试 + 人工审核

- 反思未通过且未达 `max_retry` → 退回执行
- 关键节点 `interrupt()` 支持人工确认（`AUTO_APPROVE_HUMAN` 可跳过）

### 5. 量化评测

报告末尾输出：首轮通过率、最终通过率、重试提升、LLM 调用次数、token 合计、RAG 检索次数。

---

## 流程图（Supervisor 模式）

```
Supervisor ⇄ document_loader（Chroma 入库）
         ⇄ dispatcher（规则拆分任务）
         ⇄ human_review_dispatch
         ⇄ executor / researcher（RAG + 工具 + LLM）
         ⇄ reflection（质检，失败重试）
         ⇄ human_review_summary
         ⇄ summary → reports/*.md
```

---

## 用户需求怎么写

无需 AI 提示词。推荐：

```
我要策划618大促，需要完成：
1. 制定满减规则；2. 设计页面；3. 写直播脚本
```

或：`任务A、任务B、任务C`（顿号分隔）。网页有实时书写提示。

---

## 测试

```bash
python main.py
```

内置测试（无 LLM）：任务拆分、**RAG 入库检索**、工具调用、Supervisor 路由、重试、HITL、SQLite 断点续跑。

---

## 业务文档管理

```bash
# 放入 data/documents/ 后
curl -X POST http://localhost:8000/documents/reload

# 或 API 上传
curl -F "file=@rules.md" http://localhost:8000/documents/upload
```
