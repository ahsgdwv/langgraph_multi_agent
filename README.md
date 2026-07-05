# 快消渠道数据分析 Agent

饮料快消行业里，渠道周报有大量重复劳动：拉各渠道销量、排 SKU、翻政策文档、拼成一份 Markdown。本项目将上述流程抽象为**通用饮品快消场景**（不绑定单一品牌），做成可跑的 Agent 流程，并用 Skill 固定输出格式，减少每次从零写提示词。

技术栈：Python、LangGraph、Chroma、pandas、SQLite、Gradio。

## 运行

```bash
pip install -r requirements.txt
copy .env.example .env    # 填 DEEPSEEK_API_KEY
python ui.py              # http://localhost:7860
python main.py            # 无 LLM 的集成测试
```

API：`uvicorn api:app --port 8000`，`GET /skills` 可查看 Skill 列表。

仓库：[github.com/ahsgdwv/langgraph_multi_agent](https://github.com/ahsgdwv/langgraph_multi_agent)

## 演示品牌（可选）

默认使用**通用快消饮品**文案，不绑定单一品牌。如需面试/demo 时切换品牌名，在 `.env` 中设置：

```bash
BRAND_NAME=清泉饮品      # 可选，留空则为通用场景
BRAND_REGION=华东区      # 可选，影响示例前缀
```

设置后 Gradio 标题与「完整示例」会自动变为「清泉饮品华东区渠道分析…」。

## 流程

```
用户输入（中文任务列表）
  → 任务拆分（规则解析，不依赖 LLM）
  → Supervisor 路由到 data_analyst / researcher / executor
  → 子任务执行（Skill 或 LLM+工具）
  → 质量复核，失败可重试
  → 汇总为《快消业务分析周报》
```

Supervisor 按关键词分流：含「政策、陈列」走 researcher；含「销量、排行、SQL」走 data_analyst；其余走 executor（库存查询、竞品情报等）。

## Skill 资产库

高频分析做成 5 个 Skill，用触发词匹配，输出章节固定（数据概览 / 关键发现 / 建议动作）：

| Skill | 做什么 | 数据从哪来 |
|-------|--------|------------|
| channel_compare | 各渠道销售额对比 | SQL |
| sku_ranking | SKU TOP 排行 | pandas |
| sales_trend | 月度销量趋势 | pandas |
| policy_lookup | 陈列/定价政策 | RAG |
| integrated_report | 周报整合 | SQL + pandas + RAG |

新增一种分析：在 `skills/builtin.py` 里加触发词和执行函数，注册进 `register_builtin_skills()` 即可。

## 数据与文档

- `data/analytics/channel_sales.csv`：样例销售数据，启动时导入 SQLite
- `data/documents/`：渠道政策、产品目录、分析 playbook（RAG 检索源）

业务文档更新后执行 `ingest_documents(force=True)` 或调 API `POST /documents/reload` 重建向量库。

## 目录

```
config.py                            全局参数（RAG/分析库/Skill）
skills/base.py                       BaseSkill 基类 + SkillContext
skills/builtin.py                    5 个内置 Skill 实现
agents.py / graph.py / supervisor.py LangGraph 节点与路由
data_store.py / data_tools.py        SQLite + pandas
rag_store.py                         Chroma 检索（含索引校验）
```

## 开发笔记

- 测试时去掉 API Key，走 `without_llm()`，避免 LLM 输出飘导致断言失败
- 中文「1. xxx；2. xxx」拆分不准，在 `tools.py` 加了前缀剥离和两套编号解析
- Checkpoint 存自定义 Pydantic 模型报错，需注册到 `JsonPlusSerializer`
- dispatcher 合并 flags 时曾把 `documents_loaded` 覆盖回 false，已改为只更新必要字段

## 示例输入

```
华东区饮品渠道分析，需要完成：
1. 统计便利店、商超、电商、特通渠道销量与销售额对比；
2. 输出气泡水及电解质水 SKU 销售 TOP 排行；
3. 检索渠道陈列费与定价政策；
4. 整合 SQL 销售数据、pandas 统计与政策文档生成周报。
```

MIT License
