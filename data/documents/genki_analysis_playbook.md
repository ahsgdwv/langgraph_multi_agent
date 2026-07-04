# 元气森林业务分析工作流

本文档描述饮料快消行业常见的重复性分析任务，供 Agent Skill 与 RAG 检索参考。

## 日常分析场景

### 1. 渠道销量分析
- 输入：区域 + 时间范围
- 输出：便利店/商超/电商/特通销售额与份额对比
- 数据源：销售 SQLite 库（pandas / SQL）
- 对应 Skill：`channel_compare`

### 2. SKU 结构分析
- 输入：品类或区域
- 输出：销售额 TOP SKU、长尾 SKU 预警
- 数据源：销售 SQLite 库
- 对应 Skill：`sku_ranking`

### 3. 趋势与同比环比
- 输入：月度序列
- 输出：销量走势、峰值月份、增速
- 对应 Skill：`sales_trend`

### 4. 政策合规核对
- 输入：渠道类型、拟申请费用
- 输出：陈列费上限、定价底线、审批要求
- 数据源：渠道运营政策文档（RAG）
- 对应 Skill：`policy_lookup`

### 5. 多源整合周报
- 整合：SQL 渠道汇总 + pandas SKU 排行 + 政策文档摘要
- 输出固定章节：数据概览 → 关键发现 → 建议动作
- 对应 Skill：`integrated_report`

## 业务关键词（触发 Skill）

| 关键词 | 建议 Skill |
|--------|------------|
| 渠道对比、便利店、商超 | channel_compare |
| SKU、排行、TOP | sku_ranking |
| 趋势、环比、月度 | sales_trend |
| 政策、陈列、费用 | policy_lookup |
| 整合、周报、复盘 | integrated_report |

## 人工复核要点

- 数据异常：某渠道销售额环比波动 > 30% 需标注
- 政策冲突：促销方案费用率超过 12% 需升级审批
- 库存联动：TOP SKU 需同步查询渠道库存是否充足
