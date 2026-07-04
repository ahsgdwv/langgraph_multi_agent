"""内置 Skill 实现，均继承 BaseSkill。"""
from __future__ import annotations

from config import CFG
from data_store import load_sales_dataframe, run_readonly_sql
from rag_store import format_retrieved_context, retrieve_context
from skills.base import BaseSkill, SkillContext
from skills.models import SkillOutput
from skills.registry import register_skill_instance


def _table_md(records: list[dict], columns: list[str] | None = None) -> str:
    if not records:
        return "（无数据）"
    cols = columns or list(records[0].keys())
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in records]
    return "\n".join([header, sep, *rows])


def _make_output(skill: BaseSkill, summary: str, sections: dict[str, str], sources: list[str]) -> SkillOutput:
    return SkillOutput(
        skill_id=skill.skill_id,
        skill_name=skill.name,
        summary=summary,
        sections=sections,
        markdown="",
        data_sources=sources,
    )


class ChannelCompareSkill(BaseSkill):
    skill_id = "channel_compare"
    name = "渠道对比分析"
    description = "对比便利店/商超/电商/特通渠道销售额"
    triggers = [r"渠道", r"便利店", r"商超", r"电商", r"特通", r"渠道对比", r"份额"]
    output_sections = ["数据概览", "关键发现", "建议动作"]

    def _execute(self, text: str, ctx: SkillContext) -> SkillOutput:
        df = run_readonly_sql(
            "SELECT channel, SUM(units) AS units, SUM(revenue) AS revenue "
            "FROM channel_sales GROUP BY channel ORDER BY revenue DESC"
        )
        records = df.to_dict(orient="records")
        if not records:
            raise ValueError("销售库无渠道数据")
        leader = records[0]
        total = sum(r["revenue"] for r in records)
        share = leader["revenue"] / total * 100 if total else 0
        return _make_output(
            self,
            f"对比 {len(records)} 个渠道",
            {
                "数据概览": _table_md(records),
                "关键发现": (
                    f"- 领先渠道：{leader['channel']}，份额约 {share:.1f}%\n"
                    f"- 总销售额 {total:,.0f} 元"
                ),
                "建议动作": (
                    f"- 巩固 {leader['channel']} 核心 SKU 陈列\n"
                    "- 尾部渠道做费比复盘"
                ),
            },
            ["sqlite"],
        )


class SkuRankingSkill(BaseSkill):
    skill_id = "sku_ranking"
    name = "SKU 销售排行"
    description = "按销售额输出 TOP SKU"
    triggers = [r"SKU", r"排行", r"TOP", r"爆款", r"单品", r"品类"]
    output_sections = ["数据概览", "关键发现", "建议动作"]

    def _execute(self, text: str, ctx: SkillContext) -> SkillOutput:
        top_n = CFG.analytics.sku_top_n
        df = (
            load_sales_dataframe()
            .groupby("product_name", as_index=False)
            .agg(units=("units", "sum"), revenue=("revenue", "sum"))
            .sort_values("revenue", ascending=False)
            .head(top_n)
        )
        records = df.to_dict(orient="records")
        if not records:
            raise ValueError("销售库无 SKU 数据")
        top = records[0]
        return _make_output(
            self,
            f"输出 TOP{len(records)} SKU",
            {
                "数据概览": _table_md(records),
                "关键发现": (
                    f"- TOP1：{top['product_name']}，销售额 {float(top['revenue']):,.0f} 元\n"
                    f"- 合计销量 {int(df['units'].sum()):,} 件"
                ),
                "建议动作": "- 核心 SKU 检查安全库存\n- 长尾 SKU 评估下架",
            },
            ["pandas"],
        )


class SalesTrendSkill(BaseSkill):
    skill_id = "sales_trend"
    name = "销量趋势分析"
    description = "按月汇总销量与销售额"
    triggers = [r"趋势", r"同比", r"环比", r"月度", r"走势", r"销量变化"]
    output_sections = ["数据概览", "关键发现", "建议动作"]

    def _execute(self, text: str, ctx: SkillContext) -> SkillOutput:
        df = (
            load_sales_dataframe()
            .groupby("sale_month", as_index=False)
            .agg(units=("units", "sum"), revenue=("revenue", "sum"))
            .sort_values("sale_month")
        )
        records = df.to_dict(orient="records")
        if not records:
            raise ValueError("销售库无月度数据")
        peak = df.loc[df["revenue"].idxmax()]
        growth = (
            (df["revenue"].iloc[-1] - df["revenue"].iloc[0]) / df["revenue"].iloc[0] * 100
            if len(df) > 1 else 0
        )
        return _make_output(
            self,
            f"覆盖 {len(df)} 个月",
            {
                "数据概览": _table_md(records, ["sale_month", "units", "revenue"]),
                "关键发现": (
                    f"- 峰值月：{peak['sale_month']}，销售额 {float(peak['revenue']):,.0f} 元\n"
                    f"- 首尾月变化：{growth:+.1f}%"
                ),
                "建议动作": "- 峰值月前两周加备货\n- 低增速月拆渠道定位原因",
            },
            ["pandas"],
        )


class PolicyLookupSkill(BaseSkill):
    skill_id = "policy_lookup"
    name = "渠道政策检索"
    description = "检索陈列费、定价、进场政策"
    triggers = [r"政策", r"陈列", r"定价", r"合规", r"费用", r"进场"]
    output_sections = ["政策摘要", "适用建议", "执行清单"]

    def _execute(self, text: str, ctx: SkillContext) -> SkillOutput:
        docs = retrieve_context(text, k=CFG.rag.default_k)
        rag_text = format_retrieved_context(docs)
        sources = [d.source for d in docs]
        return _make_output(
            self,
            f"检索到 {len(docs)} 条片段",
            {
                "政策摘要": rag_text,
                "适用建议": "- 下发前核对陈列费上限\n- 新渠道完成合规备案",
                "执行清单": "1. 确认渠道类型\n2. 核对 SKU 定价\n3. 留存审批记录",
            },
            ["rag", *sources],
        )


class IntegratedReportSkill(BaseSkill):
    skill_id = "integrated_report"
    name = "多源整合报告"
    description = "SQL + pandas + RAG 合成周报"
    triggers = [r"整合", r"多源", r"汇总报告", r"周报", r"月报", r"复盘"]
    output_sections = ["数据概览", "业务上下文", "关键发现", "建议动作"]

    def _execute(self, text: str, ctx: SkillContext) -> SkillOutput:
        channel_df = run_readonly_sql(
            "SELECT channel, SUM(revenue) AS revenue FROM channel_sales "
            "GROUP BY channel ORDER BY revenue DESC"
        )
        sku_df = (
            load_sales_dataframe()
            .groupby("product_name", as_index=False)
            .agg(revenue=("revenue", "sum"))
            .sort_values("revenue", ascending=False)
            .head(5)
        )
        docs = retrieve_context(text, k=CFG.rag.default_k)
        rag_text = format_retrieved_context(docs)
        if channel_df.empty or sku_df.empty:
            raise ValueError("整合报告缺少销售数据")
        return _make_output(
            self,
            "完成三源整合",
            {
                "数据概览": (
                    "**渠道（SQL）**\n" + _table_md(channel_df.to_dict(orient="records")) + "\n\n"
                    "**SKU TOP5（pandas）**\n" + _table_md(sku_df.to_dict(orient="records"))
                ),
                "业务上下文": rag_text,
                "关键发现": (
                    f"- 领先渠道：{channel_df.iloc[0]['channel']}\n"
                    f"- 核心 SKU：{sku_df.iloc[0]['product_name']}"
                ),
                "建议动作": "- 制定下月铺货计划\n- 校验促销费比合规",
            },
            ["sqlite", "pandas", "rag"],
        )


_BUILTIN_SKILLS: list[BaseSkill] = [
    ChannelCompareSkill(),
    SkuRankingSkill(),
    SalesTrendSkill(),
    PolicyLookupSkill(),
    IntegratedReportSkill(),
]


def register_builtin_skills() -> None:
    for skill in _BUILTIN_SKILLS:
        register_skill_instance(skill)
