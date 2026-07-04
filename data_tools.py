"""数据分析工具：SQL 查询、pandas 统计、多源整合（供 Agent Function Calling）。"""
from __future__ import annotations

import json
from typing import Literal

from langchain_core.tools import tool

from config import CFG
from data_store import load_sales_dataframe, run_readonly_sql
from rag_store import format_retrieved_context, retrieve_context

AnalysisType = Literal["trend", "ranking", "channel_compare", "region_summary"]


def _df_to_markdown_table(df, max_rows: int = 10) -> str:
    if df.empty:
        return "（无数据）"
    view = df.head(max_rows)
    headers = "| " + " | ".join(str(c) for c in view.columns) + " |"
    sep = "| " + " | ".join("---" for _ in view.columns) + " |"
    rows = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in view.itertuples(index=False, name=None)
    ]
    return "\n".join([headers, sep, *rows])


@tool
def query_sales_sql(sql: str) -> str:
    """只读 SQL 查询快消渠道销售数据（表 channel_sales，字段含 sale_month/channel/region/product_name/units/revenue）。"""
    try:
        df = run_readonly_sql(sql)
        return json.dumps(
            {
                "rows": len(df),
                "columns": list(df.columns),
                "preview": df.to_dict(orient="records"),
                "markdown_table": _df_to_markdown_table(df),
            },
            ensure_ascii=False,
            indent=2,
        )
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def analyze_sales_pandas(
    analysis_type: AnalysisType = "channel_compare",
    group_by: str = "channel",
) -> str:
    """pandas 销售分析：trend(月度趋势) / ranking(SKU排行) / channel_compare(渠道对比) / region_summary(区域汇总)。"""
    try:
        df = load_sales_dataframe()
        if df.empty:
            return json.dumps({"error": "销售数据为空"}, ensure_ascii=False)

        result: dict = {"analysis_type": analysis_type, "group_by": group_by}

        if analysis_type == "trend":
            agg = (
                df.groupby("sale_month", as_index=False)
                .agg(units=("units", "sum"), revenue=("revenue", "sum"))
                .sort_values("sale_month")
            )
            result["summary"] = f"覆盖 {len(agg)} 个月，总销量 {int(agg['units'].sum())} 件"
            result["table"] = agg.to_dict(orient="records")

        elif analysis_type == "ranking":
            agg = (
                df.groupby(["product_name"], as_index=False)
                .agg(units=("units", "sum"), revenue=("revenue", "sum"))
                .sort_values("revenue", ascending=False)
                .head(10)
            )
            result["summary"] = f"TOP {len(agg)} SKU 按销售额排序"
            result["table"] = agg.to_dict(orient="records")

        elif analysis_type == "channel_compare":
            agg = (
                df.groupby("channel", as_index=False)
                .agg(units=("units", "sum"), revenue=("revenue", "sum"))
                .sort_values("revenue", ascending=False)
            )
            top = agg.iloc[0]
            result["summary"] = (
                f"共 {len(agg)} 个渠道，领先渠道 {top['channel']} "
                f"销售额 {float(top['revenue']):,.0f} 元"
            )
            result["table"] = agg.to_dict(orient="records")

        elif analysis_type == "region_summary":
            col = group_by if group_by in df.columns else "region"
            agg = (
                df.groupby(col, as_index=False)
                .agg(units=("units", "sum"), revenue=("revenue", "sum"))
                .sort_values("revenue", ascending=False)
            )
            result["summary"] = f"按 {col} 汇总 {len(agg)} 组"
            result["table"] = agg.to_dict(orient="records")

        result["markdown_table"] = _df_to_markdown_table(
            __import__("pandas").DataFrame(result["table"])
        )
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def integrate_data_sources(task_description: str) -> str:
    """多源数据整合：SQLite 渠道汇总 + pandas SKU 排行 + 业务文档 RAG 检索。"""
    try:
        sql_df = run_readonly_sql(
            "SELECT channel, SUM(units) AS units, SUM(revenue) AS revenue "
            "FROM channel_sales GROUP BY channel ORDER BY revenue DESC"
        )
        sku_df = (
            load_sales_dataframe()
            .groupby("product_name", as_index=False)
            .agg(units=("units", "sum"), revenue=("revenue", "sum"))
            .sort_values("revenue", ascending=False)
            .head(5)
        )
        docs = retrieve_context(task_description, k=CFG.rag.default_k)
        rag_text = format_retrieved_context(docs)

        payload = {
            "task": task_description,
            "sources": ["sqlite", "pandas", "rag"],
            "channel_summary": sql_df.to_dict(orient="records"),
            "top_skus": sku_df.to_dict(orient="records"),
            "policy_context": rag_text,
            "markdown": (
                "### 渠道销售汇总（SQL）\n"
                f"{_df_to_markdown_table(sql_df)}\n\n"
                "### SKU TOP5（pandas）\n"
                f"{_df_to_markdown_table(sku_df)}\n\n"
                "### 业务文档检索（RAG）\n"
                f"{rag_text}"
            ),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


DATA_TOOLS = [query_sales_sql, analyze_sales_pandas, integrate_data_sources]
