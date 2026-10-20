from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

from config import CFG
from data_store import load_sales_dataframe, run_readonly_sql
from paths import PROJECT_ROOT, TOOL_OUTPUT_DIR
from rag_store import format_retrieved_context, retrieve_context

out_dir = TOOL_OUTPUT_DIR

# demo 库存，接真 WMS 前凑合用
_wms = {
    "SKU-001": {"name": "零糖气泡水480ml", "stock": 52000, "reserved": 12000},
    "SKU-002": {"name": "无糖茶500ml", "stock": 31000, "reserved": 6500},
    "SKU-003": {"name": "电解质水500ml", "stock": 28000, "reserved": 5800},
    "DEFAULT": {"name": "通用饮料SKU", "stock": 45000, "reserved": 9000},
}

_intel_snippets = {
    "竞品": [
        "竞品A（0糖气泡水）：便利店铺货率提升，夏季电商礼盒装主推。",
        "竞品B（电解质饮料）：健身房特通买2赠1，运动场景渗透加快。",
    ],
    "气泡水": [
        "即饮气泡水品类年增速约12%，无糖/低糖占比持续提升。",
        "零糖气泡水在便利店渠道单店月均动销约180件，夏季峰值可达280件。",
    ],
    "default": [
        "饮料快消复盘指标：动销率、渠道费比、毛利率、库存周转天数。",
        "周报结构建议：渠道TOP/BOTTOM → SKU排行 → 异常预警 → 下周动作。",
    ],
}


def _md_table(frame: pd.DataFrame, rows=10) -> str:
    if frame is None or frame.empty:
        return "（无数据）"
    v = frame.head(rows)
    hdr = "| " + " | ".join(str(c) for c in v.columns) + " |"
    sep = "| " + " | ".join("---" for _ in v.columns) + " |"
    body = ["| " + " | ".join(str(x) for x in row) + " |" for row in v.itertuples(index=False, name=None)]
    return "\n".join([hdr, sep, *body])


def _safe_out(path: str) -> Path:
    root = out_dir.resolve()
    p = Path(path)
    p = (root / p).resolve() if not p.is_absolute() else p.resolve()
    try:
        p.relative_to(root)
    except ValueError:
        raise ValueError("只能写 outputs 目录")
    return p


def _err(msg: str) -> str:
    return json.dumps({"error": msg}, ensure_ascii=False)


@tool
def read_local_file(path: str) -> str:
    """读 data/documents 或 outputs 下文本"""
    if not path or not str(path).strip():
        return _err("path 空")
    requested = Path(path)
    for base in (PROJECT_ROOT / "data" / "documents", out_dir):
        root = base.resolve()
        fp = (root / requested).resolve() if not requested.is_absolute() else requested.resolve()
        try:
            fp.relative_to(root)
        except ValueError:
            continue
        if fp.is_file():
            return fp.read_text(encoding="utf-8")
    return _err(f"文件不存在: {path}")


@tool
def write_local_file(path: str, content: str) -> str:
    """写到 data/outputs"""
    if not path:
        return _err("path 空")
    try:
        fp = _safe_out(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content or "", encoding="utf-8")
        return json.dumps({"ok": True, "path": str(fp.relative_to(PROJECT_ROOT))}, ensure_ascii=False)
    except Exception as e:
        return _err(str(e))


@tool
def query_channel_inventory(sku: str = "DEFAULT") -> str:
    """查渠道库存 demo"""
    key = sku if sku in _wms else "DEFAULT"
    row = _wms[key]
    avail = row["stock"] - row["reserved"]
    return json.dumps({
        "sku": key, "name": row["name"], "stock": row["stock"],
        "reserved": row["reserved"], "available": avail,
        "suggestion": "库存充足" if avail > 5000 else "建议补货",
    }, ensure_ascii=False)


@tool
def simple_web_search(query: str) -> str:
    """竞品/品类情报 mock"""
    q = (query or "").lower()
    hits = []
    for kw, lines in _intel_snippets.items():
        if kw != "default" and kw.lower() in q:
            hits.extend(lines)
    if not hits:
        hits = list(_intel_snippets["default"])
    return json.dumps({"query": query or "", "results": hits[:3]}, ensure_ascii=False, indent=2)


@tool
def query_sales_sql(sql: str) -> str:
    """只读查 channel_sales"""
    try:
        df = run_readonly_sql(sql or "")
        return json.dumps({
            "rows": len(df),
            "columns": list(df.columns),
            "preview": df.to_dict(orient="records"),
            "markdown_table": _md_table(df),
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return _err(str(e))


@tool
def analyze_sales_pandas(analysis_type: str = "channel_compare", group_by: str = "channel") -> str:
    """pandas 渠道/SKU/趋势"""
    sales = load_sales_dataframe()
    if sales.empty:
        return _err("销售数据空")

    atype = analysis_type or "channel_compare"
    report = {"analysis_type": atype, "group_by": group_by or "channel"}

    if atype == "trend":
        monthly = sales.groupby("sale_month", as_index=False).agg(units=("units", "sum"), revenue=("revenue", "sum")).sort_values("sale_month")
        report["summary"] = f"覆盖 {len(monthly)} 个月"
        report["table"] = monthly.to_dict(orient="records")

    elif atype == "ranking":
        tops = sales.groupby("product_name", as_index=False).agg(units=("units", "sum"), revenue=("revenue", "sum")).sort_values("revenue", ascending=False).head(10)
        report["summary"] = f"TOP {len(tops)} SKU"
        report["table"] = tops.to_dict(orient="records")

    elif atype == "channel_compare":
        by_ch = sales.groupby("channel", as_index=False).agg(units=("units", "sum"), revenue=("revenue", "sum")).sort_values("revenue", ascending=False)
        if by_ch.empty:
            return _err("渠道汇总空")
        lead = by_ch.iloc[0]
        report["summary"] = f"领先 {lead['channel']} 销售额 {float(lead['revenue']):,.0f}"
        report["table"] = by_ch.to_dict(orient="records")

    elif atype == "region_summary":
        dim = group_by if group_by in sales.columns else "region"
        by_r = sales.groupby(dim, as_index=False).agg(units=("units", "sum"), revenue=("revenue", "sum")).sort_values("revenue", ascending=False)
        report["summary"] = f"按 {dim} {len(by_r)} 组"
        report["table"] = by_r.to_dict(orient="records")
    else:
        # 乱传 type 时默认渠道
        by_ch = sales.groupby("channel", as_index=False).agg(revenue=("revenue", "sum")).sort_values("revenue", ascending=False)
        report["summary"] = "未知 analysis_type，已回退 channel_compare"
        report["table"] = by_ch.to_dict(orient="records")

    report["markdown_table"] = _md_table(pd.DataFrame(report.get("table") or []))
    return json.dumps(report, ensure_ascii=False, indent=2)


@tool
def integrate_data_sources(task_description: str) -> str:
    """SQL+pandas+RAG 拼一起"""
    desc = task_description or ""
    try:
        ch = run_readonly_sql(
            "SELECT channel, SUM(units) AS units, SUM(revenue) AS revenue "
            "FROM channel_sales GROUP BY channel ORDER BY revenue DESC"
        )
        skus = load_sales_dataframe().groupby("product_name", as_index=False).agg(units=("units", "sum"), revenue=("revenue", "sum")).sort_values("revenue", ascending=False).head(5)
        policy = format_retrieved_context(retrieve_context(desc, k=CFG.rag.default_k))
        payload = {
            "task": desc,
            "sources": ["sqlite", "pandas", "rag"],
            "channel_summary": ch.to_dict(orient="records") if not ch.empty else [],
            "top_skus": skus.to_dict(orient="records") if not skus.empty else [],
            "policy_context": policy,
            "markdown": f"### 渠道\n{_md_table(ch)}\n\n### SKU TOP5\n{_md_table(skus)}\n\n### 政策\n{policy}",
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception as e:
        return _err(str(e))


BASE_TOOLS = [read_local_file, write_local_file, query_channel_inventory, simple_web_search]
DATA_TOOLS = [query_sales_sql, analyze_sales_pandas, integrate_data_sources]
EXECUTOR_TOOLS = BASE_TOOLS + DATA_TOOLS
TOOL_MAP = {t.name: t for t in EXECUTOR_TOOLS}
