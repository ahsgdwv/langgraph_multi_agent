"""外部工具：文件读写、库存查询、模拟搜索（Function Calling）。"""
from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import tool

from paths import PROJECT_ROOT, TOOL_OUTPUT_DIR

OUTPUT_DIR = TOOL_OUTPUT_DIR

# 模拟 618 活动库存
_MOCK_INVENTORY = {
    "SKU-618-001": {"name": "爆款蓝牙耳机", "stock": 1280, "reserved": 320},
    "SKU-618-002": {"name": "智能手环 Pro", "stock": 860, "reserved": 140},
    "SKU-618-003": {"name": "618 限定礼盒", "stock": 420, "reserved": 95},
    "DEFAULT": {"name": "通用活动SKU", "stock": 5000, "reserved": 800},
}

# 模拟搜索引擎结果库
_MOCK_SEARCH_INDEX = {
    "618": [
        "2025年618大促平台规则：跨店满300减50，可与店铺券叠加，上限40%。",
        "618直播带货峰值时段：6月17日20:00-24:00，建议提前2小时预热。",
    ],
    "竞品": [
        "竞品A：618主打「满200减30+会员95折」，重点投放短视频。",
        "竞品B：618采用「定金膨胀+尾款立减」，直播间专属券引流。",
    ],
    "default": [
        "电商大促常见策略：预热种草、爆发冲刺、返场清仓三阶段。",
        "活动复盘核心指标：GMV、ROI、转化率、客单价、退货率。",
    ],
}


def _safe_output_path(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = OUTPUT_DIR / candidate
    candidate = candidate.resolve()
    if not str(candidate).startswith(str(OUTPUT_DIR.resolve())):
        raise ValueError("仅允许写入 data/outputs 目录")
    return candidate


@tool
def read_local_file(path: str) -> str:
    """读取本地文本文件（data/documents 或 data/outputs 下）。"""
    for base in (PROJECT_ROOT / "data" / "documents", OUTPUT_DIR):
        candidate = base / path if not Path(path).is_absolute() else Path(path)
        if candidate.exists() and candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    return json.dumps({"error": f"文件不存在: {path}"}, ensure_ascii=False)


@tool
def write_local_file(path: str, content: str) -> str:
    """写入本地文本文件到 data/outputs 目录。"""
    try:
        target = _safe_output_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return json.dumps(
            {"success": True, "path": str(target.relative_to(PROJECT_ROOT))},
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
def query_activity_inventory(sku: str = "DEFAULT") -> str:
    """查询 618 活动商品模拟库存。"""
    key = sku if sku in _MOCK_INVENTORY else "DEFAULT"
    item = _MOCK_INVENTORY[key]
    available = item["stock"] - item["reserved"]
    return json.dumps(
        {
            "sku": key,
            "name": item["name"],
            "stock": item["stock"],
            "reserved": item["reserved"],
            "available": available,
            "suggestion": "库存充足" if available > 200 else "建议补货",
        },
        ensure_ascii=False,
    )


@tool
def simple_web_search(query: str) -> str:
    """模拟搜索引擎，返回与 query 相关的预设摘要（演示用）。"""
    query_lower = query.lower()
    hits: list[str] = []
    for key, snippets in _MOCK_SEARCH_INDEX.items():
        if key != "default" and key.lower() in query_lower:
            hits.extend(snippets)
    if not hits:
        hits = list(_MOCK_SEARCH_INDEX["default"])
    return json.dumps(
        {"query": query, "results": hits[:3]},
        ensure_ascii=False,
        indent=2,
    )


EXECUTOR_TOOLS = [
    read_local_file,
    write_local_file,
    query_activity_inventory,
    simple_web_search,
]

TOOL_MAP = {t.name: t for t in EXECUTOR_TOOLS}
