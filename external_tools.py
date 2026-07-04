"""外部工具：库存查询、竞品情报、文件读写。"""
from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import tool

from data_tools import DATA_TOOLS, DATA_TOOL_MAP
from paths import PROJECT_ROOT, TOOL_OUTPUT_DIR

OUTPUT_DIR = TOOL_OUTPUT_DIR

# 快消 SKU 模拟库存
_MOCK_INVENTORY = {
    "YQ-001": {"name": "元气森林气泡水480ml", "stock": 52000, "reserved": 12000},
    "YQ-002": {"name": "燃茶500ml", "stock": 31000, "reserved": 6500},
    "YQ-003": {"name": "外星人电解质水", "stock": 28000, "reserved": 5800},
    "DEFAULT": {"name": "通用饮料SKU", "stock": 45000, "reserved": 9000},
}

# 竞品与市场情报（模拟检索）
_MOCK_SEARCH_INDEX = {
    "竞品": [
        "竞品A（0糖气泡水）：便利店铺货率提升，夏季电商礼盒装主推。",
        "竞品B（电解质饮料）：健身房特通买2赠1，运动场景渗透加快。",
    ],
    "气泡水": [
        "即饮气泡水品类年增速约12%，无糖/低糖占比持续提升。",
        "元气森林气泡水在便利店渠道单店月均动销约180件，夏季峰值可达280件。",
    ],
    "default": [
        "饮料快消复盘指标：动销率、渠道费比、毛利率、库存周转天数。",
        "周报结构建议：渠道TOP/BOTTOM → SKU排行 → 异常预警 → 下周动作。",
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
def query_channel_inventory(sku: str = "DEFAULT") -> str:
    """查询快消 SKU 渠道库存（模拟 WMS 数据）。"""
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
            "suggestion": "库存充足" if available > 5000 else "建议补货",
        },
        ensure_ascii=False,
    )


@tool
def simple_web_search(query: str) -> str:
    """模拟市场情报检索，返回竞品/品类相关摘要。"""
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


# 兼容旧工具名
query_activity_inventory = query_channel_inventory

BASE_TOOLS = [
    read_local_file,
    write_local_file,
    query_channel_inventory,
    simple_web_search,
]

EXECUTOR_TOOLS = BASE_TOOLS + DATA_TOOLS
TOOL_MAP = {t.name: t for t in EXECUTOR_TOOLS}
TOOL_MAP.update(DATA_TOOL_MAP)
