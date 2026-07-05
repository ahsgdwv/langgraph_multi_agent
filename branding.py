"""演示品牌配置：通过环境变量切换 UI 标题与示例文案。"""
from __future__ import annotations

import os

DEFAULT_REGION = "华东区"
FALLBACK_TITLE = "快消渠道数据分析 Agent"


def get_brand_name() -> str:
    """BRAND_NAME 为空时使用通用快消文案。"""
    return os.getenv("BRAND_NAME", "").strip()


def get_brand_region() -> str:
    return os.getenv("BRAND_REGION", DEFAULT_REGION).strip() or DEFAULT_REGION


def get_display_title() -> str:
    brand = get_brand_name()
    if brand:
        return f"{brand}渠道数据分析 Agent"
    return FALLBACK_TITLE


def get_example_analysis_prefix() -> str:
    brand = get_brand_name()
    region = get_brand_region()
    if brand:
        return f"{brand}{region}渠道分析，需要完成："
    return f"{region}饮品渠道分析，需要完成："


def get_subtitle() -> str:
    brand = get_brand_name()
    if brand:
        return f"当前演示品牌：**{brand}** · 渠道销量统计 · SKU 排行 · 政策检索 · 多源整合周报"
    return "通用饮品快消场景 · 渠道销量统计 · SKU 排行 · 政策检索 · 多源整合周报"
