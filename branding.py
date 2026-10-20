from __future__ import annotations

import os

DEFAULT_REGION = "华东区"
FALLBACK_TITLE = "快消渠道数据分析 Agent"


def get_brand_name() -> str:
    return os.getenv("BRAND_NAME", "").strip()


def get_brand_region() -> str:
    raw = os.getenv("BRAND_REGION", DEFAULT_REGION).strip()
    return raw or DEFAULT_REGION


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
        return (
            f"当前演示品牌：**{brand}** · 渠道销量统计 · SKU 排行 · "
            f"政策检索 · 多源整合周报"
        )
    return "通用饮品快消场景 · 渠道销量统计 · SKU 排行 · 政策检索 · 多源整合周报"
