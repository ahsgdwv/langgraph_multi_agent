from __future__ import annotations

import os
from dataclasses import dataclass, field

from paths import ANALYTICS_DIR, PROJECT_ROOT


@dataclass(frozen=True)
class RagConfig:
    chroma_dir: str = str(PROJECT_ROOT / "data" / "chroma")
    collection_name: str = "business_docs"
    chunk_size: int = 500
    chunk_overlap: int = 80
    default_k: int = 4
    researcher_k: int = 6
    analyst_k: int = 3


@dataclass(frozen=True)
class AnalyticsConfig:
    csv_path: str = str(ANALYTICS_DIR / "channel_sales.csv")
    db_path: str = str(ANALYTICS_DIR / "sales.db")
    table_name: str = "channel_sales"
    sql_row_limit: int = 50
    sku_top_n: int = 10


@dataclass(frozen=True)
class SkillConfig:
    # 各 skill 预估省多少人工分钟，周报里会用到
    manual_minutes: dict[str, int] = field(
        default_factory=lambda: {
            "inventory_lookup": 10,
            "channel_compare": 25,
            "sku_ranking": 20,
            "sales_trend": 30,
            "policy_lookup": 15,
            "integrated_report": 45,
        }
    )


@dataclass(frozen=True)
class BrandConfig:
    name: str = field(default_factory=lambda: os.getenv("BRAND_NAME", "").strip())
    region: str = field(
        default_factory=lambda: os.getenv("BRAND_REGION", "华东区").strip() or "华东区"
    )


@dataclass(frozen=True)
class AppConfig:
    rag: RagConfig = field(default_factory=RagConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    skill: SkillConfig = field(default_factory=SkillConfig)
    brand: BrandConfig = field(default_factory=BrandConfig)


CFG = AppConfig()
