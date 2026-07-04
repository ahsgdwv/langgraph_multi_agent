"""全局配置：路径引用 paths，参数集中在此修改。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from paths import ANALYTICS_DIR, DOCUMENTS_DIR, PROJECT_ROOT, REPORTS_DIR, TOOL_OUTPUT_DIR


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
    # 各 Skill 预估节省的人工分钟数（用于周报执行统计）
    manual_minutes: dict[str, int] = field(
        default_factory=lambda: {
            "channel_compare": 25,
            "sku_ranking": 20,
            "sales_trend": 30,
            "policy_lookup": 15,
            "integrated_report": 45,
        }
    )


@dataclass(frozen=True)
class AppConfig:
    documents_dir: str = str(DOCUMENTS_DIR)
    reports_dir: str = str(REPORTS_DIR)
    tool_output_dir: str = str(TOOL_OUTPUT_DIR)
    rag: RagConfig = field(default_factory=RagConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    skill: SkillConfig = field(default_factory=SkillConfig)
    llm_tool_rounds: int = 3
    use_supervisor: bool = field(
        default_factory=lambda: os.getenv("USE_SUPERVISOR", "true").lower()
        in ("1", "true", "yes")
    )
    auto_approve_human: bool = field(
        default_factory=lambda: os.getenv("AUTO_APPROVE_HUMAN", "").lower()
        in ("1", "true", "yes")
    )


CFG = AppConfig()
