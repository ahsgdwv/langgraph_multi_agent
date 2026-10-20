from __future__ import annotations

from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

REPORTS_DIR = PROJECT_ROOT / "reports"          # 用户看的最终报告
TOOL_OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
ANALYTICS_DIR = PROJECT_ROOT / "data" / "analytics"


def ensure_user_dirs() -> None:
    for d in (REPORTS_DIR, TOOL_OUTPUT_DIR, DOCUMENTS_DIR, ANALYTICS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def save_user_report(content: str, thread_id: str = "") -> Path:
    ensure_user_dirs()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = (thread_id or "run")[:8]
    path = REPORTS_DIR / f"report_{ts}_{short_id}.md"
    path.write_text(content, encoding="utf-8")
    return path
