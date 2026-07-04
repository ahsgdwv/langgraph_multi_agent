"""项目路径常量。"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

# 用户查看/下载的最终报告（网页、CLI 保存）
REPORTS_DIR = PROJECT_ROOT / "reports"

# Agent 工具写入的运行时文件
TOOL_OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"

# RAG 业务文档
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"


def ensure_user_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    TOOL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)


def save_user_report(content: str, thread_id: str = "") -> Path:
    """保存 Markdown 报告到 reports/ 目录，供用户直接查看。"""
    ensure_user_dirs()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = (thread_id or "run")[:8]
    path = REPORTS_DIR / f"report_{ts}_{short_id}.md"
    path.write_text(content, encoding="utf-8")
    return path
