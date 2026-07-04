from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path

import pandas as pd

from config import CFG

logger = logging.getLogger(__name__)

CSV_PATH = Path(CFG.analytics.csv_path)
DB_PATH = Path(CFG.analytics.db_path)
TABLE_NAME = CFG.analytics.table_name
_READ_ONLY_SQL = re.compile(r"^\s*(SELECT|WITH)\b", re.I)


def ensure_analytics_db() -> dict:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists():
        return {"rows": 0, "message": f"缺少 {CSV_PATH.name}", "ok": False}

    try:
        conn = sqlite3.connect(str(DB_PATH))
        try:
            existing = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (TABLE_NAME,),
            ).fetchone()
            if existing:
                count = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
                return {"rows": count, "message": f"分析库就绪（{count} 行）", "ok": True}
            df = pd.read_csv(CSV_PATH, encoding="utf-8")
            df.to_sql(TABLE_NAME, conn, index=False, if_exists="replace")
            return {"rows": len(df), "message": f"导入 {len(df)} 行", "ok": True}
        finally:
            conn.close()
    except Exception as exc:
        logger.error("分析库初始化失败: %s", exc)
        return {"rows": 0, "message": str(exc), "ok": False}


def get_readonly_connection() -> sqlite3.Connection:
    ensure_analytics_db()
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


def load_sales_dataframe() -> pd.DataFrame:
    init = ensure_analytics_db()
    if not init.get("ok"):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(str(DB_PATH))
        try:
            return pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
        finally:
            conn.close()
    except Exception as exc:
        logger.error("读取销售数据失败: %s", exc)
        return pd.DataFrame()


def run_readonly_sql(sql: str, *, limit: int | None = None) -> pd.DataFrame:
    limit = limit or CFG.analytics.sql_row_limit
    sql = sql.strip().rstrip(";")
    if not _READ_ONLY_SQL.match(sql):
        raise ValueError("仅允许 SELECT / WITH 查询")
    if "limit" not in sql.lower():
        sql = f"{sql} LIMIT {limit}"
    conn = get_readonly_connection()
    try:
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()
