from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path

import pandas as pd

from config import CFG

log = logging.getLogger(__name__)

csv_file = Path(CFG.analytics.csv_path)
db_file = Path(CFG.analytics.db_path)
sales_table = CFG.analytics.table_name
_select_ok = re.compile(r"^\s*(SELECT|WITH)\b", re.I)


def _csv_signature() -> str:
    st = csv_file.stat()
    return f"{st.st_mtime_ns}:{st.st_size}"


def ensure_analytics_db():
    db_file.parent.mkdir(parents=True, exist_ok=True)
    if not csv_file.exists():
        return {"rows": 0, "message": f"缺 {csv_file.name}", "ok": False}

    try:
        conn = sqlite3.connect(str(db_file))
        try:
            has_table = bool(
                conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (sales_table,),
                ).fetchone()
            )
            signature = _csv_signature()
            if has_table:
                try:
                    row = conn.execute(
                        "SELECT value FROM _ingest_meta WHERE key='csv_signature'"
                    ).fetchone()
                except sqlite3.OperationalError:
                    row = None
                if row and row[0] == signature:
                    n = conn.execute(f"SELECT COUNT(*) FROM {sales_table}").fetchone()[0]
                    return {"rows": n, "message": f"分析库就绪（{n} 行）", "ok": True}

            df = pd.read_csv(csv_file, encoding="utf-8")
            df.to_sql(sales_table, conn, index=False, if_exists="replace")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS _ingest_meta "
                "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            conn.execute(
                "INSERT OR REPLACE INTO _ingest_meta(key, value) VALUES('csv_signature', ?)",
                (signature,),
            )
            conn.commit()
            return {"rows": len(df), "message": f"导入/刷新 {len(df)} 行", "ok": True}
        finally:
            conn.close()
    except Exception as e:
        log.error("分析库挂了: %s", e)
        return {"rows": 0, "message": str(e), "ok": False}


def load_sales_dataframe() -> pd.DataFrame:
    boot = ensure_analytics_db()
    if not boot.get("ok"):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(str(db_file))
        try:
            return pd.read_sql_query(f"SELECT * FROM {sales_table}", conn)
        finally:
            conn.close()
    except Exception as e:
        log.error("读销售表失败: %s", e)
        return pd.DataFrame()


def run_readonly_sql(sql: str, *, limit: int | None = None) -> pd.DataFrame:
    cap = limit or CFG.analytics.sql_row_limit
    sql = (sql or "").strip().rstrip(";")
    if not sql:
        raise ValueError("sql 不能为空")
    if not _select_ok.match(sql):
        raise ValueError("仅允许 SELECT / WITH")
    if "limit" not in sql.lower():
        sql = f"{sql} LIMIT {cap}"

    ensure_analytics_db()
    conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
    try:
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()
