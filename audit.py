"""
Audit logger — persists every request outcome to a SQLite database.
GET /audit returns recent entries for visibility.
"""

import sqlite3
import json
import time
import os
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "audit.db")


class AuditLogger:
    def __init__(self):
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts        REAL    NOT NULL,
                    tenant_id TEXT    NOT NULL,
                    api       TEXT    NOT NULL DEFAULT 'openai',
                    outcome   TEXT    NOT NULL,
                    reason    TEXT,
                    request   TEXT
                )
            """)
            # Add api column if upgrading from v0.1
            try:
                conn.execute("ALTER TABLE audit_log ADD COLUMN api TEXT NOT NULL DEFAULT 'openai'")
            except Exception:
                pass

    def _conn(self):
        return sqlite3.connect(DB_PATH)

    def log(self, tenant_id: str, outcome: str, reason: Optional[str], body: dict, api: str = "openai"):
        messages = body.get("messages", [])
        try:
            with self._conn() as conn:
                conn.execute(
                    "INSERT INTO audit_log (ts, tenant_id, api, outcome, reason, request) VALUES (?,?,?,?,?,?)",
                    (time.time(), tenant_id, api, outcome, reason, json.dumps(messages)),
                )
        except Exception as e:
            print(f"[audit] write failed: {e}")

    def get_recent(
        self,
        limit: int = 50,
        tenant_id: Optional[str] = None,
        api: Optional[str] = None,
    ) -> list[dict]:
        conditions = []
        params: list = []

        if tenant_id:
            conditions.append("tenant_id = ?")
            params.append(tenant_id)
        if api:
            conditions.append("api = ?")
            params.append(api)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT id, ts, tenant_id, api, outcome, reason, request FROM audit_log {where} ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()

        return [
            {
                "id": row["id"],
                "timestamp": row["ts"],
                "tenant_id": row["tenant_id"],
                "api": row["api"],
                "outcome": row["outcome"],
                "reason": row["reason"],
                "messages": json.loads(row["request"] or "[]"),
            }
            for row in rows
        ]

    def get_stats(self, tenant_id: Optional[str] = None) -> dict:
        """Summary stats for the dashboard."""
        conditions = []
        params: list = []
        if tenant_id:
            conditions.append("tenant_id = ?")
            params.append(tenant_id)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        with self._conn() as conn:
            total = conn.execute(f"SELECT COUNT(*) FROM audit_log {where}", params).fetchone()[0]
            blocked = conn.execute(
                f"SELECT COUNT(*) FROM audit_log {where} {'AND' if where else 'WHERE'} outcome != 'passed'",
                params
            ).fetchone()[0]
            by_guardrail = conn.execute(
                f"SELECT reason, COUNT(*) as cnt FROM audit_log {where} {'AND' if where else 'WHERE'} outcome != 'passed' GROUP BY reason ORDER BY cnt DESC",
                params
            ).fetchall()

        return {
            "total_requests": total,
            "blocked_requests": blocked,
            "pass_rate": round((total - blocked) / total * 100, 1) if total else 100.0,
            "top_block_reasons": [{"reason": r[0], "count": r[1]} for r in by_guardrail[:5]],
        }
