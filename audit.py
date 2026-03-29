"""
Audit logger — persists every request outcome to a database.

Backend is selected automatically:
  SQLite   (default)  — no config needed, file stored at audit.db
  Postgres            — set DATABASE_URL=postgresql://user:pass@host/db

Both backends expose the same interface so nothing else in the codebase changes.

Postgres notes:
  - psycopg2-binary must be installed (included in requirements.txt)
  - DATABASE_URL may use either postgres:// or postgresql:// scheme
  - A small connection pool (1–10 connections) is used automatically
"""

import json
import os
import sqlite3
import time
from typing import Optional


# ── Shared helpers ────────────────────────────────────────────────────────────

def _row_to_dict(row) -> dict:
    return {
        "id":               row["id"],
        "timestamp":        row["ts"],
        "tenant_id":        row["tenant_id"],
        "api":              row["api"],
        "outcome":          row["outcome"],
        "reason":           row["reason"],
        "messages":         json.loads(row["request"] or "[]"),
        "prompt_tokens":    row["prompt_tokens"],
        "completion_tokens": row["completion_tokens"],
    }


# ── SQLite backend ────────────────────────────────────────────────────────────

_DATA_DIR = os.environ.get("DATA_DIR", os.path.dirname(__file__))


class SQLiteBackend:
    _DB_PATH = os.path.join(_DATA_DIR, "audit.db")

    def __init__(self):
        self._init_db()

    def _conn(self):
        return sqlite3.connect(self._DB_PATH)

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts               REAL    NOT NULL,
                    tenant_id        TEXT    NOT NULL,
                    api              TEXT    NOT NULL DEFAULT 'openai',
                    outcome          TEXT    NOT NULL,
                    reason           TEXT,
                    request          TEXT,
                    prompt_tokens    INTEGER,
                    completion_tokens INTEGER
                )
            """)
            for col, defn in [
                ("api", "TEXT NOT NULL DEFAULT 'openai'"),
                ("prompt_tokens", "INTEGER"),
                ("completion_tokens", "INTEGER"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE audit_log ADD COLUMN {col} {defn}")
                except Exception:
                    pass  # column already exists

    def log(self, tenant_id, outcome, reason, body, api, usage=None):
        messages = body.get("messages", [])
        pt = usage.get("prompt_tokens") if usage else None
        ct = usage.get("completion_tokens") if usage else None
        try:
            with self._conn() as conn:
                conn.execute(
                    "INSERT INTO audit_log (ts, tenant_id, api, outcome, reason, request, prompt_tokens, completion_tokens) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (time.time(), tenant_id, api, outcome, reason, json.dumps(messages), pt, ct),
                )
        except Exception as e:
            print(f"[audit/sqlite] write failed: {e}")

    def get_recent(self, limit, tenant_id, api):
        conditions, params = [], []
        if tenant_id:
            conditions.append("tenant_id = ?")
            params.append(tenant_id)
        if api:
            conditions.append("api = ?")
            params.append(api)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"SELECT id, ts, tenant_id, api, outcome, reason, request, prompt_tokens, completion_tokens "
                f"FROM audit_log {where} ORDER BY id DESC LIMIT ?",
                params,
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def get_stats(self, tenant_id):
        conditions, params = [], []
        if tenant_id:
            conditions.append("tenant_id = ?")
            params.append(tenant_id)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        not_passed = f"{where} {'AND' if where else 'WHERE'} outcome != 'passed'"

        with self._conn() as conn:
            total   = conn.execute(f"SELECT COUNT(*) FROM audit_log {where}", params).fetchone()[0]
            blocked = conn.execute(f"SELECT COUNT(*) FROM audit_log {not_passed}", params).fetchone()[0]
            reasons = conn.execute(
                f"SELECT reason, COUNT(*) FROM audit_log {not_passed} GROUP BY reason ORDER BY COUNT(*) DESC LIMIT 5",
                params,
            ).fetchall()

        return {
            "total_requests":   total,
            "blocked_requests": blocked,
            "pass_rate":        round((total - blocked) / total * 100, 1) if total else 100.0,
            "top_block_reasons": [{"reason": r[0], "count": r[1]} for r in reasons],
        }

    def get_monthly_request_count(self, tenant_id: str) -> int:
        """Count passed requests for a tenant since the start of the current calendar month."""
        import calendar
        now = time.gmtime()
        month_start = time.mktime((now.tm_year, now.tm_mon, 1, 0, 0, 0, 0, 0, -1))
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM audit_log "
                "WHERE tenant_id = ? AND outcome = 'passed' AND ts >= ?",
                (tenant_id, month_start),
            ).fetchone()
        return row[0] if row else 0

    def get_token_metrics(self, tenant_id, days):
        since = time.time() - days * 86400
        conditions = ["ts >= ?"]
        params = [since]
        if tenant_id:
            conditions.append("tenant_id = ?")
            params.append(tenant_id)
        where = "WHERE " + " AND ".join(conditions)

        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT tenant_id, api, "
                f"COUNT(*) as requests, "
                f"COALESCE(SUM(prompt_tokens), 0) as input_tokens, "
                f"COALESCE(SUM(completion_tokens), 0) as output_tokens "
                f"FROM audit_log {where} "
                f"GROUP BY tenant_id, api ORDER BY input_tokens + output_tokens DESC",
                params,
            ).fetchall()

        result = {}
        for r in rows:
            tid = r[0]
            if tid not in result:
                result[tid] = {"tenant_id": tid, "requests": 0, "input_tokens": 0, "output_tokens": 0, "by_api": {}}
            result[tid]["requests"]     += r[2]
            result[tid]["input_tokens"] += r[3]
            result[tid]["output_tokens"] += r[4]
            result[tid]["by_api"][r[1]]  = {"requests": r[2], "input_tokens": r[3], "output_tokens": r[4]}

        tenants = list(result.values())
        for t in tenants:
            t["total_tokens"] = t["input_tokens"] + t["output_tokens"]

        totals = {
            "requests":     sum(t["requests"] for t in tenants),
            "input_tokens": sum(t["input_tokens"] for t in tenants),
            "output_tokens": sum(t["output_tokens"] for t in tenants),
            "total_tokens": sum(t["total_tokens"] for t in tenants),
        }
        return {"period_days": days, "tenants": tenants, "totals": totals}


# ── Postgres backend ──────────────────────────────────────────────────────────

class PostgresBackend:
    def __init__(self, url: str):
        try:
            import psycopg2
            import psycopg2.pool
            import psycopg2.extras
        except ImportError:
            raise RuntimeError(
                "psycopg2-binary is required for Postgres support. "
                "Run: pip install psycopg2-binary"
            )

        self._psycopg2 = psycopg2
        self._extras   = psycopg2.extras

        # Normalise postgres:// → postgresql:// (Railway, Heroku, Render use the short form)
        if url.startswith("postgres://"):
            url = "postgresql://" + url[len("postgres://"):]

        self._pool = psycopg2.pool.SimpleConnectionPool(1, 10, dsn=url)
        self._init_db()

    def _conn(self):
        return self._pool.getconn()

    def _release(self, conn):
        self._pool.putconn(conn)

    def _init_db(self):
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS audit_log (
                        id                SERIAL PRIMARY KEY,
                        ts                DOUBLE PRECISION NOT NULL,
                        tenant_id         TEXT             NOT NULL,
                        api               TEXT             NOT NULL DEFAULT 'openai',
                        outcome           TEXT             NOT NULL,
                        reason            TEXT,
                        request           TEXT,
                        prompt_tokens     INTEGER,
                        completion_tokens INTEGER
                    )
                """)
                for col, defn in [
                    ("api", "TEXT NOT NULL DEFAULT 'openai'"),
                    ("prompt_tokens", "INTEGER"),
                    ("completion_tokens", "INTEGER"),
                ]:
                    cur.execute(f"""
                        DO $$ BEGIN
                            ALTER TABLE audit_log ADD COLUMN {col} {defn};
                        EXCEPTION WHEN duplicate_column THEN NULL;
                        END $$
                    """)
            conn.commit()
        finally:
            self._release(conn)

    def log(self, tenant_id, outcome, reason, body, api, usage=None):
        messages = body.get("messages", [])
        pt = usage.get("prompt_tokens") if usage else None
        ct = usage.get("completion_tokens") if usage else None
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO audit_log (ts, tenant_id, api, outcome, reason, request, prompt_tokens, completion_tokens) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                    (time.time(), tenant_id, api, outcome, reason, json.dumps(messages), pt, ct),
                )
            conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"[audit/postgres] write failed: {e}")
        finally:
            self._release(conn)

    def get_recent(self, limit, tenant_id, api):
        conditions, params = [], []
        if tenant_id:
            conditions.append("tenant_id = %s")
            params.append(tenant_id)
        if api:
            conditions.append("api = %s")
            params.append(api)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        conn = self._conn()
        try:
            with conn.cursor(cursor_factory=self._extras.RealDictCursor) as cur:
                cur.execute(
                    f"SELECT id, ts, tenant_id, api, outcome, reason, request, prompt_tokens, completion_tokens "
                    f"FROM audit_log {where} ORDER BY id DESC LIMIT %s",
                    params,
                )
                rows = cur.fetchall()
        finally:
            self._release(conn)

        return [_row_to_dict(r) for r in rows]

    def get_stats(self, tenant_id):
        conditions, params = [], []
        if tenant_id:
            conditions.append("tenant_id = %s")
            params.append(tenant_id)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        not_passed = f"{where} {'AND' if where else 'WHERE'} outcome != 'passed'"

        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM audit_log {where}", params)
                total = cur.fetchone()[0]

                cur.execute(f"SELECT COUNT(*) FROM audit_log {not_passed}", params)
                blocked = cur.fetchone()[0]

                cur.execute(
                    f"SELECT reason, COUNT(*) FROM audit_log {not_passed} "
                    f"GROUP BY reason ORDER BY COUNT(*) DESC LIMIT 5",
                    params,
                )
                reasons = cur.fetchall()
        finally:
            self._release(conn)

        return {
            "total_requests":   total,
            "blocked_requests": blocked,
            "pass_rate":        round((total - blocked) / total * 100, 1) if total else 100.0,
            "top_block_reasons": [{"reason": r[0], "count": r[1]} for r in reasons],
        }

    def get_monthly_request_count(self, tenant_id: str) -> int:
        import time as _time
        now = _time.gmtime()
        month_start = _time.mktime((now.tm_year, now.tm_mon, 1, 0, 0, 0, 0, 0, -1))
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM audit_log "
                    "WHERE tenant_id = %s AND outcome = 'passed' AND ts >= %s",
                    (tenant_id, month_start),
                )
                row = cur.fetchone()
        finally:
            self._release(conn)
        return row[0] if row else 0

    def get_token_metrics(self, tenant_id, days):
        since = time.time() - days * 86400
        conditions = ["ts >= %s"]
        params = [since]
        if tenant_id:
            conditions.append("tenant_id = %s")
            params.append(tenant_id)
        where = "WHERE " + " AND ".join(conditions)

        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT tenant_id, api, "
                    f"COUNT(*) as requests, "
                    f"COALESCE(SUM(prompt_tokens), 0) as input_tokens, "
                    f"COALESCE(SUM(completion_tokens), 0) as output_tokens "
                    f"FROM audit_log {where} "
                    f"GROUP BY tenant_id, api ORDER BY input_tokens + output_tokens DESC",
                    params,
                )
                rows = cur.fetchall()
        finally:
            self._release(conn)

        result = {}
        for r in rows:
            tid = r[0]
            if tid not in result:
                result[tid] = {"tenant_id": tid, "requests": 0, "input_tokens": 0, "output_tokens": 0, "by_api": {}}
            result[tid]["requests"]      += r[2]
            result[tid]["input_tokens"]  += r[3]
            result[tid]["output_tokens"] += r[4]
            result[tid]["by_api"][r[1]]   = {"requests": r[2], "input_tokens": r[3], "output_tokens": r[4]}

        tenants = list(result.values())
        for t in tenants:
            t["total_tokens"] = t["input_tokens"] + t["output_tokens"]

        totals = {
            "requests":      sum(t["requests"] for t in tenants),
            "input_tokens":  sum(t["input_tokens"] for t in tenants),
            "output_tokens": sum(t["output_tokens"] for t in tenants),
            "total_tokens":  sum(t["total_tokens"] for t in tenants),
        }
        return {"period_days": days, "tenants": tenants, "totals": totals}


# ── Public interface ──────────────────────────────────────────────────────────

class AuditLogger:
    """
    Thin wrapper that selects SQLite or Postgres based on DATABASE_URL.
    All call sites use this class — the backend is an implementation detail.
    """

    def __init__(self):
        url = os.environ.get("DATABASE_URL", "")
        if url.startswith(("postgresql://", "postgres://")):
            print(f"[audit] Using Postgres backend")
            self._backend = PostgresBackend(url)
        else:
            self._backend = SQLiteBackend()

    def log(self, tenant_id: str, outcome: str, reason: Optional[str], body: dict, api: str = "openai",
            usage: Optional[dict] = None):
        self._backend.log(tenant_id, outcome, reason, body, api, usage)

    def get_recent(self, limit: int = 50, tenant_id: Optional[str] = None, api: Optional[str] = None) -> list[dict]:
        return self._backend.get_recent(limit, tenant_id, api)

    def get_stats(self, tenant_id: Optional[str] = None) -> dict:
        return self._backend.get_stats(tenant_id)

    def get_token_metrics(self, tenant_id: Optional[str] = None, days: int = 7) -> dict:
        return self._backend.get_token_metrics(tenant_id, days)

    def get_monthly_request_count(self, tenant_id: str) -> int:
        return self._backend.get_monthly_request_count(tenant_id)
