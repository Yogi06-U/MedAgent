"""Audit logging for clinical decisions and system events."""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from medagent.config import settings

logger = logging.getLogger("medagent.audit")


def get_db_connection() -> sqlite3.Connection:
    """Return an active connection to the SQLite database."""
    conn = sqlite3.connect(str(settings.DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_audit_db() -> None:
    """Initialize audit table schemas."""
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                case_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                reviewer TEXT,
                decision TEXT,
                details_json TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_case_id ON audit_logs(case_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_logs(timestamp)
        """)
        conn.commit()


init_audit_db()


def log_event(
    case_id: str,
    event_type: str,
    reviewer: Optional[str] = None,
    decision: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Persist an audit event.
    Compliance: Logs case_id at INFO level; NEVER logs full patient text/PHI at INFO level.
    """
    ts = datetime.now(timezone.utc).isoformat()
    details_str = json.dumps(details or {})

    # Info log only records case_id, event type, and reviewer
    logger.info(
        "AUDIT: event=%s case_id=%s reviewer=%s decision=%s",
        event_type, case_id, reviewer or "system", decision or "n/a"
    )

    try:
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO audit_logs (timestamp, case_id, event_type, reviewer, decision, details_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (ts, case_id, event_type, reviewer, decision, details_str)
            )
            conn.commit()
    except Exception as e:
        logger.error("Failed to write to audit log: %s", str(e))


def get_audit_trail(case_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    """Retrieve audit records, optionally filtered by case_id."""
    with get_db_connection() as conn:
        if case_id:
            cursor = conn.execute(
                """
                SELECT id, timestamp, case_id, event_type, reviewer, decision, details_json
                FROM audit_logs
                WHERE case_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (case_id, limit)
            )
        else:
            cursor = conn.execute(
                """
                SELECT id, timestamp, case_id, event_type, reviewer, decision, details_json
                FROM audit_logs
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,)
            )

        rows = cursor.fetchall()
        result = []
        for r in rows:
            entry = dict(r)
            try:
                entry["details"] = json.loads(entry.pop("details_json"))
            except Exception:
                entry["details"] = {}
            result.append(entry)
        return result
