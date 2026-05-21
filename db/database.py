"""
SQLite database layer — single source of persistence for all agent data.

Tables:
  policies       → mock policyholder data
  adjusters      → available claim adjusters
  claims         → filed claims with lifecycle status
  conversations  → full multi-turn chat history per session
  escalations    → human handoff tickets with full context
  analytics      → event log for dashboard metrics
"""

import sqlite3
import json
import uuid
import os
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "insurance.db"


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create all tables if they don't exist."""
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS policies (
            policy_id       TEXT PRIMARY KEY,
            customer_name   TEXT NOT NULL,
            coverage_types  TEXT NOT NULL,   -- JSON array
            deductible      REAL NOT NULL,
            max_payout      REAL NOT NULL,
            policy_start    TEXT NOT NULL,
            monthly_premium REAL NOT NULL,
            status          TEXT DEFAULT 'active'
        );

        CREATE TABLE IF NOT EXISTS adjusters (
            adjuster_id      TEXT PRIMARY KEY,
            name             TEXT NOT NULL,
            specialization   TEXT NOT NULL,   -- auto / home / both
            available_slots  TEXT NOT NULL    -- JSON array of ISO datetime strings
        );

        CREATE TABLE IF NOT EXISTS claims (
            claim_number     TEXT PRIMARY KEY,
            session_id       TEXT,
            policy_id        TEXT,
            intent           TEXT,
            loss_amount      REAL,
            priority         TEXT,
            description      TEXT,
            status           TEXT DEFAULT 'Submitted',
            fraud_risk       TEXT DEFAULT 'Low',
            fraud_flags      TEXT DEFAULT '[]',   -- JSON array
            adjuster_id      TEXT,
            adjuster_slot    TEXT,
            created_at       TEXT NOT NULL,
            updated_at       TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS conversations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL,
            role        TEXT NOT NULL,        -- user / assistant / tool
            content     TEXT NOT NULL,
            metadata    TEXT DEFAULT '{}',   -- JSON: intent, priority, claim_number, etc.
            created_at  TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id);

        CREATE TABLE IF NOT EXISTS escalations (
            ticket_id            TEXT PRIMARY KEY,
            session_id           TEXT NOT NULL,
            claim_number         TEXT,
            summary              TEXT,
            conversation_history TEXT,         -- JSON
            agent_outputs        TEXT,         -- JSON
            priority             TEXT,
            status               TEXT DEFAULT 'open',
            created_at           TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS analytics (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type  TEXT NOT NULL,
            session_id  TEXT,
            data        TEXT DEFAULT '{}',   -- JSON
            created_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS notifications (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id          TEXT,
            claim_number        TEXT,
            notification_type   TEXT NOT NULL,   -- claim_filed/status_changed/adjuster_assigned/fraud_flag
            message             TEXT NOT NULL,
            webhook_payload     TEXT DEFAULT '{}',
            is_read             INTEGER DEFAULT 0,
            created_at          TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_notif_session ON notifications(session_id);

        CREATE TABLE IF NOT EXISTS ab_results (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT,
            variant         TEXT NOT NULL,       -- A or B
            intent          TEXT,
            priority        TEXT,
            latency_ms      REAL,
            token_count     INTEGER,
            created_at      TEXT NOT NULL
        );
        """)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.utcnow().isoformat()

def _uid(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:8].upper()}"


# ── Policies ──────────────────────────────────────────────────────────────────

def get_policy(policy_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM policies WHERE policy_id=?", (policy_id,)).fetchone()
        if row:
            d = dict(row)
            d["coverage_types"] = json.loads(d["coverage_types"])
            return d
    return None


def list_policies() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM policies").fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["coverage_types"] = json.loads(d["coverage_types"])
            result.append(d)
        return result


# ── Adjusters ─────────────────────────────────────────────────────────────────

def get_available_adjuster(specialization: str = "both") -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM adjusters WHERE specialization=? OR specialization='both' LIMIT 1",
            (specialization,),
        ).fetchone()
        if row:
            d = dict(row)
            d["available_slots"] = json.loads(d["available_slots"])
            return d
    return None


# ── Claims ────────────────────────────────────────────────────────────────────

def file_claim(
    session_id: str,
    intent: str,
    loss_amount: float | None,
    priority: str,
    description: str,
    policy_id: str | None = None,
) -> dict:
    claim_number = f"CLM-{datetime.utcnow().strftime('%Y')}-{_uid()}"
    now = _now()
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO claims
               (claim_number, session_id, policy_id, intent, loss_amount, priority,
                description, status, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,'Submitted',?,?)""",
            (claim_number, session_id, policy_id, intent, loss_amount, priority,
             description, now, now),
        )
    log_event("claim_filed", session_id, {
        "claim_number": claim_number, "intent": intent,
        "priority": priority, "loss_amount": loss_amount,
    })
    return {"claim_number": claim_number, "status": "Submitted", "created_at": now}


def get_claim(claim_number: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM claims WHERE claim_number=?", (claim_number,)
        ).fetchone()
        if row:
            d = dict(row)
            d["fraud_flags"] = json.loads(d["fraud_flags"])
            return d
    return None


def update_claim_status(claim_number: str, status: str) -> bool:
    with get_conn() as conn:
        conn.execute(
            "UPDATE claims SET status=?, updated_at=? WHERE claim_number=?",
            (status, _now(), claim_number),
        )
    return True


def update_claim_fraud(claim_number: str, risk: str, flags: list) -> bool:
    with get_conn() as conn:
        conn.execute(
            "UPDATE claims SET fraud_risk=?, fraud_flags=?, updated_at=? WHERE claim_number=?",
            (risk, json.dumps(flags), _now(), claim_number),
        )
    return True


def assign_adjuster(claim_number: str, adjuster_id: str, slot: str) -> bool:
    with get_conn() as conn:
        conn.execute(
            "UPDATE claims SET adjuster_id=?, adjuster_slot=?, status='Adjuster Assigned', updated_at=? WHERE claim_number=?",
            (adjuster_id, slot, _now(), claim_number),
        )
    return True


def get_claims_by_session(session_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM claims WHERE session_id=? ORDER BY created_at DESC",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_recent_claims(limit: int = 20) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM claims ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


# ── Conversations (multi-turn memory) ─────────────────────────────────────────

def save_turn(session_id: str, role: str, content: str, metadata: dict = None):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO conversations (session_id, role, content, metadata, created_at) VALUES (?,?,?,?,?)",
            (session_id, role, content, json.dumps(metadata or {}), _now()),
        )


def get_history(session_id: str, last_n: int = 10) -> list[dict]:
    """Returns last N turns as list of {role, content} dicts for injection into LLM messages."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT role, content FROM conversations
               WHERE session_id=? AND role IN ('user','assistant')
               ORDER BY id DESC LIMIT ?""",
            (session_id, last_n),
        ).fetchall()
    turns = [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
    return turns


def get_full_history(session_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM conversations WHERE session_id=? ORDER BY id",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ── Escalations ───────────────────────────────────────────────────────────────

def create_escalation(
    session_id: str,
    claim_number: str | None,
    summary: str,
    conversation_history: list,
    agent_outputs: dict,
    priority: str,
) -> str:
    ticket_id = f"ESC-{_uid()}"
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO escalations
               (ticket_id, session_id, claim_number, summary, conversation_history,
                agent_outputs, priority, status, created_at)
               VALUES (?,?,?,?,?,?,'open',?,?)""",
            (ticket_id, session_id, claim_number, summary,
             json.dumps(conversation_history), json.dumps(agent_outputs),
             priority, _now()),
        )
    log_event("escalation_created", session_id, {"ticket_id": ticket_id, "priority": priority})
    return ticket_id


def get_open_escalations() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM escalations WHERE status='open' ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


# ── Analytics ─────────────────────────────────────────────────────────────────

# ── Notifications (webhook mock) ──────────────────────────────────────────────

def create_notification(
    session_id: str,
    claim_number: str,
    notification_type: str,
    message: str,
    webhook_payload: dict = None,
) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO notifications
               (session_id, claim_number, notification_type, message, webhook_payload, created_at)
               VALUES (?,?,?,?,?,?)""",
            (session_id, claim_number, notification_type, message,
             json.dumps(webhook_payload or {}), _now()),
        )
    return cur.lastrowid


def get_notifications(session_id: str = None, unread_only: bool = False) -> list[dict]:
    with get_conn() as conn:
        if session_id:
            q = "SELECT * FROM notifications WHERE session_id=?"
            params = [session_id]
        else:
            q = "SELECT * FROM notifications"
            params = []
        if unread_only:
            q += (" AND" if "WHERE" in q else " WHERE") + " is_read=0"
        q += " ORDER BY created_at DESC LIMIT 50"
        rows = conn.execute(q, params).fetchall()
    return [dict(r) for r in rows]


def mark_notifications_read(session_id: str):
    with get_conn() as conn:
        conn.execute(
            "UPDATE notifications SET is_read=1 WHERE session_id=?", (session_id,)
        )


# ── A/B Testing ───────────────────────────────────────────────────────────────

def log_ab_result(
    session_id: str,
    variant: str,
    intent: str,
    priority: str,
    latency_ms: float,
    token_count: int,
):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO ab_results
               (session_id, variant, intent, priority, latency_ms, token_count, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (session_id, variant, intent, priority, latency_ms, token_count, _now()),
        )


def get_ab_summary() -> dict:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT variant,
                      COUNT(*) as count,
                      AVG(latency_ms) as avg_latency,
                      AVG(token_count) as avg_tokens
               FROM ab_results GROUP BY variant"""
        ).fetchall()
    return {r["variant"]: dict(r) for r in rows}


def log_event(event_type: str, session_id: str = None, data: dict = None):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO analytics (event_type, session_id, data, created_at) VALUES (?,?,?,?)",
            (event_type, session_id, json.dumps(data or {}), _now()),
        )


def get_analytics_summary() -> dict:
    with get_conn() as conn:
        total_claims = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
        by_intent = conn.execute(
            "SELECT intent, COUNT(*) as cnt FROM claims GROUP BY intent"
        ).fetchall()
        by_priority = conn.execute(
            "SELECT priority, COUNT(*) as cnt FROM claims GROUP BY priority"
        ).fetchall()
        avg_loss = conn.execute(
            "SELECT AVG(loss_amount) FROM claims WHERE loss_amount IS NOT NULL"
        ).fetchone()[0]
        fraud_counts = conn.execute(
            "SELECT fraud_risk, COUNT(*) as cnt FROM claims GROUP BY fraud_risk"
        ).fetchall()
        pii_blocks = conn.execute(
            "SELECT COUNT(*) FROM analytics WHERE event_type='pii_block'"
        ).fetchone()[0]
        escalations = conn.execute(
            "SELECT COUNT(*) FROM escalations"
        ).fetchone()[0]
        status_counts = conn.execute(
            "SELECT status, COUNT(*) as cnt FROM claims GROUP BY status"
        ).fetchall()

    return {
        "total_claims": total_claims,
        "by_intent": {r["intent"]: r["cnt"] for r in by_intent},
        "by_priority": {r["priority"]: r["cnt"] for r in by_priority},
        "avg_loss_amount": round(avg_loss or 0, 2),
        "fraud_distribution": {r["fraud_risk"]: r["cnt"] for r in fraud_counts},
        "pii_blocks": pii_blocks,
        "escalations": escalations,
        "claim_statuses": {r["status"]: r["cnt"] for r in status_counts},
    }
