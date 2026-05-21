"""
Seeds the database with realistic mock policies and adjusters.
Run once at startup via init_db_with_seed().
"""

import json
from db.database import get_conn, init_db, _now


MOCK_POLICIES = [
    {
        "policy_id": "POL-AUTO-001",
        "customer_name": "Rahul Sharma",
        "coverage_types": ["collision", "comprehensive", "liability", "rental_reimbursement"],
        "deductible": 500.0,
        "max_payout": 50000.0,
        "policy_start": "2024-01-15",
        "monthly_premium": 120.0,
    },
    {
        "policy_id": "POL-AUTO-002",
        "customer_name": "Priya Patel",
        "coverage_types": ["collision", "liability"],
        "deductible": 1000.0,
        "max_payout": 30000.0,
        "policy_start": "2023-06-01",
        "monthly_premium": 85.0,
    },
    {
        "policy_id": "POL-HOME-001",
        "customer_name": "Amit Gupta",
        "coverage_types": ["fire", "flood", "theft", "windstorm", "liability"],
        "deductible": 2000.0,
        "max_payout": 300000.0,
        "policy_start": "2022-03-10",
        "monthly_premium": 210.0,
    },
    {
        "policy_id": "POL-HOME-002",
        "customer_name": "Sunita Reddy",
        "coverage_types": ["fire", "theft", "liability"],
        "deductible": 1500.0,
        "max_payout": 200000.0,
        "policy_start": "2023-11-20",
        "monthly_premium": 175.0,
    },
    {
        "policy_id": "POL-COMBO-001",
        "customer_name": "Vikram Singh",
        "coverage_types": ["collision", "comprehensive", "fire", "theft", "flood", "liability"],
        "deductible": 750.0,
        "max_payout": 150000.0,
        "policy_start": "2021-08-01",
        "monthly_premium": 290.0,
    },
]


MOCK_ADJUSTERS = [
    {
        "adjuster_id": "ADJ-001",
        "name": "Deepa Nair",
        "specialization": "auto",
        "available_slots": [
            "2026-05-21T10:00:00", "2026-05-21T14:00:00",
            "2026-05-22T09:00:00", "2026-05-22T15:00:00",
        ],
    },
    {
        "adjuster_id": "ADJ-002",
        "name": "Kiran Mehta",
        "specialization": "home",
        "available_slots": [
            "2026-05-21T11:00:00", "2026-05-22T10:00:00",
            "2026-05-23T09:00:00", "2026-05-23T14:00:00",
        ],
    },
    {
        "adjuster_id": "ADJ-003",
        "name": "Ravi Krishnan",
        "specialization": "both",
        "available_slots": [
            "2026-05-21T13:00:00", "2026-05-22T11:00:00",
            "2026-05-22T16:00:00", "2026-05-23T10:00:00",
        ],
    },
]


def seed_if_empty():
    """Insert mock data only if tables are empty."""
    with get_conn() as conn:
        if conn.execute("SELECT COUNT(*) FROM policies").fetchone()[0] == 0:
            for p in MOCK_POLICIES:
                conn.execute(
                    """INSERT INTO policies
                       (policy_id, customer_name, coverage_types, deductible,
                        max_payout, policy_start, monthly_premium)
                       VALUES (?,?,?,?,?,?,?)""",
                    (p["policy_id"], p["customer_name"],
                     json.dumps(p["coverage_types"]),
                     p["deductible"], p["max_payout"],
                     p["policy_start"], p["monthly_premium"]),
                )

        if conn.execute("SELECT COUNT(*) FROM adjusters").fetchone()[0] == 0:
            for a in MOCK_ADJUSTERS:
                conn.execute(
                    """INSERT INTO adjusters
                       (adjuster_id, name, specialization, available_slots)
                       VALUES (?,?,?,?)""",
                    (a["adjuster_id"], a["name"], a["specialization"],
                     json.dumps(a["available_slots"])),
                )


def init_db_with_seed():
    init_db()
    seed_if_empty()
