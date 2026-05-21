"""
Agentic Tool Definitions + Execution Layer.

These tools let Agent 2 (claimParser) actually DO things:
  file_claim          → create a claim record, return a claim number
  get_claim_status    → look up an existing claim
  lookup_policy       → fetch policy details for the customer
  schedule_adjuster   → book an adjuster appointment
  calculate_deductible → compute how much the customer owes

The LLM returns tool_calls in its response; we execute them here
and feed the results back into the conversation so the LLM can
incorporate them into its final reply.
"""

import json
from datetime import datetime, timedelta
from db import database as db


# ── OpenAI-format tool schemas ─────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "file_claim",
            "description": (
                "File a new insurance claim in the system. "
                "Call this when the customer wants to start a claim for an auto accident, "
                "home damage, or theft. Returns a unique claim number."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "enum": ["Auto claim", "Home claim", "Coverage inquiry"],
                        "description": "Type of claim",
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["Low", "Medium", "High"],
                    },
                    "description": {
                        "type": "string",
                        "description": "Brief description of the incident as reported by the customer",
                    },
                    "loss_amount": {
                        "type": "number",
                        "description": "Estimated loss in dollars if mentioned. Omit if not stated.",
                    },
                    "policy_id": {
                        "type": "string",
                        "description": "Policy ID if the customer mentioned one.",
                    },
                },
                "required": ["intent", "priority", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_claim_status",
            "description": "Look up the current status of an existing claim by claim number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "claim_number": {
                        "type": "string",
                        "description": "The claim number (e.g., CLM-2026-XXXX)",
                    }
                },
                "required": ["claim_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_policy",
            "description": "Fetch policy coverage details for a given policy ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "policy_id": {
                        "type": "string",
                        "description": "Policy ID (e.g., POL-AUTO-001)",
                    }
                },
                "required": ["policy_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_adjuster",
            "description": (
                "Assign and schedule an adjuster for a filed claim. "
                "Use after filing a claim when the customer wants to know when an adjuster will visit."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "claim_number": {"type": "string"},
                    "claim_type": {
                        "type": "string",
                        "enum": ["auto", "home", "both"],
                        "description": "Type of claim to match adjuster specialization",
                    },
                },
                "required": ["claim_number", "claim_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_deductible",
            "description": (
                "Calculate the customer's out-of-pocket deductible and "
                "estimated insurance payout for a given loss amount."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "policy_id": {"type": "string"},
                    "loss_amount": {
                        "type": "number",
                        "description": "Total estimated loss in dollars",
                    },
                },
                "required": ["policy_id", "loss_amount"],
            },
        },
    },
]


# ── Tool execution ─────────────────────────────────────────────────────────────

def execute_tool(name: str, arguments: dict, session_id: str = None) -> dict:
    """
    Execute a tool call and return the result as a dict.
    All tools return a 'result' key with a human-readable summary.
    """
    try:
        if name == "file_claim":
            return _file_claim(arguments, session_id)
        elif name == "get_claim_status":
            return _get_claim_status(arguments)
        elif name == "lookup_policy":
            return _lookup_policy(arguments)
        elif name == "schedule_adjuster":
            return _schedule_adjuster(arguments)
        elif name == "calculate_deductible":
            return _calculate_deductible(arguments)
        else:
            return {"error": f"Unknown tool: {name}"}
    except Exception as e:
        return {"error": str(e)}


def _file_claim(args: dict, session_id: str) -> dict:
    record = db.file_claim(
        session_id=session_id or "unknown",
        intent=args["intent"],
        loss_amount=args.get("loss_amount"),
        priority=args["priority"],
        description=args["description"],
        policy_id=args.get("policy_id"),
    )
    loss_str = f"${args['loss_amount']:,.0f}" if args.get("loss_amount") else "not specified"
    return {
        "claim_number": record["claim_number"],
        "status": "Submitted",
        "result": (
            f"Claim successfully filed. Your claim number is {record['claim_number']}. "
            f"Type: {args['intent']} | Priority: {args['priority']} | "
            f"Estimated loss: {loss_str}. "
            f"An adjuster will be assigned shortly."
        ),
    }


def _get_claim_status(args: dict) -> dict:
    claim = db.get_claim(args["claim_number"])
    if not claim:
        return {"result": f"No claim found with number {args['claim_number']}."}
    adjuster_info = ""
    if claim.get("adjuster_id") and claim.get("adjuster_slot"):
        adjuster = db.get_conn().execute(
            "SELECT name FROM adjusters WHERE adjuster_id=?", (claim["adjuster_id"],)
        ).fetchone()
        name = adjuster["name"] if adjuster else "Unknown"
        slot = claim["adjuster_slot"][:16].replace("T", " at ")
        adjuster_info = f" Adjuster {name} is scheduled for {slot}."
    return {
        "claim_number": claim["claim_number"],
        "status": claim["status"],
        "priority": claim["priority"],
        "result": (
            f"Claim {claim['claim_number']}: Status is '{claim['status']}', "
            f"Priority: {claim['priority']}.{adjuster_info}"
        ),
    }


def _lookup_policy(args: dict) -> dict:
    policy = db.get_policy(args["policy_id"])
    if not policy:
        return {"result": f"Policy {args['policy_id']} not found."}
    coverages = ", ".join(policy["coverage_types"])
    return {
        "policy_id": policy["policy_id"],
        "customer_name": policy["customer_name"],
        "coverage_types": policy["coverage_types"],
        "deductible": policy["deductible"],
        "max_payout": policy["max_payout"],
        "status": policy["status"],
        "result": (
            f"Policy {policy['policy_id']} for {policy['customer_name']}: "
            f"Covers {coverages}. "
            f"Deductible: ${policy['deductible']:,.0f}. "
            f"Max payout: ${policy['max_payout']:,.0f}. "
            f"Status: {policy['status']}."
        ),
    }


def _schedule_adjuster(args: dict) -> dict:
    claim_type_map = {"auto": "auto", "home": "home", "both": "both"}
    spec = claim_type_map.get(args.get("claim_type", "both"), "both")
    adjuster = db.get_available_adjuster(spec)
    if not adjuster or not adjuster["available_slots"]:
        return {"result": "No adjusters currently available. We will contact you within 24 hours."}
    slot = adjuster["available_slots"][0]
    db.assign_adjuster(args["claim_number"], adjuster["adjuster_id"], slot)
    slot_str = slot[:16].replace("T", " at ")
    return {
        "adjuster_name": adjuster["name"],
        "slot": slot,
        "result": (
            f"Adjuster {adjuster['name']} has been assigned to claim {args['claim_number']}. "
            f"Scheduled visit: {slot_str}. "
            f"You will receive a confirmation call 30 minutes before the visit."
        ),
    }


def _calculate_deductible(args: dict) -> dict:
    policy = db.get_policy(args["policy_id"])
    loss = float(args["loss_amount"])
    if not policy:
        return {"result": f"Policy {args['policy_id']} not found."}
    deductible = policy["deductible"]
    max_payout = policy["max_payout"]
    customer_pays = min(deductible, loss)
    insurance_pays = min(max(loss - deductible, 0), max_payout)
    return {
        "loss_amount": loss,
        "deductible": deductible,
        "customer_pays": customer_pays,
        "insurance_pays": insurance_pays,
        "result": (
            f"For a loss of ${loss:,.0f}: "
            f"Your deductible is ${deductible:,.0f}. "
            f"You pay: ${customer_pays:,.0f}. "
            f"Insurance covers: ${insurance_pays:,.0f} "
            f"(subject to policy verification and investigation)."
        ),
    }


def run_tool_loop(client, model: str, messages: list, session_id: str = None) -> tuple[list, list]:
    """
    Run the tool-calling loop: send messages to LLM, execute any tool calls,
    feed results back, and repeat until no more tool calls.
    
    Returns:
        (updated_messages, tool_results) where tool_results is a list of dicts
        with tool name + output for logging.
    """
    tool_results = []
    max_rounds = 5  # prevent infinite loops

    for _ in range(max_rounds):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.0,
            max_tokens=800,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            # No tool calls — LLM gave a direct response, stop the loop
            messages.append({"role": "assistant", "content": msg.content or ""})
            break

        # Execute each tool call
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            result = execute_tool(tc.function.name, args, session_id)
            tool_results.append({"tool": tc.function.name, "args": args, "result": result})
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            })

    return messages, tool_results
