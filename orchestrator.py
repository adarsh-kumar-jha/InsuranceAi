"""
Insurance AI Claims Inquiry Concierge — Enhanced Pipeline Orchestrator

Full agentic flow:
  [Input Validation]
        │ invalid → reject early
        ▼
  [Language Detection]  → detect & set response language
        ▼
  [RAG Hybrid Search]   → Top-K context chunks
        ▼
  [Agent 1] guardrailCheck
        │  not_insurance → BLOCK (off-topic)
        │  no_pii=False  → BLOCK (PII detected)
        │  escalation    → ESCALATE (package context + create ticket)
        ▼
  [Agent 2.5] Clarifier (optional)
        │  needs_clarification → return question to user
        ▼
  [Agent 2] claimParser  ← RAG + conversation history + tool results
        │  Agentic tool use: file_claim, lookup_policy, get_claim_status, etc.
        ▼
  [Agent 3] safetyChecks
        │  compliance_pass=False → BLOCK
        ▼
  [Agent 4] Fraud Detector
        ▼
  Final Reply → Policyholder
  [DB] save conversation turn + analytics event
"""

import time
import uuid
from typing import Optional, Any
from dotenv import load_dotenv

from agents.guardrail_check import run_guardrail_check
from agents.claim_parser import run_claim_parser
from agents.safety_checks import run_safety_checks
from agents.clarifier import run_clarifier
from agents.fraud_detector import run_fraud_detector
from agents.sentiment_agent import run_sentiment_agent
from agents.post_processor import run_post_processor
from agents.tools import run_tool_loop, TOOLS, execute_tool
from models.schemas import (
    PipelineResult, ClarifierOutput, ToolCallRecord
)
from db import database as dbq
from db.mock_data import init_db_with_seed

load_dotenv()

# Re-export so callers can still do `from orchestrator import get_client`
from api.llm_client import get_client  # noqa: E402

# Ensure DB is ready
init_db_with_seed()


def run_pipeline(
    message: str,
    client: Any = None,
    token_tracker=None,
    rag_context_str: str = "",
    rag_context_docs: list = None,
    session_id: str = None,
    language: str = "en",
) -> PipelineResult:
    """
    Full enhanced pipeline with:
      - Multi-turn conversation memory
      - Agentic tool use (file_claim, lookup_policy, etc.)
      - Agent 2.5 (Clarifier) for ambiguous messages
      - Agent 4 (Fraud Detector)
      - Language detection + multilingual responses
      - Human escalation with full context packaging
      - DB persistence for all turns and claims
    """
    if client is None:
        client = get_client()
    if session_id is None:
        session_id = f"sess_{uuid.uuid4().hex[:10]}"

    start_time = time.time()

    # ── Load conversation memory ──────────────────────────────────────────────
    conversation_history = dbq.get_history(session_id, last_n=10)

    # ── Agent 1: Intake Guardrails ────────────────────────────────────────────
    guardrail = run_guardrail_check(
        message, client,
        token_tracker=token_tracker,
        conversation_history=conversation_history,
    )

    if not guardrail.is_insurance_related:
        dbq.save_turn(session_id, "user", message)
        off_msg = _localize(
            "I'm sorry, I can only assist with insurance claims and coverage questions. "
            "Please contact us about your policy, a claim, or coverage details.",
            language,
        )
        dbq.save_turn(session_id, "assistant", off_msg,
                      {"block_reason": "off_topic"})
        dbq.log_event("off_topic", session_id, {"message": message[:100]})
        return PipelineResult(
            guardrail=guardrail,
            blocked=True,
            block_reason="off_topic",
            final_response=off_msg,
            latency_ms=_elapsed(start_time),
            rag_context=rag_context_docs,
            session_id=session_id,
            language_detected=language,
        )

    if not guardrail.no_pii:
        dbq.save_turn(session_id, "user", message)
        pii_msg = _localize(
            "For your security, please do not share sensitive identifiers such as "
            "Social Security Numbers, full policy numbers, or bank account details "
            "in this chat. Please contact us by phone or through our secure portal.",
            language,
        )
        dbq.save_turn(session_id, "assistant", pii_msg,
                      {"block_reason": "pii_detected"})
        dbq.log_event("pii_block", session_id, {"message": message[:100]})
        return PipelineResult(
            guardrail=guardrail,
            blocked=True,
            block_reason="pii_detected",
            final_response=pii_msg,
            latency_ms=_elapsed(start_time),
            rag_context=rag_context_docs,
            session_id=session_id,
            language_detected=language,
        )

    if guardrail.needs_escalation:
        dbq.save_turn(session_id, "user", message)

        # Package full context for human agent
        ticket_id = dbq.create_escalation(
            session_id=session_id,
            claim_number=None,
            summary=f"Customer requires immediate human escalation. Trigger message: {message[:200]}",
            conversation_history=conversation_history,
            agent_outputs={"guardrail": guardrail.model_dump()},
            priority="High",
        )

        esc_msg = _localize(
            "I can see you're dealing with a very serious situation. "
            "I'm immediately connecting you with a senior claims specialist who can assist you right away. "
            f"Your escalation reference is {ticket_id}. Please stay on the line — help is on the way.",
            language,
        )
        dbq.save_turn(session_id, "assistant", esc_msg,
                      {"escalation_ticket": ticket_id})
        dbq.log_event("escalation", session_id,
                      {"ticket_id": ticket_id, "message": message[:100]})

        return PipelineResult(
            guardrail=guardrail,
            blocked=False,
            block_reason="escalated_to_human",
            final_response=esc_msg,
            latency_ms=_elapsed(start_time),
            rag_context=rag_context_docs,
            session_id=session_id,
            language_detected=language,
            escalation_ticket_id=ticket_id,
        )

    # ── Agent 0.5: Sentiment Detection ───────────────────────────────────────
    sentiment_result = run_sentiment_agent(message, client, token_tracker=token_tracker)

    # ── Agent 2.5: Clarifier ──────────────────────────────────────────────────
    clarifier_result = run_clarifier(
        message, client,
        conversation_history=conversation_history,
        token_tracker=token_tracker,
    )

    if clarifier_result.needs_clarification and clarifier_result.clarification_question:
        dbq.save_turn(session_id, "user", message)
        q = _localize(clarifier_result.clarification_question, language)
        dbq.save_turn(session_id, "assistant", q, {"type": "clarification"})
        return PipelineResult(
            guardrail=guardrail,
            clarifier=clarifier_result,
            sentiment=sentiment_result,
            blocked=False,
            needs_clarification=True,
            clarification_question=q,
            final_response=q,
            latency_ms=_elapsed(start_time),
            rag_context=rag_context_docs,
            session_id=session_id,
            language_detected=language,
        )

    # ── Agentic Tool Use: tool-calling loop ───────────────────────────────────
    tool_results_raw = []
    tool_context_str = ""

    try:
        from api.llm_client import get_models
        tool_model = get_models().guardrail_model  # fast model for tool dispatch

        tool_messages = _build_tool_messages(
            message, rag_context_str, conversation_history
        )
        _, tool_results_raw = run_tool_loop(
            client, tool_model, tool_messages, session_id=session_id
        )

        if tool_results_raw:
            tool_context_str = "\n".join(
                f"[Tool: {r['tool']}] {r['result'].get('result', '')}"
                for r in tool_results_raw
            )
    except Exception:
        pass  # Tool loop failure should not block the pipeline

    # ── A/B variant selection ─────────────────────────────────────────────────
    from config import cfg as _cfg
    prompt_variant = _cfg.ab.variant if _cfg.ab.enabled else "A"

    # ── Agent 2: Inquiry Inference (claim parser) ─────────────────────────────
    # Build rich context: RAG + tool results + language + sentiment tone
    full_rag = _merge_context(
        rag_context_str, tool_context_str, language,
        sentiment_result.tone_instruction
    )

    claim_result = run_claim_parser(
        message, client,
        token_tracker=token_tracker,
        rag_context=full_rag,
        conversation_history=conversation_history,
        prompt_variant=prompt_variant,
    )

    # ── Agent 3: Compliance Review ────────────────────────────────────────────
    safety = run_safety_checks(
        claim_result.summary_response, client, token_tracker=token_tracker
    )

    # ── Post-Processor: follow-ups + checklist + settlement ───────────────────
    # Build set of questions already sent in this session to avoid repeats
    already_asked = {
        t["content"].strip().lower()
        for t in conversation_history
        if t["role"] == "user"
    }

    post_result = run_post_processor(
        intent=claim_result.intent.value,
        priority=claim_result.priority.value,
        loss_amount=claim_result.estimated_loss_amount,
        summary_response=claim_result.summary_response,
        client=client,
        token_tracker=token_tracker,
        already_asked=already_asked,
    )

    if not safety.compliance_pass:
        fallback_msg = _localize(
            "Thank you for contacting us about your claim. A claims representative "
            "will review your inquiry and reach out to you shortly. "
            "All claims are subject to policy terms, coverage verification, and investigation. "
            "This is not a determination of coverage.",
            language,
        )
        dbq.save_turn(session_id, "user", message)
        dbq.save_turn(session_id, "assistant", fallback_msg,
                      {"block_reason": "compliance_violation"})
        dbq.log_event("compliance_violation", session_id,
                      {"violations": safety.violations})
        return PipelineResult(
            guardrail=guardrail,
            claim_parser=claim_result,
            safety_check=safety,
            blocked=True,
            block_reason=f"compliance_violation: {', '.join(safety.violations)}",
            final_response=fallback_msg,
            latency_ms=_elapsed(start_time),
            rag_context=rag_context_docs,
            session_id=session_id,
            language_detected=language,
        )

    # ── Agent 4: Fraud Detection ──────────────────────────────────────────────
    session_claim_count = len(dbq.get_claims_by_session(session_id))

    fraud_result = run_fraud_detector(
        message=message,
        claim_data=claim_result.model_dump(),
        conversation_history=conversation_history,
        client=client,
        session_claims_count=session_claim_count,
        token_tracker=token_tracker,
    )

    # If claim was auto-filed via tools, extract the claim number
    claim_number = None
    for tr in tool_results_raw:
        if tr["tool"] == "file_claim" and "claim_number" in tr.get("result", {}):
            claim_number = tr["result"]["claim_number"]
            dbq.update_claim_fraud(
                claim_number,
                fraud_result.fraud_risk_score,
                fraud_result.fraud_flags,
            )
            # Fire mock notification / webhook
            dbq.create_notification(
                session_id=session_id,
                claim_number=claim_number,
                notification_type="claim_filed",
                message=f"Your claim {claim_number} has been filed successfully. "
                        f"Priority: {claim_result.priority.value}. "
                        f"We will assign an adjuster shortly.",
                webhook_payload={
                    "event": "claim_filed",
                    "claim_number": claim_number,
                    "intent": claim_result.intent.value,
                    "priority": claim_result.priority.value,
                    "loss_amount": claim_result.estimated_loss_amount,
                    "fraud_risk": fraud_result.fraud_risk_score,
                },
            )

    # Build tool call records
    tool_call_records = [
        ToolCallRecord(tool=r["tool"], args=r["args"], result=r["result"])
        for r in tool_results_raw
    ] if tool_results_raw else None

    # ── Save turn to memory ───────────────────────────────────────────────────
    dbq.save_turn(session_id, "user", message)
    dbq.save_turn(
        session_id, "assistant", claim_result.summary_response,
        {
            "intent": claim_result.intent.value,
            "priority": claim_result.priority.value,
            "loss_amount": claim_result.estimated_loss_amount,
            "claim_number": claim_number,
            "fraud_risk": fraud_result.fraud_risk_score,
        },
    )

    # ── Log A/B result ────────────────────────────────────────────────────────
    if _cfg.ab.enabled:
        dbq.log_ab_result(
            session_id=session_id,
            variant=prompt_variant,
            intent=claim_result.intent.value,
            priority=claim_result.priority.value,
            latency_ms=_elapsed(start_time),
            token_count=token_tracker.call_log[-1]["total_tokens"] if token_tracker and token_tracker.call_log else 0,
        )

    # ── Log analytics event ───────────────────────────────────────────────────
    dbq.log_event("message_processed", session_id, {
        "intent": claim_result.intent.value,
        "priority": claim_result.priority.value,
        "loss_amount": claim_result.estimated_loss_amount,
        "fraud_risk": fraud_result.fraud_risk_score,
        "tools_used": [r["tool"] for r in tool_results_raw],
        "language": language,
    })

    return PipelineResult(
        guardrail=guardrail,
        claim_parser=claim_result,
        safety_check=safety,
        fraud_check=fraud_result,
        clarifier=clarifier_result,
        sentiment=sentiment_result,
        post_processor=post_result,
        blocked=False,
        final_response=claim_result.summary_response,
        latency_ms=_elapsed(start_time),
        rag_context=rag_context_docs,
        session_id=session_id,
        language_detected=language,
        claim_number=claim_number,
        tool_calls=tool_call_records,
        needs_clarification=False,
        prompt_variant=prompt_variant,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _elapsed(start: float) -> float:
    return round((time.time() - start) * 1000, 2)


def _build_tool_messages(
    message: str,
    rag_context: str,
    history: list,
) -> list:
    """Build the message list for the tool-calling loop."""
    system = (
        "You are an insurance claims assistant. You have access to tools to look up policies, "
        "file claims, check claim status, schedule adjusters, and calculate deductibles. "
        "Use tools when the customer clearly wants to file a claim, check a status, "
        "look up their policy, or schedule an adjuster. "
        "Do not use tools for general coverage questions.\n"
    )
    if rag_context:
        system += f"\nContext:\n{rag_context}"

    messages = [{"role": "system", "content": system}]
    messages.extend(history[-6:])  # last 3 turns for context
    messages.append({"role": "user", "content": message})
    return messages


def _merge_context(rag: str, tools: str, language: str, tone: str = "") -> str:
    parts = []
    if rag:
        parts.append(rag)
    if tools:
        parts.append(f"\n=== TOOL RESULTS ===\n{tools}")
    if tone:
        parts.append(f"\n=== TONE INSTRUCTION ===\n{tone}")
    # Only inject language instruction for clearly non-English messages
    # Avoids LLM hallucinating a language from a misclassified short message
    _ENGLISH_VARIANTS = {"en", "en-us", "en-gb", "en-au"}
    if language and language.lower() not in _ENGLISH_VARIANTS:
        parts.append(
            f"\n=== LANGUAGE INSTRUCTION ===\n"
            f"The customer is writing in language '{language}'. "
            f"Respond ONLY in that same language. "
            f"Keep all insurance terms accurate but write naturally in {language}. "
            f"Do NOT mix languages in your response."
        )
    return "\n".join(parts)


def _localize(text: str, language: str) -> str:
    """For non-English, pass through LLM translation in future. For now return as-is."""
    return text


# ── Legacy support for eval harness ──────────────────────────────────────────

def run_pipeline_simple(
    message: str,
    client: Any = None,
    token_tracker=None,
    rag_context_str: str = "",
    rag_context_docs: list = None,
) -> PipelineResult:
    """
    Thin wrapper that calls run_pipeline without memory/tools.
    Used by eval/run_eval.py to keep test cases fast and deterministic.
    """
    return run_pipeline(
        message=message,
        client=client,
        token_tracker=token_tracker,
        rag_context_str=rag_context_str,
        rag_context_docs=rag_context_docs,
        session_id=f"eval_{uuid.uuid4().hex[:6]}",
    )


if __name__ == "__main__":
    import json
    import sys
    from colorama import init, Fore, Style
    init()

    print(f"\n{Fore.CYAN}{'=' * 60}")
    print("  Insurance AI - Claims Inquiry Concierge (Enhanced)")
    print(f"{'=' * 60}{Style.RESET_ALL}\n")

    test_messages = [
        "Hi, I was in a car accident yesterday. The other driver ran a red light and hit my car. Damage looks around $8,000.",
        "My house caught fire last night. We got out safely but the kitchen is destroyed. What do I do?",
        "Does my policy POL-AUTO-001 cover rental cars while mine is being repaired?",
    ]

    client = get_client()
    sid = f"demo_{uuid.uuid4().hex[:6]}"

    for msg in test_messages:
        print(f"{Fore.YELLOW}Message:{Style.RESET_ALL} {msg}\n")
        result = run_pipeline(msg, client, session_id=sid)

        print(f"{Fore.GREEN}Guardrail:{Style.RESET_ALL} {result.guardrail.model_dump()}")
        if result.needs_clarification:
            print(f"{Fore.BLUE}Clarification needed:{Style.RESET_ALL} {result.clarification_question}")
        if result.claim_parser:
            cp = result.claim_parser
            print(f"{Fore.GREEN}Intent:{Style.RESET_ALL} {cp.intent.value} | "
                  f"{Fore.GREEN}Priority:{Style.RESET_ALL} {cp.priority.value} | "
                  f"{Fore.GREEN}Loss:{Style.RESET_ALL} ${cp.estimated_loss_amount}")
        if result.fraud_check:
            print(f"{Fore.MAGENTA}Fraud Risk:{Style.RESET_ALL} {result.fraud_check.fraud_risk_score}")
        if result.tool_calls:
            for tc in result.tool_calls:
                print(f"{Fore.CYAN}Tool:{Style.RESET_ALL} {tc.tool} -> {tc.result.get('result', '')[:80]}")
        if result.claim_number:
            print(f"{Fore.GREEN}Claim Number:{Style.RESET_ALL} {result.claim_number}")
        print(f"{Fore.GREEN}Response:{Style.RESET_ALL} {result.final_response}")
        if result.blocked:
            print(f"{Fore.RED}BLOCKED:{Style.RESET_ALL} {result.block_reason}")
        print(f"{Fore.CYAN}Latency:{Style.RESET_ALL} {result.latency_ms}ms\n{'-' * 60}\n")
