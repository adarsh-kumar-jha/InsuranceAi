"""
Insurance AI — FastAPI Backend (Enhanced)

Endpoints:
  POST /api/chat              → Full pipeline (memory + tools + fraud)
  GET  /api/claim/{number}    → Get claim status
  GET  /api/claims            → List recent claims
  GET  /api/session/{id}      → Conversation history for a session
  POST /api/upload            → Upload document (PDF / image)
  GET  /api/analytics         → Dashboard metrics
  GET  /api/escalations       → Open human handoff tickets
  GET  /api/policies          → List available mock policies
  GET  /api/stats             → Token usage stats
  POST /api/stats/reset       → Reset token tracker
  GET  /api/health            → Health check
"""

import sys
import os
import uuid
import time
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel, field_validator

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from orchestrator import run_pipeline, get_client
from api.token_tracker import token_tracker
from api.input_validator import validate_input, sanitize_input
from rag.hybrid_search import rag_index
from db import database as dbq
from db.mock_data import init_db_with_seed

# Initialise DB
init_db_with_seed()

# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Insurance AI Concierge",
    description="Multi-agent insurance claims inquiry system — agentic, memory-aware",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Shared LLM client
_client = None


def get_llm_client():
    global _client
    if _client is None:
        _client = get_client()
    return _client


# ─── Language detection ───────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    """
    Detect language only for messages long enough to be reliable.
    langdetect is unreliable on < 30 characters — defaults to 'en'.
    Also ignores detection if the result is a low-confidence edge case.
    """
    # Too short to classify reliably — assume English
    if len(text.strip()) < 30 or len(text.split()) < 6:
        return "en"
    try:
        from langdetect import detect, detect_langs
        langs = detect_langs(text)
        # Only override English if top language is non-English with high confidence
        top = langs[0]
        if top.lang != "en" and top.prob >= 0.90:
            return top.lang
        return "en"
    except Exception:
        return "en"


# ─── PDF / Image extraction ───────────────────────────────────────────────────

def extract_text_from_upload(content: bytes, filename: str) -> str:
    """Extract text from PDF or return placeholder for images."""
    fname = filename.lower()
    if fname.endswith(".pdf"):
        try:
            import pdfplumber
            import io
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
            return "\n".join(pages).strip()
        except ImportError:
            return "[PDF uploaded — pdfplumber not installed. Please describe the document in the chat.]"
        except Exception as e:
            return f"[PDF extraction failed: {str(e)[:100]}]"
    elif any(fname.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".gif"]):
        return f"[Image uploaded: {filename}. Please describe the damage in your message for processing.]"
    else:
        try:
            return content.decode("utf-8", errors="ignore")[:2000]
        except Exception:
            return "[File uploaded — could not extract text automatically.]"


# ─── Request / Response Models ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    document_context: Optional[str] = None  # pre-extracted text from upload

    @field_validator("message")
    @classmethod
    def message_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Message cannot be empty")
        return v


class ChatResponse(BaseModel):
    message_id: str
    final_response: str
    blocked: bool
    block_reason: Optional[str]
    intent: Optional[str]
    priority: Optional[str]
    estimated_loss_amount: Optional[float]
    needs_escalation: bool
    needs_clarification: bool
    clarification_question: Optional[str]
    compliance_pass: Optional[bool]
    violations: Optional[list[str]]
    guardrail_details: dict
    claim_parser_details: Optional[dict]
    safety_details: Optional[dict]
    fraud_details: Optional[dict]
    sentiment_details: Optional[dict]
    tool_calls: Optional[list[dict]]
    claim_number: Optional[str]
    escalation_ticket_id: Optional[str]
    session_id: str
    language_detected: Optional[str]
    token_usage: dict
    rag_context: list[dict]
    latency_ms: float
    timestamp: str
    # New fields
    follow_up_questions: list[str]
    evidence_checklist: list[str]
    settlement_range: Optional[dict]
    prompt_variant: Optional[str]


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_ui():
    return FileResponse(str(static_dir / "index.html"))


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main pipeline endpoint with:
    - Multi-turn conversation memory (per session_id)
    - Agentic tool use (file_claim, lookup_policy, etc.)
    - Agent 2.5 (Clarifier)
    - Agent 4 (Fraud Detector)
    - Language detection
    """
    raw_message = request.message
    is_valid, error_msg = validate_input(raw_message)
    if not is_valid:
        raise HTTPException(status_code=422, detail=error_msg)

    message = sanitize_input(raw_message)
    session_id = request.session_id or f"sess_{uuid.uuid4().hex[:10]}"

    # Language detection
    language = detect_language(message)

    # If document context was provided, prepend it
    if request.document_context:
        doc_preview = request.document_context[:500]
        message = f"[Uploaded document excerpt: {doc_preview}]\n\n{message}"

    # RAG hybrid search
    rag_results = rag_index.search(message, top_k=3)
    rag_context_str = rag_index.format_context(rag_results)

    # Run full pipeline
    try:
        client = get_llm_client()
        result = run_pipeline(
            message=message,
            client=client,
            token_tracker=token_tracker,
            rag_context_str=rag_context_str,
            rag_context_docs=rag_results,
            session_id=session_id,
            language=language,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"Pipeline error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

    guardrail_dict = result.guardrail.model_dump()
    claim_parser_dict = result.claim_parser.model_dump() if result.claim_parser else None
    safety_dict = result.safety_check.model_dump() if result.safety_check else None
    fraud_dict = result.fraud_check.model_dump() if result.fraud_check else None
    sentiment_dict = result.sentiment.model_dump() if result.sentiment else None
    tool_calls_list = (
        [tc.model_dump() for tc in result.tool_calls] if result.tool_calls else None
    )
    post = result.post_processor

    # Serialize intent/priority from enum
    if claim_parser_dict:
        claim_parser_dict["intent"] = (
            claim_parser_dict["intent"].value
            if hasattr(claim_parser_dict["intent"], "value")
            else claim_parser_dict["intent"]
        )
        claim_parser_dict["priority"] = (
            claim_parser_dict["priority"].value
            if hasattr(claim_parser_dict["priority"], "value")
            else claim_parser_dict["priority"]
        )

    recent = token_tracker.call_log[-5:] if token_tracker.call_log else []
    call_tokens = {
        "prompt_tokens": sum(c["prompt_tokens"] for c in recent),
        "completion_tokens": sum(c["completion_tokens"] for c in recent),
        "total_tokens": sum(c["total_tokens"] for c in recent),
        "by_agent": {c["agent"]: c["total_tokens"] for c in recent},
    }

    return ChatResponse(
        message_id=str(uuid.uuid4()),
        final_response=result.final_response or "",
        blocked=result.blocked,
        block_reason=result.block_reason,
        intent=result.claim_parser.intent.value if result.claim_parser else None,
        priority=result.claim_parser.priority.value if result.claim_parser else None,
        estimated_loss_amount=result.claim_parser.estimated_loss_amount if result.claim_parser else None,
        needs_escalation=result.guardrail.needs_escalation,
        needs_clarification=result.needs_clarification,
        clarification_question=result.clarification_question,
        compliance_pass=result.safety_check.compliance_pass if result.safety_check else None,
        violations=result.safety_check.violations if result.safety_check else None,
        guardrail_details=guardrail_dict,
        claim_parser_details=claim_parser_dict,
        safety_details=safety_dict,
        fraud_details=fraud_dict,
        sentiment_details=sentiment_dict,
        tool_calls=tool_calls_list,
        claim_number=result.claim_number,
        escalation_ticket_id=result.escalation_ticket_id,
        session_id=session_id,
        language_detected=language,
        token_usage=call_tokens,
        rag_context=result.rag_context or [],
        latency_ms=result.latency_ms or 0,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        follow_up_questions=post.follow_up_questions if post else [],
        evidence_checklist=post.evidence_checklist if post else [],
        settlement_range=post.settlement_range if post else None,
        prompt_variant=result.prompt_variant,
    )


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...), session_id: str = Form(default="")):
    """
    Upload a PDF (police report, repair estimate) or image (damage photo).
    Returns extracted text to be included in the next chat message.
    """
    max_size = 10 * 1024 * 1024  # 10 MB
    content = await file.read()
    if len(content) > max_size:
        raise HTTPException(status_code=413, detail="File too large. Maximum 10 MB.")

    extracted = extract_text_from_upload(content, file.filename or "upload")

    dbq.log_event("document_upload", session_id or None, {
        "filename": file.filename,
        "size_bytes": len(content),
        "extracted_chars": len(extracted),
    })

    return {
        "filename": file.filename,
        "size_bytes": len(content),
        "extracted_text": extracted,
        "preview": extracted[:300],
        "message": "Document processed. The extracted text will be included in your next message.",
    }


@app.get("/api/claim/{claim_number}")
async def get_claim(claim_number: str):
    """Look up a claim by number."""
    claim = dbq.get_claim(claim_number)
    if not claim:
        raise HTTPException(status_code=404, detail=f"Claim {claim_number} not found.")
    return claim


@app.get("/api/claims")
async def list_claims(limit: int = 20):
    """List most recent claims (for the analytics dashboard)."""
    return dbq.get_recent_claims(limit=limit)


@app.get("/api/session/{session_id}/history")
async def get_session_history(session_id: str):
    """Return full conversation history for a session."""
    history = dbq.get_full_history(session_id)
    return {"session_id": session_id, "turns": history, "count": len(history)}


@app.get("/api/policies")
async def list_policies():
    """List all mock policies (for UI display / testing tool use)."""
    return dbq.list_policies()


@app.get("/api/escalations")
async def get_escalations():
    """Return all open human handoff tickets."""
    tickets = dbq.get_open_escalations()
    return {"open_count": len(tickets), "tickets": tickets}


@app.get("/api/analytics")
async def get_analytics():
    """Return aggregated analytics data for the dashboard."""
    summary = dbq.get_analytics_summary()
    recent = dbq.get_recent_claims(limit=50)
    return {
        "summary": summary,
        "recent_claims": recent,
        "token_stats": token_tracker.get_stats(),
    }


@app.get("/api/notifications")
async def get_notifications(session_id: str = "", unread_only: bool = False):
    """Get notifications (claim filed, status changes, adjuster assigned)."""
    notifs = dbq.get_notifications(session_id or None, unread_only=unread_only)
    return {"count": len(notifs), "notifications": notifs}


@app.post("/api/notifications/read")
async def mark_read(session_id: str):
    """Mark all notifications as read for a session."""
    dbq.mark_notifications_read(session_id)
    return {"status": "ok"}


@app.get("/api/ab-results")
async def get_ab_results():
    """A/B prompt testing comparison dashboard."""
    return dbq.get_ab_summary()


@app.get("/api/session/{session_id}/export", response_class=HTMLResponse)
async def export_session(session_id: str):
    """
    Export the full conversation + claim details as a printable HTML page.
    In the browser: Ctrl+P → Save as PDF.
    """
    history = dbq.get_full_history(session_id)
    claims = dbq.get_claims_by_session(session_id)
    notifs = dbq.get_notifications(session_id)

    def msg_html(turn):
        role_label = "Customer" if turn["role"] == "user" else "AI Agent"
        bg = "#1e293b" if turn["role"] == "user" else "#0f172a"
        border = "#3b82f6" if turn["role"] == "user" else "#6366f1"
        meta = json.loads(turn.get("metadata") or "{}")
        meta_html = ""
        if meta.get("intent"):
            meta_html += f'<span style="background:#1e3a5f;color:#93c5fd;padding:2px 8px;border-radius:12px;font-size:11px;margin-right:4px">{meta["intent"]}</span>'
        if meta.get("priority"):
            colors = {"High": "#7f1d1d:#fca5a5", "Medium": "#78350f:#fcd34d", "Low": "#064e3b:#6ee7b7"}
            c = colors.get(meta["priority"], "#1f2937:#d1d5db").split(":")
            meta_html += f'<span style="background:{c[0]};color:{c[1]};padding:2px 8px;border-radius:12px;font-size:11px;margin-right:4px">⚡ {meta["priority"]}</span>'
        if meta.get("claim_number"):
            meta_html += f'<span style="background:#064e3b;color:#6ee7b7;padding:2px 8px;border-radius:12px;font-size:11px">🎫 {meta["claim_number"]}</span>'
        return f"""
        <div style="background:{bg};border-left:3px solid {border};padding:12px 16px;margin:8px 0;border-radius:6px">
          <div style="color:#94a3b8;font-size:11px;margin-bottom:4px">{role_label} · {turn.get("created_at","")[:16].replace("T"," ")}</div>
          <div style="color:#e2e8f0;line-height:1.6;white-space:pre-wrap">{turn["content"]}</div>
          {"<div style='margin-top:6px'>"+meta_html+"</div>" if meta_html else ""}
        </div>"""

    claims_html = ""
    for c in claims:
        claims_html += f"""
        <div style="background:#0f172a;border:1px solid #1e293b;padding:12px;border-radius:8px;margin:6px 0">
          <div style="color:#a5b4fc;font-weight:600">{c["claim_number"]}</div>
          <div style="color:#94a3b8;font-size:12px">{c["intent"]} · {c["priority"]} priority · ${c["loss_amount"] or 0:,.0f} · {c["status"]}</div>
          <div style="color:#64748b;font-size:11px">{c.get("description","")[:100]}</div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8"/>
  <title>Claim Session Export — {session_id}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #020617; color: #e2e8f0; margin: 0; padding: 24px; }}
    @media print {{ body {{ background: white; color: black; }} }}
    h1 {{ color: #a5b4fc; font-size: 20px; margin-bottom: 4px; }}
    h2 {{ color: #6366f1; font-size: 14px; margin: 20px 0 8px; text-transform: uppercase; letter-spacing: 1px; }}
    .meta {{ color: #64748b; font-size: 12px; margin-bottom: 20px; }}
    .print-btn {{ background: #6366f1; color: white; border: none; padding: 8px 20px; border-radius: 8px; cursor: pointer; font-size: 13px; margin-bottom: 20px; }}
    @media print {{ .print-btn {{ display: none; }} }}
  </style>
</head>
<body>
  <button class="print-btn" onclick="window.print()">🖨 Print / Save as PDF</button>
  <h1>🛡 InsureAI — Session Export</h1>
  <div class="meta">Session ID: {session_id} &nbsp;·&nbsp; Exported: {time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}</div>
  {"<h2>Claims Filed</h2>" + claims_html if claims else "<p style='color:#64748b'>No claims filed in this session.</p>"}
  <h2>Conversation ({len(history)} messages)</h2>
  {"".join(msg_html(t) for t in history if t["role"] in ("user","assistant"))}
  <div style="color:#334155;font-size:11px;margin-top:24px;border-top:1px solid #1e293b;padding-top:12px">
    All claims are subject to policy terms, coverage verification, and investigation. This is not a determination of coverage.
  </div>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.get("/api/stats")
async def get_stats():
    return token_tracker.get_stats()


@app.post("/api/stats/reset")
async def reset_stats():
    token_tracker.reset()
    return {"status": "reset", "message": "Token stats have been reset."}


@app.get("/api/health")
async def health():
    from config import cfg
    try:
        _ = cfg.llm.api_key
        return {
            "status": "healthy",
            "version": "2.0.0",
            "provider": cfg.llm.provider,
            "model_guardrail": cfg.llm.guardrail_model,
            "model_parser": cfg.llm.parser_model,
            "model_safety": cfg.llm.safety_model,
            "rag_enabled": cfg.rag.enabled,
            "rag_kb_size": len(rag_index.docs),
            "token_tracker_calls": token_tracker.total_calls,
            "features": [
                "multi_turn_memory",
                "tool_use",
                "clarifier_agent",
                "fraud_detection",
                "document_upload",
                "voice_input",
                "language_detection",
                "analytics_dashboard",
                "human_handoff",
            ],
        }
    except EnvironmentError as e:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "reason": str(e)})


if __name__ == "__main__":
    import uvicorn
    from config import cfg as _cfg
    uvicorn.run(
        "api.main:app",
        host=_cfg.server.host,
        port=_cfg.server.port,
        reload=_cfg.server.reload,
        log_level=_cfg.server.log_level,
    )
