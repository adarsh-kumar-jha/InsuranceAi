from pydantic import BaseModel, Field
from typing import Literal, Optional, Any
from enum import Enum


class GuardrailOutput(BaseModel):
    is_insurance_related: bool = Field(
        description="True if the message is about claims, coverage, or policy administration."
    )
    no_pii: bool = Field(
        description="True if the message contains no unmasked sensitive identifiers."
    )
    needs_escalation: bool = Field(
        description="True if the claimant is in severe distress or describing an active safety situation."
    )


class IntentType(str, Enum):
    AUTO_CLAIM = "Auto claim"
    HOME_CLAIM = "Home claim"
    COVERAGE_INQUIRY = "Coverage inquiry"


class PriorityLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class ClaimParserOutput(BaseModel):
    intent: IntentType = Field(
        description="Classified intent of the policyholder message."
    )
    estimated_loss_amount: Optional[float] = Field(
        default=None,
        description="Estimated dollar value of the loss. Null if not mentioned or not applicable."
    )
    priority: PriorityLevel = Field(
        description="Priority level: Low, Medium, or High."
    )
    summary_response: str = Field(
        description="A calm, professional reply to send to the policyholder. Must include the required disclaimer."
    )


class SafetyCheckOutput(BaseModel):
    compliance_pass: bool = Field(
        description="True if the reply passes all compliance checks."
    )
    violations: list[str] = Field(
        default_factory=list,
        description="List of compliance violations found. Empty if compliance_pass is True."
    )


class FraudCheckOutput(BaseModel):
    fraud_risk_score: str = Field(description="Low / Medium / High")
    fraud_flags: list[str] = Field(default_factory=list)
    recommended_action: str = Field(description="auto_approve / manual_review / flag_for_investigation")
    fraud_justification: str = Field(default="")


class ClarifierOutput(BaseModel):
    needs_clarification: bool
    clarification_question: Optional[str] = None
    confidence: str = "high"


class SentimentOutput(BaseModel):
    sentiment: str = "calm"            # calm / frustrated / distressed
    tone_instruction: str = ""
    urgency_boost: bool = False


class PostProcessorOutput(BaseModel):
    follow_up_questions: list[str] = Field(default_factory=list)
    evidence_checklist: list[str] = Field(default_factory=list)
    settlement_range: Optional[dict] = None   # {min, max, note} or None


class ToolCallRecord(BaseModel):
    tool: str
    args: dict
    result: dict


class PipelineResult(BaseModel):
    guardrail: GuardrailOutput
    claim_parser: Optional[ClaimParserOutput] = None
    safety_check: Optional[SafetyCheckOutput] = None
    fraud_check: Optional[FraudCheckOutput] = None
    clarifier: Optional[ClarifierOutput] = None
    sentiment: Optional[SentimentOutput] = None
    post_processor: Optional[PostProcessorOutput] = None
    final_response: Optional[str] = None
    blocked: bool = False
    block_reason: Optional[str] = None
    latency_ms: Optional[float] = None
    token_usage: Optional[dict] = None
    rag_context: Optional[list[dict]] = None
    # Agentic extensions
    claim_number: Optional[str] = None
    tool_calls: Optional[list[ToolCallRecord]] = None
    session_id: Optional[str] = None
    language_detected: Optional[str] = None
    escalation_ticket_id: Optional[str] = None
    # Clarification flow
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    # A/B variant tracking
    prompt_variant: Optional[str] = None
