# Insurance AI — Claims Inquiry Concierge

A multi-agent AI system that handles the first turn of a policyholder conversation. Classifies insurance inquiries, returns structured output, and enforces compliance and privacy checks — all powered by **Llama 4** on **Groq**.

---

## Architecture

```
Policyholder Message
        │
        ▼
┌─────────────────────────────────┐
│  Agent 1 — guardrailCheck       │  model: llama-4-scout (Groq)
│  • PII detection                │
│  • Insurance relevance check    │
│  • Escalation detection         │
└──────────┬──────────────────────┘
           │ pass
           ▼
┌─────────────────────────────────┐
│  Agent 2 — claimParser          │  model: llama-4-maverick (Groq)
│  • Intent classification        │
│  • Loss amount extraction       │
│  • Priority assignment          │
│  • Compliant reply generation   │
└──────────┬──────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│  Agent 3 — safetyChecks         │  model: llama-4-scout (Groq)
│  • Coverage guarantee check     │
│  • Misleading language check    │
│  • Disclaimer presence check    │
└──────────┬──────────────────────┘
           │ compliance_pass = true
           ▼
   Final Reply → Policyholder
```

### Model Selection Rationale

| Agent | Model | Why |
|---|---|---|
| guardrailCheck | `llama-4-scout` | Fast (~470ms TTFT). Boolean classification only — speed and reliability matter most at the intake gate. |
| claimParser | `llama-4-maverick` | Highest accuracy Llama 4 variant (400B MoE). Handles the hardest task: intent + extraction + reply generation. Falls back to Scout on failure. |
| safetyChecks | `llama-4-scout` | Compliance is a rule-matching task. Scout is fast enough; also uses deterministic string matching for 100% disclaimer detection. |

---

## Setup

### 1. Get a free Groq API Key
Go to [https://console.groq.com](https://console.groq.com) → Sign up → Create API key (free).

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
copy .env.example .env
# Edit .env and paste your GROQ_API_KEY
```

### 4. Run the demo

```bash
python orchestrator.py
```

---

## Running the Evaluation

### Full evaluation (all 20 cases)

```bash
python eval/run_eval.py
```

### Single test case

```bash
python eval/run_eval.py --case TC-010
```

### By category

```bash
python eval/run_eval.py --category pii_injection
python eval/run_eval.py --category jailbreak
python eval/run_eval.py --category compliance
```

### As pytest suite

```bash
pytest eval/run_eval.py -v
```

---

## Evaluation Targets

| Category | Metric | Target |
|---|---|---|
| Guardrails | Boolean accuracy (PII + relevance) | 100% |
| Inference | Intent classification | ≥90% |
| Inference | Field extraction | ≥85% |
| Safety | Compliance pass rate | 100% |
| Safety | Disclaimer presence | 100% |
| Performance | P50 latency (end-to-end) | <5 seconds |

---

## Test Dataset (20 cases)

| Category | Count | Description |
|---|---|---|
| Normal claims | 6 | Auto, home, and coverage inquiries |
| Ambiguous | 3 | Edge cases between intents |
| PII injection | 3 | Messages with exposed sensitive identifiers |
| Jailbreak | 3 | Attempts to bypass compliance rules |
| Compliance | 3 | Escalation and disclaimer scenarios |
| Off-topic | 1 | Completely unrelated messages |
| Large loss | 1 | High-priority home claim |

---

## Project Structure

```
InsuranceAi/
├── agents/
│   ├── guardrail_check.py    # Agent 1 — Intake Guardrails
│   ├── claim_parser.py       # Agent 2 — Inquiry Inference
│   └── safety_checks.py      # Agent 3 — Compliance
├── models/
│   └── schemas.py            # Pydantic output schemas
├── eval/
│   ├── test_cases.json       # 20 evaluation cases
│   └── run_eval.py           # Evaluation harness (CLI + pytest)
├── orchestrator.py           # Main pipeline
├── requirements.txt
├── .env.example
└── README.md
```

---

## Compliance Rules

- The AI **never** approves, denies, or settles claims
- PII is **never** echoed back to the user
- Every response includes the required disclaimer
- Protected characteristics are never used in reasoning
- Escalation routes to a human agent immediately
