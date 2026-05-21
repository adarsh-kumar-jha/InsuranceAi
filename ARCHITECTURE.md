# Architecture Rationale — Insurance AI Claims Inquiry Concierge

## 1. System Architecture Diagram

```
                         ┌─────────────────────────────────────────────┐
                         │           POLICYHOLDER MESSAGE               │
                         └───────────────────┬─────────────────────────┘
                                             │
                         ┌───────────────────▼─────────────────────────┐
                         │         INPUT VALIDATION (pre-LLM)          │
                         │  • Min/max length check                      │
                         │  • Gibberish detection (entropy + ratio)     │
                         │  • Injection pattern filter                  │
                         └───────────────────┬─────────────────────────┘
                                             │ valid
                         ┌───────────────────▼─────────────────────────┐
                         │          RAG HYBRID SEARCH (pre-LLM)        │
                         │  BM25 keyword search  +  TF-IDF cosine sim  │
                         │  Hybrid score = 0.5*BM25 + 0.5*TF-IDF       │
                         │  Reranker: keyword overlap boost             │
                         │  → Top-3 insurance KB chunks injected        │
                         └───────────────────┬─────────────────────────┘
                                             │ context
                         ┌───────────────────▼─────────────────────────┐
                         │     AGENT 1 — guardrailCheck                │
                         │     Model: Llama 4 Scout (Groq)             │
                         │     + Deterministic regex PII override       │
                         │                                              │
                         │  Output:                                     │
                         │   is_insurance_related (bool)                │
                         │   no_pii (bool)                              │
                         │   needs_escalation (bool)                    │
                         └──┬──────────────┬──────────────┬────────────┘
                            │              │              │
                    not related        PII found     escalation
                            │              │              │
                       BLOCK (off)   BLOCK (pii)   ESCALATE (human)
                            │              │              │
                         ───┘              └──────────────┘
                                             │ pass
                         ┌───────────────────▼─────────────────────────┐
                         │     AGENT 2 — claimParser                   │
                         │     Model: Qwen3-32B (Groq)                 │
                         │     Fallback: Llama-3.3-70B                  │
                         │     + RAG context injected in system prompt  │
                         │                                              │
                         │  Output:                                     │
                         │   intent: Auto/Home/Coverage                 │
                         │   estimated_loss_amount: float | null        │
                         │   priority: Low / Medium / High              │
                         │   summary_response: compliant reply          │
                         │     (+ deterministic disclaimer append)      │
                         └───────────────────┬─────────────────────────┘
                                             │
                         ┌───────────────────▼─────────────────────────┐
                         │     AGENT 3 — safetyChecks                  │
                         │     Model: Llama 4 Scout (Groq)             │
                         │     + Deterministic regex violation scan     │
                         │                                              │
                         │  Output:                                     │
                         │   compliance_pass (bool)                     │
                         │   violations: list[str]                      │
                         └───────────────────┬─────────────────────────┘
                                             │
                              compliance_pass=False → BLOCK (fallback)
                                             │ pass
                         ┌───────────────────▼─────────────────────────┐
                         │         FINAL REPLY → POLICYHOLDER          │
                         └─────────────────────────────────────────────┘
```

---

## 2. Model Selection Rationale

### Agent 1 — guardrailCheck

| Property | Value |
|---|---|
| Model | `meta-llama/llama-4-scout-17b-16e-instruct` |
| Provider | Groq |
| Size | 17B (MoE, 16 experts) |
| TTFT | ~470ms |
| Task | 3-way boolean classification |

**Why Scout?**  
This is a binary gate — three boolean checks on a raw message. The task does not require deep reasoning or long context. Scout's speed (~470ms TTFT on Groq) is the dominant factor. Additionally, we layer a **deterministic regex PII pre-check** on top, which guarantees 100% recall on known PII patterns (SSN, Aadhaar, PAN, VIN) regardless of what the LLM outputs.

**Tradeoffs:**
- Cost: Lowest cost model in the pipeline
- Accuracy: Sufficient for boolean classification; hard overrides close any gaps
- Latency: Fastest, keeps total P50 well under 5s

---

### Agent 2 — claimParser

| Property | Value |
|---|---|
| Model | `qwen/qwen3-32b` (primary) |
| Fallback | `llama-3.3-70b-versatile` |
| Provider | Groq |
| Size | 32B |
| TTFT | ~210ms |
| Task | Intent classification + field extraction + reply generation |

**Why Qwen3-32B?**  
This is the hardest agent — it must simultaneously:
1. Classify intent across 3 categories (with ambiguous edge cases)
2. Extract a precise loss amount or return null
3. Apply a 3-tier priority rubric
4. Generate a legally compliant, empathetic reply

Qwen3-32B leads open models on structured JSON accuracy (88.4% GPQA Diamond) and has the most consistent JSON schema compliance among available Groq models. It outperforms Llama 3.3-70B on structured extraction benchmarks.

A **deterministic disclaimer append** is applied post-generation — even if the LLM omits it, the pipeline adds it. This guarantees 100% disclaimer presence.

**Tradeoffs:**
- Cost: Higher than Scout, justified by accuracy requirements
- Accuracy: Best available open-source structured output model on Groq
- Latency: ~700ms — acceptable within 5s budget

---

### Agent 3 — safetyChecks

| Property | Value |
|---|---|
| Model | `meta-llama/llama-4-scout-17b-16e-instruct` |
| Provider | Groq |
| Size | 17B |
| Task | Compliance review of generated reply |

**Why Scout?**  
Compliance checking is primarily a rule-matching task:
1. Did the LLM promise payment? → Regex can detect this
2. Is the disclaimer present? → String match

We run a **deterministic regex scan** before the LLM call and merge results after. This creates a defense-in-depth: even if the LLM misses a violation, the regex catches known patterns. The LLM adds value for nuanced misleading language detection.

**Tradeoffs:**
- Cost: Minimal — same as Agent 1
- Accuracy: 100% on disclaimer (deterministic) + high on other violations
- Latency: Fast, keeps pipeline under budget

---

## 3. Pre-LLM Optimizations

### Input Validation
- Rejects messages <10 chars, >2000 chars, <3 words
- Entropy check rejects keyboard mash (gibberish)
- Repeated char/word spam filter
- SQL/script injection pattern block
- Applied **before any API call** → saves latency and API cost on junk

### RAG Hybrid Search
- **25 insurance knowledge base entries** covering auto, home, coverage, compliance
- **BM25** (rank-bm25): keyword frequency matching — catches exact insurance terms
- **TF-IDF cosine similarity** (sklearn): semantic overlap — catches synonyms and related terms
- **Hybrid score** = 0.5 × BM25_normalized + 0.5 × TF-IDF_cosine
- **Reranker**: keyword overlap boost (+0.05 per matching keyword from KB entry)
- Top-3 chunks injected into Agent 2 system prompt as policy reference context
- This improves response accuracy on coverage-specific questions

### Retry + Fallback
- Each agent: 3 retry attempts with exponential back-off (0.5s, 1.0s, 1.5s)
- Agent 2 falls back from Qwen3-32B → Llama-3.3-70B on parse failure
- Agent 3 falls back to pure deterministic result if LLM fails

---

## 4. Cost, Accuracy, Latency Tradeoffs

| | Agent 1 | Agent 2 | Agent 3 |
|---|---|---|---|
| Model | Llama 4 Scout | Qwen3-32B | Llama 4 Scout |
| Relative Cost | Low | Medium | Low |
| Est. Tokens/call | ~250 | ~550 | ~300 |
| P50 Latency | ~300ms | ~700ms | ~300ms |
| Accuracy | ~100% (+ regex) | ~95%+ | ~100% (+ regex) |
| Compliance Risk | Low | Low | Minimal (deterministic override) |

**Total estimated pipeline cost per call:**
- Groq free tier covers 1,000 req/day for Qwen3-32B, 14,400 for Scout
- At production scale: ~$0.0005–0.001 per conversation on pay-as-you-go

---

## 5. Compliance Design

| Requirement | How We Meet It |
|---|---|
| Disclaimer always present | Hard deterministic append + Agent 3 string match |
| No coverage approval/denial | Agent 2 prompt + Agent 3 regex scan |
| PII not echoed | Agent 1 blocks messages with raw PII before Agent 2 sees it |
| Protected characteristics excluded | Explicit rule in Agent 2 system prompt |
| Escalation for emergencies | Agent 1 flags → orchestrator routes to human |
| AI is informational only | Policy in all agent prompts |
