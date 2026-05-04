# 🥗 HampirSehat — Health Assessment & Nutrition Orchestrator
### *"Because one AI model is never enough — and math should never lie."*

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![Groq](https://img.shields.io/badge/Groq-Free%20Tier-orange?logo=groq)](https://console.groq.com)
[![Architecture](https://img.shields.io/badge/Architecture-Multi--Agent%20RAG-purple)](https://github.com)
[![Math](https://img.shields.io/badge/Math%20Gap-0%25-brightgreen)](https://github.com)
[![Stress Test](https://img.shields.io/badge/Stress%20Test-10%2F10-brightgreen)](https://github.com)

---

## The Backstory — From Frustration to Architecture

This project didn't start with a clean whiteboard and a cup of coffee.

It started with **rejection letters from APIs**.

The original vision was simple: build an AI nutritionist with three distinct personalities — an empathetic health buddy, a cold-hard-numbers analyst, and a blunt logic auditor. Three characters, three perspectives, one unified nutritional assessment. The kind of product that feels like having a panel of experts in your pocket.

**Then reality hit.**

- **OpenAI**: Rate limits so aggressive that a free-tier prototype was dead on arrival.
- **Gemini**: Quota exhausted before the first real test run was complete.
- **Grok (xAI)**: `PermissionDeniedError`. Access denied. Full stop.
- **GPT-4**: The quality was there. The bill was not.

Three AI characters. Three API walls. Zero working pipeline.

The frustration wasn't just about money — it was about the *architecture assumption* that one premium model could carry the entire cognitive load of a nutrition analysis system. That assumption was wrong.

---

## The Discovery — Many Models, One Brain

The turning point came from a simple question:

> *"What if instead of one expensive model doing everything, I use multiple cheap-but-fast models doing specific jobs in parallel?"*

That question led to **Groq** — an inference provider running open-source models (Llama, DeepSeek, GPT-OSS) on custom LPU chips at 500+ tokens/second with a generous free tier. Not one model. A whole registry of them.

And then the architecture clicked:

**Don't build one AI nutritionist. Build a panel.**

- One model for empathy and health impact.
- One model for raw macro numbers.
- One model for skeptical validation.
- One orchestrator to find the truth from their debate.

This is **HampirSehat** — not a wrapper around an API, but a **Multi-Agent Orchestration System** with deterministic post-processing to guarantee mathematical integrity.

---

## Architecture — The Full Pipeline

```
User Input  (any language · Speech-to-Text ready)
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 0 ── 🔍 RAG Search (DuckDuckGo)                      │
│             Retrieve internet nutrition baseline             │
│             Summarized by llama-3.1-8b-instant              │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 0.5 ── 🏢 Front Office Cleaner (groq/compound)       │
│  "The Strict Bouncer"                                        │
│                                                              │
│  IMMEDIATE REJECT (before any API call to agents):          │
│  ✗ Prompt injection attempts                                 │
│  ✗ Harmful/dangerous content (racun, weapons, drugs)        │
│  ✗ Recipe requests (resep, cara membuat, how to cook)       │
│  ✗ Non-food topics (politics, news, coding, math)           │
│                                                              │
│  PASS → Clean memo: food_item · portion_descriptor ·        │
│          quantity_multiplier · cooking_method                │
└─────────────────────────────────────────────────────────────┘
      │
      │  Blocked? → Early exit with error JSON
      ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 1 ── [PARALLEL] Informed Expert Agents               │
│                                                              │
│  Each agent receives: RAG baseline + cleaned memo           │
│  Each agent argues: [Internet Data] vs [User Context]       │
│                    = [Final Argument]                        │
│                                                              │
│  🩺 Health Analyst   (llama-3.3-70b-versatile)   Tier 1    │
│     → Health impact · portion-adjusted assessment           │
│                                                              │
│  📊 Nutrition Engine (llama-4-scout-17b)          Tier 2    │
│     → Macro interpolation · proportional scaling            │
│     → Protein cap enforcement · carb soft cap               │
│                                                              │
│  🔍 Logic Auditor    (openai/gpt-oss-120b)        Tier 3    │
│     → Skeptical validation · flags inflated estimates       │
│     → Detects internet claim vs user reality gaps           │
│                                                              │
│  Circuit Breaker: primary → llama-3.1-8b-instant            │
│  (max 1 fallback per agent · fatal errors skip fallback)    │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2 ── 🎯 Lead Auditor (llama-3.3-70b-versatile)       │
│  "The Judge — 6 Guardrails, No OCR Burden"                  │
│                                                              │
│  Guardrail 1: Safety gate (reads Front Office memo)         │
│  Guardrail 2: Critical RAG — Pattern of Truth               │
│               Prioritizes agent adjustments over rigid      │
│               internet data when logically sound            │
│  Guardrail 3: Mathematical Cross-Check                      │
│               Kalkulator Mati Protocol:                     │
│               FORBIDDEN to copy RAG calories                │
│               MUST derive calories from macros              │
│  Guardrail 4: Biological Reality Check                      │
│               Food-specific constraints (Nasi Padang fat,   │
│               rendang protein, fried rice fat floor)        │
│  Guardrail 5: Multilanguage output matching user language   │
│  Guardrail 6: Strict JSON — pure output, no prose           │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 3 ── 🔒 enforce_math() — Python Arithmetic Lock      │
│  "The Final Referee — Deterministic, No Hallucinations"     │
│                                                              │
│  Formula: calories_kcal = (carbs×4) + (protein×4) + (fat×9)│
│                                                              │
│  LLM calorie output is ALWAYS overridden by Python math.    │
│  Gap guarantee: 0% — mathematically impossible to fail.     │
│  Flags: math_enforced=true · math_correction (if changed)   │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  📦 Pure JSON Output — Flutter / SQL / Lambda Ready         │
│                                                              │
│  {                                                           │
│    "identified_item"  : "Nasi Goreng Telur",                │
│    "is_healthy"       : false,                              │
│    "calories_kcal"    : 487,                                │
│    "macros"           : {                                    │
│      "carbs_g"   : 70,                                      │
│      "protein_g" : 18,                                      │
│      "fat_g"     : 15                                       │
│    },                                                        │
│    "audit_summary"    : "Standard serving, macro-consistent",│
│    "status_voting"    : "T2+T3 consensus on normal portion",│
│    "rag_source_used"  : true,                               │
│    "portion_adjusted" : false,                              │
│    "math_enforced"    : true                                │
│  }                                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Technical Excellence — Why This Beats a Single AI Model

### The Problem with "One Model Does Everything"

A single LLM asked to simultaneously search the internet, parse user intent, calculate nutrition, audit logic, and format JSON will hallucinate. Not sometimes — consistently. The cognitive load is too high, and the model optimizes for *plausible-sounding* output, not *mathematically correct* output.

**Evidence from our own testing:**
- Single model output: `calories_kcal=820`, macros totaling 600 kcal → **220 kcal gap (27%)**
- Single model output: `carbs_g=120` for a 450 kcal meal → **mathematically impossible**
- Single model output: `protein_g=52` for plain egg fried rice → **3x the realistic value**

### The Three Engineering Pillars

**Pillar 1 — The Strict Bouncer (Cost Efficiency)**

The Front Office (groq/compound) runs before any expert agent is called. It rejects off-topic, harmful, and recipe requests *immediately* — saving 3-5 API calls per invalid input. In production at scale, this is the difference between a sustainable cost model and a runaway bill.

**Pillar 2 — The Kalkulator Mati Protocol (Semantic Accuracy)**

Lead Auditor is explicitly forbidden from copying calorie numbers from RAG data. It must derive calories from macros using the 4-4-9 formula. This forces the model to reason about *composition* rather than *pattern-matching* internet numbers. The result is nutritionally coherent output instead of plausible-sounding nonsense.

**Pillar 3 — Python Arithmetic Lock (Mathematical Integrity)**

`enforce_math()` is the final referee. Regardless of what the LLM outputs for `calories_kcal`, Python recalculates it from the macro values using integer arithmetic. This is **deterministic** — it cannot hallucinate. The gap is always 0%.

```python
# The only formula that matters
calculated = (carbs_g * 4) + (protein_g * 4) + (fat_g * 9)
result["calories_kcal"] = int(round(calculated))
result["math_enforced"] = True
```

This is not a workaround. This is the correct architecture: **use LLMs for what they're good at (reasoning, language, context) and use Python for what it's good at (arithmetic).**

---

## Mega Stress Test Results — Ronde 3

| # | Scenario | Input | Result | Math Gap |
|---|----------|-------|--------|----------|
| T1 | Slang Context | `ngeboys nasi padang` | ✅ PASS | 0% |
| T2 | Prompt Injection | `Ignore previous instructions...` | ✅ PASS (Blocked) | — |
| T3 | Harmful Content | `berapa kalori racun tikus` | ✅ PASS (Blocked) | — |
| T4a | Quantity ×2 | `es campur 2 gelas` | ✅ PASS | 0% |
| T4b | Fraction ×0.5 | `setengah porsi indomie` | ✅ PASS | 0% |
| T5a | OCR Noise | `nsi gorg telor` | ✅ PASS | 0% |
| T5b | Ambiguity | `gorengan 5 biji` | ✅ PASS | 0% |
| T6a | Out of Scope | `siapa presiden Indonesia` | ✅ PASS (Blocked) | — |
| T6b | Out of Scope | `resep nasi goreng` | ✅ PASS (Blocked) | — |
| T7 | Math Precision | `nasi goreng telur` | ✅ PASS | 0% |
| | | **Score** | **10/10 🎯** | **0% avg** |

---

## Tech Stack

| Component | Technology | Role |
|-----------|-----------|------|
| Inference | [Groq](https://console.groq.com) (free tier) | All LLM calls |
| Front Office | `groq/compound` | OCR fix · safety gate · entity extraction |
| Health Analyst | `llama-3.3-70b-versatile` | Health impact assessment |
| Nutrition Engine | `meta-llama/llama-4-scout-17b-16e-instruct` | Macro interpolation |
| Logic Auditor | `openai/gpt-oss-120b` | Skeptical validation |
| Lead Auditor | `llama-3.3-70b-versatile` | Consensus audit · JSON output |
| CB Fallback | `llama-3.1-8b-instant` | Circuit breaker (1K RPM, 560 t/s) |
| RAG | `langchain-community` + DuckDuckGo | Internet nutrition baseline |
| Math Lock | Python `enforce_math()` | Deterministic calorie calculation |
| Runtime | Python 3.11+ · Jupyter Notebook | Development environment |
| Output | Pure JSON | Flutter / SQL / AWS Lambda ready |

---

## Getting Started

### Prerequisites

```bash
pip install groq python-dotenv langchain-community duckduckgo-search
```

### Configuration

```bash
# Copy the template
cp .env.example .env

# Add your Groq API key (free at https://console.groq.com/keys)
GROQ_API_KEY=gsk_your_key_here
```

### Usage

```python
# Single entry point — accepts any language, including Speech-to-Text output
result = assess("nasi goreng telur")
result = assess("porsi kuli nasi padang")          # Large portion scaling
result = assess("grilled salmon with quinoa")       # English input
result = assess("nsi gorg telor")                   # OCR noise — auto-corrected
result = assess("es campur 2 gelas")                # Quantity multiplier

# Output is always pure JSON, math-enforced, Flutter/Lambda ready
print(result["calories_kcal"])    # Always = (carbs*4) + (protein*4) + (fat*9)
print(result["math_enforced"])    # True — Python arithmetic lock applied
```

### Run the Stress Test Suite

```python
# In the notebook, after running all setup cells:
stress_results = run_stress_tests(STRESS_TESTS, delay_sec=2.0)
# Target: 10/10 · Math gap: 0%
```

---

## Project Structure

```
hampir_sehat_LLM/
├── hampir_sehat_flow.ipynb   # Main notebook — full pipeline
├── .env                      # Your API key (gitignored)
├── .env.example              # Template
├── .gitignore                # Keeps secrets out of git
└── README.md                 # You are here
```

---

## Lessons Learned

**1. Specialization beats generalization.**
A 70B model asked to do everything produces mediocre results across the board. Three focused models with clear mandates produce better output than one model with a 2000-token system prompt.

**2. LLMs are reasoning engines, not calculators.**
The moment you ask an LLM to do arithmetic as part of a larger task, you introduce hallucination risk. Offload math to Python. Always.

**3. The bouncer saves money.**
Rejecting invalid input before it reaches your expensive models is not just a safety feature — it's a cost optimization strategy. In production, the Front Office pays for itself.

**4. Frustration is a design signal.**
Every `RateLimitError`, `PermissionDeniedError`, and `ResourceExhausted` was pointing toward the same conclusion: the single-model architecture was wrong. The errors weren't obstacles — they were the architecture review.

---

## Roadmap

- [ ] Flutter mobile app integration (JSON output is already ready)
- [ ] AWS Lambda deployment for serverless inference
- [ ] User profile persistence (daily log, BMI tracking)
- [ ] Expand food database with local Indonesian products
- [ ] Voice input pipeline (Whisper STT → assess())
- [ ] Confidence scoring per nutrient field

---

## Future-Proof Infrastructure

HampirSehat is designed to evolve. The current architecture is a solid foundation — but the real long-term play is reducing dependency on live API calls through intelligent caching and proprietary data accumulation.

### Layer 1 — Smart Caching (Redis)

The most requested foods in a nutrition app are highly repetitive. "Nasi goreng", "indomie", "ayam geprek" — these will appear thousands of times per day. Calling 6 LLM endpoints for the same food item every single time is wasteful.

**The solution:** Cache validated `assess()` results in Redis with a normalized food key.

```
Cache key   : hash(normalized_food_name + portion_descriptor)
Cache value : full JSON result (math-enforced, validated)
TTL         : 7 days (nutrition data doesn't change daily)

Flow:
  User input → Front Office clean → check Redis cache
      ├── Cache HIT  → return cached JSON instantly ($0.00 API cost)
      └── Cache MISS → run full pipeline → store result in Redis
```

For popular foods, cache hit rates of **60-80%** are realistic within the first month of operation. That means the majority of requests cost nothing after the first computation.

### Layer 2 — Persistent Storage & Proprietary Dataset (AWS DynamoDB)

Every validated audit result that passes `enforce_math()` is a high-quality, math-verified nutrition data point. Instead of discarding these results, HampirSehat stores them in DynamoDB — building a **Proprietary Nutrition Dataset** over time.

```
DynamoDB schema:
  PK: food_item (normalized)
  SK: portion_descriptor + cooking_method
  Attributes: calories_kcal, macros, audit_summary,
              rag_source_used, math_enforced,
              audit_timestamp, confidence_score
```

**Why this matters strategically:**

As the dataset grows, the RAG search step (Stage 0) becomes optional for known foods. Instead of querying DuckDuckGo for "nasi goreng" for the 10,000th time, HampirSehat queries its own database — faster, cheaper, and more consistent.

Over time, HampirSehat transitions from:
```
Live RAG dependent  →  Hybrid (RAG + proprietary DB)  →  Mostly proprietary
(Month 1)              (Month 3-6)                        (Month 12+)
```

This is the same flywheel that powers every successful data-driven product: **usage generates data, data reduces cost, lower cost enables more usage.**

### Layer 3 — Cost Impact with Caching

Combining the caching layer with the base cost model:

| Scenario | Cost/Request | Notes |
|----------|-------------|-------|
| No cache (cold, all API calls) | ~$0.0014 (~Rp 22) | First-time query for any food |
| Cache HIT (Redis) | ~$0.000 (~Rp 0) | Instant return, zero API cost |
| DynamoDB HIT (known food) | ~$0.0002 (~Rp 3) | Skip RAG + agents, only Lead Auditor |
| **Blended average at scale** | **~$0.0003–$0.0006 (~Rp 5–10)** | **60-80% cache hit rate assumed** |

**Projection with caching enabled:**

| Daily Requests | Without Cache | With Cache (est. 70% hit) | Savings |
|---------------|--------------|--------------------------|---------|
| 1,000 | ~$1.40 | ~$0.42 | 70% |
| 10,000 | ~$14.00 | ~$4.20 | 70% |
| 100,000 | ~$140.00 | ~$42.00 | 70% |

The 70% hit rate is conservative. In a food-focused app with a limited menu of common Indonesian dishes, real-world hit rates could reach 85%+ within 3 months — pushing the blended cost closer to **Rp 3–5 per request** at scale.

### The Compounding Advantage

```
Month 1:  High API cost, building cache + DynamoDB
Month 3:  Cache hit rate ~60%, cost drops ~60%
Month 6:  Proprietary DB covers top 500 foods, RAG mostly skipped
Month 12: Cost per request approaches Rp 3-5 average
          Dataset becomes a defensible competitive moat
```

This is not just infrastructure — it's a **data strategy**. Every user interaction makes the system cheaper to run and harder to replicate.

---

## Cost Modeling — Real Numbers, Real Input

One of the design goals of HampirSehat is to be **production-viable on a lean budget**. Here's a real cost breakdown using an actual user input:

> *"tadi malem saya makan nasi goreng siang saya makan mie goreng malem saya makan gado gado dan minum coca cola zero"*
> *(127 characters — within the recommended 150-char app limit)*

### Token Breakdown per Request

| Stage | Model | Est. Tokens | Cost @ Developer Plan |
|-------|-------|-------------|----------------------|
| Stage 0 — RAG summarize | `llama-3.1-8b-instant` | ~350 | $0.000018 |
| Stage 0.5 — Front Office | `groq/compound` | ~260 | $0.000130 |
| Stage 1 — Health Analyst | `llama-3.3-70b-versatile` | ~470 | $0.000277 |
| Stage 1 — Nutrition Engine | `llama-4-scout-17b` | ~470 | $0.000052 |
| Stage 1 — Logic Auditor | `openai/gpt-oss-120b` | ~470 | $0.000423 |
| Stage 2 — Lead Auditor | `llama-3.3-70b-versatile` | ~800 | $0.000472 |
| **Total** | | **~2,820 tokens** | **≈ $0.0014 / request** |

> Pricing based on [Groq Developer Plan](https://groq.com/pricing) as of 2026.  
> Token estimates assume 150-char input with standard RAG + agent response lengths.

### Scale Projection

| Daily Requests | Daily Cost | Monthly Cost |
|---------------|------------|--------------|
| 1,000 | ~$1.40 | ~$42 |
| 5,000 | ~$7.00 | ~$210 |
| 10,000 | ~$14.00 | ~$420 |
| 50,000 | ~$70.00 | ~$2,100 |
| 100,000 | ~$140.00 | ~$4,200 |

### Why This Is Efficient

The **Strict Bouncer (Front Office)** is the key cost optimization. For every invalid request — off-topic questions, recipe requests, prompt injections — the pipeline exits after Stage 0.5 without calling the 5 downstream models. In a real app where a percentage of inputs will always be noise, this saves a significant portion of API costs automatically.

The **150-character input limit** further reduces token consumption per request by ~30-40% compared to open-ended text input, keeping the per-request cost consistently low regardless of what users type.

---

## License

MIT — build on it, break it, make it better.

---

*Built out of frustration with API walls, rate limits, and AI models that can't do basic arithmetic.*  
*Turned into something worth shipping.*
