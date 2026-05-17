# SHL Assessment Recommender — Approach

## Design Overview

Stateless FastAPI service. `GET /health` for readiness, `POST /chat` for agent interaction. Each `/chat` call receives full conversation history, returns reply, recommendations (0–10), and `end_of_conversation`. No per-conversation state.

Three-layer architecture:

1. **Retrieval (RAG)** — BM25 keyword search over 377 catalog items, pre-indexed at startup. Domain detection + anchor item injection boosts recall to 0.78@15. Top-15 candidates passed to LLM.
2. **LLM Decision** — Google Gemini 2.5 Flash receives 15 candidates + conversation history + behavioral rules. Returns structured JSON.
3. **Validation (groundedness)** — Post-LLM layer verifies each recommendation name against catalog. Hallucinated items dropped. URL and test_type mapped from catalog truth.

## Behaviors

All four required behaviors are implemented:

- **Clarify** — `_is_vague_first_turn()` counts signals (role, seniority, skills, language, context). When ≤2 signals on turn 1, returns one clarifying question with empty recommendations.
- **Recommend** — When enough context, LLM selects 1–10 items from the 15 retrieved candidates. Technical roles auto-add Verify G+ and OPQ32r.
- **Refine** — Prompt rule: "User changes constraints (add X, remove Y): update shortlist, don't restart." Tested in eval traces C4, C8, C9, C10.
- **Compare** — Prompt rule: "Contrast two items from catalog using descriptions, types, durations." Tested in eval traces C3, C5, C6, C7.

## Retrieval Setup

Custom BM25 with pre-computed inverted index. Documents indexed on: name, description, category keys, job levels, languages.

**Anchor injection**: OPQ32r, Verify G+, and other high-value assessments missed by BM25 (too generic for keyword match) are force-injected when domain keywords detected. 9 domain categories (leader, technical, graduate, contact, finance, sales, safety, healthcare, admin) each map to relevant anchor items.

**Measured retrieval recall@15**: 0.78 across 10 sample traces. Without anchors: 0.62.

Vector embeddings considered but rejected — catalog is small (377 items), keyword overlap is strong, and embedding API calls would add latency under the 30-second timeout.

## Prompt Design

System prompt ~300 tokens + ~1000 tokens for 15 candidates (pipe-delimited, descriptions at 60 chars). Format: `{id}|{name}|{type}|{keys}|{levels}|{desc}|{url}`.

Rules encode all four behaviors plus scope enforcement and legal refusal. Temperature 0.1, max_tokens 1500.

**What didn't work:**
- Full catalog injection (377 items, ~50K tokens) — context overflow, slow, budget exhaustion.
- Llama 8B models — failed on multi-turn reasoning, vague query detection, constraint refinement.
- `rank-bm25` library — had to implement custom BM25 for startup index build and scoring control.

**Token budget**: ~1.8K input + ~400 output per call.

## Evaluation

### Retrieval quality (recall@15)

`evals/test_retrieval.py` — 10 domain-specific queries, 100 expected items total. BM25 + anchor injection achieves **mean recall@15 = 0.777**.

### Recommendation relevance (recall@10)

`evals/replay.py` — 10 multi-turn conversation traces, 33 checkpoints. Each trace tested at multiple conversation depths. Best recall per trace used. Covers behavior probes: clarification on vague queries, off-topic refusal, comparison grounding, mid-conversation refinement.

### Groundedness

`_parse_response()` in `src/agent.py` — every LLM recommendation is validated against the catalog. Only items matching by name or URL are returned. Hallucinated items are silently dropped.

## Deployment

Render free tier. `render.yaml` configures build and start. API URL: https://shl-recommender-uxfj.onrender.com

## AI Tools Used

Claude (opencode) used for initial scaffolding, prompt iteration, and code generation. All design decisions, prompt rules, and evaluation methodology authored and verified by me.
