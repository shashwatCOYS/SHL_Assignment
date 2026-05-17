# SHL Assessment Recommender — Approach

## Design Overview

Stateless FastAPI service. `GET /health` for readiness, `POST /chat` for agent interaction. Each `/chat` call receives full conversation history, returns reply, recommendations (0-10), and `end_of_conversation`. No per-conversation state.

Three-layer architecture:

1. **Retrieval** — BM25 keyword search over 377 catalog items, pre-indexed at startup. Domain detection + anchor item injection boosts recall to 0.78@15. Top-15 candidates passed to LLM.
2. **LLM Decision** — Groq Llama 3.3 70B receives 15 candidates + conversation history + behavioral rules. Returns structured JSON via `response_format: json_object`.
3. **Validation** — Post-LLM layer verifies each recommendation name against catalog. Hallucinated items dropped. URL and test_type mapped from catalog truth.

## Retrieval Setup

Custom BM25 with pre-computed inverted index. Documents indexed on: name, description, category keys, job levels, languages.

**Anchor injection**: OPQ32r, Verify G+, and other high-value assessments missed by BM25 (too generic for keyword match) are force-injected when domain keywords detected. 9 domain categories (leader, technical, graduate, contact, finance, sales, safety, healthcare, admin) each map to relevant anchor items.

**Measured retrieval recall@15**: 0.78 across 10 sample traces. Without anchors: 0.62.

Vector embeddings considered but rejected — catalog is small (377 items), keyword overlap is strong, and embedding API calls would add latency under the 30-second timeout.

## Prompt Design

System prompt ~300 tokens + ~1000 tokens for 15 candidates (pipe-delimited, descriptions at 50 chars). Format: `{id}|{name}|{type}|{keys}|{levels}|{desc}|{url}`.

Rules encode all four behaviors (clarify, recommend, refine, compare) plus scope enforcement. Temperature 0.1, max_tokens 1500.

**What didn't work:**
- Full catalog injection (377 items, ~50K tokens) — context overflow, slow, budget exhaustion.
- Llama 8B models — failed on multi-turn reasoning, vague query detection, constraint refinement. Always asked clarifying questions, never committed to recommendations.
- `rank-bm25` library — had to implement custom BM25 for startup index build and scoring control.

**Token budget**: ~1.8K input + ~400 output per call. 33 eval turns × ~2.2K = ~73K, fits under Groq 100K daily free tier.

## Evaluation

Retrieval eval: 10 traces, BM25 + anchor injection, mean recall@15 = 0.78.

LLM eval harness (evals/replay.py): 10 traces, 33 total turn-checkpoints. Each trace tested at multiple conversation depths. Best recall per trace used. Behavior probes: clarification on vague queries, off-topic refusal, comparison grounding, mid-conversation refinement.

## Deployment

Render free tier. `render.yaml` configures build and start. Cold start ~2 minutes (within evaluator allowance).

## AI Tools Used

Claude (opencode) used for initial scaffolding, prompt iteration, and code generation. All design decisions, prompt rules, and evaluation methodology authored and verified by me.
