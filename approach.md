# SHL Assessment Recommender — Approach

## Design Overview

The system is a stateless FastAPI service with two endpoints: `GET /health` and `POST /chat`. Each `/chat` call receives full conversation history and returns the next reply, optional recommendations (1-10), and an `end_of_conversation` flag. No per-conversation state is stored.

The architecture has three layers:

1. **Retrieval** — BM25 keyword search over 377 catalog items (name, description, keys, job levels, languages) pre-indexed at startup. Top-15 candidates extracted per query.
2. **LLM Decision** — Groq Llama 3.3 70B receives the 15 candidates + conversation history + rules. Returns structured JSON with reply, recommendations, and end-of-conversation signal.
3. **Validation** — Post-LLM layer verifies every recommended name exists in the catalog, maps to correct URL and test_type. Hallucinated items are silently dropped.

## Retrieval Setup

BM25 with pre-computed inverted index at startup (O(N) build, O(query_tokens × matching_docs) per query). Each catalog item is indexed on: name, description, category keys, job levels, and supported languages. This covers the main signals: skill keywords ("Java", "sales"), role levels ("director", "graduate"), and domain terms ("contact centre", "safety").

Vector embeddings were considered but BM25 proved sufficient: the catalog is small (377 items), domain-specific, and keyword overlap is a strong signal. Vector similarity would add an embedding API call per request, increasing latency under the 30-second timeout.

## Prompt Design

The system prompt is ~500 tokens + ~1000 tokens for 15 candidates (pipe-delimited format, descriptions truncated to 80 chars). Rules cover all four conversational behaviors plus scope enforcement.

Format: `{id}.|{name}|{type}|{duration}|{keys}|{levels}|{desc}|{url}` — dense, low token count, LLM parses reliably.

JSON output enforced via `response_format: json_object`. The validation layer provides a safety net: any name not in the catalog is rejected by exact name match or URL match.

**What didn't work:**
- Full catalog injection (377 items, ~50K tokens) — context overflow, slow response, budget exhaustion.
- 8B models (Llama 3.1 8B) — failed on multi-turn reasoning, vague query detection, and constraint refinement.
- rank-bm25 library — had to implement custom BM25 for control over index building and scoring.

**Measured improvement:** Pre-filtering to 15 candidates reduced per-call token usage from ~50K to ~1.5K, enabling all 33 eval turns to complete within the 100K daily Groq free-tier budget.

## Evaluation

Local replay harness simulates all 10 sample conversation traces. For each trace, multiple turn checkpoints are sent (simulating different conversational paths), and Recall@10 is computed against the expected shortlist. The best recall across all turns per trace is used.

Behavior probes tested: vague query clarification (no recommendations on turn 1), off-topic refusal, comparison grounded in catalog data, constraint refinement (add/remove mid-conversation).

## Deployment

Render free tier (Python web service). Cold start up to 2 minutes (within evaluator's allowance). `render.yaml` configures build, start command, and environment variables.
