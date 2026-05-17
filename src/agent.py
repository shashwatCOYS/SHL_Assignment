import json
import os
import time
import urllib.request
import urllib.error
from typing import List, Dict, Any

from src.catalog import get_catalog
from src.retriever import retrieve_candidates, build_index

# Provider detection (check all common env vars)
LLM_PROVIDER_RAW = os.environ.get("LLM_PROVIDER", "auto")  # "openai", "gemini", "groq", "auto"
LLM_PROVIDER = LLM_PROVIDER_RAW
API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GROQ_API_KEY", "")
MODEL = os.environ.get("LLM_MODEL", "auto")
BASE_URL = os.environ.get("LLM_BASE_URL", "")  # For OpenAI-compatible providers (OpenRouter, etc.)

# Resolve provider
if LLM_PROVIDER == "auto":
    if os.environ.get("GEMINI_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        LLM_PROVIDER = "gemini"
    elif os.environ.get("GROQ_API_KEY"):
        LLM_PROVIDER = "groq"
    elif os.environ.get("OPENAI_API_KEY"):
        LLM_PROVIDER = "openai"
    elif os.environ.get("LLM_API_KEY"):
        # Default to gemini when only LLM_API_KEY is set (e.g. on Render)
        LLM_PROVIDER = "gemini"
    else:
        LLM_PROVIDER = "openai"  # last resort

# Resolve model
if MODEL == "auto":
    if LLM_PROVIDER == "gemini":
        MODEL = "gemini-2.5-flash"
    elif LLM_PROVIDER == "groq":
        MODEL = "llama-3.3-70b-versatile"
    else:
        MODEL = "gpt-4o-mini"

print(f"[agent] Provider: {LLM_PROVIDER}, Model: {MODEL}, Key set: {bool(API_KEY)}", flush=True)

_client = None


def get_client():
    global _client
    if _client is not None:
        return _client

    if LLM_PROVIDER == "gemini":
        _client = None  # Direct HTTP, no SDK
    else:
        # OpenAI-compatible (openai, groq, openrouter, etc.)
        from openai import OpenAI
        if LLM_PROVIDER == "groq":
            _client = OpenAI(api_key=API_KEY, base_url="https://api.groq.com/openai/v1")
        elif BASE_URL:
            _client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        else:
            _client = OpenAI(api_key=API_KEY)
    return _client


def init():
    build_index()


def _build_system_prompt(candidates: List[Dict[str, Any]]) -> str:
    lines = []
    for i, item in enumerate(candidates, 1):
        desc = item.get("description", "")[:60].replace("\n", " ").replace("|", "")
        levels = ",".join(item.get("job_levels", [])[:2])
        keys = ",".join(item.get("keys", []))
        url = item["url"]
        lines.append(f"{i}|{item['name']}|{item['test_type']}|{keys}|{levels}|{desc}|{url}")
    catalog = "\n".join(lines)

    return f"""SHL Assessment Recommender. Pick assessments from catalog.

CATALOG (id|name|type|keys|levels|desc|url):
{catalog}

BEHAVIOR RULES (CRITICAL):
1) VAGUE FIRST TURN → CLARIFY, NEVER RECOMMEND. If this is turn 1 and the user gives only 1-2 signals, you MUST ask ONE clarifying question. Set recommendations to []. Examples of vague queries:
   - "We need a solution for senior leadership" → ask "What decision type? (selection, development, benchmarking)"
   - "I need an assessment" → ask "What role and seniority level?"
   - "We need a screening solution" → ask "What role, seniority, and language?"
   - "I am hiring for X" → ask about seniority, specific skills, or test types needed
   A query needs at least 3 specific signals (role, seniority, skills/test-types, language, volume) to recommend.

2) ENOUGH CONTEXT → RECOMMEND. When user provides 3+ signals, return 1-10 items.
3) REFINE → User changes constraints ("add X","remove Y","replace Z"): update shortlist, don't restart.
4) COMPARE → Contrast two items from catalog using descriptions, types, durations. No recommendations.
5) REFUSE → Off-topic, legal, general hiring: refuse. You only cover SHL assessments.
6) MISSING SKILL → Suggest closest tests, note gap in catalog.
7) TECHNICAL IC → Add Verify G+ and OPQ32r unless told otherwise.
8) MAX 10 recommendations per response.
9) end_of_conversation: true when user confirms or turn 7+.

OUTPUT: Valid JSON object only. No markdown wrapping, no code blocks.
{{
  "reply": "your natural language response",
  "recommendations": [],
  "end_of_conversation": false
}}
recommendations: array of objects with exact "name", "url", "test_type" from catalog."""


def _count_user_signals(text: str) -> int:
    signals = 0
    t = text.lower()
    if any(w in t for w in ["java", "python", "rust", "aws", "sql", "spring", "react", "docker", "finance", "excel", "word", "hipaa", "medical", "contact", "call", "admin", "engineering", "developer", "operator", "sales", "nursing", "accounting", "statistics"]):
        signals += 1
    if any(w in t for w in ["senior", "junior", "entry", "graduate", "mid", "lead", "manager", "director", "cxo", "executive", "trainee", "fresh", "final-year"]):
        signals += 1
    if any(w in t for w in ["numerical", "cognitive", "personality", "situational", "knowledge", "simulation", "spoken", "verbal", "reasoning", "behavior", "aptitude", "safety"]):
        signals += 1
    if any(w in t for w in ["english", "spanish", "french", "german", "chinese", "hindi", "japanese", "indian", "australian"]):
        signals += 1
    if any(w in t for w in ["hiring", "screening", "assessment", "assess", "battery", "test", "solution", "hire", "fill", "role", "evaluating", "interview", "candidate", "recruit"]):
        signals += 1
    if len(t.split()) > 10:
        signals += 1
    return signals


def _is_vague_first_turn(messages: List[Dict[str, str]]) -> bool:
    if len(messages) != 1:
        return False
    if messages[0]["role"] != "user":
        return False
    text = messages[0]["content"]
    if "difference" in text.lower() or "compare" in text.lower() or "between" in text.lower():
        return False
    if "legal" in text.lower() or "required" in text.lower() or "law" in text.lower():
        return False
    return _count_user_signals(text) <= 2


def _clarify_reply(text: str) -> Dict[str, Any]:
    t = text.lower()
    if any(w in t for w in ["leader", "cxo", "director", "executive", "management"]):
        reply = "What is the primary decision type — selection (comparing candidates), development (growth planning), or benchmarking against a standard?"
    elif any(w in t for w in ["hire", "hiring", "role", "fill", "position", "solution", "assessment"]):
        reply = "What role and seniority level are you hiring for? Any specific skills or test types you need?"
    else:
        reply = "What role are you assessing, and what seniority level?"
    return {"reply": reply, "recommendations": [], "end_of_conversation": False}


def _extract_query(messages: List[Dict[str, str]]) -> str:
    return " ".join(m["content"] for m in messages)


def call_agent(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    if _is_vague_first_turn(messages):
        return _clarify_reply(messages[0]["content"])

    query = _extract_query(messages)
    candidates = retrieve_candidates(query, top_k=15)
    system_prompt = _build_system_prompt(candidates)

    turn_count = len(messages)
    user_msgs = []
    for m in messages:
        role = "U" if m["role"] == "user" else "A"
        user_msgs.append(f"{role}: {m['content']}")

    last = messages[-1]["content"] if messages else ""
    hist = "\n".join(user_msgs[:-1])

    prompt = f"""Conversation history:
{hist}

Latest message: {last}
Total turns: {turn_count // 2}

Respond with valid JSON only. No markdown, no code blocks, no explanation."""

    client = get_client()

    if LLM_PROVIDER == "gemini":
        return _call_gemini(client, system_prompt, prompt)
    else:
        return _call_openai_compat(client, system_prompt, prompt)


def _call_gemini(client, system_prompt: str, prompt: str) -> Dict[str, Any]:
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"
        body = json.dumps({
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1500, "responseMimeType": "application/json"},
        }).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
        raw = result["candidates"][0]["content"]["parts"][0]["text"]
        return _parse_response(raw)
    except urllib.error.HTTPError as e:
        print(f"Gemini HTTP error: {e.code} {e.reason}", flush=True)
        return _error_response("service is busy")
    except Exception as e:
        print(f"Gemini error: {e}", flush=True)
        return _error_response("service is busy")


def _call_openai_compat(client, system_prompt: str, prompt: str) -> Dict[str, Any]:
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1500,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            return _parse_response(raw)
        except Exception as e:
            err_str = str(e).lower()
            if "rate" in err_str or "429" in err_str:
                wait = min(10, 3 * (attempt + 1))
                print(f"Rate limit (attempt {attempt+1}). Waiting {wait}s", flush=True)
                time.sleep(wait)
            else:
                print(f"LLM error: {e}", flush=True)
                break
    return _error_response("service is busy")


def _error_response(msg: str) -> Dict[str, Any]:
    return {
        "reply": f"I apologize, {msg}. Please try again shortly.",
        "recommendations": [],
        "end_of_conversation": False,
    }


def _parse_response(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw[:-3]

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return _error_response("I had a processing error. Could you rephrase?")

    recommendations = data.get("recommendations", [])
    if not isinstance(recommendations, list):
        recommendations = []

    catalog = get_catalog()
    name_url_map = {item["name"]: item for item in catalog}
    url_map = {item["url"]: item for item in catalog}

    validated: List[Dict[str, str]] = []
    seen_names = set()
    for rec in recommendations[:10]:
        if not isinstance(rec, dict):
            continue
        name = rec.get("name", "")
        if name and name not in seen_names:
            if name in name_url_map:
                item = name_url_map[name]
                validated.append({
                    "name": item["name"],
                    "url": item["url"],
                    "test_type": item["test_type"],
                })
                seen_names.add(name)
            elif rec.get("url") and rec["url"] in url_map:
                item = url_map[rec["url"]]
                if item["name"] not in seen_names:
                    validated.append({
                        "name": item["name"],
                        "url": item["url"],
                        "test_type": item["test_type"],
                    })
                    seen_names.add(item["name"])

    end_conv = data.get("end_of_conversation", False)

    return {
        "reply": data.get("reply", ""),
        "recommendations": validated,
        "end_of_conversation": bool(end_conv),
    }
