import json
import os
import time
import re
from typing import List, Dict, Any
from google import genai

from src.catalog import get_catalog
from src.retriever import retrieve_candidates, build_index

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL = "gemini-2.5-flash"

_client: genai.Client | None = None


def get_gemini_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
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
    if any(w in t for w in ["hiring", "screening", "assessment", "battery", "test", "solution", "hire", "fill", "role"]):
        signals += 1
    if len(t.split()) > 15:
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

    client = get_gemini_client()
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config={
                    "system_instruction": system_prompt,
                    "temperature": 0.1,
                    "max_output_tokens": 1500,
                    "response_mime_type": "application/json",
                },
            )
            raw = response.text
            result = _parse_response(raw, messages)
            return result
        except Exception as e:
            err_str = str(e).lower()
            if "rate" in err_str or "429" in err_str:
                wait = min(15, 3 * (attempt + 1))
                print(f"Rate limit (attempt {attempt+1}). Waiting {wait}s", flush=True)
                time.sleep(wait)
            elif "503" in err_str or "unavailable" in err_str:
                time.sleep(5)
            else:
                print(f"Unexpected error: {e}", flush=True)
                return {
                    "reply": "I apologize, I had a processing error. Please try again.",
                    "recommendations": [],
                    "end_of_conversation": False,
                }
    return {
        "reply": "I apologize, service is busy. Please try again shortly.",
        "recommendations": [],
        "end_of_conversation": False,
    }


def _parse_response(raw: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw[:-3]

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "reply": "I apologize, I had a processing error. Could you rephrase?",
            "recommendations": [],
            "end_of_conversation": False,
        }

    recommendations = data.get("recommendations", [])
    if not isinstance(recommendations, list):
        recommendations = []

    catalog = get_catalog()
    name_url_map = {item["name"]: item for item in catalog}
    url_map = {item["url"]: item for item in catalog}

    validated: List[Dict[str, str]] = []
    for rec in recommendations[:10]:
        if not isinstance(rec, dict):
            continue
        name = rec.get("name", "")
        if name in name_url_map:
            item = name_url_map[name]
            validated.append({
                "name": item["name"],
                "url": item["url"],
                "test_type": item["test_type"],
            })
        elif rec.get("url") and rec["url"] in url_map:
            item = url_map[rec["url"]]
            validated.append({
                "name": item["name"],
                "url": item["url"],
                "test_type": item["test_type"],
            })

    end_conv = data.get("end_of_conversation", False)
    if len(messages) >= 14:
        end_conv = True

    return {
        "reply": data.get("reply", ""),
        "recommendations": validated,
        "end_of_conversation": bool(end_conv),
    }
