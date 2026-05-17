import json
import os
from typing import List, Dict, Any
from groq import Groq
import httpx

from src.catalog import get_catalog
from src.retriever import retrieve_candidates, build_index

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL = "llama-3.3-70b-versatile"

_client: Groq | None = None


def get_groq_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(
            api_key=GROQ_API_KEY,
            timeout=httpx.Timeout(25.0, connect=5.0),
            max_retries=1,
        )
    return _client


def init():
    build_index()


def _build_system_prompt(candidates: List[Dict[str, Any]]) -> str:
    lines = []
    for i, item in enumerate(candidates, 1):
        desc = item.get("description", "")[:50].replace("\n", " ").replace("|", "")
        levels = ",".join(item.get("job_levels", [])[:2])
        keys = ",".join(item.get("keys", []))
        url = item["url"]
        lines.append(f"{i}|{item['name']}|{item['test_type']}|{keys}|{levels}|{desc}|{url}")
    catalog = "\n".join(lines)

    return f"""SHL Assessment Recommender. Pick assessments from catalog.

CATALOG (id|name|type|keys|levels|desc|url):
{catalog}

RULES:
1) ONLY recommend items from catalog above. No invented names/URLs.
2) Vague query: ask ONE clarifying question. No recommendations.
3) Enough context: recommend 1-10 items.
4) Constraint change ("add X","remove Y"): update, don't restart.
5) Compare: contrast items from catalog.
6) Off-topic/legal: refuse.
7) Missing skill: suggest closest tests, note gap.
8) Technical IC: add Verify G+ and OPQ32r unless told otherwise.
9) Max 10 recommendations.
10) end_of_conversation: true when user confirms or turn 7+.

OUTPUT: JSON only.
{{"reply":"text","recommendations":[],"end_of_conversation":false}}
recommendations: array of {{name,url,test_type}}. name must match catalog."""


def _extract_query(messages: List[Dict[str, str]]) -> str:
    return " ".join(m["content"] for m in messages)


def call_agent(messages: List[Dict[str, str]]) -> Dict[str, Any]:
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

    prompt = f"""Hist:
{hist}

Last: {last}
Turns: {turn_count // 2}
JSON only, no markdown."""

    import time
    client = get_groq_client()
    last_error = None
    for attempt in range(10):
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
            break
        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            if "rate_limit" in err_str or "429" in err_str or "rate limit" in err_str:
                if "try again in" in err_str:
                    import re
                    match = re.search(r"(\d+)m(\d+)", err_str)
                    if match:
                        wait = int(match.group(1)) * 60 + int(match.group(2))
                    else:
                        wait = min(300, 30 * (attempt + 1))
                else:
                    wait = min(300, 30 * (attempt + 1))
                print(f"Rate limit. Waiting {wait}s (attempt {attempt+1})", flush=True)
                time.sleep(wait)
            else:
                raise
    else:
        return {
            "reply": f"I apologize, I'm experiencing high demand. Please try again shortly.",
            "recommendations": [],
            "end_of_conversation": False,
        }

    raw = response.choices[0].message.content
    result = _parse_response(raw, messages)
    return result


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
            "reply": "I apologize, I had a processing error. Could you rephrase your request?",
            "recommendations": [],
            "end_of_conversation": False,
        }

    recommendations = data.get("recommendations", [])
    if not isinstance(recommendations, list):
        recommendations = []

    catalog = get_catalog()
    name_url_map = {item["name"]: item for item in catalog}

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
        elif rec.get("url"):
            for item in catalog:
                if item["url"] == rec["url"]:
                    validated.append({
                        "name": item["name"],
                        "url": item["url"],
                        "test_type": item["test_type"],
                    })
                    break

    end_conv = data.get("end_of_conversation", False)
    turn_count = len(messages)
    if turn_count >= 14:
        end_conv = True

    return {
        "reply": data.get("reply", ""),
        "recommendations": validated,
        "end_of_conversation": bool(end_conv),
    }
