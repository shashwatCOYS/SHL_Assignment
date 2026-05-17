import re
import math
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict

catalog_cache: List[Dict[str, Any]] | None = None
_index: Dict[str, Any] | None = None


def get_catalog():
    global catalog_cache
    if catalog_cache is None:
        from src.catalog import get_catalog as _load
        catalog_cache = _load()
    return catalog_cache


def tokenize(text: str) -> List[str]:
    return re.findall(r'\b[a-z0-9]{2,}\b', text.lower())


def _make_text(item: Dict[str, Any]) -> str:
    parts = [
        item.get("name", ""),
        item.get("description", ""),
        " ".join(item.get("keys", [])),
        " ".join(item.get("job_levels", [])),
        " ".join(item.get("languages", [])),
    ]
    return " ".join(parts)


def build_index() -> Dict[str, Any]:
    """Build pre-computed BM25 index at startup."""
    global _index
    if _index is not None:
        return _index

    catalog = get_catalog()
    N = len(catalog)

    # Pre-tokenize all documents
    doc_token_lists: List[List[str]] = []
    doc_lengths: List[int] = []
    for item in catalog:
        tokens = tokenize(_make_text(item))
        doc_token_lists.append(tokens)
        doc_lengths.append(len(tokens))

    avgdl = sum(doc_lengths) / N

    # Inverted index: token -> set of doc ids
    inverted: Dict[str, set] = defaultdict(set)
    for doc_id, tokens in enumerate(doc_token_lists):
        for token in tokens:
            inverted[token].add(doc_id)

    _index = {
        "N": N,
        "avgdl": avgdl,
        "doc_lengths": doc_lengths,
        "doc_token_lists": doc_token_lists,
        "inverted": inverted,
        "catalog": catalog,
    }
    return _index


def search_index(query: str, top_k: int = 30) -> List[Dict[str, Any]]:
    """Fast BM25 search using pre-computed index."""
    idx = build_index()
    catalog = idx["catalog"]
    N = idx["N"]
    avgdl = idx["avgdl"]
    doc_lengths = idx["doc_lengths"]
    inverted = idx["inverted"]

    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    query_counter = Counter(query_tokens)

    # Pre-compute IDF for all query tokens
    idf: Dict[str, float] = {}
    for token in query_counter:
        df = len(inverted.get(token, set()))
        idf[token] = math.log((N - df + 0.5) / (df + 0.5) + 1)

    # Score documents
    scored: List[Tuple[float, int]] = []
    for doc_id, tokens in enumerate(idx["doc_token_lists"]):
        doc_counter = Counter(tokens)
        dl = doc_lengths[doc_id]

        score = 0.0
        for token in query_counter:
            freq = doc_counter.get(token, 0)
            if freq == 0:
                continue
            k1, b = 1.5, 0.75
            numerator = freq * (k1 + 1)
            denominator = freq + k1 * (1 - b + b * dl / avgdl)
            score += idf[token] * numerator / denominator

        if score > 0:
            scored.append((score, doc_id))

    scored.sort(key=lambda x: -x[0])
    return [catalog[doc_id] for _, doc_id in scored[:top_k]]


ANCHOR_BOOST = {
    "Occupational Personality Questionnaire OPQ32r": 2.5,
    "SHL Verify Interactive G+": 2.0,
    "SHL Verify Interactive - Numerical Reasoning": 1.5,
    "Graduate Scenarios": 1.5,
    "Smart Interview Live Coding": 1.8,
    "Global Skills Assessment": 1.5,
    "Dependability and Safety Instrument (DSI)": 1.5,
}

ROLE_KEYWORDS = {
    "leader": ["director", "executive", "cxo", "senior leadership", "management", "lead"],
    "technical": ["engineer", "developer", "programmer", "coding", "software", "devops", "architecture", "java", "python", "rust", "aws", "docker", "spring"],
    "graduate": ["graduate", "trainee", "entry-level", "final-year", "fresh graduate", "recent graduate"],
    "contact": ["contact centre", "call centre", "customer service", "sva", "spoken"],
    "finance": ["financial", "accounting", "accountant", "finance", "statistics", "numerical"],
    "sales": ["sales", "selling", "revenue", "account executive", "bdr"],
    "safety": ["safety", "plant", "chemical", "manufacturing", "industrial", "operator", "dependability"],
    "healthcare": ["healthcare", "medical", "hipaa", "patient", "hospital", "nursing"],
    "admin": ["admin", "office", "assistant", "excel", "word", "powerpoint"],
}

DOMAIN_ANCHORS = {
    "leader": ["Occupational Personality Questionnaire OPQ32r"],
    "technical": ["Occupational Personality Questionnaire OPQ32r", "SHL Verify Interactive G+"],
    "graduate": ["Occupational Personality Questionnaire OPQ32r"],
    "contact": [],
    "finance": ["Occupational Personality Questionnaire OPQ32r", "Graduate Scenarios"],
    "sales": ["Occupational Personality Questionnaire OPQ32r", "Global Skills Assessment"],
    "safety": ["Dependability and Safety Instrument (DSI)"],
    "healthcare": ["Occupational Personality Questionnaire OPQ32r", "Dependability and Safety Instrument (DSI)"],
    "admin": ["Occupational Personality Questionnaire OPQ32r"],
}


def _detect_domains(query: str) -> set:
    q = query.lower()
    found = set()
    for domain, keywords in ROLE_KEYWORDS.items():
        if any(k in q for k in keywords):
            found.add(domain)
    return found


def retrieve_candidates(query: str, top_k: int = 30) -> List[Dict[str, Any]]:
    raw = search_index(query, top_k=40)
    domains = _detect_domains(query)
    catalog = get_catalog()

    name_map = {item["name"]: item for item in catalog}
    existing_names = {item["name"] for item in raw}

    for domain in domains:
        for anchor_name in DOMAIN_ANCHORS.get(domain, []):
            if anchor_name not in existing_names and anchor_name in name_map:
                raw.insert(0, name_map[anchor_name])
                existing_names.add(anchor_name)

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for item in raw:
        s = 1.0
        name = item.get("name", "")
        s *= ANCHOR_BOOST.get(name, 1.0)
        scored.append((s, item))

    scored.sort(key=lambda x: -x[0])
    return [item for _, item in scored[:top_k]]
