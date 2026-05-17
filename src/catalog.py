import json
import os
from typing import List, Dict, Any

CATALOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "catalog.json")

_catalog: List[Dict[str, Any]] | None = None
_name_index: Dict[str, Dict[str, Any]] = {}


def load_catalog() -> List[Dict[str, Any]]:
    global _catalog, _name_index
    with open(CATALOG_PATH, "r") as f:
        _catalog = json.load(f)
    for item in _catalog:
        normalized = item["name"].lower()
        _name_index[normalized] = item
    return _catalog


def get_catalog() -> List[Dict[str, Any]]:
    if _catalog is None:
        load_catalog()
    return _catalog  # type: ignore


def find_by_name(name: str) -> Dict[str, Any] | None:
    return _name_index.get(name.lower())


def format_item_for_prompt(item: Dict[str, Any]) -> str:
    return (
        f"- {item['name']} | Type: {item['test_type']} | "
        f"Duration: {item['duration']} | Levels: {', '.join(item['job_levels'][:3])} | "
        f"Keys: {', '.join(item['keys'])} | "
        f"URL: {item['url']}"
    )
