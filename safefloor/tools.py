"""Tools the agent can call, with their OpenAI function-calling schemas."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable

from .store import Store

SEVERITY_ORDER = ["low", "medium", "high", "critical"]

SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_zones",
            "description": "List the warehouse zones, their cameras and their safety rules.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_observations",
            "description": "Search camera observations. Timestamps are ISO 8601 UTC, e.g. 2026-10-29T07:00:00Z.",
            "parameters": {
                "type": "object",
                "properties": {
                    "zone_id": {"type": "string", "description": "Zone id, e.g. dock-a"},
                    "since": {"type": "string", "description": "Earliest timestamp, inclusive"},
                    "until": {"type": "string", "description": "Latest timestamp, inclusive"},
                    "query": {"type": "string", "description": "Keyword to match in the summary or objects"},
                    "only_violations": {"type": "boolean", "description": "Only observations with a rule violation"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_violations",
            "description": "Count rule violations grouped by zone and rule, most frequent first.",
            "parameters": {
                "type": "object",
                "properties": {"zone_id": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_incident_report",
            "description": "File an incident report backed by one or more observation ids. Only cite ids returned by search_observations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "observation_ids": {"type": "array", "items": {"type": "integer"}},
                    "body": {"type": "string", "description": "What happened, where, when, risk, recommended action"},
                },
                "required": ["title", "observation_ids", "body"],
            },
        },
    },
]


def make_tools(store: Store) -> dict[str, Callable[..., Any]]:
    def list_zones() -> dict[str, Any]:
        return {"zones": [vars(z) for z in store.zones.values()]}

    def search_observations(zone_id=None, since=None, until=None, query=None, only_violations=False):
        return {"observations": store.observations(zone_id, since, until, query, bool(only_violations))}

    def count_violations(zone_id=None):
        return {"counts": store.violation_counts(zone_id)}

    def file_incident_report(title, observation_ids, body):
        found = [store.get_observation(int(i)) for i in observation_ids]
        found = [o for o in found if o]
        if not found:
            return {"error": "none of those observation ids exist; search first"}
        severities = [v["severity"] for o in found for v in o["violations"]] or ["low"]
        severity = max(severities, key=SEVERITY_ORDER.index)
        created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        ids = [o["id"] for o in found]
        incident_id = store.add_incident(created, title, severity, ids, body)
        return {"incident_id": incident_id, "severity": severity, "observation_ids": ids}

    return {
        "list_zones": list_zones,
        "search_observations": search_observations,
        "count_violations": count_violations,
        "file_incident_report": file_incident_report,
    }


def run_tool(tools: dict[str, Callable[..., Any]], name: str, arguments: str) -> str:
    if name not in tools:
        return json.dumps({"error": f"unknown tool {name}"})
    try:
        args = json.loads(arguments or "{}")
        return json.dumps(tools[name](**args), default=str)
    except Exception as exc:  # report tool errors back to the model instead of crashing the loop
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
