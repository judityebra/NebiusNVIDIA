"""Tools the agent can call, with their OpenAI function-calling schemas."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable

from .regulations import RegulationLookup
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
            "description": "Search camera observations. Timestamps are ISO 8601 UTC, e.g. 2026-10-29T07:00:00Z. "
            "Returns at most the 50 newest matches; if truncated is true, narrow the query (zone, since/until, only_violations) and search again.",
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
            "name": "lookup_regulation",
            "description": "Look up the OSHA regulation behind a zone rule (live web search restricted to osha.gov). "
            "Call it for the violated rule before filing an incident report, and cite only what it returns.",
            "parameters": {
                "type": "object",
                "properties": {"rule_id": {"type": "string", "description": "Rule id, e.g. forklift-gap"}},
                "required": ["rule_id"],
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
                    "regulation_refs": {
                        "type": "array",
                        "description": "Regulation links to cite. Only urls returned by lookup_regulation in this conversation are accepted.",
                        "items": {"type": "object", "properties": {"title": {"type": "string"}, "url": {"type": "string"}}, "required": ["url"]},
                    },
                },
                "required": ["title", "observation_ids", "body"],
            },
        },
    },
]


class RunState:
    """What the tools retrieved during one ask(): regulation urls the agent may cite."""

    def __init__(self) -> None:
        self.regulations: dict[str, str] = {}  # url -> title


def make_tools(
    store: Store, regulations: RegulationLookup | None = None, run: RunState | None = None
) -> dict[str, Callable[..., Any]]:
    regulations = regulations or RegulationLookup()
    run = run or RunState()

    def list_zones() -> dict[str, Any]:
        return {"zones": [vars(z) for z in store.zones.values()]}

    def search_observations(zone_id=None, since=None, until=None, query=None, only_violations=False):
        flag = bool(only_violations)
        rows = store.observations(zone_id, since, until, query, flag)
        total = store.count_observations(zone_id, since, until, query, flag)
        return {"observations": rows, "truncated": total > len(rows), "total_matching": total}

    def count_violations(zone_id=None):
        return {"counts": store.violation_counts(zone_id)}

    def lookup_regulation(rule_id):
        rule = next((r for z in store.zones.values() for r in z.rules if r["id"] == rule_id), None)
        if rule is None:
            return {"error": f"unknown rule {rule_id!r}", "results": []}
        result = regulations.lookup(rule)
        for r in result.get("results", []):
            run.regulations[r["url"]] = r["title"]
        return result

    def file_incident_report(title, observation_ids, body, regulation_refs=None):
        refs = []
        for ref in regulation_refs or []:
            url = ref.get("url") if isinstance(ref, dict) else None
            if url not in run.regulations:
                return {"error": f"regulation url {url!r} was not returned by lookup_regulation in this conversation; call it first"}
            refs.append({"title": run.regulations[url], "url": url})
        found = [store.get_observation(int(i)) for i in observation_ids]
        found = [o for o in found if o]
        if not found:
            return {"error": "none of those observation ids exist; search first"}
        severities = [v["severity"] for o in found for v in o["violations"]] or ["low"]
        severity = max(severities, key=SEVERITY_ORDER.index)
        created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        ids = [o["id"] for o in found]
        incident_id = store.add_incident(created, title, severity, ids, body, refs)
        return {"incident_id": incident_id, "severity": severity, "observation_ids": ids, "regulation_refs": refs}

    return {
        "list_zones": list_zones,
        "search_observations": search_observations,
        "count_violations": count_violations,
        "lookup_regulation": lookup_regulation,
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
