"""Tool-calling agent loop over the observation store."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Callable

from .llm import LLM
from .regulations import RegulationLookup
from .store import Store
from .tools import SCHEMAS, RunState, make_tools, run_tool

SYSTEM = """You are SafeFloor, a safety copilot for warehouse supervisors.
You answer questions about what the site cameras saw, using only the tools provided.
Rules:
- Always look the facts up with a tool before answering. Never invent observations.
- Cite observation ids in brackets, e.g. [obs 3], for every claim about an event.
- Be brief and concrete: time, zone, what happened, why it matters.
- Only file an incident report when the user asks for one, and only cite ids you retrieved.
- When filing a report, call lookup_regulation for the violated rule first and pass the links it returned as regulation_refs; never cite a regulation any other way. If the lookup is unavailable, say so and file without one.
- If the data does not answer the question, say so.
- Current UTC time: {now}. All timestamps are UTC; convert "today", "yesterday", "this morning" to ISO ranges before searching.
- If a search result says truncated, narrow the query before concluding anything."""

MAX_STEPS = 6

CITATION = re.compile(r"\[obs\s+(\d+(?:\s*,\s*\d+)*)\s*\]")


def cited_ids(text: str) -> list[int]:
    """Observation ids cited as [obs N] or [obs N, M], in order of first appearance."""
    seen: list[int] = []
    for m in CITATION.finditer(text):
        for i in re.findall(r"\d+", m.group(1)):
            if int(i) not in seen:
                seen.append(int(i))
    return seen


def retrieved_ids(result_json: str) -> set[int]:
    """Observation ids a tool result actually returned or cited."""
    try:
        data = json.loads(result_json)
    except json.JSONDecodeError:
        return set()
    if not isinstance(data, dict):
        return set()
    ids = {o["id"] for o in data.get("observations") or [] if isinstance(o, dict) and "id" in o}
    ids |= {i for i in data.get("observation_ids") or [] if isinstance(i, int)}
    return ids


def mark_unverified(text: str, unverified: set[int]) -> str:
    def repl(m: re.Match) -> str:
        ids = [int(i) for i in re.findall(r"\d+", m.group(1))]
        if not unverified.intersection(ids):
            return m.group(0)
        return " ".join(f"[obs {i} UNVERIFIED]" if i in unverified else f"[obs {i}]" for i in ids)

    return CITATION.sub(repl, text)


class Agent:
    def __init__(
        self,
        llm: LLM,
        store: Store,
        clock: Callable[[], datetime] | None = None,
        regulations: RegulationLookup | None = None,
    ):
        self.llm = llm
        self.store = store
        self.regulations = regulations or RegulationLookup()
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def ask(self, question: str, history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM.format(now=self.clock().strftime("%Y-%m-%dT%H:%M:%SZ"))}]
        messages += history or []
        messages.append({"role": "user", "content": question})
        trace: list[dict[str, Any]] = []
        retrieved: set[int] = set()
        tools = make_tools(self.store, self.regulations, RunState())  # fresh per ask(): regulation urls are per run
        for _ in range(MAX_STEPS):
            reply = self.llm.chat(messages, SCHEMAS)
            if not reply["tool_calls"]:
                return self._verified(reply["content"] or "", messages, retrieved, trace)
            messages.append(
                {
                    "role": "assistant",
                    "content": reply["content"],
                    "tool_calls": [
                        {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                        for c in reply["tool_calls"]
                    ],
                }
            )
            for call in reply["tool_calls"]:
                result = run_tool(tools, call["name"], call["arguments"])
                retrieved |= retrieved_ids(result)
                trace.append({"tool": call["name"], "arguments": call["arguments"], "result": result[:2000]})
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
        answer = "I stopped after too many tool calls without reaching an answer."
        return {"answer": answer, "trace": trace, "citations": _citations([], [], False)}

    def _verified(self, answer: str, messages: list[dict[str, Any]], retrieved: set[int], trace: list) -> dict[str, Any]:
        """Check every [obs N] against what the tools returned; retry once, then flag what is left."""
        cited = cited_ids(answer)
        bad = [i for i in cited if i not in retrieved]
        retried = False
        if bad:
            retried = True
            messages.append({"role": "assistant", "content": answer})
            messages.append(
                {
                    "role": "system",
                    "content": "Your answer cited observation ids that no tool returned in this conversation: "
                    + ", ".join(map(str, bad))
                    + ". Rewrite the answer citing only ids you retrieved, or remove those claims.",
                }
            )
            try:
                second = self.llm.chat(messages, None)["content"]
            except Exception:  # a server that rejects the retry must not lose the first answer
                second = None
            if second:
                answer = second
                cited = cited_ids(answer)
                bad = [i for i in cited if i not in retrieved]
        if bad:
            answer = mark_unverified(answer, set(bad))
        return {
            "answer": answer,
            "trace": trace,
            "citations": _citations(cited, [i for i in cited if i not in bad], retried),
        }


def _citations(cited: list[int], verified: list[int], retried: bool) -> dict[str, Any]:
    return {"cited": cited, "verified": verified, "unverified": [i for i in cited if i not in verified], "retried": retried}
