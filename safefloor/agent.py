"""Tool-calling agent loop over the observation store."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from .llm import LLM
from .store import Store
from .tools import SCHEMAS, make_tools, run_tool

SYSTEM = """You are SafeFloor, a safety copilot for warehouse supervisors.
You answer questions about what the site cameras saw, using only the tools provided.
Rules:
- Always look the facts up with a tool before answering. Never invent observations.
- Cite observation ids in brackets, e.g. [obs 3], for every claim about an event.
- Be brief and concrete: time, zone, what happened, why it matters.
- Only file an incident report when the user asks for one, and only cite ids you retrieved.
- If the data does not answer the question, say so.
- Current UTC time: {now}. All timestamps are UTC; convert "today", "yesterday", "this morning" to ISO ranges before searching.
- If a search result says truncated, narrow the query before concluding anything."""

MAX_STEPS = 6


class Agent:
    def __init__(self, llm: LLM, store: Store, clock: Callable[[], datetime] | None = None):
        self.llm = llm
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.tools = make_tools(store)

    def ask(self, question: str, history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM.format(now=self.clock().strftime("%Y-%m-%dT%H:%M:%SZ"))}]
        messages += history or []
        messages.append({"role": "user", "content": question})
        trace: list[dict[str, Any]] = []
        for _ in range(MAX_STEPS):
            reply = self.llm.chat(messages, SCHEMAS)
            if not reply["tool_calls"]:
                return {"answer": reply["content"] or "", "trace": trace}
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
                result = run_tool(self.tools, call["name"], call["arguments"])
                trace.append({"tool": call["name"], "arguments": call["arguments"], "result": result[:2000]})
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
        return {"answer": "I stopped after too many tool calls without reaching an answer.", "trace": trace}
