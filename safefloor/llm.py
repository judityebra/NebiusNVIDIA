"""Model access: Nebius Token Factory (OpenAI-compatible) or an offline mock.

Both clients expose the same two calls so the rest of the code does not care
which one is active:

- ``chat(messages, tools)`` returns ``{"content": str | None, "tool_calls": [...]}``
- ``vision(prompt, image_b64, mime)`` returns the model's text reply
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from .config import Settings

THINK = re.compile(r"<think>.*?(</think>|$)", re.DOTALL)


def strip_reasoning(text: str | None) -> str | None:
    """Drop the <think>...</think> block that Nemotron reasoning models can emit."""
    if text is None:
        return None
    return THINK.sub("", text).strip() or None


class LLM(Protocol):
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> dict[str, Any]: ...

    def vision(self, prompt: str, image_b64: str, mime: str = "image/jpeg") -> str: ...


class TokenFactoryLLM:
    def __init__(self, settings: Settings):
        from openai import OpenAI

        self.settings = settings
        self.client = OpenAI(api_key=settings.api_key, base_url=settings.base_url, timeout=90, max_retries=2)

    def chat(self, messages, tools=None):
        kwargs: dict[str, Any] = {"model": self.settings.agent_model, "messages": messages, "temperature": 0.2}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        msg = self.client.chat.completions.create(**kwargs).choices[0].message
        calls = [
            {"id": c.id, "name": c.function.name, "arguments": c.function.arguments or "{}"}
            for c in (msg.tool_calls or [])
        ]
        return {"content": strip_reasoning(msg.content), "tool_calls": calls}

    def vision(self, prompt, image_b64, mime="image/jpeg"):
        resp = self.client.chat.completions.create(
            model=self.settings.vision_model,
            temperature=0.1,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
                    ],
                }
            ],
        )
        return strip_reasoning(resp.choices[0].message.content) or ""


class MockLLM:
    """Scripted stand-in so the app, the UI and the tests run without a key.

    It plans tool calls from keywords in the question and then writes a short
    answer from the tool results, so the agent loop is exercised end to end.
    """

    ZONE_WORDS = {"dock": "dock-a", "aisle": "aisle-7", "charg": "charging"}

    def chat(self, messages, tools=None):
        last = messages[-1]
        if last["role"] == "tool":
            report = self._report_call(messages)
            if report:
                return {"content": None, "tool_calls": [report]}
            return {"content": self._answer(messages), "tool_calls": []}
        question = last["content"].lower()
        zone = next((z for w, z in self.ZONE_WORDS.items() if w in question), None)
        args: dict[str, Any] = {"zone_id": zone} if zone else {}
        if any(w in question for w in ("report", "incident", "write up")):
            name = "search_observations"
            args["only_violations"] = True
        elif any(w in question for w in ("how many", "count", "most", "worst", "summary", "summar")):
            name = "count_violations"
        else:
            name = "search_observations"
            if "forklift" in question:
                args["query"] = "forklift"
            if any(w in question for w in ("violation", "unsafe", "risk", "near miss", "near-miss")):
                args["only_violations"] = True
        return {"content": None, "tool_calls": [{"id": "call_0", "name": name, "arguments": json.dumps(args)}]}

    @staticmethod
    def _wants_report(messages):
        question = next(m["content"] for m in reversed(messages) if m["role"] == "user").lower()
        return any(w in question for w in ("report", "incident", "write up"))

    def _report_call(self, messages):
        """After a search for a report request, file it from the most severe hit."""
        if not self._wants_report(messages) or any(
            c["function"]["name"] == "file_incident_report"
            for m in messages
            if m["role"] == "assistant"
            for c in m.get("tool_calls") or []
        ):
            return None
        obs = json.loads(messages[-1]["content"]).get("observations", [])
        if not obs:
            return None
        rank = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        worst = max(obs, key=lambda o: max((rank[v["severity"]] for v in o["violations"]), default=-1))
        body = f"What happened: {worst['summary']}\nWhere/when: {worst['zone_id']} at {worst['ts']}"
        args = {"title": f"Safety event in {worst['zone_id']}", "observation_ids": [worst["id"]], "body": body}
        return {"id": "call_1", "name": "file_incident_report", "arguments": json.dumps(args)}

    def _answer(self, messages):
        result = json.loads(messages[-1]["content"])
        if isinstance(result, dict) and "incident_id" in result:
            return (
                f"Filed incident #{result['incident_id']} ({result['severity']}) "
                f"citing [obs {', '.join(map(str, result['observation_ids']))}] (mock model)."
            )
        if isinstance(result, dict) and "counts" in result:
            counts = result["counts"]
            if not counts:
                return "No rule violations recorded for that scope."
            lines = [f"- {c['zone_id']} / {c['rule_id']} ({c['severity']}): {c['n']}" for c in counts]
            return "Violations by zone and rule (mock model):\n" + "\n".join(lines)
        obs = result.get("observations", []) if isinstance(result, dict) else []
        if not obs:
            return "I found no matching observations."
        lines = [f"- [{o['ts']}] {o['zone_id']}: {o['summary']} (obs {o['id']})" for o in obs]
        return f"Found {len(obs)} observation(s) (mock model):\n" + "\n".join(lines)

    def vision(self, prompt, image_b64, mime="image/jpeg"):
        rule_ids = re.findall(r"- (\S+) \(", prompt)
        violations = [{"rule_id": rule_ids[0], "confidence": 0.6}] if rule_ids else []
        return json.dumps(
            {
                "summary": "Mock perception: one person and one forklift visible in the frame.",
                "objects": ["person", "forklift"],
                "violations": violations,
            }
        )


def make_llm(settings: Settings) -> LLM:
    return MockLLM() if settings.mock else TokenFactoryLLM(settings)
