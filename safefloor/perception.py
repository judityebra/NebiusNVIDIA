"""Turn a camera frame into a structured observation with a Nemotron vision model."""

from __future__ import annotations

import base64
import json
import re
from datetime import datetime, timezone
from typing import Any

from .llm import LLM
from .store import Zone

PROMPT = """You are the perception module of a warehouse safety system.
Camera {camera} watches the zone "{zone_name}". The zone's safety rules are:
{rules}

Look at the image and reply with ONLY a JSON object, no prose, in this shape:
{{"summary": "<one sentence describing what is happening>",
  "objects": ["<short noun>", ...],
  "violations": [{{"rule_id": "<one of the rule ids above>", "confidence": <0..1>}}]}}

List a violation only when the image shows evidence of it. If nothing is wrong, return an empty list."""


def build_prompt(camera: str, zone: Zone, prompt_mode: str = "zone-rules") -> str:
    """``rule-ids-only`` drops the written rule texts (ablation: what do the zone rules add?)."""
    if prompt_mode == "rule-ids-only":
        rules = "\n".join(f"- {r['id']} ({r['severity']})" for r in zone.rules)
    else:
        rules = "\n".join(f"- {r['id']} ({r['severity']}): {r['text']}" for r in zone.rules)
    return PROMPT.format(camera=camera, zone_name=zone.name, rules=rules)


class VisionError(Exception):
    """The vision model returned nothing we can parse."""


def _balanced_objects(text: str):
    """Yield every balanced ``{...}`` span, scanning braces and respecting quoted strings."""
    for start, ch in enumerate(text):
        if ch != "{":
            continue
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    yield text[start : i + 1]
                    break


def extract_json(text: str) -> dict[str, Any] | None:
    """First parseable JSON object in a model reply, ignoring think blocks and code fences."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE)
    text = text.rsplit("</think>", 1)[-1]  # unclosed opener: keep what follows the closing tag
    text = re.sub(r"```[a-zA-Z]*", "", text)
    for span in _balanced_objects(text):
        try:
            data = json.loads(span)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def parse_reply(text: str, zone: Zone) -> dict[str, Any]:
    """Pull the JSON object out of the reply and drop anything malformed."""
    data = extract_json(text) or {}
    known = {r["id"].lower(): r["id"] for r in zone.rules}
    violations = []
    for v in data.get("violations") or []:
        rule_id = known.get(str(v.get("rule_id", "")).strip().lower()) if isinstance(v, dict) else None
        if rule_id:
            try:
                conf = min(max(float(v.get("confidence", 0.5)), 0.0), 1.0)
            except (TypeError, ValueError):
                conf = 0.5
            violations.append({"rule_id": rule_id, "confidence": conf})
    objects = [str(o) for o in data.get("objects") or [] if isinstance(o, (str, int, float))]
    summary = str(data.get("summary") or text.strip()[:300] or "No description returned.")
    return {"summary": summary, "objects": objects, "violations": violations}


def describe_frame(llm: LLM, image: bytes, camera: str, zone: Zone, mime: str = "image/jpeg", ts: str | None = None, prompt_mode: str = "zone-rules") -> dict[str, Any]:
    prompt, b64 = build_prompt(camera, zone, prompt_mode), base64.b64encode(image).decode()
    reply = llm.vision(prompt, b64, mime)
    if extract_json(reply) is None:  # one more try before giving up
        reply = llm.vision(prompt, b64, mime)
        if extract_json(reply) is None:
            raise VisionError("vision model returned no parseable JSON")
    obs = parse_reply(reply, zone)
    obs["camera"] = camera
    obs["ts"] = ts or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return obs
