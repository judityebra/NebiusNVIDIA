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


def build_prompt(camera: str, zone: Zone) -> str:
    rules = "\n".join(f"- {r['id']} ({r['severity']}): {r['text']}" for r in zone.rules)
    return PROMPT.format(camera=camera, zone_name=zone.name, rules=rules)


def first_json_object(text: str) -> dict[str, Any]:
    """Return the first JSON object embedded in the text, or {} if there is none."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            data, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return {}


def parse_reply(text: str, zone: Zone) -> dict[str, Any]:
    """Pull the JSON object out of the reply and drop anything malformed."""
    data = first_json_object(text)
    known = {r["id"] for r in zone.rules}
    violations = []
    for v in data.get("violations") or []:
        if isinstance(v, dict) and v.get("rule_id") in known:
            try:
                conf = min(max(float(v.get("confidence", 0.5)), 0.0), 1.0)
            except (TypeError, ValueError):
                conf = 0.5
            violations.append({"rule_id": v["rule_id"], "confidence": conf})
    objects = [str(o) for o in data.get("objects") or [] if isinstance(o, (str, int, float))]
    summary = str(data.get("summary") or text.strip()[:300] or "No description returned.")
    return {"summary": summary, "objects": objects, "violations": violations}


def describe_frame(llm: LLM, image: bytes, camera: str, zone: Zone, mime: str = "image/jpeg", ts: str | None = None) -> dict[str, Any]:
    reply = llm.vision(build_prompt(camera, zone), base64.b64encode(image).decode(), mime)
    obs = parse_reply(reply, zone)
    obs["camera"] = camera
    obs["ts"] = ts or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return obs
