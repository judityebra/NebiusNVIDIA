"""Merge per-frame observations of a clip into timed safety events."""

from __future__ import annotations

from typing import Any

MAX_GAP = 1  # non-violating frames that may sit inside one event


def fmt_t(t: float) -> str:
    s = int(round(t))
    return f"{s // 60:02d}:{s % 60:02d}"


def _score(frame: dict[str, Any]) -> float:
    return sum(v["confidence"] for v in frame["violations"])


def build_events(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn ordered per-frame observations into events.

    ``frames`` are ``{"t", "summary", "objects", "violations": [{"rule_id", "confidence"}], "jpeg"}``.
    An event is a maximal run of violating frames in which each frame shares at least one violated
    rule with the previous violating frame and at most ``MAX_GAP`` non-violating frame sits between
    them. Each event keeps its peak frame (highest summed confidence) as the evidence frame. A clip
    with no violations yields one "no violations" event whose evidence is the middle frame.
    """
    if not frames:
        return []
    runs: list[list[int]] = []
    for i, f in enumerate(frames):
        if not f["violations"]:
            continue
        rules = {v["rule_id"] for v in f["violations"]}
        if runs:
            prev = runs[-1][-1]
            prev_rules = {v["rule_id"] for v in frames[prev]["violations"]}
            if i - prev <= MAX_GAP + 1 and rules & prev_rules:
                runs[-1].append(i)
                continue
        runs.append([i])
    if not runs:
        mid = frames[len(frames) // 2]
        start, end = frames[0]["t"], frames[-1]["t"]
        return [
            {
                "start_t": start,
                "end_t": end,
                "rules": {},
                "violations": [],
                "summary": f"{fmt_t(start)} - {fmt_t(end)}: No violations in clip.",
                "objects": _objects(frames),
                "jpeg": mid["jpeg"],
                "frame_count": len(frames),
            }
        ]
    events = []
    for run in runs:
        first, last = run[0], run[-1]
        peak = max((frames[i] for i in run), key=_score)
        rules: dict[str, float] = {}
        for i in run:
            for v in frames[i]["violations"]:
                rules[v["rule_id"]] = max(rules.get(v["rule_id"], 0.0), v["confidence"])
        start, end = frames[first]["t"], frames[last]["t"]
        events.append(
            {
                "start_t": start,
                "end_t": end,
                "rules": rules,
                "violations": [{"rule_id": r, "confidence": c} for r, c in rules.items()],
                "summary": f"{fmt_t(start)} - {fmt_t(end)}: {peak['summary']}",
                "objects": _objects(frames[first : last + 1]),
                "jpeg": peak["jpeg"],
                "frame_count": last - first + 1,
            }
        )
    return events


def _objects(frames: list[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for f in frames:
        for o in f["objects"]:
            if o not in seen:
                seen.append(o)
    return seen
