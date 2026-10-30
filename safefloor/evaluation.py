"""Accuracy of the vision step against hand-labelled frames: per-rule precision, recall and F1."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .llm import LLM
from .perception import describe_frame
from .store import Zone

PROMPT_MODES = ("zone-rules", "rule-ids-only")


def load_labels(path: Path) -> list[dict[str, Any]]:
    """Read a labels file. Image paths are relative to the labels file."""
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError("labels file must hold a JSON list")
    for i, it in enumerate(items):
        for key in ("image", "camera", "violations"):
            if key not in it:
                raise ValueError(f"label {i} is missing {key!r}")
    return items


def run_eval(
    llm: LLM,
    zones: dict[str, Zone],
    labels: list[dict[str, Any]],
    base_dir: Path,
    prompt_mode: str = "zone-rules",
) -> list[dict[str, Any]]:
    """Run the vision step on every labelled image; one result row per label."""
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"prompt mode must be one of {PROMPT_MODES}")
    by_camera = {z.camera: z for z in zones.values()}
    rows = []
    for item in labels:
        zone = by_camera.get(item["camera"])
        if zone is None:
            raise ValueError(f"unknown camera {item['camera']!r} for {item['image']}")
        truth = set(item["violations"])
        unknown = truth - {r["id"] for r in zone.rules}
        if unknown:
            raise ValueError(f"{item['image']}: rules {sorted(unknown)} do not exist in zone {zone.id}")
        image = (Path(base_dir) / item["image"]).read_bytes()
        row: dict[str, Any] = {"image": item["image"], "zone_id": zone.id, "truth": sorted(truth), "note": item.get("note", "")}
        try:
            obs = describe_frame(llm, image, item["camera"], zone, prompt_mode=prompt_mode)
            row.update(pred=sorted({v["rule_id"] for v in obs["violations"]}), summary=obs["summary"], error=None)
        except Exception as exc:  # a failed call counts as "predicted nothing" and is listed
            row.update(pred=[], summary="", error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    return rows


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def _prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    f1 = _ratio(2 * tp, 2 * tp + fp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def compute_metrics(rows: list[dict[str, Any]], zones: dict[str, Zone]) -> dict[str, Any]:
    """Per-rule and micro-averaged metrics. A rule is scored only on frames of zones that have it."""
    per_rule: dict[str, dict[str, int]] = {}
    for row in rows:
        for rule in zones[row["zone_id"]].rules:
            c = per_rule.setdefault(rule["id"], {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
            t, p = rule["id"] in row["truth"], rule["id"] in row["pred"]
            c["tp" if t and p else "fp" if p else "fn" if t else "tn"] += 1
    rules = {rid: {**_prf(c["tp"], c["fp"], c["fn"]), "tn": c["tn"]} for rid, c in sorted(per_rule.items())}
    tp, fp, fn = (sum(c[k] for c in per_rule.values()) for k in ("tp", "fp", "fn"))
    clean = [r for r in rows if not r["truth"]]
    flagged = [r for r in clean if r["pred"]]
    return {
        "frames": len(rows),
        "frames_with_violation": len(rows) - len(clean),
        "clean_frames": len(clean),
        "clean_frames_flagged": len(flagged),
        "clean_false_alarm_rate": _ratio(len(flagged), len(clean)),
        "errors": sum(1 for r in rows if r["error"]),
        "overall": _prf(tp, fp, fn),
        "rules": rules,
        "misses": [r for r in rows if set(r["truth"]) - set(r["pred"])],
        "false_positives": [r for r in rows if set(r["pred"]) - set(r["truth"])],
    }


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def render_markdown(report: dict[str, Any], meta: dict[str, Any]) -> str:
    m = report
    lines = []
    if meta["mock"]:
        lines += ["> **MOCK RUN, not a measurement.** The scripted mock model answered, not Nemotron.", ""]
    lines += [
        "# Vision accuracy on labelled frames",
        "",
        f"- Date: {meta['date']}",
        f"- Vision model: `{meta['vision_model']}`" + (" (mock)" if meta["mock"] else ""),
        f"- Prompt mode: `{meta['prompt_mode']}`",
        f"- Frames: {m['frames']} ({m['frames_with_violation']} with a violation, {m['clean_frames']} clean); "
        f"model call errors: {m['errors']}",
        "",
        "## Per rule",
        "",
        "| Rule | TP | FP | FN | TN | Precision | Recall | F1 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for rid, r in m["rules"].items():
        lines.append(f"| {rid} | {r['tp']} | {r['fp']} | {r['fn']} | {r['tn']} | {_pct(r['precision'])} | {_pct(r['recall'])} | {_pct(r['f1'])} |")
    o = m["overall"]
    lines += [
        f"| **overall (micro)** | {o['tp']} | {o['fp']} | {o['fn']} | | {_pct(o['precision'])} | {_pct(o['recall'])} | {_pct(o['f1'])} |",
        "",
        f"Clean frames wrongly flagged: {m['clean_frames_flagged']} of {m['clean_frames']} "
        f"(false-alarm rate {_pct(m['clean_false_alarm_rate'])}).",
        "",
        "## Confusion per rule (truth x predicted)",
        "",
        "| Rule | truth + predicted | truth + not predicted | no truth + predicted | no truth + not predicted |",
        "|---|---|---|---|---|",
    ]
    for rid, r in m["rules"].items():
        lines.append(f"| {rid} | {r['tp']} | {r['fn']} | {r['fp']} | {r['tn']} |")
    for title, rows in (("Misses (labelled violation not predicted)", m["misses"]), ("False positives (predicted, not labelled)", m["false_positives"])):
        lines += ["", f"## {title}", ""]
        if not rows:
            lines.append("None.")
        for r in rows:
            lines.append(f"- `{r['image']}`: truth {r['truth'] or 'none'}, predicted {r['pred'] or 'none'}. "
                         f"Model said: {r['summary'] or r['error'] or '(nothing)'}")
    return "\n".join(lines) + "\n"


def report_json(report: dict[str, Any], meta: dict[str, Any]) -> str:
    return json.dumps({"meta": meta, **report}, indent=2, ensure_ascii=False) + "\n"


def make_meta(vision_model: str, prompt_mode: str, mock: bool) -> dict[str, Any]:
    return {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "vision_model": vision_model,
        "prompt_mode": prompt_mode,
        "mock": mock,
    }
