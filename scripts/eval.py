"""Measure the vision step against hand-labelled frames and write docs/EVAL.md.

    python scripts/eval.py --labels data/eval/labels.json
    python scripts/eval.py --labels data/eval/labels.json --prompt rule-ids-only
    python scripts/eval.py --labels data/eval/labels.json --vision-model <other id> --out docs/EVAL-other.md

Needs NEBIUS_API_KEY (live run). Without a key the app is in mock mode and the script refuses to
write numbers unless --allow-mock is given; the file is then marked "MOCK RUN, not a measurement".
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safefloor.config import Settings  # noqa: E402
from safefloor.evaluation import (  # noqa: E402
    PROMPT_MODES, compute_metrics, load_labels, make_meta, render_markdown, report_json, run_eval,
)
from safefloor.llm import make_llm  # noqa: E402
from safefloor.store import load_zones  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=Path("data/eval/labels.json"))
    ap.add_argument("--prompt", choices=PROMPT_MODES, default="zone-rules")
    ap.add_argument("--vision-model", help="override NEBIUS_VISION_MODEL for this run")
    ap.add_argument("--out", type=Path, default=Path("docs/EVAL.md"))
    ap.add_argument("--allow-mock", action="store_true", help="write a clearly labelled MOCK report")
    args = ap.parse_args(argv)

    settings = Settings.from_env()
    if args.vision_model:
        settings = dataclasses.replace(settings, vision_model=args.vision_model)
    if settings.mock and not args.allow_mock:
        print("Mock mode is active (no NEBIUS_API_KEY): refusing to write accuracy numbers. "
              "Set a key, or pass --allow-mock for a labelled dry run.", file=sys.stderr)
        return 2

    zones = load_zones(settings.zones_path)
    labels = load_labels(args.labels)
    rows = run_eval(make_llm(settings), zones, labels, args.labels.parent, args.prompt)
    report = compute_metrics(rows, zones)
    meta = make_meta(settings.vision_model, args.prompt, settings.mock)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(report, meta), encoding="utf-8")
    args.out.with_suffix(".json").write_text(report_json(report, meta), encoding="utf-8")
    o = report["overall"]
    print(f"{report['frames']} frames, precision {o['precision']}, recall {o['recall']}, F1 {o['f1']} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
