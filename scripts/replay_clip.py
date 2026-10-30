"""Post a video clip to a running SafeFloor server and print the events it found.

    python scripts/replay_clip.py clip.mp4 --camera cam-01 --server http://127.0.0.1:8000 --fps 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--camera", default="cam-01")
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--ts", help="clip start time, ISO 8601 UTC (default: now)")
    args = ap.parse_args()

    data = {"camera": args.camera, "fps": str(args.fps)}
    if args.ts:
        data["ts"] = args.ts
    with args.video.open("rb") as fh:
        res = httpx.post(
            f"{args.server.rstrip('/')}/api/clips",
            data=data,
            files={"video": (args.video.name, fh, "video/mp4")},
            timeout=600,
        )
    if res.status_code != 200:
        print(f"error {res.status_code}: {res.text}", file=sys.stderr)
        return 1
    for obs in res.json():
        rules = ", ".join(f"{v['rule_id']} ({v['severity']}, {round(v['confidence'] * 100)}%)" for v in obs["violations"]) or "none"
        print(f"obs {obs['id']}  {obs['ts']} -> {obs.get('end_ts')}  rules: {rules}\n    {obs['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
