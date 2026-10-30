# Labelling frames for the accuracy benchmark

`scripts/eval.py` compares what the vision model flags with what you labelled. The numbers are only
as good as the labels, so label carefully and never invent results.

## What to label

- About **20 frames with a violation and 20 clean frames**, taken from the real clips.
- At least a few examples per rule (one line per rule in `data/zones.json`).
- Never use client footage. Use clips we are allowed to publish.
- Small enough to commit: put the JPEGs in `data/eval/frames/` (about 768 px long side is plenty).
  `data/eval/frames/` and `data/eval/labels.json` are not gitignored on purpose: they are evidence.

## Format

`data/eval/labels.json` is a list, see `labels.example.json`:

```json
{"image": "frames/clip1_0007.jpg", "camera": "cam-01", "violations": ["ppe-vest"], "note": "..."}
```

- `image`: path relative to the labels file.
- `camera`: the camera of the zone the frame comes from (`cam-01` dock, `cam-02` aisle, `cam-03` charging).
- `violations`: rule ids from that zone that are clearly visible. An empty list means a clean frame.
- `note`: free text, shown next to misses and false positives.

Label only what you can see in that single frame. If you are unsure, leave the frame out.

## Run it

```bash
python scripts/eval.py --labels data/eval/labels.json                       # zone rules in the prompt
python scripts/eval.py --labels data/eval/labels.json --prompt rule-ids-only --out docs/EVAL-ablation.md
python scripts/eval.py --labels data/eval/labels.json --vision-model <other id> --out docs/EVAL-other.md
```

It needs `NEBIUS_API_KEY`. Without a key it refuses to write numbers (add `--allow-mock` for a dry run
that is stamped "MOCK RUN, not a measurement"). Output: `docs/EVAL.md` plus `docs/EVAL.json`.
