# 07 Experiments

Purpose: log of live measurements and the raw material for the required tool-feedback section.

Nothing has been measured live yet, so every row is `TODO`. Do not fill rows from memory or from the mock model.

## Log

| Date | Idea | Setup | Result | Next |
|---|---|---|---|---|
| TODO | Vision JSON reliability | Real frames via "Analyse frame", count replies needing the retry or a 502 | TODO | TODO |
| TODO | Agent grounding | The three sample questions with a live key; read the citation badges | TODO | TODO |
| TODO | Zone rules vs bare ids | `scripts/eval.py --prompt rule-ids-only` against the default | TODO | TODO |
| TODO | Model comparison | `scripts/eval.py --vision-model <id>` | TODO | TODO |
| TODO | Clip latency | `scripts/replay_clip.py` on a 30 s clip: total time, frames skipped | TODO | TODO |

## Feedback prompts (for the Devpost "feedback on Nebius and NVIDIA tools" field)

Each item: what happened, and a suggested fix. Status: TODO for all, nothing measured yet.

- Model-id mismatches: ids that differ between the docs, `check_models.py` output and the defaults in `.env.example`.
- Latency per call: vision call and agent step, p50 and worst case.
- Tool-calling failures: malformed arguments, calls to unknown tools, answers without tool use.
- Vision JSON reliability: replies with prose, think blocks, code fences, wrong rule ids.
- Multi-image or video support: whether one call can take several frames (we use one call per frame).
- Documentation gaps: anything that blocked setup.
