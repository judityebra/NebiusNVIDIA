# 05 Roadmap

Purpose: milestones to the deadline (2026-10-30 10:00 PDT, per `docs/SUBMISSION.md`).

Built and merged (pull requests #1-#8):
- [x] Markdown structure and docs index (#1)
- [x] CI: pytest on every PR, auto squash-merge when green (#2)
- [x] Live-path robustness: timeouts, tolerant JSON parsing, 502 on unusable vision reply (#3)
- [x] Evidence frames, citation verifier, source labels (#4)
- [x] Video clips to timed events (#5)
- [x] Accuracy benchmark harness `scripts/eval.py` (#6)
- [x] Tavily regulation lookup grounded into incident reports (#7)
- [x] Critical alerts, draft incidents, webhook, live SSE feed (#8)

Still open (nothing has run against the real API yet):
- [ ] Token Factory key; run `scripts/check_models.py`; set model ids in `.env` (owner: TODO)
- [ ] Tavily key in `.env` (owner: TODO)
- [ ] Run live ("Live" badge) and test the three sample questions (owner: TODO)
- [ ] Analyse at least one real warehouse frame and one clip (owner: TODO)
- [ ] Label about 40 frames and run `scripts/eval.py`; paste the table in the README (owner: TODO)
- [ ] Record and upload the demo video (owner: TODO)
- [ ] Make the repo public (owner: judityebra, Settings)
- [ ] Write the tool-feedback section (owner: TODO)
- [ ] Submit on the event platform (owner: TODO)

## Risks
- Model ids change; vision JSON reliability and tool-calling quality are unproven live.
- Video analysis is one vision call per sampled frame; multi-image input support on Token Factory is unverified.
- The system message used by the citation retry may be rejected by some chat templates; the code falls back to flagging the citation.
- The repo is private until the owner makes it public.
