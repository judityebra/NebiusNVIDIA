# SafeFloor (NebiusNVIDIA) — Claude instructions

Warehouse safety copilot: Nemotron vision turns camera frames into structured safety observations, a Nemotron agent answers supervisor questions with cited evidence. Entry for the Nebius x NVIDIA Global AI Hackathon 2026 (Physical AI track).

Read `docs/00-OVERVIEW.md` first, then `docs/README.md` for the index.

## Stack
Python 3.12, FastAPI + uvicorn, SQLite, OpenAI-compatible client (Nebius Token Factory), plain HTML/JS in `static/`, pytest. Docker supported.

## Run and test
- `python -m venv .venv`, activate, `pip install -r requirements.txt`
- `cp .env.example .env` and set `NEBIUS_API_KEY` (without it the app runs in offline mock mode)
- `python -m safefloor`, then open http://127.0.0.1:8000
- `pytest -q` (all tests run in mock mode, no network)

## Conventions
- Surgical edits; keep changes scoped to the task.
- Severity comes from zone rules in `data/zones.json`, never from the model.
- Never invent facts about the hackathon: unknown items stay `TODO` + `Status: Unconfirmed`.

## Git rules
- No secrets or `.env` in git (`.env.example` only).
- Commit as Isaac: `git -c user.name="Isaac Boqué Carbonell" -c user.email="isboque19@gmail.com" commit ...`
- Never push to main directly. Use work branches + PR; merge only when CI is green (or the repo has no checks).
