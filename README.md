# SafeFloor

A warehouse safety copilot built on NVIDIA Nemotron models served by Nebius Token Factory.
Entry for the **Nebius x NVIDIA Global AI Hackathon 2026**, Physical AI track.

![SafeFloor dashboard](docs/screenshot.png)

## What it does

Warehouse cameras record many hours of footage, and safety teams only look at it after something
goes wrong. SafeFloor turns camera frames into a searchable log of safety events, and gives
supervisors an agent they can ask questions in plain language:

- "Which zone had the most violations today?"
- "Show near misses with forklifts at the dock."
- "Write an incident report for the worst dock event."

The agent answers from the recorded observations only, cites the observation ids behind every
claim, and can file incident reports backed by that evidence.

## How it works

```
camera frame ──► Nemotron vision model ──► structured observation ──► SQLite
                 (zone rules in prompt)     summary, objects,            │
                                            rule violations              │
                                                                         ▼
supervisor question ──► Nemotron agent ──tool calls──► list_zones
                        (Token Factory,                search_observations
                         function calling)             count_violations
                                                       file_incident_report
```

1. **Perception** (`safefloor/perception.py`). Each frame goes to a Nemotron vision-language model
   with the safety rules of that camera's zone. The model returns JSON: a one-line summary, the
   objects it sees and any rule violations with a confidence score. The parser drops rule ids the
   zone does not have and clamps confidences, so bad model output never reaches the database.
2. **Store** (`safefloor/store.py`). Observations, violations and incidents live in SQLite. Every
   violation takes its severity from the zone rule, not from the model.
3. **Agent** (`safefloor/agent.py`, `safefloor/tools.py`). A Nemotron model on Token Factory runs an
   OpenAI-style function-calling loop over four tools. Tool errors go back to the model as data,
   and the loop stops after six steps.
4. **Dashboard** (`static/`). Zones and rules, the live observation feed, frame upload, the chat
   with an expandable trace of every tool call, and the incident list.

## Nebius and NVIDIA usage

| Piece | What it uses |
|---|---|
| Inference | Nebius Token Factory, OpenAI-compatible API (`NEBIUS_BASE_URL`) |
| Agent model | NVIDIA Nemotron (`NEBIUS_AGENT_MODEL`, default `nvidia/nemotron-3-super-120b-a12b`) |
| Vision model | NVIDIA Nemotron VL (`NEBIUS_VISION_MODEL`) |
| Hosting (optional) | The Dockerfile runs on a Nebius AI Cloud VM or any container host |

Model ids change. Run `python scripts/check_models.py` with your key and copy the exact NVIDIA ids it
prints into `.env`.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env    # then set NEBIUS_API_KEY
```

Load the variables and start the server:

```bash
set -a; source .env; set +a
python -m safefloor
```

Open http://127.0.0.1:8000. On Windows PowerShell, set the variables with
`$env:NEBIUS_API_KEY="..."` before `python -m safefloor`.

Without `NEBIUS_API_KEY` the app starts in **mock mode**: a scripted model stands in for Nemotron so
the UI, the agent loop and the tests run offline. The badge in the header shows which mode is on.

The app seeds one shift of sample observations (`data/sample_observations.json`) on first start.
Zones and rules are in `data/zones.json`; edit them to match a real site.

Docker:

```bash
docker build -t safefloor .
docker run -p 8000:8000 --env-file .env safefloor
```

## Tests

```bash
pytest -q
```

The tests cover the store filters, the rule-id validation, the reply parser, the incident severity
logic, the agent loop and the HTTP API, all in mock mode.

## API

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/status` | | mode and model ids |
| GET | `/api/zones` | | zones with rules |
| GET | `/api/observations?zone_id=&only_violations=` | | observations with violations |
| POST | `/api/frames` | multipart `camera`, `image` | the new observation |
| POST | `/api/ask` | `{"question": "...", "history": [...]}` | `{"answer", "trace"}` |
| GET | `/api/incidents` | | filed incident reports |

## Limits and next steps

- Frames are analysed one at a time. The next step is sampling a live RTSP stream and tracking
  people and forklifts across frames, which is what NVIDIA VSS does at production scale.
- Distances such as "3 m from a forklift" are estimated by the vision model, not measured. A depth or
  calibrated-camera stage would make those rules reliable.
- Confidence scores come from the model and are not calibrated.

## License

MIT, see [LICENSE](LICENSE).
