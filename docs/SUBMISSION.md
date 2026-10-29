# Devpost submission draft

Fill the bracketed parts before you submit. The deadline is 2026-10-30 at 10:00 PT (17:00 UTC,
18:00 in Barcelona).

## Checklist

- [ ] Get a Token Factory key and run `python scripts/check_models.py`; set the two model ids in `.env`.
- [ ] Run the app with the key (header badge says "Live") and test the three sample questions.
- [ ] Upload at least one real warehouse frame through "Analyse frame" (live vision model).
- [ ] Deploy somewhere public for the demo URL (Nebius AI Cloud VM with the Dockerfile, or any host).
- [ ] Record the demo video (outline below) and upload it to YouTube as public or unlisted.
- [ ] Make this GitHub repo public. It already has an MIT license.
- [ ] Pick the track: **Physical AI**.
- [ ] Write the tool feedback section from your own experience.

## Tagline

A safety copilot that turns warehouse camera frames into searchable, cited safety events, built on
NVIDIA Nemotron and Nebius Token Factory.

## Inspiration

[Write in your own words. Suggested angle: your client work on a warehouse vision agent with NVIDIA
VSS showed that safety teams review footage only after an accident. Do not name the client.]

## What it does

SafeFloor sends each camera frame to a Nemotron vision model together with the safety rules of that
zone, and stores the result as a structured observation: what is happening, what objects are
present and which rules were broken. Supervisors then ask a Nemotron agent questions in plain
language. The agent looks everything up through tools, cites the observation ids behind each claim
and can file incident reports backed by that evidence.

## How we built it

- Nebius Token Factory through its OpenAI-compatible API for every model call.
- A Nemotron vision-language model for perception, prompted with the zone's rules and constrained to
  JSON. A validating parser drops unknown rule ids and clamps confidences.
- A Nemotron model for the agent, with function calling over four tools: list zones, search
  observations, count violations and file an incident report.
- FastAPI and SQLite for the backend, a plain HTML and JavaScript dashboard, pytest for the tests.
- A mock model mode so the full loop runs and is tested without network access.

## Challenges

[Your own notes. Candidates: getting reliable JSON from the vision model, keeping the agent from
inventing events, mapping model output to rules it is allowed to report.]

## Accomplishments

- Every answer is grounded: the agent cites observation ids and can only file reports on ids that exist.
- Severity comes from the site's rules, not from the model.
- The tool trace is visible in the UI, so a supervisor can check how each answer was produced.

## What we learned

[Your own words.]

## What's next

Sampling live RTSP streams, tracking people and forklifts across frames, measured distances from
calibrated cameras, and alerts pushed to the shift lead in real time.

## Feedback on Nebius and NVIDIA tools (required)

[Write from your own use: setup of Token Factory, the model catalogue, latency, function-calling
quality of Nemotron, the vision model's JSON reliability, docs, anything that blocked you.]

## Demo video outline (about 2.5 minutes)

1. 0:00–0:20. The problem: hours of warehouse footage nobody watches until after an incident.
2. 0:20–0:50. The dashboard: zones and rules, then the shift of observations with violations flagged.
3. 0:50–1:30. Upload a frame. Show the Nemotron vision result with the rule it flagged.
4. 1:30–2:15. Ask the three sample questions. Expand the tool trace on one answer. File the incident report.
5. 2:15–2:30. Architecture slide (README diagram) and the Token Factory model ids.
