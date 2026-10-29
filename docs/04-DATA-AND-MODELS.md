# 04 Data and models

Purpose: datasets, models, compute and costs.

## Data
- `data/sample_observations.json`: sample shift seeded on first start. Provenance: TODO. Status: Unconfirmed
- `data/zones.json`: zones and safety rules (editable).
- Real warehouse frames for the demo: TODO. Status: Unconfirmed

## Models
- Agent: NVIDIA Nemotron via Nebius Token Factory (`NEBIUS_AGENT_MODEL`, default `nvidia/nemotron-3-super-120b-a12b`).
- Vision: NVIDIA Nemotron VL (`NEBIUS_VISION_MODEL`); exact id TODO via `scripts/check_models.py`. Status: Unconfirmed
- Mock model for offline tests.

## Compute and licences
- Inference on Token Factory (OpenAI-compatible). Optional hosting on a Nebius AI Cloud VM via the Dockerfile.
- Model licences: TODO. Credits and costs: TODO. Status: Unconfirmed
