"""List the models your Token Factory key can use, so you can set the model env vars.

Usage: NEBIUS_API_KEY=... python scripts/check_models.py
"""

import os

from openai import OpenAI

client = OpenAI(
    api_key=os.environ["NEBIUS_API_KEY"],
    base_url=os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"),
)
ids = sorted(m.id for m in client.models.list())
for model_id in ids:
    marker = "  <- NVIDIA" if model_id.lower().startswith("nvidia/") else ""
    print(f"{model_id}{marker}")
