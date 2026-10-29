"""List the models your Token Factory key can use, so you can set the model env vars.

Usage: python scripts/check_models.py   (reads NEBIUS_API_KEY from the shell or .env)
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

client = OpenAI(
    api_key=os.environ["NEBIUS_API_KEY"],
    base_url=os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"),
)
ids = sorted(m.id for m in client.models.list())
for model_id in ids:
    marker = "  <- NVIDIA" if model_id.lower().startswith("nvidia/") else ""
    print(f"{model_id}{marker}")
