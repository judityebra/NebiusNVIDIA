"""Runtime settings, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    base_url: str
    agent_model: str
    vision_model: str
    db_path: Path
    zones_path: Path
    mock: bool
    timeout: float = 60.0
    frames_dir: Path | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        api_key = os.getenv("NEBIUS_API_KEY") or None
        # Mock mode runs the whole app offline with scripted model replies.
        # It is on by default when no key is configured.
        mock_env = os.getenv("SAFEFLOOR_MOCK")
        mock = (mock_env.lower() in {"1", "true", "yes"}) if mock_env else api_key is None
        return cls(
            api_key=api_key,
            base_url=os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"),
            agent_model=os.getenv("NEBIUS_AGENT_MODEL", "nvidia/nemotron-3-super-120b-a12b"),
            vision_model=os.getenv("NEBIUS_VISION_MODEL", "nvidia/Nemotron-3.5-Super-VL"),
            db_path=Path(os.getenv("SAFEFLOOR_DB", str(DATA_DIR / "safefloor.db"))),
            zones_path=Path(os.getenv("SAFEFLOOR_ZONES", str(DATA_DIR / "zones.json"))),
            mock=mock,
            timeout=float(os.getenv("SAFEFLOOR_TIMEOUT", "60")),
            frames_dir=Path(os.getenv("SAFEFLOOR_FRAMES", str(DATA_DIR / "frames"))),
        )
