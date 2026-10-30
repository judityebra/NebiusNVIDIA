"""FastAPI service: frame ingestion, the agent endpoint and the dashboard."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agent import Agent
from .config import DATA_DIR, ROOT, Settings
from .llm import make_llm
from .perception import VisionError, describe_frame
from .store import Store, load_zones


def seed(store: Store) -> int:
    """Load the sample observations so the demo has a shift of history."""
    if store.observations(limit=1):
        return 0
    samples = json.loads((DATA_DIR / "sample_observations.json").read_text())
    for obs in samples:
        store.add_observation({**obs, "source": "seed"})
    return len(samples)


FRAME_TYPES = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp", "gif": "image/gif", "bmp": "image/bmp"}
MIME_EXT = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp", "image/gif": "gif", "image/bmp": "bmp"}


def frame_ext(content_type: str | None, filename: str | None) -> str:
    ext = MIME_EXT.get((content_type or "").split(";")[0].strip().lower())
    if not ext and filename and "." in filename:
        guess = filename.rsplit(".", 1)[1].lower()
        ext = guess if guess in FRAME_TYPES else None
    return ext or "jpg"


def parse_ts(value: str) -> str:
    """Validate an ISO 8601 timestamp and normalise it to UTC, e.g. 2026-10-30T12:00:00Z."""
    try:
        dt = datetime.fromisoformat(value.strip())
    except ValueError as exc:
        raise HTTPException(400, f"ts must be ISO 8601, got {value!r}") from exc
    dt = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


class Question(BaseModel):
    question: str
    history: list[dict[str, Any]] | None = None


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    store = Store(settings.db_path, load_zones(settings.zones_path))
    seed(store)
    llm = make_llm(settings)
    frames_dir = settings.frames_dir or settings.db_path.parent / "frames"
    agent = Agent(llm, store)

    app = FastAPI(title="SafeFloor")

    @app.get("/api/status")
    def status():
        return {
            "mock": settings.mock,
            "agent_model": settings.agent_model,
            "vision_model": settings.vision_model,
            "base_url": settings.base_url,
        }

    @app.get("/api/zones")
    def zones():
        return [vars(z) for z in store.zones.values()]

    @app.get("/api/observations")
    def observations(zone_id: str | None = None, only_violations: bool = False):
        return store.observations(zone_id=zone_id, only_violations=only_violations, limit=200)

    @app.get("/api/observations/{obs_id}")
    def observation(obs_id: int):
        obs = store.get_observation(obs_id)
        if obs is None:
            raise HTTPException(404, f"no observation {obs_id}")
        return obs

    @app.get("/api/observations/{obs_id}/frame")
    def observation_frame(obs_id: int):
        obs = store.get_observation(obs_id)
        path = frames_dir / f"{obs_id}.{obs['frame_ext']}" if obs and obs["frame_ext"] else None
        if path is None or not path.is_file():
            raise HTTPException(404, f"no frame for observation {obs_id}")
        return FileResponse(path, media_type=FRAME_TYPES.get(obs["frame_ext"], "application/octet-stream"))

    @app.get("/api/incidents")
    def incidents():
        return store.incidents()

    @app.post("/api/frames")
    async def ingest(camera: str = Form(...), image: UploadFile = File(...), ts: str | None = Form(None)):
        zone = store.zone_for_camera(camera)
        if zone is None:
            raise HTTPException(404, f"unknown camera {camera}")
        data = await image.read()
        if not data:
            raise HTTPException(400, "empty image")
        captured = parse_ts(ts) if ts else None
        try:
            obs = describe_frame(llm, data, camera, zone, image.content_type or "image/jpeg", ts=captured)
        except VisionError as exc:
            raise HTTPException(502, str(exc)) from exc
        except Exception as exc:  # network or API failure: visible error, nothing stored
            raise HTTPException(502, f"vision model call failed: {type(exc).__name__}") from exc
        obs["source"] = "mock" if settings.mock else "nemotron"
        obs["frame_ext"] = frame_ext(image.content_type, image.filename)
        obs_id = store.add_observation(obs)
        try:
            frames_dir.mkdir(parents=True, exist_ok=True)
            (frames_dir / f"{obs_id}.{obs['frame_ext']}").write_bytes(data)
        except OSError:  # keep the observation, just without an evidence frame
            store.clear_frame(obs_id)
        return store.get_observation(obs_id)

    @app.post("/api/ask")
    def ask(q: Question):
        if not q.question.strip():
            raise HTTPException(400, "empty question")
        return agent.ask(q.question, q.history)

    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

    @app.get("/")
    def index():
        return FileResponse(ROOT / "static" / "index.html")

    return app
