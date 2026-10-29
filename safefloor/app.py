"""FastAPI service: frame ingestion, the agent endpoint and the dashboard."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from openai import OpenAIError
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agent import Agent
from .config import DATA_DIR, ROOT, Settings
from .llm import make_llm
from .perception import describe_frame
from .store import Store, load_zones


def seed(store: Store, day: str | None = None) -> int:
    """Load the sample observations so the demo has a shift of history.

    With ``day`` (YYYY-MM-DD) the shift is moved onto that date, so questions
    about "today" find it whenever the demo runs.
    """
    if store.observations(limit=1):
        return 0
    samples = json.loads((DATA_DIR / "sample_observations.json").read_text())
    for obs in samples:
        if day:
            obs["ts"] = day + obs["ts"][10:]
        store.add_observation(obs)
    return len(samples)


class Question(BaseModel):
    question: str
    history: list[dict[str, Any]] | None = None


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    store = Store(settings.db_path, load_zones(settings.zones_path))
    seed(store, datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    llm = make_llm(settings)
    agent = Agent(llm, store)

    app = FastAPI(title="SafeFloor")

    @app.exception_handler(OpenAIError)
    async def model_error(request: Request, exc: OpenAIError):
        # A wrong model id, a bad key or a timeout should read clearly in the UI.
        return JSONResponse(status_code=502, content={"detail": f"Token Factory call failed: {exc}"})

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

    @app.get("/api/incidents")
    def incidents():
        return store.incidents()

    @app.post("/api/frames")
    async def ingest(camera: str = Form(...), image: UploadFile = File(...)):
        zone = store.zone_for_camera(camera)
        if zone is None:
            raise HTTPException(404, f"unknown camera {camera}")
        data = await image.read()
        if not data:
            raise HTTPException(400, "empty image")
        obs = describe_frame(llm, data, camera, zone, image.content_type or "image/jpeg")
        obs_id = store.add_observation(obs)
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
