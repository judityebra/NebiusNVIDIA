import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from safefloor.agent import Agent, cited_ids, mark_unverified
from safefloor.app import create_app, seed
from safefloor.config import DATA_DIR, Settings
from safefloor.store import Store, load_zones

OLD_SCHEMA = """
CREATE TABLE observations (id INTEGER PRIMARY KEY AUTOINCREMENT, camera TEXT NOT NULL, zone_id TEXT NOT NULL,
    ts TEXT NOT NULL, summary TEXT NOT NULL, objects TEXT NOT NULL);
CREATE TABLE violations (id INTEGER PRIMARY KEY AUTOINCREMENT, observation_id INTEGER NOT NULL,
    rule_id TEXT NOT NULL, severity TEXT NOT NULL, confidence REAL NOT NULL);
CREATE TABLE incidents (id INTEGER PRIMARY KEY AUTOINCREMENT, created_ts TEXT NOT NULL, title TEXT NOT NULL,
    severity TEXT NOT NULL, observation_ids TEXT NOT NULL, body TEXT NOT NULL);
"""

JPEG = b"\xff\xd8\xff\xe0tiny-jpeg-bytes\xff\xd9"


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "t.db", load_zones(DATA_DIR / "zones.json"))
    seed(s)
    return s


@pytest.fixture
def client(tmp_path):
    settings = Settings(
        None, "http://unused/v1/", "m", "v", tmp_path / "app.db", DATA_DIR / "zones.json", True,
        frames_dir=tmp_path / "frames",
    )
    return TestClient(create_app(settings))


def test_migration_keeps_old_rows(tmp_path):
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript(OLD_SCHEMA)
    conn.execute("INSERT INTO observations (camera, zone_id, ts, summary, objects) VALUES ('cam-01','dock-a','2026-10-29T07:00:00Z','old','[]')")
    conn.commit()
    conn.close()
    s = Store(db, load_zones(DATA_DIR / "zones.json"))
    s2 = Store(db, load_zones(DATA_DIR / "zones.json"))  # second open must not fail
    row = s.get_observation(1)
    assert row["summary"] == "old" and row["source"] == "unknown" and row["has_frame"] is False
    assert s2.get_observation(1)["summary"] == "old"
    new_id = s.add_observation({"camera": "cam-01", "ts": "2026-10-29T08:00:00Z", "summary": "n", "objects": [], "source": "mock"})
    assert s.get_observation(new_id)["source"] == "mock"


def test_seed_rows_are_labelled_seed(store):
    assert {o["source"] for o in store.observations(limit=100)} == {"seed"}


def test_frame_round_trip_and_ts(client, tmp_path):
    r = client.post(
        "/api/frames",
        data={"camera": "cam-01", "ts": "2026-10-30T09:15:00+02:00"},
        files={"image": ("shot.jpg", JPEG, "image/jpeg")},
    )
    obs = r.json()
    assert r.status_code == 200
    assert obs["ts"] == "2026-10-30T07:15:00Z" and obs["source"] == "mock" and obs["has_frame"] is True
    frame = client.get(f"/api/observations/{obs['id']}/frame")
    assert frame.status_code == 200 and frame.content == JPEG and frame.headers["content-type"] == "image/jpeg"
    assert (tmp_path / "frames" / f"{obs['id']}.jpg").read_bytes() == JPEG
    assert client.get(f"/api/observations/{obs['id']}").json()["id"] == obs["id"]


def test_png_keeps_extension_and_bad_ts_is_rejected(client):
    r = client.post("/api/frames", data={"camera": "cam-01"}, files={"image": ("a.png", b"\x89PNG", "image/png")})
    assert client.get(f"/api/observations/{r.json()['id']}/frame").headers["content-type"] == "image/png"
    bad = client.post("/api/frames", data={"camera": "cam-01", "ts": "yesterday"}, files={"image": ("a.jpg", JPEG, "image/jpeg")})
    assert bad.status_code == 400


def test_seed_rows_have_no_frame(client):
    assert client.get("/api/observations/1/frame").status_code == 404
    assert client.get("/api/observations/9999/frame").status_code == 404
    assert client.get("/api/observations/9999").status_code == 404
    first = client.get("/api/observations").json()[0]
    assert first["source"] == "seed" and first["has_frame"] is False


class FakeChat:
    """Serves scripted chat replies in order and records what the agent sent."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.sent = []

    def chat(self, messages, tools=None):
        self.sent.append([dict(m) for m in messages])
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def vision(self, *a, **k):
        raise NotImplementedError


def search_call():
    args = json.dumps({"zone_id": "dock-a", "only_violations": True})  # returns obs 2 and 8
    return {"content": None, "tool_calls": [{"id": "c1", "name": "search_observations", "arguments": args}]}


def say(text):
    return {"content": text, "tool_calls": []}


def test_citation_parsing_and_marking():
    assert cited_ids("a [obs 3] b [obs 4, 3] [obs  7 ]") == [3, 4, 7]
    assert mark_unverified("x [obs 2, 3] y [obs 5]", {3}) == "x [obs 2] [obs 3 UNVERIFIED] y [obs 5]"


def test_correct_citations_are_verified(store):
    llm = FakeChat(search_call(), say("Forklift near miss [obs 2] and again [obs 8]."))
    r = Agent(llm, store).ask("q")
    assert r["citations"] == {"cited": [2, 8], "verified": [2, 8], "unverified": [], "retried": False}
    assert "UNVERIFIED" not in r["answer"]


def test_unretrieved_citation_is_retried_and_can_be_fixed(store):
    llm = FakeChat(search_call(), say("It was [obs 3]."), say("Correction: [obs 2]."))
    r = Agent(llm, store).ask("q")
    assert r["answer"] == "Correction: [obs 2]."
    assert r["citations"] == {"cited": [2], "verified": [2], "unverified": [], "retried": True}
    correction = llm.sent[-1][-1]
    assert correction["role"] == "system" and "3" in correction["content"]


def test_repeated_unretrieved_citation_is_flagged(store):
    llm = FakeChat(search_call(), say("It was [obs 3]."), say("Still [obs 3] and [obs 2]."))
    r = Agent(llm, store).ask("q")
    assert r["answer"] == "Still [obs 3 UNVERIFIED] and [obs 2]."
    assert r["citations"] == {"cited": [3, 2], "verified": [2], "unverified": [3], "retried": True}


def test_failed_retry_keeps_the_first_answer_flagged(store):
    llm = FakeChat(search_call(), say("It was [obs 3]."), RuntimeError("system role rejected"))
    r = Agent(llm, store).ask("q")
    assert r["answer"] == "It was [obs 3 UNVERIFIED]." and r["citations"]["unverified"] == [3]


def test_answer_without_citations_has_empty_citations(store):
    r = Agent(FakeChat(say("Nothing to report.")), store).ask("q")
    assert r["citations"] == {"cited": [], "verified": [], "unverified": [], "retried": False}


def test_filed_incident_ids_count_as_retrieved(store):
    args = json.dumps({"title": "t", "observation_ids": [2], "body": "b"})
    call = {"content": None, "tool_calls": [{"id": "c1", "name": "file_incident_report", "arguments": args}]}
    r = Agent(FakeChat(call, say("Filed, see [obs 2].")), store).ask("q")
    assert r["citations"]["verified"] == [2]


def test_mock_answers_now_carry_verifiable_citations(client):
    ans = client.post("/api/ask", json={"question": "Show near misses with forklifts at the dock"}).json()
    assert ans["citations"]["cited"] and not ans["citations"]["unverified"]
