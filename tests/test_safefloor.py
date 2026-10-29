import json

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import APIConnectionError

from safefloor.agent import Agent
from safefloor.app import create_app, seed
from safefloor.config import DATA_DIR, Settings
from safefloor.llm import MockLLM, strip_reasoning
from safefloor.perception import build_prompt, describe_frame, parse_reply
from safefloor.store import Store, load_zones
from safefloor.tools import make_tools, run_tool


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "t.db", load_zones(DATA_DIR / "zones.json"))
    seed(s)
    return s


@pytest.fixture
def client(tmp_path):
    settings = Settings(
        api_key=None,
        base_url="http://unused/v1/",
        agent_model="m",
        vision_model="v",
        db_path=tmp_path / "app.db",
        zones_path=DATA_DIR / "zones.json",
        mock=True,
    )
    return TestClient(create_app(settings))


def test_seed_loads_samples_once(store):
    assert len(store.observations(limit=100)) == 8
    assert seed(store) == 0


def test_unknown_rule_ids_are_dropped(store):
    obs_id = store.add_observation(
        {"camera": "cam-01", "ts": "2026-10-29T11:00:00Z", "summary": "x", "objects": [],
         "violations": [{"rule_id": "made-up", "confidence": 0.9}, {"rule_id": "ppe-vest", "confidence": 0.7}]}
    )
    assert [v["rule_id"] for v in store.get_observation(obs_id)["violations"]] == ["ppe-vest"]


def test_search_filters(store):
    dock = store.observations(zone_id="dock-a", only_violations=True)
    assert {o["id"] for o in dock} == {2, 8}
    assert all("forklift" in o["summary"].lower() or "forklift" in o["objects"] for o in store.observations(query="forklift"))
    assert [o["id"] for o in store.observations(since="2026-10-29T09:00:00Z")] == [6, 7, 8]


def test_violation_counts(store):
    counts = store.violation_counts()
    top = counts[0]
    assert (top["zone_id"], top["rule_id"], top["n"]) == ("dock-a", "forklift-gap", 2)


def test_incident_takes_worst_severity_and_rejects_bad_ids(store):
    tools = make_tools(store)
    res = json.loads(run_tool(tools, "file_incident_report", json.dumps({"title": "t", "observation_ids": [2, 3], "body": "b"})))
    assert res["severity"] == "critical"
    bad = json.loads(run_tool(tools, "file_incident_report", json.dumps({"title": "t", "observation_ids": [999], "body": "b"})))
    assert "error" in bad
    assert "error" in json.loads(run_tool(tools, "nope", "{}"))
    assert "error" in json.loads(run_tool(tools, "count_violations", "not json"))


def test_parse_reply_handles_prose_and_bad_values(store):
    zone = store.zones["dock-a"]
    text = 'Sure! {"summary": "s", "objects": ["person"], "violations": [{"rule_id": "ppe-vest", "confidence": 3}, {"rule_id": "zzz"}, "junk"]} done'
    out = parse_reply(text, zone)
    assert out == {"summary": "s", "objects": ["person"], "violations": [{"rule_id": "ppe-vest", "confidence": 1.0}]}
    assert parse_reply("no json here", zone)["violations"] == []


def test_reasoning_is_stripped_before_parsing(store):
    assert strip_reasoning("<think>plan {not json}</think>\nHello") == "Hello"
    assert strip_reasoning("<think>cut off mid thought") is None
    text = '<think>maybe {"a": 1}?</think> {"summary": "s", "objects": [], "violations": [{"rule_id": "ppe-vest"}]}'
    out = parse_reply(strip_reasoning(text), store.zones["dock-a"])
    assert out["summary"] == "s" and out["violations"][0]["rule_id"] == "ppe-vest"
    assert parse_reply('prose {bad} then {"summary": "ok"}', store.zones["dock-a"])["summary"] == "ok"


def test_seed_moves_shift_onto_given_day(tmp_path):
    s = Store(tmp_path / "d.db", load_zones(DATA_DIR / "zones.json"))
    seed(s, "2027-01-15")
    assert all(o["ts"].startswith("2027-01-15T") for o in s.observations(limit=100))


def test_model_errors_become_502(client, monkeypatch):
    def fail(*args, **kwargs):
        raise APIConnectionError(request=httpx.Request("POST", "http://unused"))

    monkeypatch.setattr(MockLLM, "chat", fail)
    r = client.post("/api/ask", json={"question": "anything"})
    assert r.status_code == 502 and "Token Factory" in r.json()["detail"]


def test_prompt_lists_zone_rules(store):
    prompt = build_prompt("cam-03", store.zones["charging"])
    assert "no-entry (critical)" in prompt and "Battery charging bay" in prompt


def test_describe_frame_with_mock(store):
    obs = describe_frame(MockLLM(), b"\xff\xd8fake", "cam-01", store.zones["dock-a"], ts="2026-10-29T12:00:00Z")
    assert obs["camera"] == "cam-01" and obs["violations"][0]["rule_id"] == "ppe-vest"


def test_agent_counts_and_reports(store):
    agent = Agent(MockLLM(), store)
    r = agent.ask("Which zone had the most violations?")
    assert r["trace"][0]["tool"] == "count_violations" and "forklift-gap" in r["answer"]
    r = agent.ask("Write an incident report for the worst dock event")
    assert [t["tool"] for t in r["trace"]] == ["search_observations", "file_incident_report"]
    assert store.incidents()[0]["severity"] == "critical"


def test_api_end_to_end(client):
    assert client.get("/api/status").json()["mock"] is True
    assert len(client.get("/api/zones").json()) == 3
    assert len(client.get("/api/observations", params={"only_violations": True}).json()) == 5
    r = client.post("/api/frames", data={"camera": "cam-02"}, files={"image": ("f.jpg", b"\xff\xd8x", "image/jpeg")})
    assert r.status_code == 200 and r.json()["zone_id"] == "aisle-7"
    assert client.post("/api/frames", data={"camera": "cam-99"}, files={"image": ("f.jpg", b"x", "image/jpeg")}).status_code == 404
    ans = client.post("/api/ask", json={"question": "Show near misses with forklifts at the dock"}).json()
    assert "obs 2" in ans["answer"]
    assert client.get("/").status_code == 200
