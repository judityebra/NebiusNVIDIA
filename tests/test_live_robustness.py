import json
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from safefloor import app as app_module
from safefloor.agent import Agent
from safefloor.app import create_app
from safefloor.config import DATA_DIR, Settings
from safefloor.llm import MockLLM
from safefloor.perception import VisionError, describe_frame, extract_json, parse_reply
from safefloor.store import Store, load_zones

GOOD = '{"summary": "s", "objects": ["person"], "violations": [{"rule_id": "ppe-vest", "confidence": 0.8}]}'


@pytest.fixture
def zone():
    return load_zones(DATA_DIR / "zones.json")["dock-a"]


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "t.db", load_zones(DATA_DIR / "zones.json"))


class ScriptedLLM:
    """Vision replies are served in order; chat is a no-op."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.systems = []

    def vision(self, prompt, image_b64, mime="image/jpeg"):
        self.calls += 1
        return self.replies.pop(0)

    def chat(self, messages, tools=None):
        self.systems.append(messages[0]["content"])
        return {"content": "ok", "tool_calls": []}


def test_extract_json_variants():
    assert extract_json("```json\n" + GOOD + "\n```")["summary"] == "s"
    assert extract_json("<think>I should output {not json}</think>" + GOOD)["summary"] == "s"
    assert extract_json("reasoning without opener</think>" + GOOD)["summary"] == "s"
    assert extract_json("Here you go: " + GOOD + " Hope that helps {").get("objects") == ["person"]
    nested = '{"summary": "a } brace { in a string", "extra": {"k": [1, 2]}, "violations": []}'
    assert extract_json("x " + nested)["summary"] == "a } brace { in a string"
    assert extract_json("{broken json} then " + GOOD)["summary"] == "s"
    for junk in ("", "no json here", "{", "[1, 2]", "{oops}"):
        assert extract_json(junk) is None


def test_rule_ids_match_case_insensitively(zone):
    text = '{"summary": "s", "violations": [{"rule_id": " PPE-Vest ", "confidence": 0.5}, {"rule_id": "nope"}]}'
    assert parse_reply(text, zone)["violations"] == [{"rule_id": "ppe-vest", "confidence": 0.5}]


def test_describe_frame_retries_once_then_succeeds(zone):
    llm = ScriptedLLM(["I cannot help with that", GOOD])
    obs = describe_frame(llm, b"x", "cam-01", zone)
    assert llm.calls == 2 and obs["violations"][0]["rule_id"] == "ppe-vest"


def test_describe_frame_gives_up_after_second_failure(zone):
    llm = ScriptedLLM(["nope", "still nope"])
    with pytest.raises(VisionError):
        describe_frame(llm, b"x", "cam-01", zone)
    assert llm.calls == 2


def _client(tmp_path, monkeypatch, llm):
    settings = Settings(None, "http://unused/v1/", "m", "v", tmp_path / "app.db", DATA_DIR / "zones.json", True)
    monkeypatch.setattr(app_module, "make_llm", lambda s: llm)
    return TestClient(create_app(settings))


def test_api_returns_502_and_stores_nothing(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, ScriptedLLM(["junk", "junk"]))
    before = len(client.get("/api/observations").json())
    r = client.post("/api/frames", data={"camera": "cam-01"}, files={"image": ("f.jpg", b"x", "image/jpeg")})
    assert r.status_code == 502 and r.json() == {"detail": "vision model returned no parseable JSON"}
    assert len(client.get("/api/observations").json()) == before


def test_api_returns_502_when_the_call_itself_fails(tmp_path, monkeypatch):
    class Boom(ScriptedLLM):
        def vision(self, *a, **k):
            raise TimeoutError("slow")

    client = _client(tmp_path, monkeypatch, Boom([]))
    r = client.post("/api/frames", data={"camera": "cam-01"}, files={"image": ("f.jpg", b"x", "image/jpeg")})
    assert r.status_code == 502 and "TimeoutError" in r.json()["detail"]


def test_api_accepts_good_reply_after_retry(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, ScriptedLLM(["junk", GOOD]))
    r = client.post("/api/frames", data={"camera": "cam-01"}, files={"image": ("f.jpg", b"x", "image/jpeg")})
    assert r.status_code == 200 and r.json()["violations"][0]["rule_id"] == "ppe-vest"


def test_current_time_is_in_the_system_prompt(store):
    llm = ScriptedLLM([])
    clock = lambda: datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)  # noqa: E731
    Agent(llm, store, clock=clock).ask("hello")
    assert "Current UTC time: 2026-10-30T12:00:00Z" in llm.systems[0]


def _add(store, n):
    for i in range(n):
        store.add_observation(
            {"camera": "cam-01", "ts": f"2026-10-29T{i // 60:02d}:{i % 60:02d}:00Z", "summary": f"obs {i}", "objects": []}
        )


def test_newest_rows_are_kept_in_chronological_order(store):
    _add(store, 120)
    rows = store.observations(limit=50)
    assert [r["summary"] for r in rows] == [f"obs {i}" for i in range(70, 120)]
    assert store.count_observations() == 120


def test_search_tool_reports_truncation(store):
    from safefloor.tools import make_tools, run_tool

    _add(store, 60)
    tools = make_tools(store)
    res = json.loads(run_tool(tools, "search_observations", "{}"))
    assert res["truncated"] is True and res["total_matching"] == 60 and len(res["observations"]) == 50
    res = json.loads(run_tool(tools, "search_observations", json.dumps({"since": "2026-10-29T00:20:00Z"})))
    assert res["truncated"] is False and res["total_matching"] == 40


def test_mock_agent_still_works_with_the_new_tool_shape(store):
    _add(store, 3)
    r = Agent(MockLLM(), store).ask("Show forklift events")
    assert r["trace"][0]["tool"] == "search_observations"
