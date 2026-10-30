import json

import pytest

from safefloor.agent import Agent
from safefloor.app import seed
from safefloor.config import DATA_DIR
from safefloor.regulations import RegulationLookup
from safefloor.store import Store, load_zones
from safefloor.tools import RunState, make_tools, run_tool

OSHA = "https://www.osha.gov/powered-industrial-trucks"


class FakeTavily:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return {"results": self.results}


def results(n=5, host="www.osha.gov"):
    return [{"title": f"T{i}", "url": f"https://{host}/p{i}", "content": "x" * 500, "score": 0.9} for i in range(n)]


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "t.db", load_zones(DATA_DIR / "zones.json"))
    seed(s)
    return s


def tools_with(store, fake=None, **kw):
    run = RunState()
    lookup = RegulationLookup(api_key="k", factory=lambda: fake, **kw) if fake else RegulationLookup(**kw)
    return make_tools(store, lookup, run), run


def call(tools, name, **args):
    return json.loads(run_tool(tools, name, json.dumps(args)))


def test_every_rule_has_a_plain_regulation_query():
    for z in load_zones(DATA_DIR / "zones.json").values():
        for r in z.rules:
            assert r["regulation_query"].startswith("OSHA ") and not any(ch.isdigit() for ch in r["regulation_query"])


def test_results_are_mapped_capped_and_domain_restricted(store):
    fake = FakeTavily(results())
    tools, run = tools_with(store, fake)
    out = call(tools, "lookup_regulation", rule_id="forklift-gap")
    assert len(out["results"]) == 3 and set(out["results"][0]) == {"title", "url", "snippet"}
    assert all(len(r["snippet"]) <= 300 for r in out["results"])
    query, kwargs = fake.calls[0]
    assert query == "OSHA powered industrial trucks pedestrian safety forklift"
    assert kwargs == {"max_results": 3, "include_domains": ["osha.gov"]}
    assert set(run.regulations) == {r["url"] for r in out["results"]}


def test_off_domain_results_are_dropped(store):
    fake = FakeTavily(results(2, host="evil-osha.gov.example.com") + results(1, host="osha.gov"))
    out = call(tools_with(store, fake)[0], "lookup_regulation", rule_id="ppe-vest")
    assert [r["url"] for r in out["results"]] == ["https://osha.gov/p0"]


def test_missing_key_and_mock_mode_and_unknown_rule(store):
    assert call(tools_with(store)[0], "lookup_regulation", rule_id="ppe-vest") == {"error": "TAVILY_API_KEY not set", "results": []}
    mock = call(tools_with(store, mock=True)[0], "lookup_regulation", rule_id="ppe-vest")
    assert mock["results"] == [] and mock["note"] == "mock mode: no regulation lookup"
    assert "error" in call(tools_with(store, FakeTavily([]))[0], "lookup_regulation", rule_id="nope")


def file_report(tools, refs):
    return call(tools, "file_incident_report", title="t", observation_ids=[2], body="b", regulation_refs=refs)


def test_report_citing_an_unretrieved_url_is_rejected(store):
    tools, _ = tools_with(store, FakeTavily(results()))
    out = file_report(tools, [{"title": "Made up", "url": "https://www.osha.gov/invented"}])
    assert "error" in out and store.incidents() == []


def test_report_citing_a_retrieved_url_is_stored_and_listed(store):
    tools, _ = tools_with(store, FakeTavily(results()))
    call(tools, "lookup_regulation", rule_id="forklift-gap")
    out = file_report(tools, [{"title": "my own title", "url": "https://www.osha.gov/p1"}])
    assert out["regulation_refs"] == [{"title": "T1", "url": "https://www.osha.gov/p1"}]  # title comes from the search
    assert store.incidents()[0]["regulation_refs"] == out["regulation_refs"]


def test_report_without_refs_still_works(store):
    out = file_report(tools_with(store)[0], None)
    assert out["incident_id"] and store.incidents()[0]["regulation_refs"] == []


def test_urls_do_not_leak_between_ask_runs(store):
    fake = FakeTavily(results())

    class Script:
        def __init__(self, replies):
            self.replies = replies

        def chat(self, messages, tools=None):
            return self.replies.pop(0)

    def tool_call(name, args):
        return {"content": None, "tool_calls": [{"id": "c", "name": name, "arguments": json.dumps(args)}]}

    lookup = RegulationLookup(api_key="k", factory=lambda: fake)
    first = Script([tool_call("lookup_regulation", {"rule_id": "forklift-gap"}), {"content": "done", "tool_calls": []}])
    Agent(first, store, regulations=lookup).ask("q")
    refs = [{"url": "https://www.osha.gov/p0"}]
    second = Script([tool_call("file_incident_report", {"title": "t", "observation_ids": [2], "body": "b", "regulation_refs": refs}), {"content": "ok", "tool_calls": []}])
    agent = Agent(second, store, regulations=lookup)
    agent.ask("q")
    assert store.incidents() == []  # p0 was only retrieved in the first run


def test_old_incident_rows_get_empty_refs(tmp_path):
    s = Store(tmp_path / "x.db", load_zones(DATA_DIR / "zones.json"))
    s.conn.execute("INSERT INTO incidents (created_ts, title, severity, observation_ids, body) VALUES ('t','a','low','[]','b')")
    assert s.incidents()[0]["regulation_refs"] == []
