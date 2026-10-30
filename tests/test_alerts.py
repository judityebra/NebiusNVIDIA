import socket
import threading
import time

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

from safefloor import app as app_module
from safefloor.app import create_app
from safefloor.config import DATA_DIR, Settings

JPEG = b"\xff\xd8tiny\xff\xd9"


def make_settings(tmp_path, **kw):
    return Settings(
        None, "http://unused/v1/", "m", "v", tmp_path / "app.db", DATA_DIR / "zones.json", True,
        frames_dir=tmp_path / "frames", **kw,
    )


def upload(client, camera):
    return client.post("/api/frames", data={"camera": camera}, files={"image": ("f.jpg", JPEG, "image/jpeg")})


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(make_settings(tmp_path)))


def test_critical_violation_creates_one_alert_and_one_draft(client):
    obs = upload(client, "cam-03").json()  # mock flags no-entry (critical) in the charging bay
    alerts = client.get("/api/alerts").json()
    assert len(alerts) == 1
    a = alerts[0]
    assert (a["observation_id"], a["rule_id"], a["zone_id"], a["status"]) == (obs["id"], "no-entry", "charging", "draft")
    p = a["payload"]
    assert p["severity"] == "critical" and p["incident_id"] == a["incident_id"] and p["frame_url"].endswith("/frame")
    drafts = [i for i in client.get("/api/incidents").json() if i["status"] == "draft"]
    assert len(drafts) == 1 and drafts[0]["id"] == a["incident_id"] and drafts[0]["severity"] == "critical"
    assert drafts[0]["observation_ids"] == [obs["id"]]


def test_non_critical_and_seed_data_create_nothing(client):
    assert client.get("/api/alerts").json() == []  # seeded history does not alert
    assert upload(client, "cam-01").status_code == 200  # ppe-vest is high, not critical
    assert client.get("/api/alerts").json() == []
    assert all(i["status"] == "filed" for i in client.get("/api/incidents").json())


def test_confirm_flow(client):
    upload(client, "cam-03")
    incident_id = client.get("/api/alerts").json()[0]["incident_id"]
    assert client.post("/api/incidents/9999/confirm").status_code == 404
    ok = client.post(f"/api/incidents/{incident_id}/confirm")
    assert ok.status_code == 200 and ok.json()["status"] == "filed"
    assert client.post(f"/api/incidents/{incident_id}/confirm").status_code == 409
    assert client.get("/api/alerts").json()[0]["status"] == "filed"


def test_agent_filed_incidents_are_filed_already(client):
    client.post("/api/ask", json={"question": "Write an incident report for the worst dock event"})
    incidents = client.get("/api/incidents").json()
    assert incidents and incidents[0]["status"] == "filed"
    assert client.post(f"/api/incidents/{incidents[0]['id']}/confirm").status_code == 409


def test_webhook_gets_the_payload_and_a_failing_one_does_not_break_ingestion(tmp_path):
    sent = []
    (tmp_path / "a").mkdir()
    good = TestClient(
        create_app(
            make_settings(tmp_path / "a", webhook_url="http://hook.invalid/x"),
            hub_sender=lambda url, payload: sent.append((url, payload)),
        )
    )
    upload(good, "cam-03")
    good.app.state.hub.flush()
    assert len(sent) == 1 and sent[0][0] == "http://hook.invalid/x"
    assert sent[0][1]["rule_id"] == "no-entry" and sent[0][1]["alert_id"]

    def boom(url, payload):
        raise ConnectionError("down")

    (tmp_path / "b").mkdir()
    bad = TestClient(create_app(make_settings(tmp_path / "b", webhook_url="http://hook.invalid/x"), hub_sender=boom))
    assert upload(bad, "cam-03").status_code == 200
    bad.app.state.hub.flush()
    assert len(bad.get("/api/alerts").json()) == 1


def test_no_webhook_without_env_even_if_request_asks(tmp_path):
    sent = []
    c = TestClient(create_app(make_settings(tmp_path), hub_sender=lambda u, p: sent.append(u)))
    c.post("/api/frames", data={"camera": "cam-03", "webhook_url": "http://evil.example/x"}, files={"image": ("f.jpg", JPEG, "image/jpeg")})
    c.app.state.hub.flush()
    assert sent == []


def test_old_incident_rows_default_to_filed(tmp_path):
    from safefloor.store import Store, load_zones

    s = Store(tmp_path / "old.db", load_zones(DATA_DIR / "zones.json"))
    s.conn.execute("INSERT INTO incidents (created_ts, title, severity, observation_ids, body) VALUES ('t','a','low','[]','b')")
    assert s.incidents()[0]["status"] == "filed"


@pytest.fixture
def live(tmp_path):
    """A real uvicorn server: TestClient cannot stream an endless SSE response."""
    app = create_app(make_settings(tmp_path))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    assert server.started
    yield f"http://127.0.0.1:{port}", app
    server.should_exit = True
    thread.join(10)


def read_until(lines, wanted, limit=50):
    seen = []
    for line in lines:
        seen.append(line)
        if line.startswith(wanted):
            return seen
        if len(seen) > limit:
            break
    pytest.fail(f"never saw {wanted!r}: {seen}")


def test_sse_delivers_observation_and_alert_after_an_upload(live):
    url, app = live
    with httpx.stream("GET", f"{url}/api/stream", timeout=10) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        assert r.headers["cache-control"] == "no-cache" and r.headers["x-accel-buffering"] == "no"
        lines = r.iter_lines()
        read_until(lines, ": connected")
        up = httpx.post(f"{url}/api/frames", data={"camera": "cam-03"}, files={"image": ("f.jpg", JPEG, "image/jpeg")}, timeout=10)
        assert up.status_code == 200
        read_until(lines, "event: observation")
        data = next(lines)
        assert data.startswith("data: ") and '"zone_id": "charging"' in data
        read_until(lines, "event: alert")
    for _ in range(60):  # the subscriber is cleaned up after the client disconnects
        if app.state.hub.subscriber_count == 0:
            break
        time.sleep(0.1)
    assert app.state.hub.subscriber_count == 0


def test_sse_sends_heartbeats(live, monkeypatch):
    url, _ = live
    monkeypatch.setattr(app_module, "HEARTBEAT_SECONDS", 0.2)
    with httpx.stream("GET", f"{url}/api/stream", timeout=10) as r:
        read_until(r.iter_lines(), ": heartbeat")
