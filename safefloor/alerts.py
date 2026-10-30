"""Closed loop without hardware: critical violations raise alerts, draft incidents, a webhook and SSE events."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Callable

from .store import Store

log = logging.getLogger("safefloor.alerts")


def post_webhook(url: str, payload: dict[str, Any]) -> None:
    import httpx

    httpx.post(url, json=payload, timeout=3.0).raise_for_status()


class AlertHub:
    """Creates alerts and draft incidents for critical violations and fans events out to SSE clients.

    The webhook URL comes only from the environment (``SAFEFLOOR_WEBHOOK_URL``), never from a request.
    """

    def __init__(
        self,
        store: Store,
        webhook_url: str | None = None,
        sender: Callable[[str, dict[str, Any]], None] = post_webhook,
    ):
        self.store = store
        self.webhook_url = webhook_url
        self.sender = sender
        self._subscribers: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = []
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []

    # --- live feed -------------------------------------------------------------------------
    def subscribe(self) -> asyncio.Queue:
        """Call from the event loop that will consume the queue."""
        q: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._subscribers.append((asyncio.get_running_loop(), q))
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            self._subscribers = [(loop, s) for loop, s in self._subscribers if s is not q]

    def publish(self, event: str, data: Any) -> None:
        with self._lock:
            targets = list(self._subscribers)
        for loop, q in targets:
            try:
                loop.call_soon_threadsafe(q.put_nowait, (event, data))
            except RuntimeError:  # that client's loop is already closed
                self.unsubscribe(q)

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    # --- ingestion hook --------------------------------------------------------------------
    def on_observation(self, obs: dict[str, Any]) -> list[dict[str, Any]]:
        """Announce a stored observation; raise one alert and one draft incident per critical violation."""
        self.publish("observation", obs)
        alerts = []
        zone = self.store.zones[obs["zone_id"]]
        for v in obs["violations"]:
            if v["severity"] != "critical":  # severity comes from zones.json, never from the model
                continue
            created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            incident_id = self.store.add_incident(
                created,
                f"[DRAFT] Critical: {v['rule']} ({zone.name})",
                "critical",
                [obs["id"]],
                (
                    f"What happened: {obs['summary']}\n"
                    f"Where/when: {zone.name} ({obs['camera']}) at {obs['ts']}\n"
                    f"Rule: {v['rule']} [{v['rule_id']}, critical], model confidence {round(v['confidence'] * 100)}%\n"
                    "Status: draft raised automatically; a supervisor must confirm it."
                ),
                status="draft",
            )
            payload = {
                "type": "critical_violation",
                "observation_id": obs["id"],
                "incident_id": incident_id,
                "rule_id": v["rule_id"],
                "rule": v["rule"],
                "severity": "critical",
                "zone_id": zone.id,
                "zone_name": zone.name,
                "camera": obs["camera"],
                "ts": obs["ts"],
                "summary": obs["summary"],
                "confidence": v["confidence"],
                "source": obs.get("source"),
                "frame_url": f"/api/observations/{obs['id']}/frame" if obs.get("has_frame") else None,
            }
            alert_id = self.store.add_alert(obs["id"], v["rule_id"], zone.id, created, {**payload})
            payload["alert_id"] = alert_id
            alert = self.store.get_alert(alert_id)
            alerts.append(alert)
            self.publish("alert", alert)
            self.publish("incident", {"id": incident_id, "status": "draft"})
            self._notify(payload)
        return alerts

    def _notify(self, payload: dict[str, Any]) -> None:
        if not self.webhook_url:
            return
        t = threading.Thread(target=self._send, args=(self.webhook_url, payload), daemon=True)
        self._threads.append(t)
        t.start()

    def _send(self, url: str, payload: dict[str, Any]) -> None:
        try:
            self.sender(url, payload)
        except Exception as exc:  # a dead webhook must never break ingestion
            log.warning("webhook failed: %s: %s", type(exc).__name__, exc)

    def flush(self, timeout: float = 5.0) -> None:
        """Wait for in-flight webhook threads (used by tests)."""
        threads, self._threads = self._threads, []
        for t in threads:
            t.join(timeout)


def sse_format(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"
