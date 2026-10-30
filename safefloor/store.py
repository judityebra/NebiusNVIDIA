"""SQLite store for observations, rule violations and incident reports."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera TEXT NOT NULL,
    zone_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    summary TEXT NOT NULL,
    objects TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS violations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    observation_id INTEGER NOT NULL REFERENCES observations(id),
    rule_id TEXT NOT NULL,
    severity TEXT NOT NULL,
    confidence REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    observation_id INTEGER NOT NULL REFERENCES observations(id),
    rule_id TEXT NOT NULL,
    zone_id TEXT NOT NULL,
    created_ts TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS incidents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_ts TEXT NOT NULL,
    title TEXT NOT NULL,
    severity TEXT NOT NULL,
    observation_ids TEXT NOT NULL,
    body TEXT NOT NULL
);
"""


# Columns added after the first release: (table, column, type). Applied to existing DBs on open.
MIGRATIONS: list[tuple[str, str, str]] = [
    ("observations", "source", "TEXT"),
    ("observations", "frame_ext", "TEXT"),
    ("observations", "end_ts", "TEXT"),
    ("observations", "clip", "TEXT"),
    ("observations", "frame_count", "INTEGER"),
    ("incidents", "regulation_refs", "TEXT"),
    ("incidents", "status", "TEXT DEFAULT 'filed'"),  # existing rows count as filed
]


@dataclass
class Zone:
    id: str
    name: str
    camera: str
    rules: list[dict[str, str]]

    def rule(self, rule_id: str) -> dict[str, str] | None:
        return next((r for r in self.rules if r["id"] == rule_id), None)


def load_zones(path: Path) -> dict[str, Zone]:
    raw = json.loads(path.read_text())
    return {z["id"]: Zone(**z) for z in raw["zones"]}


class Store:
    def __init__(self, db_path: Path | str, zones: dict[str, Zone]):
        self.zones = zones
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Add any column an older database file is missing, keeping its rows."""
        have: dict[str, set[str]] = {}
        for table, column, ctype in MIGRATIONS:
            if table not in have:
                have[table] = {r["name"] for r in self.conn.execute(f"PRAGMA table_info({table})")}
            if column not in have[table]:
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ctype}")
                have[table].add(column)
        self.conn.commit()

    def zone_for_camera(self, camera: str) -> Zone | None:
        return next((z for z in self.zones.values() if z.camera == camera), None)

    def add_observation(self, obs: dict[str, Any]) -> int:
        zone = self.zone_for_camera(obs["camera"])
        if zone is None:
            raise ValueError(f"unknown camera {obs['camera']!r}")
        cur = self.conn.execute(
            "INSERT INTO observations (camera, zone_id, ts, summary, objects, source, frame_ext, end_ts, clip, frame_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                obs["camera"], zone.id, obs["ts"], obs["summary"], json.dumps(obs.get("objects", [])),
                obs.get("source"), obs.get("frame_ext"), obs.get("end_ts"), obs.get("clip"), obs.get("frame_count"),
            ),
        )
        obs_id = cur.lastrowid
        for v in obs.get("violations", []):
            rule = zone.rule(v["rule_id"])
            if rule is None:
                continue  # the model named a rule this zone does not have
            self.conn.execute(
                "INSERT INTO violations (observation_id, rule_id, severity, confidence) VALUES (?, ?, ?, ?)",
                (obs_id, rule["id"], rule["severity"], float(v.get("confidence", 0.5))),
            )
        self.conn.commit()
        return obs_id

    def observations(
        self,
        zone_id: str | None = None,
        since: str | None = None,
        until: str | None = None,
        query: str | None = None,
        only_violations: bool = False,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """The newest ``limit`` matching observations, in chronological order."""
        where, args = self._where(zone_id, since, until, query, only_violations)
        sql = (
            f"SELECT * FROM (SELECT * FROM observations WHERE {where} ORDER BY ts DESC, id DESC LIMIT ?) "
            "ORDER BY ts, id"
        )
        rows = self.conn.execute(sql, [*args, limit]).fetchall()
        return [self._with_violations(dict(r)) for r in rows]

    def count_observations(
        self,
        zone_id: str | None = None,
        since: str | None = None,
        until: str | None = None,
        query: str | None = None,
        only_violations: bool = False,
    ) -> int:
        where, args = self._where(zone_id, since, until, query, only_violations)
        return self.conn.execute(f"SELECT COUNT(*) FROM observations WHERE {where}", args).fetchone()[0]

    @staticmethod
    def _where(zone_id, since, until, query, only_violations) -> tuple[str, list[Any]]:
        sql = "1=1"
        args: list[Any] = []
        if zone_id:
            sql += " AND zone_id = ?"
            args.append(zone_id)
        if since:
            sql += " AND ts >= ?"
            args.append(since)
        if until:
            sql += " AND ts <= ?"
            args.append(until)
        if query:
            sql += " AND (summary LIKE ? OR objects LIKE ?)"
            args += [f"%{query}%", f"%{query}%"]
        if only_violations:
            sql += " AND id IN (SELECT observation_id FROM violations)"
        return sql, args

    def clear_frame(self, obs_id: int) -> None:
        self.conn.execute("UPDATE observations SET frame_ext = NULL WHERE id = ?", (obs_id,))
        self.conn.commit()

    def get_observation(self, obs_id: int) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM observations WHERE id = ?", (obs_id,)).fetchone()
        return self._with_violations(dict(row)) if row else None

    def _with_violations(self, row: dict[str, Any]) -> dict[str, Any]:
        row["objects"] = json.loads(row["objects"])
        row["source"] = row.get("source") or "unknown"
        row["has_frame"] = bool(row.get("frame_ext"))
        vs = self.conn.execute(
            "SELECT rule_id, severity, confidence FROM violations WHERE observation_id = ?", (row["id"],)
        ).fetchall()
        zone = self.zones[row["zone_id"]]
        row["violations"] = [
            {**dict(v), "rule": (zone.rule(v["rule_id"]) or {}).get("text", "")} for v in vs
        ]
        return row

    def violation_counts(self, zone_id: str | None = None) -> list[dict[str, Any]]:
        sql = (
            "SELECT o.zone_id, v.rule_id, v.severity, COUNT(*) AS n FROM violations v "
            "JOIN observations o ON o.id = v.observation_id"
        )
        args: list[Any] = []
        if zone_id:
            sql += " WHERE o.zone_id = ?"
            args.append(zone_id)
        sql += " GROUP BY o.zone_id, v.rule_id ORDER BY n DESC"
        return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def add_incident(
        self,
        created_ts: str,
        title: str,
        severity: str,
        observation_ids: list[int],
        body: str,
        regulation_refs: list[dict[str, str]] | None = None,
        status: str = "filed",
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO incidents (created_ts, title, severity, observation_ids, body, regulation_refs, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (created_ts, title, severity, json.dumps(observation_ids), body, json.dumps(regulation_refs or []), status),
        )
        self.conn.commit()
        return cur.lastrowid

    @staticmethod
    def _incident(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["observation_ids"] = json.loads(d["observation_ids"])
        d["regulation_refs"] = json.loads(d["regulation_refs"] or "[]")
        d["status"] = d.get("status") or "filed"
        return d

    def incidents(self) -> list[dict[str, Any]]:
        return [self._incident(r) for r in self.conn.execute("SELECT * FROM incidents ORDER BY id DESC").fetchall()]

    def get_incident(self, incident_id: int) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,)).fetchone()
        return self._incident(row) if row else None

    def confirm_incident(self, incident_id: int) -> None:
        self.conn.execute("UPDATE incidents SET status = 'filed' WHERE id = ?", (incident_id,))
        self.conn.commit()

    def add_alert(self, observation_id: int, rule_id: str, zone_id: str, created_ts: str, payload: dict[str, Any]) -> int:
        cur = self.conn.execute(
            "INSERT INTO alerts (observation_id, rule_id, zone_id, created_ts, payload) VALUES (?, ?, ?, ?, ?)",
            (observation_id, rule_id, zone_id, created_ts, json.dumps(payload)),
        )
        self.conn.commit()
        return cur.lastrowid

    def _alert(self, row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d["payload"] = json.loads(d["payload"])
        d["incident_id"] = d["payload"].get("incident_id")
        incident = self.get_incident(d["incident_id"]) if d["incident_id"] else None
        d["status"] = incident["status"] if incident else "unknown"
        return d

    def get_alert(self, alert_id: int) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,)).fetchone()
        return self._alert(row) if row else None

    def alerts(self) -> list[dict[str, Any]]:
        return [self._alert(r) for r in self.conn.execute("SELECT * FROM alerts ORDER BY id DESC").fetchall()]
