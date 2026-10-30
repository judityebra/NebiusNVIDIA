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
CREATE TABLE IF NOT EXISTS incidents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_ts TEXT NOT NULL,
    title TEXT NOT NULL,
    severity TEXT NOT NULL,
    observation_ids TEXT NOT NULL,
    body TEXT NOT NULL
);
"""


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

    def zone_for_camera(self, camera: str) -> Zone | None:
        return next((z for z in self.zones.values() if z.camera == camera), None)

    def add_observation(self, obs: dict[str, Any]) -> int:
        zone = self.zone_for_camera(obs["camera"])
        if zone is None:
            raise ValueError(f"unknown camera {obs['camera']!r}")
        cur = self.conn.execute(
            "INSERT INTO observations (camera, zone_id, ts, summary, objects) VALUES (?, ?, ?, ?, ?)",
            (obs["camera"], zone.id, obs["ts"], obs["summary"], json.dumps(obs.get("objects", []))),
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

    def get_observation(self, obs_id: int) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM observations WHERE id = ?", (obs_id,)).fetchone()
        return self._with_violations(dict(row)) if row else None

    def _with_violations(self, row: dict[str, Any]) -> dict[str, Any]:
        row["objects"] = json.loads(row["objects"])
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

    def add_incident(self, created_ts: str, title: str, severity: str, observation_ids: list[int], body: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO incidents (created_ts, title, severity, observation_ids, body) VALUES (?, ?, ?, ?, ?)",
            (created_ts, title, severity, json.dumps(observation_ids), body),
        )
        self.conn.commit()
        return cur.lastrowid

    def incidents(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM incidents ORDER BY id DESC").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["observation_ids"] = json.loads(d["observation_ids"])
            out.append(d)
        return out
