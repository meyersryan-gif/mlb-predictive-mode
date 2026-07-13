from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


BASE = "https://statsapi.mlb.com/api/v1"
CACHE_DIR = Path("mlb") / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ScheduleGame:
    game_pk: int
    game_date: str
    game_time_utc: str
    home_team: str
    away_team: str
    home_probable_id: Optional[int]
    home_probable_name: Optional[str]
    away_probable_id: Optional[int]
    away_probable_name: Optional[str]


def _cache_path(name: str) -> Path:
    return CACHE_DIR / name


def _get_json(url: str, params: Dict[str, Any], cache_key: str, ttl_seconds: int = 3600) -> Dict[str, Any]:
    p = _cache_path(cache_key)
    if p.exists():
        age = time.time() - p.stat().st_mtime
        if age <= ttl_seconds:
            with p.open("r", encoding="utf-8") as f:
                return json.load(f)

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f)

    return data


def get_schedule_with_probables(
    date_iso: str,
    game_types: str = "R",
    sport_id: int = 1,
) -> List[ScheduleGame]:
    url = f"{BASE}/schedule"
    params = {
        "sportId": sport_id,
        "startDate": date_iso,
        "endDate": date_iso,
        "gameTypes": game_types,
        "hydrate": "probablePitcher",
    }
    cache_key = f"schedule_{date_iso}_{game_types.replace(',', '-')}.json"
    data = _get_json(url, params=params, cache_key=cache_key, ttl_seconds=1800)

    out: List[ScheduleGame] = []
    for d in data.get("dates", []) or []:
        for g in d.get("games", []) or []:
            teams = g.get("teams", {})
            home = teams.get("home", {}).get("team", {}).get("name")
            away = teams.get("away", {}).get("team", {}).get("name")

            home_pp = teams.get("home", {}).get("probablePitcher")
            away_pp = teams.get("away", {}).get("probablePitcher")

            out.append(
                ScheduleGame(
                    game_pk=int(g.get("gamePk")),
                    game_date=d.get("date"),
                    game_time_utc=g.get("gameDate"),
                    home_team=home,
                    away_team=away,
                    home_probable_id=int(home_pp.get("id")) if isinstance(home_pp, dict) and home_pp.get("id") else None,
                    home_probable_name=home_pp.get("fullName") if isinstance(home_pp, dict) else None,
                    away_probable_id=int(away_pp.get("id")) if isinstance(away_pp, dict) and away_pp.get("id") else None,
                    away_probable_name=away_pp.get("fullName") if isinstance(away_pp, dict) else None,
                )
            )
    return out


def get_player_throws(player_id: int) -> Optional[str]:
    url = f"{BASE}/people/{player_id}"
    params = {"hydrate": "currentTeam"}
    cache_key = f"player_{player_id}.json"
    data = _get_json(url, params=params, cache_key=cache_key, ttl_seconds=86400)

    people = data.get("people", [])
    if not people:
        return None
    hand = people[0].get("pitchHand", {}).get("code")
    if hand in ("R", "L"):
        return hand
    return None


def _roster_request(team_id: int, season: int, roster_type: str) -> List[Dict[str, Any]]:
    url = f"{BASE}/teams/{team_id}/roster"
    params = {"season": season, "rosterType": roster_type}
    cache_key = f"roster_{team_id}_{season}_{roster_type}.json"
    data = _get_json(url, params=params, cache_key=cache_key, ttl_seconds=86400)

    out = []
    for r in data.get("roster", []) or []:
        person = r.get("person", {}) or {}
        pid = person.get("id")
        full_name = person.get("fullName")
        if pid and full_name:
            out.append({
                "player_id": pid,
                "full_name": full_name,
            })
    return out


def get_team_roster(team_id: int, season: int) -> List[Dict[str, Any]]:
    """
    Try multiple roster types so early-season / pre-lock roster quirks don't kill mapping.
    Priority:
      1) active
      2) 40Man
      3) depthChart
    Deduplicates by player_id.
    """
    roster_types = ["active", "40Man", "depthChart"]
    seen: Dict[int, Dict[str, Any]] = {}

    for rt in roster_types:
        try:
            rows = _roster_request(team_id, season, rt)
        except Exception:
            rows = []

        for row in rows:
            pid = row.get("player_id")
            if pid and pid not in seen:
                seen[pid] = row

    return list(seen.values())


def get_teams_for_date(date_iso: str, game_types: str = "R", sport_id: int = 1) -> List[Dict[str, Any]]:
    url = f"{BASE}/schedule"
    params = {
        "sportId": sport_id,
        "startDate": date_iso,
        "endDate": date_iso,
        "gameTypes": game_types,
    }
    cache_key = f"teams_{date_iso}_{game_types.replace(',', '-')}.json"
    data = _get_json(url, params=params, cache_key=cache_key, ttl_seconds=1800)

    seen = {}
    for d in data.get("dates", []) or []:
        for g in d.get("games", []) or []:
            teams = g.get("teams", {})
            for side in ("home", "away"):
                t = teams.get(side, {}).get("team", {}) or {}
                tid = t.get("id")
                name = t.get("name")
                if tid and name:
                    seen[int(tid)] = {"team_id": int(tid), "team_name": name}
    return list(seen.values())
