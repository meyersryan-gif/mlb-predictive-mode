"""
mlb/features/project_lineups.py
================================
Projects batting orders for teams whose lineups haven't been posted yet.
Uses last N games of lineup history from the MLB Stats API to determine
each player's most likely batting spot and starting probability.

Used as a fallback in attach_hitter_lineup_status when:
  - lineup_status == "not_posted"
  - is_starting is None

Output fields (same as confirmed lineup):
  is_starting         : True if player starts 70%+ of recent games
  batting_order_spot  : most common batting spot (1-9)
  lineup_status       : "projected"
  lineup_confidence   : float 0-1, how consistent the projection is
"""
from __future__ import annotations

import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
PROJECTION_LOOKBACK_DAYS = 14   # how many days back to look
PROJECTION_MIN_GAMES     = 3    # min games needed to project
STARTER_THRESHOLD        = 0.60 # player must start 60%+ of games to be projected as starter


def _get_json(url: str, params: dict = None) -> dict:
    try:
        r = requests.get(url, params=params or {}, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def _get_team_id(team_name: str, season: int) -> Optional[int]:
    """Resolve team name to MLB team_id."""
    data = _get_json(f"{MLB_API_BASE}/teams", {"sportId": 1, "season": season})
    name_lower = str(team_name).lower()
    for t in data.get("teams", []):
        t_name = str(t.get("name", "")).lower()
        t_nick = str(t.get("teamName", "")).lower()
        if t_name == name_lower or t_nick in name_lower or name_lower in t_name:
            return t.get("id")
    return None


def _get_recent_game_pks(team_id: int, date_iso: str, lookback_days: int = 14) -> List[int]:
    """Get game_pks for a team's recent completed games."""
    end_dt   = datetime.strptime(date_iso, "%Y-%m-%d") - timedelta(days=1)
    start_dt = end_dt - timedelta(days=lookback_days)
    params = {
        "teamId":    team_id,
        "startDate": start_dt.strftime("%Y-%m-%d"),
        "endDate":   end_dt.strftime("%Y-%m-%d"),
        "sportId":   1,
        "gameTypes": "R",
    }
    data = _get_json(f"{MLB_API_BASE}/schedule", params)
    pks = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            status = g.get("status", {}).get("abstractGameState", "")
            if status == "Final":
                try:
                    pks.append(int(g["gamePk"]))
                except Exception:
                    pass
    return pks


def _get_lineup_from_boxscore(game_pk: int, team_id: int) -> List[Tuple[int, int]]:
    """
    Returns list of (player_id, batting_order_spot) for a team in a completed game.
    batting_order_spot is 1-9.
    """
    data  = _get_json(f"{MLB_API_BASE}/game/{game_pk}/boxscore")
    teams = data.get("teams", {})
    result = []
    for side in ("home", "away"):
        t_data = teams.get(side, {})
        t_info = t_data.get("team", {})
        if t_info.get("id") == team_id:
            order = t_data.get("battingOrder", [])
            for spot, pid in enumerate(order[:9], start=1):
                try:
                    result.append((int(pid), spot))
                except Exception:
                    pass
            break
    return result


def build_team_lineup_projection(
    team_name: str,
    date_iso: str,
    season: int = 2026,
    lookback_days: int = PROJECTION_LOOKBACK_DAYS,
    min_games: int = PROJECTION_MIN_GAMES,
) -> Dict[int, Dict]:
    """
    Build a projected lineup for a team based on recent game history.

    Returns dict keyed by player_id:
    {
        player_id: {
            "batting_order_spot": int,       # most common spot
            "start_rate":         float,     # fraction of games started
            "spot_consistency":   float,     # fraction of starts in this spot
            "games_seen":         int,
            "is_projected_starter": bool,
        }
    }
    Returns empty dict if insufficient data.
    """
    team_id = _get_team_id(team_name, season)
    if not team_id:
        return {}

    game_pks = _get_recent_game_pks(team_id, date_iso, lookback_days)
    if not game_pks:
        return {}

    # Collect appearances per player
    # player_id -> list of batting spots (or None if didn't start)
    appearances: Dict[int, List[Optional[int]]] = defaultdict(list)
    total_games = len(game_pks)

    for pk in game_pks[-10:]:  # cap at last 10 to avoid stale data
        lineup = _get_lineup_from_boxscore(pk, team_id)
        starters = {pid for pid, _ in lineup}
        for pid, spot in lineup:
            appearances[pid].append(spot)
        time.sleep(0.05)

    if not appearances:
        return {}

    projection = {}
    for pid, spots in appearances.items():
        games_seen   = len(spots)
        start_rate   = games_seen / min(total_games, 10)

        if games_seen < min_games:
            continue

        spot_counts      = Counter(spots)
        most_common_spot = spot_counts.most_common(1)[0][0]
        spot_consistency = spot_counts[most_common_spot] / games_seen

        projection[pid] = {
            "batting_order_spot":    most_common_spot,
            "start_rate":            round(start_rate, 3),
            "spot_consistency":      round(spot_consistency, 3),
            "games_seen":            games_seen,
            "is_projected_starter":  start_rate >= STARTER_THRESHOLD,
        }

    return projection


# ---------------------------------------------------------------------------
# Cache so we don't re-fetch the same team multiple times per run
# ---------------------------------------------------------------------------
_PROJECTION_CACHE: Dict[str, Dict] = {}


def get_cached_projection(
    team_name: str,
    date_iso: str,
    season: int = 2026,
) -> Dict[int, Dict]:
    key = f"{team_name}_{date_iso}"
    if key not in _PROJECTION_CACHE:
        _PROJECTION_CACHE[key] = build_team_lineup_projection(
            team_name, date_iso, season
        )
    return _PROJECTION_CACHE[key]


def attach_projected_lineup_status(
    hitter_df: pd.DataFrame,
    date_iso: str,
    season: int = 2026,
) -> pd.DataFrame:
    """
    For hitter rows where is_starting is None (lineup not yet posted),
    fill in projected lineup data from recent game history.

    Only fills rows where lineup_status == "not_posted" or is_starting is None.
    Adds a "projected" lineup_status and a confidence penalty flag.

    Call this AFTER attach_hitter_lineup_status in map_hitters.py.
    """
    out = df = hitter_df.copy()

    # Only process rows that still need lineup data
    needs_projection = (
        out["is_starting"].isna() |
        out["lineup_status"].isin(["not_posted", "unavailable", None])
    )

    if not needs_projection.any():
        print("[lineup_project] No rows need projection — all confirmed")
        return out

    n_needs = needs_projection.sum()
    print(f"[lineup_project] Projecting lineups for {n_needs} hitter rows")

    projected = 0
    not_found = 0

    for idx, row in out[needs_projection].iterrows():
        batter_id   = row.get("batter_id")
        batter_team = row.get("batter_team")

        if not batter_team:
            not_found += 1
            continue

        proj = get_cached_projection(batter_team, date_iso, season)

        if not proj:
            not_found += 1
            continue

        try:
            bid = int(float(batter_id)) if batter_id and str(batter_id) != "nan" else None
        except Exception:
            bid = None

        if bid and bid in proj:
            p = proj[bid]
            out.at[idx, "is_starting"]       = p["is_projected_starter"]
            out.at[idx, "batting_order_spot"] = p["batting_order_spot"] if p["is_projected_starter"] else None
            out.at[idx, "lineup_status"]      = "projected"
            # Store projection metadata
            if "lineup_confidence" not in out.columns:
                out["lineup_confidence"] = None
            out.at[idx, "lineup_confidence"] = round(
                p["start_rate"] * p["spot_consistency"], 3
            )
            projected += 1
        else:
            # Batter not seen in recent games — likely bench/new player
            out.at[idx, "is_starting"]  = False
            out.at[idx, "lineup_status"] = "not_in_recent_lineups"
            not_found += 1

    print(f"[lineup_project] projected={projected}  not_found={not_found}  total_processed={n_needs}")
    return out