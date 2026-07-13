from __future__ import annotations

"""
team_implied_runs.py — Derives team implied run totals from game totals and moneylines.

Uses the Odds API h2h and totals markets (separate from pitcher prop markets).
Computes implied runs per team per game and converts to slate percentiles.

Logic:
  1. Fetch game total (e.g. 8.5) from totals market
  2. Fetch moneyline for each team from h2h market
  3. Convert moneylines to implied win probabilities
  4. Normalize probabilities to split the total between teams
  5. team_implied_runs = total * team_share
  6. Convert to percentile across today's slate

Example:
  CLE -172, LAA +144, total 7.5
  CLE implied prob = 172/(172+100) = 63.2%
  LAA implied prob = 100/(144+100) = 41.0%
  Normalized: CLE = 63.2/(63.2+41.0) = 60.7%
  CLE implied runs = 7.5 * 0.607 = 4.55
  LAA implied runs = 7.5 * 0.393 = 2.95
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
import requests


def _american_to_implied_prob(odds: float) -> float:
    if odds >= 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def _fetch_game_lines(event_id: str, api_key: str) -> dict:
    """Fetch h2h and totals for a single game from Odds API."""
    url = f"https://api.the-odds-api.com/v4/sports/baseball_mlb/events/{event_id}/odds"
    params = {
        "apiKey":     api_key,
        "regions":    "us",
        "markets":    "h2h,totals",
        "bookmakers": "fanduel,draftkings,betmgm",
        "oddsFormat": "american",
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def _parse_game_lines(data: dict, home_team: str, away_team: str) -> dict:
    """
    Parse h2h and totals from Odds API response.
    Returns dict with home_implied_runs, away_implied_runs.
    """
    books = data.get("bookmakers") or []

    # Prefer FanDuel, fall back to DraftKings, then first available
    book = None
    for preferred in ("fanduel", "draftkings", "betmgm"):
        book = next((b for b in books if b.get("key") == preferred), None)
        if book:
            break
    if not book and books:
        book = books[0]
    if not book:
        return {}

    markets = {m.get("key"): m.get("outcomes", []) for m in (book.get("markets") or [])}

    # Parse total
    total = None
    for outcome in markets.get("totals", []):
        if outcome.get("name") == "Over":
            total = float(outcome.get("point", 0))
            break

    if not total:
        return {}

    # Parse moneylines
    h2h = markets.get("h2h", [])
    home_odds = away_odds = None
    home_norm = home_team.lower().strip()
    away_norm = away_team.lower().strip()

    for outcome in h2h:
        name = str(outcome.get("name", "")).lower().strip()
        price = float(outcome.get("price", 0))
        if name == home_norm or home_norm in name or name in home_norm:
            home_odds = price
        elif name == away_norm or away_norm in name or name in away_norm:
            away_odds = price

    if home_odds is None or away_odds is None:
        # Fallback — assign evenly
        return {
            "total":              total,
            "home_implied_runs":  round(total * 0.5, 2),
            "away_implied_runs":  round(total * 0.5, 2),
        }

    home_prob = _american_to_implied_prob(home_odds)
    away_prob = _american_to_implied_prob(away_odds)
    total_prob = home_prob + away_prob
    home_share = home_prob / total_prob
    away_share = away_prob / total_prob

    return {
        "total":              total,
        "home_moneyline":     home_odds,
        "away_moneyline":     away_odds,
        "home_implied_runs":  round(total * home_share, 2),
        "away_implied_runs":  round(total * away_share, 2),
    }


def fetch_team_implied_runs(events: list[dict], api_key: str) -> dict[str, dict]:
    """
    Fetch implied runs for all games on today's slate.
    Returns dict keyed by game_id with home/away implied runs.
    """
    results = {}
    fetched = failed = 0

    for event in events:
        event_id  = event.get("id", "")
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")

        if not event_id:
            continue

        data   = _fetch_game_lines(event_id, api_key)
        parsed = _parse_game_lines(data, home_team, away_team)

        if parsed:
            results[event_id] = {
                "home_team":         home_team,
                "away_team":         away_team,
                "total":             parsed.get("total"),
                "home_implied_runs": parsed.get("home_implied_runs"),
                "away_implied_runs": parsed.get("away_implied_runs"),
            }
            fetched += 1
        else:
            failed += 1

    print(f"[team_implied_runs] fetched={fetched}  failed={failed}  total_games={len(events)}")
    return results


def attach_team_implied_runs(
    df: pd.DataFrame,
    events: list[dict],
    api_key: str,
) -> pd.DataFrame:
    """
    Attach team_implied_runs and pctl_team_implied_runs to props DataFrame.
    Works for both pitcher props and hitter lotto DataFrames.
    Uses home_away and home_team/away_team to assign correct team runs.
    """
    out = df.copy()

    if not api_key:
        print("[team_implied_runs] WARNING: No API key — skipping")
        return out

    game_lines = fetch_team_implied_runs(events, api_key)

    if not game_lines:
        print("[team_implied_runs] No game lines retrieved — skipping")
        return out

    # Build lookup: game_id -> {home_team_name: runs, away_team_name: runs}
    def _get_implied_runs(row) -> Optional[float]:
        game_id   = str(row.get("game_id", "") or "")
        home_away = str(row.get("home_away", "") or "").upper()

        if game_id not in game_lines:
            return None

        gl = game_lines[game_id]
        if home_away == "HOME":
            return gl.get("home_implied_runs")
        elif home_away == "AWAY":
            return gl.get("away_implied_runs")
        else:
            # Try matching by team name
            batter_team   = str(row.get("batter_team",   "") or "").lower()
            pitcher_team  = str(row.get("pitcher_team",  "") or "").lower()
            home_team     = str(gl.get("home_team",      "") or "").lower()
            away_team     = str(gl.get("away_team",      "") or "").lower()

            team = batter_team or pitcher_team
            if team and (team in home_team or home_team in team):
                return gl.get("home_implied_runs")
            elif team and (team in away_team or away_team in team):
                return gl.get("away_implied_runs")
            return None

    out["team_implied_runs"] = out.apply(_get_implied_runs, axis=1)

    # Convert to slate percentile
    implied = pd.to_numeric(out["team_implied_runs"], errors="coerce")
    if implied.notna().sum() > 1:
        out["pctl_team_implied_runs"] = (
            implied.rank(pct=True, method="average") * 100.0
        ).fillna(50.0).round(1)
    else:
        out["pctl_team_implied_runs"] = 50.0

    populated = implied.notna().sum()
    print(f"[team_implied_runs] populated={populated}/{len(out)} rows")

    return out
