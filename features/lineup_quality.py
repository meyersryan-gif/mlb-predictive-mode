from __future__ import annotations

import time
from typing import Optional

import pandas as pd
import requests

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

HIGH_K_THRESH = 0.28
LOW_K_THRESH  = 0.16

_BATTER_K_CACHE: dict = {}
_TEAM_K_CACHE:   dict = {}


def _get_json(url: str, params: dict = None) -> dict:
    try:
        r = requests.get(url, params=params or {}, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def _fetch_batter_k_pct(player_id: int, season: int) -> Optional[float]:
    if player_id in _BATTER_K_CACHE:
        return _BATTER_K_CACHE[player_id]
    url    = MLB_API_BASE + f"/people/{player_id}/stats"
    params = {"stats": "season", "group": "hitting", "season": season, "sportIds": 1}
    data   = _get_json(url, params)
    try:
        splits = data.get("stats", [{}])[0].get("splits", [])
        if not splits:
            return None
        stat  = splits[0].get("stat", {})
        ab    = int(stat.get("atBats",      0))
        so    = int(stat.get("strikeOuts",  0))
        bb    = int(stat.get("baseOnBalls", 0))
        hbp   = int(stat.get("hitByPitch",  0))
        pa    = ab + bb + hbp
        if pa < 30:
            return None
        k_pct = so / pa
        _BATTER_K_CACHE[player_id] = round(k_pct, 4)
        return _BATTER_K_CACHE[player_id]
    except Exception:
        return None


def _fetch_confirmed_lineup(game_pk: int) -> Optional[list]:
    url  = MLB_API_BASE + f"/game/{game_pk}/boxscore"
    data = _get_json(url)
    teams = data.get("teams", {})
    for side in ("home", "away"):
        batting_order = teams.get(side, {}).get("battingOrder", [])
        if batting_order:
            return [int(pid) for pid in batting_order[:9]]
    return None


def _fetch_team_avg_k_pct(team_name: str, season: int) -> float:
    cache_key = f"{team_name}_{season}"
    if cache_key in _TEAM_K_CACHE:
        return _TEAM_K_CACHE[cache_key]

    url    = MLB_API_BASE + "/teams"
    params = {"sportId": 1, "season": season}
    data   = _get_json(url, params)

    team_id    = None
    name_lower = str(team_name).lower()
    for team in data.get("teams", []):
        t_name = str(team.get("name", "")).lower()
        t_nick = str(team.get("teamName", "")).lower()
        if t_name == name_lower or t_nick in name_lower or name_lower in t_name:
            team_id = team.get("id")
            break

    if not team_id:
        return 0.215

    url2    = MLB_API_BASE + "/stats"
    params2 = {"stats": "season", "group": "hitting", "season": season, "teamId": team_id, "sportId": 1}
    data2   = _get_json(url2, params2)

    try:
        splits = data2.get("stats", [{}])[0].get("splits", [])
        if not splits:
            return 0.215
        stat  = splits[0].get("stat", {})
        ab    = int(stat.get("atBats",      0))
        so    = int(stat.get("strikeOuts",  0))
        bb    = int(stat.get("baseOnBalls", 0))
        hbp   = int(stat.get("hitByPitch",  0))
        pa    = ab + bb + hbp
        k_pct = (so / pa) if pa > 0 else 0.215
        _TEAM_K_CACHE[cache_key] = round(k_pct, 4)
        return _TEAM_K_CACHE[cache_key]
    except Exception:
        return 0.215


def _compute_lineup_metrics(k_pcts: list) -> dict:
    if not k_pcts:
        return {"lineup_k_pct_avg": 0.215, "lineup_high_k_count": 4, "lineup_low_k_count": 2}
    avg_k        = sum(k_pcts) / len(k_pcts)
    high_k_count = sum(1 for k in k_pcts if k >= HIGH_K_THRESH)
    low_k_count  = sum(1 for k in k_pcts if k <= LOW_K_THRESH)
    return {
        "lineup_k_pct_avg":    round(avg_k, 4),
        "lineup_high_k_count": high_k_count,
        "lineup_low_k_count":  low_k_count,
    }


def attach_lineup_quality(df: pd.DataFrame, season: int = 2026) -> pd.DataFrame:
    out = df.copy()

    for col in ["lineup_status", "lineup_k_pct_avg", "lineup_high_k_count", "lineup_low_k_count"]:
        if col not in out.columns:
            out[col] = None

    lineup_cache: dict = {}
    confirmed = estimated = unavailable = 0

    for idx, row in out.iterrows():
        opp_team  = str(row.get("opponent_team", "") or "")
        game_id   = str(row.get("game_id",       "") or "")
        cache_key = f"{game_id}_{opp_team}"

        if cache_key in lineup_cache:
            metrics = lineup_cache[cache_key]
        else:
            metrics = None

            game_pk = None
            try:
                game_pk = int(game_id)
            except (ValueError, TypeError):
                pass

            if game_pk:
                batter_ids = _fetch_confirmed_lineup(game_pk)
                if batter_ids:
                    k_pcts = []
                    for bid in batter_ids:
                        k = _fetch_batter_k_pct(bid, season)
                        if k is not None:
                            k_pcts.append(k)
                        time.sleep(0.05)
                    if len(k_pcts) >= 6:
                        metrics = _compute_lineup_metrics(k_pcts)
                        metrics["lineup_status"] = "confirmed"
                        confirmed += 1

            if metrics is None and opp_team:
                team_k        = _fetch_team_avg_k_pct(opp_team, season)
                avg_k         = team_k
                high_k_count  = round(9 * max(0, (avg_k - 0.16) / (0.32 - 0.16)))
                low_k_count   = round(9 * max(0, (0.22 - avg_k)  / (0.22 - 0.16)))
                metrics = {
                    "lineup_status":       "estimated",
                    "lineup_k_pct_avg":    round(avg_k, 4),
                    "lineup_high_k_count": int(high_k_count),
                    "lineup_low_k_count":  int(low_k_count),
                }
                estimated += 1
                time.sleep(0.1)

            if metrics is None:
                metrics = {
                    "lineup_status":       "unavailable",
                    "lineup_k_pct_avg":    0.215,
                    "lineup_high_k_count": 4,
                    "lineup_low_k_count":  2,
                }
                unavailable += 1

            lineup_cache[cache_key] = metrics

        out.at[idx, "lineup_status"]       = metrics.get("lineup_status")
        out.at[idx, "lineup_k_pct_avg"]    = metrics.get("lineup_k_pct_avg")
        out.at[idx, "lineup_high_k_count"] = metrics.get("lineup_high_k_count")
        out.at[idx, "lineup_low_k_count"]  = metrics.get("lineup_low_k_count")

    print(f"[lineup_quality] confirmed={confirmed}  estimated={estimated}  unavailable={unavailable}")
    return out


def _norm_team(name: str) -> str:
    """Normalize team name to last word (city-stripped) for fuzzy matching."""
    if not name:
        return ""
    # Use last word — e.g. "Oakland Athletics" -> "athletics"
    return str(name).strip().lower().split()[-1]


def attach_hitter_lineup_status(
    hitter_df: pd.DataFrame,
    games,
    season: int = 2026,
) -> pd.DataFrame:
    """
    Attach is_starting, batting_order_spot, and lineup_status to hitter rows
    by matching batter_id against the confirmed MLB boxscore lineup.

    Falls back to is_starting=None / lineup_status='not_posted' if lineup
    is not yet posted (pre-game) or batter_id is missing.

    Call this at the end of attach_hitter_context, after roster mapping.
    """
    out = hitter_df.copy()

    for col in ["is_starting", "batting_order_spot", "lineup_status"]:
        if col not in out.columns:
            out[col] = None

    # Build game_pk lookup two ways:
    # 1) exact: (home_team, away_team) -> game_pk
    # 2) normalized: (norm_home, norm_away) -> game_pk  (handles "Athletics" vs "Oakland Athletics")
    game_pk_map: dict = {}
    game_pk_map_norm: dict = {}
    print(f"[hitter_lineup_debug] games_received={len(games) if games else 0}")
    for g in games:
        try:
            pk = int(g.game_pk)
            game_pk_map[(g.home_team, g.away_team)] = pk
            game_pk_map_norm[(_norm_team(g.home_team), _norm_team(g.away_team))] = pk
        except Exception as e:
            print(f"[hitter_lineup_debug] game_pk_map_fail type={type(g)} attrs={dir(g)} err={e}")
            break
    print(f"[hitter_lineup_debug] game_pk_map_size={len(game_pk_map)}")
    if game_pk_map:
        sample = list(game_pk_map.items())[:2]
        print(f"[hitter_lineup_debug] sample={sample}")
    # Also check first hitter row team names
    if len(hitter_df):
        r0 = hitter_df.iloc[0]
        print(f"[hitter_lineup_debug] first_hitter home={r0.get('home_team')} away={r0.get('away_team')}")

    # Cache confirmed lineups per game_pk: {game_pk: {player_id: order_spot}}
    lineup_cache: dict = {}
    confirmed_games = 0
    unavailable_games = 0

    def _get_lineup_for_game(game_pk: int) -> dict:
        """
        Returns {player_id: batting_order_spot (1-9)} or empty dict.
        Tries two endpoints:
          1) /schedule?gamePk=X&hydrate=lineups  -- pre-game lineups if posted
          2) /game/{game_pk}/boxscore            -- live/post-game batting order
        """
        if game_pk in lineup_cache:
            return lineup_cache[game_pk]

        result = {}

        # --- Strategy 1: pre-game lineups hydration ---
        try:
            url1 = MLB_API_BASE + "/schedule"
            params1 = {"gamePk": game_pk, "hydrate": "lineups"}
            data1 = _get_json(url1, params1)
            for d in data1.get("dates", []):
                for g in d.get("games", []):
                    lineups = g.get("lineups", {}) or {}
                    for side in ("homePlayers", "awayPlayers"):
                        players = lineups.get(side, []) or []
                        for entry in players:
                            pid = entry.get("id") or (entry.get("person", {}) or {}).get("id")
                            spot = entry.get("battingOrder") or entry.get("lineupPosition")
                            if pid and spot:
                                try:
                                    result[int(pid)] = int(spot)
                                except Exception:
                                    pass
        except Exception:
            pass

        # --- Strategy 2: boxscore batting order (live/post-game) ---
        if not result:
            try:
                url2 = MLB_API_BASE + f"/game/{game_pk}/boxscore"
                data2 = _get_json(url2)
                teams = data2.get("teams", {})
                for side in ("home", "away"):
                    order = teams.get(side, {}).get("battingOrder", [])
                    for spot, pid in enumerate(order[:9], start=1):
                        try:
                            result[int(pid)] = spot
                        except Exception:
                            pass
            except Exception:
                pass

        lineup_cache[game_pk] = result
        if game_pk == list(lineup_cache.keys())[0]:  # debug first game only
            print(f"[hitter_lineup_debug] game_pk={game_pk} result_size={len(result)}")
        return result

    confirmed = estimated = unavailable = 0

    for idx, row in out.iterrows():
        home_team  = row.get("home_team")
        away_team  = row.get("away_team")
        batter_id  = row.get("batter_id")
        batter_team = row.get("batter_team")

        game_pk = game_pk_map.get((home_team, away_team))
        if game_pk is None:
            game_pk = game_pk_map_norm.get((_norm_team(home_team), _norm_team(away_team)))

        if not game_pk:
            out.at[idx, "is_starting"]       = None
            out.at[idx, "batting_order_spot"] = None
            out.at[idx, "lineup_status"]      = "unavailable"
            unavailable += 1
            continue

        lineup = _get_lineup_for_game(game_pk)

        if not lineup:
            # Lineup not posted yet
            out.at[idx, "is_starting"]       = None
            out.at[idx, "batting_order_spot"] = None
            out.at[idx, "lineup_status"]      = "not_posted"
            unavailable += 1
            continue

        try:
            bid_int = int(float(batter_id)) if batter_id is not None and str(batter_id) != "nan" else None
        except Exception:
            bid_int = None
        if bid_int and bid_int in lineup:
            spot = lineup[bid_int]
            out.at[idx, "is_starting"]       = True
            out.at[idx, "batting_order_spot"] = spot
            out.at[idx, "lineup_status"]      = "confirmed"
            confirmed += 1
        else:
            # Batter not in lineup — either bench or ID mismatch
            out.at[idx, "is_starting"]       = False
            out.at[idx, "batting_order_spot"] = None
            out.at[idx, "lineup_status"]      = "not_in_lineup"
            estimated += 1

    print(f"[hitter_lineup] confirmed={confirmed}  not_in_lineup={estimated}  unavailable={unavailable}  total={len(out)}")
    return out