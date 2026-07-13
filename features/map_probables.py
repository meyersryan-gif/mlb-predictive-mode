from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Tuple

import pandas as pd

from mlb.data_sources.mlb_stats_api import ScheduleGame, get_player_throws


def _normalize_name(s: Optional[str]) -> str:
    """
    Normalize pitcher names for matching:
    - strip accents/diacritics
    - lowercase
    - remove punctuation
    - collapse to alnum only
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return "".join(c for c in s if c.isalnum())


def build_probable_index(games: List[ScheduleGame]) -> Dict[Tuple[str, str, str], Dict[str, object]]:
    """
    Build a lookup keyed by (home_team, away_team, normalized_pitcher_name)
    Returns dict with pitcher_id, pitcher_team, opponent_team, home_away.
    """
    idx: Dict[Tuple[str, str, str], Dict[str, object]] = {}

    for g in games:
        if g.home_probable_name:
            key = (g.home_team, g.away_team, _normalize_name(g.home_probable_name))
            idx[key] = {
                "pitcher_id": g.home_probable_id,
                "pitcher_team": g.home_team,
                "opponent_team": g.away_team,
                "home_away": "HOME",
            }
        if g.away_probable_name:
            key = (g.home_team, g.away_team, _normalize_name(g.away_probable_name))
            idx[key] = {
                "pitcher_id": g.away_probable_id,
                "pitcher_team": g.away_team,
                "opponent_team": g.home_team,
                "home_away": "AWAY",
            }
    return idx


def attach_probables_to_pitcher_props(df: pd.DataFrame, games: List[ScheduleGame]) -> pd.DataFrame:
    """
    Attach pitcher_id/team/opponent/home_away onto pitcher props rows.
    Matching strategy:
      1) (home_team, away_team, pitcher_name_normalized) exact match
      2) fallback: try swapping home/away (in case odds feed flips)
    """
    idx = build_probable_index(games)

    pitcher_ids = []
    pitcher_teams = []
    opponent_teams = []
    home_aways = []
    notes = []

    for _, r in df.iterrows():
        home = r.get("home_team")
        away = r.get("away_team")
        pname = _normalize_name(r.get("pitcher_name"))

        hit = idx.get((home, away, pname))
        if not hit:
            hit = idx.get((away, home, pname))  # fallback

        if hit:
            pitcher_ids.append(hit.get("pitcher_id"))
            pitcher_teams.append(hit.get("pitcher_team"))
            opponent_teams.append(hit.get("opponent_team"))
            home_aways.append(hit.get("home_away"))
            notes.append(None)
        else:
            pitcher_ids.append(None)
            pitcher_teams.append(None)
            opponent_teams.append(None)
            home_aways.append(None)
            notes.append("no_probable_match")

    df = df.copy()
    df["pitcher_id"] = pitcher_ids
    df["pitcher_team"] = pitcher_teams
    df["opponent_team"] = opponent_teams
    df["home_away"] = home_aways

    if "notes" not in df.columns:
        df["notes"] = None
    df["notes"] = df["notes"].fillna("").astype(str)
    df["notes"] = df["notes"].where(df["notes"] != "None", "")
    df["notes"] = (df["notes"].str.strip() + " " + pd.Series(notes).fillna("").astype(str)).str.strip()
    df.loc[df["notes"] == "", "notes"] = None

    return df


def attach_pitcher_hand(df: pd.DataFrame, polite_sleep_s: float = 0.15) -> pd.DataFrame:
    """
    Fetch pitchHand for mapped pitcher_id values.
    Uses cached MLB Stats API responses; still sleeps lightly to be polite.
    """
    df = df.copy()
    if "pitcher_id" not in df.columns:
        df["pitcher_hand"] = None
        return df

    unique_ids = sorted({int(x) for x in df["pitcher_id"].dropna().unique()})
    id_to_hand: Dict[int, Optional[str]] = {}

    import time
    for pid in unique_ids:
        hand = get_player_throws(pid)
        id_to_hand[pid] = hand
        time.sleep(polite_sleep_s)

    df["pitcher_hand"] = df["pitcher_id"].apply(lambda x: id_to_hand.get(int(x)) if pd.notna(x) else None)
    return df
