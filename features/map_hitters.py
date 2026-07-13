from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

from mlb.data_sources.mlb_stats_api import ScheduleGame, get_player_throws, get_team_roster, get_teams_for_date
from mlb.features.hitter_baselines import load_hitter_baseline_stats, attach_hitter_baseline_stats
from mlb.features.pitcher_suppression import load_pitcher_suppression_stats
from mlb.features.lineup_quality import attach_hitter_lineup_status
from mlb.features.project_lineups import attach_projected_lineup_status


def _norm_name(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _canon_name(s: pd.Series) -> pd.Series:
    def one(v):
        if pd.isna(v):
            return ""
        v = str(v).strip().lower()
        if "," in v:
            parts = [p.strip() for p in v.split(",", 1)]
            if len(parts) == 2:
                v = f"{parts[1]} {parts[0]}"
        v = re.sub(r"[^a-z ]", "", v)
        v = re.sub(r"\s+", " ", v).strip()
        return v.replace(" ", "")
    return s.astype(str).apply(one)


def _build_roster_name_index(date_iso: str, season: int, game_types: str = "R") -> Dict[Tuple[str, str], Dict[str, object]]:
    teams = get_teams_for_date(date_iso, game_types=game_types)
    idx: Dict[Tuple[str, str], Dict[str, object]] = {}

    for t in teams:
        team_id = t["team_id"]
        team_name = t["team_name"]
        roster = get_team_roster(team_id, season)
        for p in roster:
            full_name = p.get("full_name")
            pid = p.get("player_id")
            if full_name and pid:
                idx[(team_name, _norm_name(full_name))] = {
                    "player_id": pid,
                    "full_name": full_name,
                    "team_name": team_name,
                }
    return idx


def _attach_pitcher_suppression(mapped: pd.DataFrame, season: int) -> pd.DataFrame:
    if mapped is None or len(mapped) == 0:
        return mapped.copy()

    supp = load_pitcher_suppression_stats(season, season + 1, qual=0)

    if supp is None or supp.empty:
        out = mapped.copy()
        for col in [
            "pctl_pitcher_k_pct",
            "pctl_pitcher_bb_pct",
            "pctl_pitcher_hr_per_9",
            "pctl_pitcher_fip",
            "pctl_pitcher_xfip",
        ]:
            out[col] = 50.0
        print(f"[pitcher_supp_attach] name_matches=0/{len(out)}")
        return out

    out = mapped.copy()
    out["pitcher_name_norm"] = _canon_name(out["pitcher_name"]) if "pitcher_name" in out.columns else ""

    supp = supp.copy()
    if "pitcher_name_norm" in supp.columns:
        supp["pitcher_name_norm"] = _canon_name(supp["pitcher_name_norm"])
    elif "pitcher_name" in supp.columns:
        supp["pitcher_name_norm"] = _canon_name(supp["pitcher_name"])
    else:
        supp["pitcher_name_norm"] = ""

    if "pctl_pitcher_hr9" in supp.columns and "pctl_pitcher_hr_per_9" not in supp.columns:
        supp["pctl_pitcher_hr_per_9"] = pd.to_numeric(supp["pctl_pitcher_hr9"], errors="coerce")

    keep = [c for c in [
        "pitcher_name_norm",
        "pctl_pitcher_k_pct",
        "pctl_pitcher_bb_pct",
        "pctl_pitcher_hr_per_9",
        "pctl_pitcher_fip",
        "pctl_pitcher_xfip",
    ] if c in supp.columns]

    supp = supp[keep].copy()

    merged = out.merge(
        supp,
        on="pitcher_name_norm",
        how="left",
        suffixes=("", "_y")
    )

    cols = [
        "pctl_pitcher_k_pct",
        "pctl_pitcher_bb_pct",
        "pctl_pitcher_hr_per_9",
        "pctl_pitcher_fip",
        "pctl_pitcher_xfip",
    ]

    for col in cols:
        ycol = f"{col}_y"
        if ycol in merged.columns:
            merged[col] = pd.to_numeric(merged[ycol], errors="coerce")

    for col in cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(50.0)

    matches = merged["pctl_pitcher_k_pct"].notna().sum()
    print(f"[pitcher_supp_attach] name_matches={matches}/{len(merged)}")

    return merged


def attach_hitter_context(
    hitter_df: pd.DataFrame,
    games: List[ScheduleGame],
    date_iso: str,
    season: int,
    game_types: str = "R",
) -> pd.DataFrame:
    """
    Infer batter_team/opponent/home_away by matching batter name against active rosters
    for the two teams in the event. Then attach opposing probable pitcher context,
    hitter baseline features, and pitcher suppression features.
    """
    if hitter_df is None or len(hitter_df) == 0:
        return hitter_df.copy()

    roster_idx = _build_roster_name_index(date_iso=date_iso, season=season, game_types=game_types)

    rows = []
    for _, r in hitter_df.iterrows():
        home_team = r.get("home_team")
        away_team = r.get("away_team")
        batter_name = r.get("batter_name")
        batter_norm = _norm_name(batter_name)

        home_hit = roster_idx.get((home_team, batter_norm))
        away_hit = roster_idx.get((away_team, batter_norm))

        batter_team = None
        opponent_team = None
        home_away = None
        batter_id = None

        if home_hit and not away_hit:
            batter_team = home_team
            opponent_team = away_team
            home_away = "HOME"
            batter_id = home_hit.get("player_id")
        elif away_hit and not home_hit:
            batter_team = away_team
            opponent_team = home_team
            home_away = "AWAY"
            batter_id = away_hit.get("player_id")
        elif home_hit and away_hit:
            batter_team = home_team
            opponent_team = away_team
            home_away = "HOME"
            batter_id = home_hit.get("player_id")

        pitcher_id = None
        pitcher_name = None
        pitcher_hand = None

        for g in games:
            if g.home_team == home_team and g.away_team == away_team:
                if home_away == "HOME":
                    pitcher_id = g.away_probable_id
                    pitcher_name = g.away_probable_name
                elif home_away == "AWAY":
                    pitcher_id = g.home_probable_id
                    pitcher_name = g.home_probable_name
                break

        if pitcher_id:
            pitcher_hand = get_player_throws(int(pitcher_id))

        out = r.to_dict()
        out["batter_id"] = batter_id
        out["batter_team"] = batter_team
        out["opponent_team"] = opponent_team
        out["home_away"] = home_away
        out["pitcher_id"] = pitcher_id
        out["pitcher_name"] = pitcher_name
        out["pitcher_hand"] = pitcher_hand
        rows.append(out)

    mapped = pd.DataFrame(rows)

    hitter_base = load_hitter_baseline_stats(season, season, qual=0)
    mapped = attach_hitter_baseline_stats(mapped, hitter_base)

    mapped = _attach_pitcher_suppression(mapped, season)

    mapped = attach_hitter_lineup_status(mapped, games, season=season)

    # For rows where lineup not yet posted (west coast games etc),
    # fill in projected lineup from recent game history
    mapped = attach_projected_lineup_status(mapped, date_iso=date_iso, season=season)

    return mapped