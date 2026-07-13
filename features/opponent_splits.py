from __future__ import annotations

import re
import numpy as np
import pandas as pd
import requests


MLB_API_BASE = "https://statsapi.mlb.com/api/v1"


# ---------------------------------------------------------------------------
# Fallback hardcoded table — used only if MLB API call fails entirely.
# These are approximate 2024 values.
# ---------------------------------------------------------------------------
_FALLBACK_ROWS = [
    ("ARI", 52, 46, 58, 57, 49), ("ATL", 44, 57, 72, 70, 52),
    ("BAL", 47, 50, 66, 64, 50), ("BOS", 55, 51, 60, 59, 50),
    ("CHC", 53, 52, 55, 53, 49), ("CIN", 58, 45, 54, 56, 51),
    ("CLE", 42, 54, 48, 44, 47), ("COL", 56, 44, 59, 61, 52),
    ("CWS", 63, 41, 35, 33, 54), ("DET", 51, 47, 46, 45, 49),
    ("HOU", 41, 58, 69, 67, 48), ("KC",  46, 49, 52, 50, 49),
    ("LAA", 57, 43, 47, 48, 51), ("LAD", 43, 59, 78, 75, 50),
    ("MIA", 54, 42, 40, 39, 52), ("MIL", 50, 48, 56, 55, 50),
    ("MIN", 59, 50, 57, 58, 50), ("NYM", 49, 53, 61, 60, 49),
    ("NYY", 48, 55, 73, 74, 51), ("OAK", 61, 40, 38, 36, 53),
    ("PHI", 45, 54, 68, 66, 49), ("PIT", 55, 43, 42, 41, 51),
    ("SD",  47, 52, 63, 62, 49), ("SEA", 62, 47, 53, 54, 50),
    ("SF",  52, 51, 50, 48, 49), ("STL", 46, 50, 51, 49, 48),
    ("TB",  58, 46, 49, 50, 50), ("TEX", 51, 48, 65, 63, 50),
    ("TOR", 50, 53, 62, 61, 50), ("WSH", 54, 44, 41, 40, 51),
]

_TEAM_ABBREV_MAP = {
    "ARIZONA DIAMONDBACKS": "ARI", "ARI": "ARI",
    "ATLANTA BRAVES": "ATL", "ATL": "ATL",
    "BALTIMORE ORIOLES": "BAL", "BAL": "BAL",
    "BOSTON RED SOX": "BOS", "BOS": "BOS",
    "CHICAGO CUBS": "CHC", "CHC": "CHC",
    "CINCINNATI REDS": "CIN", "CIN": "CIN",
    "CLEVELAND GUARDIANS": "CLE", "CLE": "CLE",
    "COLORADO ROCKIES": "COL", "COL": "COL",
    "CHICAGO WHITE SOX": "CWS", "CWS": "CWS",
    "DETROIT TIGERS": "DET", "DET": "DET",
    "HOUSTON ASTROS": "HOU", "HOU": "HOU",
    "KANSAS CITY ROYALS": "KC", "KC": "KC",
    "LOS ANGELES ANGELS": "LAA", "LAA": "LAA",
    "LOS ANGELES DODGERS": "LAD", "LAD": "LAD",
    "MIAMI MARLINS": "MIA", "MIA": "MIA",
    "MILWAUKEE BREWERS": "MIL", "MIL": "MIL",
    "MINNESOTA TWINS": "MIN", "MIN": "MIN",
    "NEW YORK METS": "NYM", "NYM": "NYM",
    "NEW YORK YANKEES": "NYY", "NYY": "NYY",
    "ATHLETICS": "OAK", "OAKLAND ATHLETICS": "OAK", "OAK": "OAK",
    "PHILADELPHIA PHILLIES": "PHI", "PHI": "PHI",
    "PITTSBURGH PIRATES": "PIT", "PIT": "PIT",
    "SAN DIEGO PADRES": "SD", "SD": "SD",
    "SEATTLE MARINERS": "SEA", "SEA": "SEA",
    "SAN FRANCISCO GIANTS": "SF", "SF": "SF",
    "ST LOUIS CARDINALS": "STL", "ST. LOUIS CARDINALS": "STL", "STL": "STL",
    "TAMPA BAY RAYS": "TB", "TB": "TB",
    "TEXAS RANGERS": "TEX", "TEX": "TEX",
    "TORONTO BLUE JAYS": "TOR", "TOR": "TOR",
    "WASHINGTON NATIONALS": "WSH", "WSH": "WSH",
}

# MLB Stats API team ID to abbreviation mapping
_MLBAPI_TEAM_ID_MAP = {
    109: "ARI", 144: "ATL", 110: "BAL", 111: "BOS", 112: "CHC",
    113: "CIN", 114: "CLE", 115: "COL", 145: "CWS", 116: "DET",
    117: "HOU", 118: "KC",  108: "LAA", 119: "LAD", 146: "MIA",
    158: "MIL", 142: "MIN", 121: "NYM", 147: "NYY", 133: "OAK",
    143: "PHI", 134: "PIT", 135: "SD",  136: "SEA", 137: "SF",
    138: "STL", 139: "TB",  140: "TEX", 141: "TOR", 120: "WSH",
}


def _canon_team(v) -> str:
    if pd.isna(v):
        return ""
    return _TEAM_ABBREV_MAP.get(str(v).strip().upper(), str(v).strip().upper())


def _pct_rank(s: pd.Series, ascending: bool = True, default=50.0) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(default, index=s.index, dtype="float64")
    pct = s.rank(pct=True, method="average", ascending=ascending) * 100.0
    return pct.fillna(default).clip(0, 100)


def _build_fallback_df() -> pd.DataFrame:
    rows = []
    for team, k_pct, bb_pct, woba, iso_, ppa in _FALLBACK_ROWS:
        for hand in ["L", "R"]:
            rows.append({
                "opponent_team":              team,
                "opp_hand":                   hand,
                "opp_pctl_k_pct_vs_hand":     float(k_pct),
                "opp_pctl_bb_pct_vs_hand":    float(bb_pct),
                "opp_pctl_woba_vs_hand":      float(woba),
                "opp_pctl_iso_vs_hand":       float(iso_),
                "opp_pctl_pitches_pa_vs_hand":float(ppa),
            })
    return pd.DataFrame(rows)


def _fetch_mlbapi_team_batting(season: int) -> pd.DataFrame:
    """
    Pull team batting splits from MLB Stats API — vs LHP and vs RHP separately.
    Uses statSplits endpoint with sitCodes vl/vr for real handedness splits.
    Paginates with offset 0/40/80 to ensure all 30 teams are captured.
    """
    url = MLB_API_BASE + "/stats"
    all_rows = []

    for sitcode in ["vl", "vr"]:
        pitcher_hand = "L" if sitcode == "vl" else "R"
        page_splits  = []

        for offset in [0, 40]:
            params = {
                "stats":    "statSplits",
                "group":    "hitting",
                "gameType": "R",
                "season":   season,
                "sportId":  1,
                "sitCodes": sitcode,
                "limit":    40,
                "offset":   offset,
            }
            try:
                r = requests.get(url, params=params, timeout=15)
                r.raise_for_status()
                page = (r.json().get("stats") or [{}])[0].get("splits") or []
                page_splits.extend(page)
                if len(page) < 40:
                    break
            except Exception as e:
                print(f"[opponent_splits] MLB API page failed ({sitcode} offset={offset}): {e}")
                break

        for split in page_splits:
            team = split.get("team") or {}
            stat = split.get("stat") or {}

            team_id   = team.get("id")
            team_abbr = _MLBAPI_TEAM_ID_MAP.get(team_id)
            if not team_abbr:
                team_name = str(team.get("name", "")).upper()
                team_abbr = _TEAM_ABBREV_MAP.get(team_name)
            if not team_abbr:
                continue

            pa  = int(stat.get("plateAppearances", 0) or 0)
            ab  = int(stat.get("atBats",           0) or 0)
            so  = int(stat.get("strikeOuts",        0) or 0)
            bb  = int(stat.get("baseOnBalls",       0) or 0)
            h   = int(stat.get("hits",              0) or 0)
            hr  = int(stat.get("homeRuns",          0) or 0)
            tb  = int(stat.get("totalBases",        0) or 0)
            hbp = int(stat.get("hitByPitch",        0) or 0)

            denom = pa if pa > 0 else ab
            if denom == 0:
                continue

            k_rate     = so / denom
            bb_rate    = bb / denom
            singles    = h - hr
            woba_num   = (bb * 0.69) + (hbp * 0.72) + (singles * 0.89) + (hr * 2.10)
            woba_proxy = woba_num / denom if denom > 0 else 0.0
            iso_proxy  = (tb - h) / ab if ab > 0 else 0.0

            all_rows.append({
                "opponent_team": team_abbr,
                "pitcher_hand":  pitcher_hand,
                "k_rate":        k_rate,
                "bb_rate":       bb_rate,
                "woba_proxy":    woba_proxy,
                "iso_proxy":     iso_proxy,
            })

    if not all_rows:
        print(f"[opponent_splits] MLB API returned no team batting rows for {season}")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Aggregate duplicate team+hand rows by averaging rates
    df = df.groupby(["opponent_team", "pitcher_hand"], as_index=False).agg({
        "k_rate":     "mean",
        "bb_rate":    "mean",
        "woba_proxy": "mean",
        "iso_proxy":  "mean",
    })

    # opp_hand = pitcher_hand — batters face a pitcher of this hand
    df["opp_hand"] = df["pitcher_hand"]

    # Compute percentiles within each hand group separately
    for hand in ["L", "R"]:
        mask = df["pitcher_hand"] == hand
        if mask.sum() > 1:
            df.loc[mask, "opp_pctl_k_pct_vs_hand"]      = _pct_rank(df.loc[mask, "k_rate"],     ascending=True).values
            df.loc[mask, "opp_pctl_bb_pct_vs_hand"]     = _pct_rank(df.loc[mask, "bb_rate"],    ascending=False).values
            df.loc[mask, "opp_pctl_woba_vs_hand"]       = _pct_rank(df.loc[mask, "woba_proxy"], ascending=True).values
            df.loc[mask, "opp_pctl_iso_vs_hand"]        = _pct_rank(df.loc[mask, "iso_proxy"],  ascending=True).values
        else:
            df.loc[mask, "opp_pctl_k_pct_vs_hand"]      = 50.0
            df.loc[mask, "opp_pctl_bb_pct_vs_hand"]     = 50.0
            df.loc[mask, "opp_pctl_woba_vs_hand"]       = 50.0
            df.loc[mask, "opp_pctl_iso_vs_hand"]        = 50.0

    df["opp_pctl_pitches_pa_vs_hand"] = 50.0

    teams_l = (df["pitcher_hand"] == "L").sum()
    teams_r = (df["pitcher_hand"] == "R").sum()
    print(f"[opponent_splits] MLB API team batting loaded: {teams_l} teams vs LHP, {teams_r} teams vs RHP for {season}")

    return df[[
        "opponent_team", "opp_hand",
        "opp_pctl_k_pct_vs_hand",
        "opp_pctl_bb_pct_vs_hand",
        "opp_pctl_woba_vs_hand",
        "opp_pctl_iso_vs_hand",
        "opp_pctl_pitches_pa_vs_hand",
    ]]


def _build_live_df(start_season: int, end_season: int) -> pd.DataFrame:
    """
    Pull team batting stats from MLB Stats API (replaces FanGraphs/pybaseball).
    Uses most recent season with data — falls back one year if current returns empty.
    """
    for season in range(end_season, start_season - 1, -1):
        df = _fetch_mlbapi_team_batting(season)
        if df is not None and not df.empty and len(df) >= 20:
            # Expand to both pitcher hands
            # Hand-specific splits not available from season aggregates
            # Same values applied to L and R until a hand-split source is added
            rows = []
            for _, row in df.iterrows():
                for hand in ["L", "R"]:
                    r = row.to_dict()
                    r["opp_hand"] = hand
                    rows.append(r)
            return pd.DataFrame(rows)

    return pd.DataFrame()


def load_team_opponent_tendencies(
    start_season: int,
    end_season: int | None = None,
    qual: int = 0,
) -> pd.DataFrame:
    """
    Load team opponent tendencies using MLB Stats API as primary source.
    Falls back to hardcoded 2024 estimates only if API call fails entirely.
    """
    if end_season is None:
        end_season = start_season

    live = _build_live_df(start_season, end_season)
    if live is not None and not live.empty:
        return live

    print("[opponent_splits] MLB API failed — using fallback table")
    return _build_fallback_df()


def attach_opponent_tendencies(df: pd.DataFrame, opp_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out

    if opp_df is None or opp_df.empty:
        opp_df = load_team_opponent_tendencies(2025, 2025, 0)

    out["_opp_team_key"] = out["opponent_team"].apply(_canon_team) if "opponent_team" in out.columns else ""
    if "opp_hand" not in out.columns:
        out["opp_hand"] = "R"
    out["opp_hand"] = out["opp_hand"].astype(str).str.upper().replace({"": "R"}).fillna("R")

    base = opp_df.copy()
    base["_opp_team_key"] = base["opponent_team"].apply(_canon_team)
    base["opp_hand"]      = base["opp_hand"].astype(str).str.upper()

    base = base.rename(columns={
        "opp_pctl_k_pct_vs_hand":      "base_opp_pctl_k_pct_vs_hand",
        "opp_pctl_bb_pct_vs_hand":     "base_opp_pctl_bb_pct_vs_hand",
        "opp_pctl_woba_vs_hand":       "base_opp_pctl_woba_vs_hand",
        "opp_pctl_iso_vs_hand":        "base_opp_pctl_iso_vs_hand",
        "opp_pctl_pitches_pa_vs_hand": "base_opp_pctl_pitches_pa_vs_hand",
    })

    merged = out.merge(
        base[[
            "_opp_team_key", "opp_hand",
            "base_opp_pctl_k_pct_vs_hand",
            "base_opp_pctl_bb_pct_vs_hand",
            "base_opp_pctl_woba_vs_hand",
            "base_opp_pctl_iso_vs_hand",
            "base_opp_pctl_pitches_pa_vs_hand",
        ]],
        on=["_opp_team_key", "opp_hand"],
        how="left"
    )

    mapping = {
        "opp_pctl_k_pct_vs_hand":      "base_opp_pctl_k_pct_vs_hand",
        "opp_pctl_bb_pct_vs_hand":     "base_opp_pctl_bb_pct_vs_hand",
        "opp_pctl_woba_vs_hand":       "base_opp_pctl_woba_vs_hand",
        "opp_pctl_iso_vs_hand":        "base_opp_pctl_iso_vs_hand",
        "opp_pctl_pitches_pa_vs_hand": "base_opp_pctl_pitches_pa_vs_hand",
    }

    for live_col, base_col in mapping.items():
        merged[base_col] = pd.to_numeric(merged.get(base_col), errors="coerce")
        if live_col in merged.columns:
            merged[live_col] = pd.to_numeric(merged[live_col], errors="coerce").fillna(merged[base_col]).fillna(50.0)
        else:
            merged[live_col] = merged[base_col].fillna(50.0)

    if "_opp_team_key" in merged.columns:
        merged = merged.drop(columns=["_opp_team_key"])

    # Deduplicate — each pitcher should appear once per market/side
    if "pitcher_name" in merged.columns and "market" in merged.columns and "side" in merged.columns:
        merged = merged.drop_duplicates(subset=["pitcher_name", "market", "side"], keep="first")

    return merged













