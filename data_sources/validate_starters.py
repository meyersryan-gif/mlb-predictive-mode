from __future__ import annotations

"""
validate_starters.py
--------------------
Cross-validates probable starters from the odds feed against the
MLB Stats API probable pitcher endpoint.

Adds two columns to the pitcher props DataFrame:
  starter_confirmed  : bool  — both sources agree on this pitcher
  starter_conflict   : bool  — sources disagree (mismatch or one is missing)

Pitchers flagged as starter_conflict=True receive a confidence cap
(UNCONFIRMED_CONF_CAP, below both the flagship and parlay-leg thresholds
in config.py) so they cannot reach a flagship or parlay tier pick. The
cap value here is an illustrative placeholder — see config.py.

Usage (in main.py, after props are built but before score_pitcher_props):
    from mlb.data_sources.validate_starters import validate_probable_starters
    props_df = validate_probable_starters(props_df, game_date)
"""

import re
import unicodedata
import requests
import pandas as pd

from mlb.data_sources.mlb_stats_api import BASE

UNCONFIRMED_CONF_CAP = 60.0


def _canon(name: str) -> str:
    if not name or pd.isna(name):
        return ""
    v = str(name)
    v = unicodedata.normalize("NFKD", v).encode("ascii", "ignore").decode("ascii")
    v = v.strip().lower()
    if "," in v:
        parts = [p.strip() for p in v.split(",", 1)]
        if len(parts) == 2:
            v = f"{parts[1]} {parts[0]}"
    v = re.sub(r"[^a-z0-9 ]+", "", v)
    v = re.sub(r"\s+", " ", v).strip()
    return v.replace(" ", "")


def _fetch_mlbapi_starters(game_date: str) -> dict[str, int]:
    url = BASE + "/schedule"
    params = {
        "sportId":  1,
        "date":     game_date,
        "hydrate":  "probablePitcher",
        "gameType": "R",
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[validate_starters] MLB API fetch failed: {e}")
        return {}

    starters: dict[str, int] = {}
    for date_block in data.get("dates", []):
        for game in date_block.get("games", []):
            for side in ("home", "away"):
                team_data = game.get("teams", {}).get(side, {})
                probable  = team_data.get("probablePitcher")
                if not probable:
                    continue
                pid  = probable.get("id")
                name = probable.get("fullName", "")
                if pid and name:
                    key = _canon(name)
                    if key:
                        starters[key] = int(pid)
    return starters


def validate_probable_starters(
    df: pd.DataFrame,
    game_date: str,
) -> pd.DataFrame:
    out = df.copy()

    mlbapi_starters = _fetch_mlbapi_starters(game_date)
    n_api = len(mlbapi_starters)

    if n_api == 0:
        print(f"[validate_starters] MLB API returned 0 starters — skipping validation")
        out["starter_confirmed"] = False
        out["starter_conflict"]  = False
        out["starter_source"]    = "api_unavailable"
        return out

    print(f"[validate_starters] MLB API starters loaded: {n_api} pitchers for {game_date}")

    pitcher_col = None
    for c in ["pitcher_name", "Pitcher", "pitcher", "PitcherName", "player_name"]:
        if c in out.columns:
            pitcher_col = c
            break

    if pitcher_col is None:
        print(f"[validate_starters] No pitcher name column found — skipping validation")
        out["starter_confirmed"] = False
        out["starter_conflict"]  = False
        out["starter_source"]    = "no_name_col"
        return out

    out["_name_key"] = out[pitcher_col].apply(_canon)

    confirmed = out["_name_key"].isin(mlbapi_starters)
    conflict  = ~confirmed

    out["starter_confirmed"] = confirmed
    out["starter_conflict"]  = conflict
    out["starter_source"]    = confirmed.map({True: "mlbapi_confirmed", False: "mlbapi_not_found"})

    if "confidence_score" in out.columns:
        before = out.loc[conflict, "confidence_score"].copy()
        out.loc[conflict, "confidence_score"] = out.loc[
            conflict, "confidence_score"
        ].clip(upper=UNCONFIRMED_CONF_CAP)

        n_capped = (before > UNCONFIRMED_CONF_CAP).sum()
        if n_capped:
            print(
                f"[validate_starters] Capped {n_capped} unconfirmed starters "
                f"to conf={UNCONFIRMED_CONF_CAP}"
            )

        if "notes" not in out.columns:
            out["notes"] = ""
        out["notes"] = out["notes"].fillna("").astype(str)
        capped_mask = conflict & (before > UNCONFIRMED_CONF_CAP)
        out.loc[capped_mask, "notes"] = (
            out.loc[capped_mask, "notes"].str.strip() + " starter-unconfirmed-cap;"
        ).str.strip()
    else:
        out["starter_conf_cap"] = out["starter_conflict"].map(
            {True: UNCONFIRMED_CONF_CAP, False: float("inf")}
        )

    out.drop(columns=["_name_key"], inplace=True)

    n_confirmed = int(confirmed.sum())
    n_conflict  = int(conflict.sum())
    print(
        f"[validate_starters] confirmed={n_confirmed}/{len(out)}  "
        f"conflict={n_conflict}/{len(out)}"
    )

    return out
