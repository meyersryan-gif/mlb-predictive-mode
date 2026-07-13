from __future__ import annotations

"""
build_k_ladder_parlay.py — Builds a speculative alt-line K over parlay.

For each pitcher on the slate, evaluates all available alt K lines below
the posted line and finds the one with the best EV. Selects top 3-5 legs
from different pitchers/games with genuine positive EV at the alt line.

This is a fun/speculative tier — lower confidence threshold, no juice gate,
fixed 0.10u stake (same as lotto).
"""

import pandas as pd
from typing import Optional

K_LADDER_MIN_LEGS    = 3
K_LADDER_MAX_LEGS    = 5
K_LADDER_UNITS       = 0.10
K_LADDER_MIN_CONF    = 75.0   # relaxed from 82
K_LADDER_MAX_JUICE   = -600   # exclude -3000 type locks
K_LADDER_MIN_EV      = 0.02   # must have some positive EV


def build_k_ladder_parlay(df: pd.DataFrame) -> Optional[dict]:
    """
    Build K Ladder Parlay from K_ALT rows in the scored pitcher props.
    Selects best alt line per pitcher (highest EV), then picks top legs.
    """
    if df is None or df.empty:
        return None

    # Filter to K_ALT overs only with positive edge
    alt = df[
        (df["market"].astype(str).str.upper() == "K_ALT") &
        (df["side"].astype(str).str.lower() == "over") &
        (df["hard_gate_pass"] == True) &
        (df["ev_per_unit"] >= K_LADDER_MIN_EV) &
        (df["confidence_score"] >= K_LADDER_MIN_CONF)
    ].copy()

    if alt.empty:
        return None

    # Filter out extreme juice
    alt = alt[pd.to_numeric(alt["odds_american"], errors="coerce") >= K_LADDER_MAX_JUICE]

    if alt.empty:
        return None

    # For each pitcher keep only the best EV alt line
    best = (
        alt.sort_values("ev_per_unit", ascending=False)
        .drop_duplicates(subset=["pitcher_name"], keep="first")
    )

    # Sort by EV descending and select legs
    best = best.sort_values("ev_per_unit", ascending=False)

    selected   = []
    used_games = set()

    for _, row in best.iterrows():
        if len(selected) >= K_LADDER_MAX_LEGS:
            break

        game_id = str(row.get("game_id", "") or
                      str(row.get("pitcher_team","")) + "_" + str(row.get("opponent_team","")))

        if game_id in used_games:
            continue

        selected.append(row)
        used_games.add(game_id)

    if len(selected) < K_LADDER_MIN_LEGS:
        return None

    legs = []
    for row in selected:
        posted_k_line = None
        # Try to find the matching K main line for this pitcher
        k_rows = df[
            (df["pitcher_name"] == row["pitcher_name"]) &
            (df["market"].astype(str).str.upper() == "K") &
            (df["side"].astype(str).str.lower() == "over")
        ]
        if not k_rows.empty:
            posted_k_line = float(k_rows.iloc[0]["line"])

        legs.append({
            "pitcher_name":     row.get("pitcher_name"),
            "pitcher_team":     row.get("pitcher_team"),
            "opponent_team":    row.get("opponent_team"),
            "alt_line":         float(row.get("line", 0)),
            "posted_line":      posted_k_line,
            "odds_american":    int(row.get("odds_american", -110)),
            "proj":             round(float(row.get("proj", 0)), 1),
            "confidence_score": row.get("confidence_score"),
            "ev_per_unit":      round(float(row.get("ev_per_unit", 0)), 4),
        })

    return {
        "type":     "K_LADDER_PARLAY",
        "legs":     legs,
        "num_legs": len(legs),
        "units":    K_LADDER_UNITS,
        "note":     "Alt-line K over parlay — each leg below posted line · Speculative · 0.10u",
    }
