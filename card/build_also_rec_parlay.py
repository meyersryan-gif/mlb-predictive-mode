from __future__ import annotations

"""
build_also_rec_parlay.py — Builds an alt-line parlay from Also Recommended K overs.

Rules:
  - Only K Over picks from Also Recommended
  - Line drops 0.5 (alt line = posted line - 0.5)
  - Different pitchers, different games
  - Minimum 3 legs, maximum 5 legs
  - No juice gate — this is a fun/speculative parlay
  - 0.25u stake
"""

from itertools import combinations
from typing import Optional


ALSO_REC_PARLAY_MIN_LEGS = 3
ALSO_REC_PARLAY_MAX_LEGS = 5
ALSO_REC_PARLAY_UNITS    = 0.25


def build_also_rec_parlay(also_recommended: list[dict]) -> Optional[dict]:
    """
    Build an alt-line parlay from Also Recommended K overs.
    Selects up to MAX_LEGS highest-confidence K over picks
    with different pitchers from different games.
    """
    if not also_recommended:
        return None

    # Filter to K overs only
    k_overs = [
        p for p in also_recommended
        if str(p.get("market", "")).upper() == "K"
        and str(p.get("side", "")).lower() == "over"
    ]

    if len(k_overs) < ALSO_REC_PARLAY_MIN_LEGS:
        return None

    # Sort by confidence descending
    k_overs = sorted(k_overs, key=lambda x: float(x.get("confidence_score", 0)), reverse=True)

    # Select legs — different pitchers and different games
    selected = []
    used_pitchers = set()
    used_games    = set()

    for pick in k_overs:
        if len(selected) >= ALSO_REC_PARLAY_MAX_LEGS:
            break

        pitcher = str(pick.get("pitcher_name", "")).strip().lower()
        game_id = str(pick.get("game_id", "") or
                      pick.get("pitcher_team", "") + "_" + pick.get("opponent_team", ""))

        if pitcher in used_pitchers:
            continue
        if game_id in used_games:
            continue

        selected.append(pick)
        used_pitchers.add(pitcher)
        used_games.add(game_id)

    if len(selected) < ALSO_REC_PARLAY_MIN_LEGS:
        return None

    # Build legs with alt lines (0.5 below posted line)
    legs = []
    for pick in selected:
        posted_line = float(pick.get("line", 0))
        alt_line    = posted_line - 0.5
        legs.append({
            "pitcher_name":     pick.get("pitcher_name"),
            "pitcher_id":       pick.get("pitcher_id"),
            "pitcher_team":     pick.get("pitcher_team"),
            "opponent_team":    pick.get("opponent_team"),
            "home_away":        pick.get("home_away"),
            "market":           "K",
            "side":             "Over",
            "posted_line":      posted_line,
            "alt_line":         alt_line,
            "confidence_score": pick.get("confidence_score"),
            "ev_per_unit":      pick.get("ev_per_unit"),
        })

    return {
        "type":     "ALSO_REC_K_OVER_PARLAY",
        "legs":     legs,
        "num_legs": len(legs),
        "units":    ALSO_REC_PARLAY_UNITS,
        "note":     "Alt-line parlay — each leg is 0.5 below posted Also Recommended line",
    }
