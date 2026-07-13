from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional

import pandas as pd

from mlb.config import (
    THRESH_PARLAY_LEG,
    THRESH_FLAGSHIP,
    FLAGSHIP_MIN_EV,
    FLAGSHIP_MAX_JUICE,
    FLAGSHIP_MIN_JUICE,
    FLAGSHIP_USE_JUICE_WINDOW,
    PARLAY_REQUIRE_SAME_SIDE,
)


def _same_pitcher(row1: pd.Series, row2: pd.Series) -> bool:
    pid1 = row1.get("pitcher_id")
    pid2 = row2.get("pitcher_id")
    return pd.notna(pid1) and pd.notna(pid2) and int(pid1) == int(pid2)


def _same_game(row1: pd.Series, row2: pd.Series) -> bool:
    g1 = row1.get("game_id")
    g2 = row2.get("game_id")
    return pd.notna(g1) and pd.notna(g2) and str(g1) == str(g2)


def _aligned_side(row1: pd.Series, row2: pd.Series) -> bool:
    return str(row1.get("side")) == str(row2.get("side"))


def _is_flagship_pick(row: pd.Series, flagship: dict | None) -> bool:
    if flagship is None:
        return False
    f_pid    = flagship.get("_pitcher_id")
    f_market = flagship.get("_market")
    f_side   = flagship.get("_side")
    f_name   = flagship.get("pitcher_name")
    row_pid    = row.get("pitcher_id")
    row_market = row.get("market")
    row_side   = row.get("side")
    row_name   = row.get("pitcher_name")
    if pd.notna(f_pid) and pd.notna(row_pid):
        return (
            int(f_pid) == int(row_pid) and
            str(f_market) == str(row_market) and
            str(f_side).lower() == str(row_side).lower()
        )
    return (
        str(f_name).strip().lower() == str(row_name).strip().lower() and
        str(f_market) == str(row_market) and
        str(f_side).lower() == str(row_side).lower()
    )


# Maximum juice allowed on any single parlay leg — beyond this the book
# is pricing more confidence than our model has, killing expected value.
# Illustrative value; production threshold is tuned via backtest.
PARLAY_MAX_JUICE = -115


def _valid_pitcher_parlay_pair(row1: pd.Series, row2: pd.Series) -> tuple[bool, str, int]:
    """
    Valid parlay pairs on FanDuel — K props only, different pitchers, different games.
    OUTS cannot be parlayed on FanDuel at all.
    Each leg must also clear the juice gate (default -130 max).
    """
    m1 = str(row1.get("market"))
    m2 = str(row2.get("market"))

    # Parlay legs are drawn from the same qualifying pool as flagship
    # single-picks (confidence, EV, and juice-window gates defined in
    # config.py). This mirrors production logic; exact threshold values
    # are illustrative in this portfolio version.
    for tag, row in (("leg1", row1), ("leg2", row2)):
        if float(row.get("confidence_score", 0) or 0) < THRESH_FLAGSHIP:
            return False, f"{tag}_below_flagship_conf", 0
        if float(row.get("ev_per_unit", -1) or -1) <= FLAGSHIP_MIN_EV:
            return False, f"{tag}_ev_not_positive", 0
        odds = float(row.get("odds_american", -110) or -110)
        if FLAGSHIP_USE_JUICE_WINDOW and not (FLAGSHIP_MAX_JUICE <= odds <= FLAGSHIP_MIN_JUICE):
            return False, f"{tag}_outside_juice_window", 0

    if m1 == "OUTS" or m2 == "OUTS":
        return False, "outs_not_parlay_eligible_fanduel", 0

    if _same_pitcher(row1, row2):
        return False, "same_pitcher_blocked_correlated", 0

    if _same_game(row1, row2):
        return False, "same_game_blocked_correlated", 0

    # Same-side is optional now (default off — see PARLAY_REQUIRE_SAME_SIDE).
    same_side = _aligned_side(row1, row2)
    if PARLAY_REQUIRE_SAME_SIDE and not same_side:
        return False, "opposite_side_blocked", 0

    if m1 == "K" and m2 == "K":
        if same_side:
            return True, "two_pitcher_k_same_side", 4
        return True, "two_pitcher_k_mixed_side", 0

    return False, "invalid_combo", 0


def build_parlay_candidates(
    scored_df: pd.DataFrame,
    exclude_flagship: dict | None = None,
) -> pd.DataFrame:
    if len(scored_df) == 0:
        return pd.DataFrame([])

    eligible = scored_df[
        (scored_df["hard_gate_pass"] == True) &
        (pd.to_numeric(scored_df["confidence_score"], errors="coerce") >= THRESH_FLAGSHIP)
    ].copy()

    if len(eligible) < 2:
        return pd.DataFrame([])

    rows: List[Dict[str, object]] = []

    for idx1, idx2 in combinations(eligible.index.tolist(), 2):
        r1 = eligible.loc[idx1]
        r2 = eligible.loc[idx2]

        # Overlap with flagship singles is allowed: the parlay may pair the
        # day's best flagship-grade legs. exclude_flagship is kept in the
        # signature for backward compatibility but no longer filters.

        is_valid, corr_type, corr_bonus = _valid_pitcher_parlay_pair(r1, r2)
        if not is_valid:
            continue

        combo_score = float(r1["confidence_score"]) + float(r2["confidence_score"]) + corr_bonus
        ev1 = float(r1.get("ev_per_unit") or 0.0)
        ev2 = float(r2.get("ev_per_unit") or 0.0)

        rows.append({
            "leg1_idx":           idx1,
            "leg2_idx":           idx2,
            "leg1_ev":            round(ev1, 4),
            "leg2_ev":            round(ev2, 4),
            "parlay_ev_rank":     round(ev1 + ev2, 4),
            "leg1_pitcher":       r1.get("pitcher_name"),
            "leg1_pitcher_id":    r1.get("pitcher_id"),
            "leg1_pitcher_team":  r1.get("pitcher_team"),
            "leg1_opponent_team": r1.get("opponent_team"),
            "leg1_home_away":     r1.get("home_away"),
            "leg1_market":        r1.get("market"),
            "leg1_side":          r1.get("side"),
            "leg1_line":          r1.get("line"),
            "leg1_conf":          r1.get("confidence_score"),
            "leg1_odds_american":  float(r1.get("odds_american") or -110),
            "leg2_pitcher":       r2.get("pitcher_name"),
            "leg2_pitcher_id":    r2.get("pitcher_id"),
            "leg2_pitcher_team":  r2.get("pitcher_team"),
            "leg2_opponent_team": r2.get("opponent_team"),
            "leg2_home_away":     r2.get("home_away"),
            "leg2_market":        r2.get("market"),
            "leg2_side":          r2.get("side"),
            "leg2_line":          r2.get("line"),
            "leg2_conf":          r2.get("confidence_score"),
            "leg2_odds_american":  float(r2.get("odds_american") or -110),
            "corr_type":          corr_type,
            "corr_bonus":         corr_bonus,
            "combo_score":        round(combo_score, 2),
        })

    if not rows:
        return pd.DataFrame([])

    return pd.DataFrame(rows).sort_values(
        by=["parlay_ev_rank", "combo_score"],
        ascending=[False, False]
    ).reset_index(drop=True)


def select_best_parlay(candidates_df: pd.DataFrame) -> Optional[Dict[str, object]]:
    if candidates_df is None or len(candidates_df) == 0:
        return None
    return candidates_df.sort_values(
        by=["combo_score", "corr_bonus"],
        ascending=[False, False]
    ).iloc[0].to_dict()
