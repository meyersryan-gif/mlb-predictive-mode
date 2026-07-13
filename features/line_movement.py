from __future__ import annotations

"""
line_movement.py — Computes line movement signals for pitcher props.

Sources line_open and line_latest from the props DataFrame (already
populated by build_pitcher_props.py from the Odds API) and computes:

  line_delta          : line_open - line_latest (positive = line moved down)
  line_move_direction : 'toward' or 'away' relative to the pick side
  pctl_line_delta     : percentile rank of favorable line movement vs slate
  pctl_line_move      : same, alias for schema compatibility

A line moving toward your side means sharps bet that side.
A line moving away from your side is a warning.
No movement = neutral (50th percentile).
"""

import numpy as np
import pandas as pd


def compute_line_movement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute line movement features for each pitcher prop row.

    Requires columns: line_open, line_latest, side
    Adds columns:
        line_delta          - raw change (open - latest), positive = moved down
        line_move_toward    - True if line moved toward pick side
        line_move_abs       - absolute magnitude of movement
        pctl_line_delta     - percentile of favorable movement vs slate
        pctl_line_move      - alias for pctl_line_delta
    """
    out = df.copy()

    line_open   = pd.to_numeric(out.get("line_open"),   errors="coerce")
    line_latest = pd.to_numeric(out.get("line_latest"), errors="coerce")
    side        = out.get("side", pd.Series("", index=out.index)).astype(str).str.lower()

    # Raw delta: positive means line moved down
    line_delta = (line_open - line_latest).fillna(0.0)
    out["line_delta"]    = line_delta.round(3)
    out["line_move_abs"] = line_delta.abs().round(3)

    # Favorable movement: toward our side
    # Under pick + line moved down (delta > 0) = confirming
    # Over pick  + line moved up   (delta < 0) = confirming
    favorable = pd.Series(0.0, index=out.index, dtype="float64")

    under_mask = side.str.contains("under", case=False, na=False)
    over_mask  = side.str.contains("over",  case=False, na=False)

    # Positive delta = line moved down = good for unders
    favorable[under_mask] = line_delta[under_mask]
    # Negative delta = line moved up = good for overs (flip sign so positive = good)
    favorable[over_mask]  = -line_delta[over_mask]

    out["line_move_toward"] = (favorable > 0)

    # Percentile rank of favorable movement across the slate
    # 0 movement = 50th percentile (neutral)
    # Positive favorable = above 50 (confirming)
    # Negative favorable = below 50 (warning)
    if favorable.abs().sum() == 0:
        # No line movement today — all neutral
        out["pctl_line_delta"] = 50.0
        out["pctl_line_move"]  = 50.0
        print("[line_movement] No line movement detected today — all neutral")
    else:
        pct = favorable.rank(pct=True, method="average") * 100.0
        out["pctl_line_delta"] = pct.fillna(50.0).clip(0, 100).round(1)
        out["pctl_line_move"]  = out["pctl_line_delta"]

    moved     = (line_delta.abs() > 0.01).sum()
    favorable_count = (favorable > 0).sum()

    # Hard gate — large unfavorable moves indicate sharp money against our side
    # Threshold: 50+ point odds move against the pick = hard gate fail
    UNFAVORABLE_GATE_THRESHOLD = 50.0
    if "odds_open" in out.columns and "odds_american" in out.columns:
        odds_now  = pd.to_numeric(out["odds_american"], errors="coerce")
        odds_open = pd.to_numeric(out["odds_open"],     errors="coerce")
        odds_delta = odds_now - odds_open

        # Unfavorable = odds moved against our side
        # For overs: odds got more negative (book pricing it higher) = unfavorable
        # For unders: same logic
        unfavorable_move = -favorable  # negative favorable = unfavorable magnitude

        large_unfavorable = (unfavorable_move >= UNFAVORABLE_GATE_THRESHOLD) & odds_open.notna()
        gated_count = large_unfavorable.sum()

        if "hard_gate_pass" not in out.columns:
            out["hard_gate_pass"] = True
        out.loc[large_unfavorable, "hard_gate_pass"] = False
        out.loc[large_unfavorable, "notes"] = out.loc[large_unfavorable, "notes"].fillna("").astype(str) + " unfavorable-move-gate;"

        if gated_count > 0:
            print(f"[line_movement] unfavorable_gate={gated_count} plays hard-gated (move >= {UNFAVORABLE_GATE_THRESHOLD} pts against pick)")

    print(f"[line_movement] lines_moved={moved}/{len(out)}  favorable_for_pick={favorable_count}/{len(out)}")

    return out


def attach_line_movement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wrapper that checks for required columns and applies line movement.
    Safe to call even if line_open/line_latest are missing.
    """
    out = df.copy()

    has_open   = "line_open"   in out.columns and out["line_open"].notna().any()
    has_latest = "line_latest" in out.columns and out["line_latest"].notna().any()

    if not has_open or not has_latest:
        print("[line_movement] WARNING: line_open or line_latest missing — movement defaulting to neutral")
        out["line_delta"]      = 0.0
        out["line_move_abs"]   = 0.0
        out["line_move_toward"]= False
        out["pctl_line_delta"] = 50.0
        out["pctl_line_move"]  = 50.0
        return out

    return compute_line_movement(out)

