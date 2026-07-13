from __future__ import annotations

# ---------------------------------------------------------------------------
# NOTE: This is a portfolio/showcase version of the hitter scoring module.
# The blend weights below (e.g. the coefficients in hr_hitter,
# contact_hitter, hr_pitcher_vuln, etc.) are illustrative placeholders that
# preserve the shape of the production multi-factor model. The production
# weights are tuned against a running backtest and kept private. All gating
# logic, thresholds, and architecture are unchanged from production.
# ---------------------------------------------------------------------------

import pandas as pd


def _num(df: pd.DataFrame, col: str, default=50.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")


def _american_to_implied_prob(odds: pd.Series) -> pd.Series:
    odds = pd.to_numeric(odds, errors="coerce")
    out = pd.Series(0.50, index=odds.index, dtype="float64")
    pos = odds > 0
    neg = odds < 0
    out[pos] = 100.0 / (odds[pos] + 100.0)
    out[neg] = (-odds[neg]) / ((-odds[neg]) + 100.0)
    return out.fillna(0.50).clip(0.01, 0.99)


def _lineup_points(order_spot: pd.Series, lineup_status: pd.Series) -> pd.Series:
    """Score a batter's expected plate-appearance volume from lineup slot
    and lineup-confirmation status. Illustrative point values below."""
    spot   = pd.to_numeric(order_spot, errors="coerce")
    status = lineup_status.astype(str).str.lower()
    pts    = pd.Series(50.0, index=spot.index, dtype="float64")
    pts = pts.where(~spot.between(1, 3, inclusive="both"), 80.0)
    pts = pts.where(~spot.between(4, 5, inclusive="both"), 72.0)
    pts = pts.where(~spot.between(6, 7, inclusive="both"), 58.0)
    pts = pts.where(~spot.between(8, 9, inclusive="both"), 42.0)
    pts = pts.where(~status.eq("confirmed"),                pts + 8.0)
    pts = pts.where(~status.isin(["unknown", "tbd", ""]),   pts - 8.0)
    return pts.clip(20.0, 90.0)


def _bool_series(df: pd.DataFrame, col: str, default: bool) -> pd.Series:
    if col in df.columns:
        s = df[col]
        if s.dtype == bool:
            return s.fillna(default)
        return s.astype("object").where(s.notna(), default).astype(bool)
    return pd.Series(default, index=df.index, dtype=bool)


def _has_real_value(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    return pd.to_numeric(df[col], errors="coerce").notna()


def score_hitter_lotto(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score hitter "lotto" prop bets (HR / total bases / 2+ hits / RBI) from
    percentile-ranked batter, opposing-pitcher, and environment inputs.

    Architecture:
      1. Build sub-scores: batter power/contact skill, opposing-pitcher
         vulnerability, lineup-slot volume, park/weather environment.
      2. Blend sub-scores into a market-specific lotto_score.
      3. Apply hard gates (must be an Over, must be starting, line caps,
         missing-data gates) that can zero out eligibility.
      4. Apply market-specific adjustments (e.g. elite-strikeout-pitcher
         penalty for HR bets) and clip to a display range.

    All coefficients below are illustrative; production weights are tuned
    against a running backtest and kept private.
    """
    out = df.copy()
    if out.empty:
        return out

    market        = out.get("market", pd.Series("", index=out.index)).astype(str).str.upper()
    side          = out.get("side",   pd.Series("", index=out.index)).astype(str).str.upper()
    line          = pd.to_numeric(out.get("line", pd.Series(pd.NA, index=out.index)), errors="coerce")
    odds_american = _num(out, "odds_american", default=100.0)

    pctl_iso      = _num(out, "pctl_iso_vs_hand")
    pctl_woba     = _num(out, "pctl_woba_vs_hand")
    pctl_ops      = _num(out, "pctl_ops_vs_hand")
    pctl_hr_rate  = _num(out, "pctl_hr_rate")
    pctl_barrel   = _num(out, "pctl_barrel_pct")
    pctl_hardhit  = _num(out, "pctl_hardhit_pct")
    pctl_k        = _num(out, "pctl_k_pct")
    pctl_bb       = _num(out, "pctl_bb_pct")
    pctl_avg      = _num(out, "pctl_avg_vs_hand")
    pctl_xba      = _num(out, "pctl_xba_vs_hand")

    pctl_pitcher_k     = _num(out, "pctl_pitcher_k_pct")
    pctl_pitcher_bb    = _num(out, "pctl_pitcher_bb_pct")
    pctl_pitcher_hr9   = _num(out, "pctl_pitcher_hr_per_9")
    pctl_pitcher_fip   = _num(out, "pctl_pitcher_fip")
    pctl_pitcher_xfip  = _num(out, "pctl_pitcher_xfip")

    if "pctl_pitcher_barrel_allowed" in out.columns:
        pctl_pitcher_barrel_allowed = _num(out, "pctl_pitcher_barrel_allowed")
    else:
        pctl_pitcher_barrel_allowed = 0.60 * pctl_pitcher_hr9 + 0.40 * pctl_pitcher_xfip

    if "pctl_pitcher_hardhit_allowed" in out.columns:
        pctl_pitcher_hardhit_allowed = _num(out, "pctl_pitcher_hardhit_allowed")
    else:
        pctl_pitcher_hardhit_allowed = 0.55 * pctl_pitcher_fip + 0.45 * pctl_pitcher_hr9

    if "pctl_pitcher_gb_pct" in out.columns:
        pctl_pitcher_gb = _num(out, "pctl_pitcher_gb_pct")
    else:
        pctl_pitcher_gb = pd.Series(50.0, index=out.index, dtype="float64")

    if "pctl_pitcher_woba_allowed" in out.columns:
        pctl_pitcher_woba_allowed = _num(out, "pctl_pitcher_woba_allowed")
    else:
        pctl_pitcher_woba_allowed = 0.50 * pctl_pitcher_fip + 0.50 * pctl_pitcher_xfip

    pctl_team_runs = _num(out, "pctl_team_implied_runs")
    pctl_park_hr   = _num(out, "pctl_park_hr")
    pctl_park_runs = _num(out, "pctl_park_runs")
    wind_mph       = _num(out, "wind_mph",  default=0.0)
    temp_f         = _num(out, "temp_f",    default=70.0)

    delay_risk   = _bool_series(out, "delay_risk_flag", False)
    platoon_risk = _bool_series(out, "platoon_risk_flag", False)

    # is_starting defaults to False — unknown lineup status should not pass gate
    is_starting = _bool_series(out, "is_starting", False)

    # null situational gates for HR market
    has_team_runs = _has_real_value(out, "pctl_team_implied_runs")
    has_park_hr   = _has_real_value(out, "pctl_park_hr")

    # pitcher with missing-pitcher-signal-cap = unreliable opponent read
    pitcher_notes             = out.get("notes", pd.Series("", index=out.index)).fillna("").astype(str)
    has_missing_pitcher_cap   = pitcher_notes.str.contains("missing-pitcher-signal-cap", na=False)

    lineup_status      = out.get("lineup_status",      pd.Series("", index=out.index)).fillna("").astype(str)
    batting_order_spot = out.get("batting_order_spot", pd.Series(pd.NA, index=out.index))
    batter_team        = out.get("batter_team",        pd.Series(pd.NA, index=out.index))

    implied_prob  = _american_to_implied_prob(odds_american)
    price_points  = (100.0 - implied_prob * 100.0).clip(25.0, 75.0)

    # --- Batter skill sub-scores (illustrative weights) ---
    hr_hitter = (
        0.30 * pctl_iso      +
        0.25 * pctl_hr_rate  +
        0.20 * pctl_barrel   +
        0.10 * pctl_hardhit  +
        0.10 * pctl_woba     +
        0.05 * pctl_bb       -
        0.10 * (100.0 - pctl_k)
    )

    contact_hitter = (
        0.25 * pctl_avg      +
        0.20 * pctl_xba      +
        0.20 * pctl_woba     +
        0.15 * pctl_ops      +
        0.10 * pctl_hardhit  +
        0.10 * pctl_bb       -
        0.10 * (100.0 - pctl_k)
    )

    # --- Opposing-pitcher vulnerability sub-scores (illustrative weights) ---
    hr_pitcher_vuln = (
        0.35 * pctl_pitcher_hr9              +
        0.20 * pctl_pitcher_barrel_allowed   +
        0.15 * pctl_pitcher_hardhit_allowed  +
        0.10 * pctl_pitcher_woba_allowed     +
        0.10 * pctl_pitcher_bb               +
        0.10 * pctl_pitcher_fip              -
        0.20 * pctl_pitcher_k                -
        0.20 * pctl_pitcher_gb
    )

    contact_pitcher_vuln = (
        0.30 * pctl_pitcher_woba_allowed     +
        0.20 * pctl_pitcher_hardhit_allowed  +
        0.15 * pctl_pitcher_barrel_allowed   +
        0.10 * pctl_pitcher_bb               +
        0.10 * pctl_pitcher_fip              -
        0.15 * pctl_pitcher_k                -
        0.10 * pctl_pitcher_gb
    )

    lineup_pts = _lineup_points(batting_order_spot, lineup_status)

    # --- Environment sub-scores (illustrative weights) ---
    env_hr = (
        0.40 * pctl_park_hr   +
        0.25 * pctl_team_runs +
        0.20 * (temp_f - 60.0).clip(lower=0, upper=35) * (100.0 / 35.0) +
        0.15 * wind_mph.clip(lower=0, upper=20) * (100.0 / 20.0)
    ).clip(0.0, 100.0)

    env_contact = (
        0.35 * pctl_park_runs +
        0.30 * pctl_team_runs +
        0.10 * (temp_f - 60.0).clip(lower=0, upper=35) * (100.0 / 35.0)
    ).clip(0.0, 100.0)

    # --- Final blended scores (illustrative weights) ---
    hr_score = (
        0.30 * hr_hitter          +
        0.35 * hr_pitcher_vuln    +
        0.15 * lineup_pts         +
        0.15 * env_hr             +
        0.05 * price_points
    )

    contact_score = (
        0.35 * contact_hitter        +
        0.25 * contact_pitcher_vuln  +
        0.18 * lineup_pts            +
        0.12 * env_contact           +
        0.10 * price_points
    )

    out["lotto_score"] = contact_score
    out.loc[market.eq("HR"),         "lotto_score"] = hr_score.loc[market.eq("HR")]
    out.loc[market.eq("TB"),         "lotto_score"] = (0.64 * contact_score + 0.36 * hr_score).loc[market.eq("TB")]
    out.loc[market.eq("RBI"),        "lotto_score"] = (0.48 * contact_score + 0.18 * hr_score + 0.34 * pctl_team_runs).loc[market.eq("RBI")]
    out.loc[market.eq("HITS_2PLUS"), "lotto_score"] = (contact_score + 0.10 * lineup_pts).loc[market.eq("HITS_2PLUS")]
    out.loc[market.eq("HITS"),       "lotto_score"] = contact_score.loc[market.eq("HITS")]

    # ---- hard gates ----
    out["hard_gate_pass"] = True

    # Missing batter team
    out.loc[batter_team.isna(), "lotto_score"]    -= 18.0
    out.loc[batter_team.isna(), "hard_gate_pass"]  = False

    # Must be an over
    out.loc[~side.eq("OVER"), "hard_gate_pass"]  = False
    out.loc[~side.eq("OVER"), "lotto_score"]     -= 20.0

    # Line gates — beyond these lines, treat as the "alt ladder" tier
    out.loc[market.eq("HR")       & (line > 0.5), "hard_gate_pass"] = False
    out.loc[market.eq("HR")       & (line > 0.5), "lotto_score"]    -= 20.0
    out.loc[market.eq("HITS")     & (line > 1.5), "hard_gate_pass"] = False
    out.loc[market.eq("HITS")     & (line > 1.5), "lotto_score"]    -= 16.0
    out.loc[market.eq("TB")       & (line > 2.5), "hard_gate_pass"] = False
    out.loc[market.eq("TB")       & (line > 2.5), "lotto_score"]    -= 16.0
    out.loc[market.eq("RBI")      & (line > 1.5), "hard_gate_pass"] = False
    out.loc[market.eq("RBI")      & (line > 1.5), "lotto_score"]    -= 16.0

    # Unknown lineup status gates
    out.loc[~is_starting, "hard_gate_pass"] = False
    out.loc[~is_starting, "lotto_score"]    -= 15.0

    # Null situational context gates for HR
    out.loc[market.eq("HR") & ~has_park_hr, "hard_gate_pass"] = False
    out.loc[market.eq("HR") & ~has_park_hr, "lotto_score"]    -= 10.0
    out.loc[market.eq("HR") & ~has_team_runs, "lotto_score"] -= 5.0

    # Unreliable pitcher opponent
    out.loc[has_missing_pitcher_cap, "hard_gate_pass"] = False
    out.loc[has_missing_pitcher_cap, "lotto_score"]    -= 8.0

    # Platoon and delay penalties
    out.loc[platoon_risk, "lotto_score"] -= 8.0
    out.loc[delay_risk,   "lotto_score"] -= 8.0

    # HR-specific adjustments
    hr_mask = market.eq("HR")
    out.loc[hr_mask & (pctl_hr_rate  <  35), "lotto_score"] -= 10.0
    out.loc[hr_mask & (pctl_iso      <  35), "lotto_score"] -=  8.0
    out.loc[hr_mask & (odds_american < 300), "lotto_score"] -= 10.0
    out.loc[hr_mask & (odds_american > 1500),"lotto_score"] -= 10.0

    out.loc[hr_mask & (pctl_pitcher_k    >= 75), "lotto_score"] -= 16.0
    out.loc[hr_mask & (pctl_pitcher_hr9  <= 30), "lotto_score"] -= 14.0
    out.loc[hr_mask & (pctl_pitcher_xfip <= 25), "lotto_score"] -= 10.0
    out.loc[hr_mask & (pctl_pitcher_fip  <= 25), "lotto_score"] -=  5.0
    out.loc[hr_mask & (pctl_pitcher_gb   >= 65), "lotto_score"] -=  5.0

    out.loc[hr_mask & (pctl_pitcher_hr9  >= 70), "lotto_score"] +=  6.0
    out.loc[hr_mask & (pctl_pitcher_bb   >= 65), "lotto_score"] +=  3.0
    out.loc[hr_mask & (pctl_pitcher_xfip >= 70), "lotto_score"] +=  4.0

    contact_mask = market.isin(["TB", "HITS_2PLUS", "RBI", "HITS"])
    order_num    = pd.to_numeric(batting_order_spot, errors="coerce").fillna(9)
    out.loc[contact_mask & (order_num >= 8), "lotto_score"] -= 8.0
    out.loc[contact_mask & (pctl_avg  <  35),"lotto_score"] -= 8.0

    out["lotto_score"]      = out["lotto_score"].clip(20.0, 85.0)
    out["confidence_score"] = out["lotto_score"].round(1)

    out["caps_applied"] = ""
    out["notes"]        = ""

    out.loc[~is_starting,                               "notes"] = out["notes"].astype(str) + " not-starting-gate;"
    out.loc[delay_risk,                                 "notes"] = out["notes"].astype(str) + " delay-risk;"
    out.loc[platoon_risk,                               "notes"] = out["notes"].astype(str) + " platoon-risk;"
    out.loc[batter_team.isna(),                         "notes"] = out["notes"].astype(str) + " missing-team;"
    out.loc[market.eq("HR") & ~has_team_runs,           "notes"] = out["notes"].astype(str) + " missing-team-runs-gate;"
    out.loc[market.eq("HR") & ~has_park_hr,             "notes"] = out["notes"].astype(str) + " missing-park-hr-gate;"
    out.loc[has_missing_pitcher_cap,                    "notes"] = out["notes"].astype(str) + " missing-pitcher-cap-gate;"
    out.loc[hr_mask & (pctl_pitcher_k    >= 75),        "notes"] = out["notes"].astype(str) + " elite-k-pitcher;"
    out.loc[hr_mask & (pctl_pitcher_hr9  <= 30),        "notes"] = out["notes"].astype(str) + " hr-suppressor;"
    out.loc[hr_mask & (pctl_pitcher_hr9  >= 70),        "notes"] = out["notes"].astype(str) + " weak-hr-pitcher;"
    out.loc[market.eq("HR")       & (line > 0.5),       "notes"] = out["notes"].astype(str) + " alt-hr-ladder;"
    out.loc[market.eq("HITS")     & (line > 1.5),       "notes"] = out["notes"].astype(str) + " alt-hits-ladder;"
    out.loc[market.eq("TB")       & (line > 2.5),       "notes"] = out["notes"].astype(str) + " alt-tb-ladder;"
    out.loc[market.eq("RBI")      & (line > 1.5),       "notes"] = out["notes"].astype(str) + " alt-rbi-ladder;"

    out["notes"] = out["notes"].str.strip()

    return out
