from __future__ import annotations

# ---------------------------------------------------------------------------
# NOTE: This is a portfolio/showcase version of the pitcher scoring module.
# The point weights and calibration curve below are illustrative
# placeholders that preserve the shape and architecture of the production
# scorer (multi-factor point allocation -> confidence score -> calibrated
# hit-rate -> EV -> fractional Kelly sizing). The production coefficients
# are tuned against a running backtest and are kept private.
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd

from mlb.config import (
    THRESH_FLAGSHIP, THRESH_PARLAY_LEG, THRESH_LOTTO,
    KELLY_FRACTION, KELLY_MAX_BET_PCT, BANKROLL,
)


def _safe_num(x, default=0.0) -> float:
    try:
        if pd.isna(x):
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _norm_side(side: str) -> str:
    s = str(side).strip().lower()
    if s == "over":  return "Over"
    if s == "under": return "Under"
    return "Over"


# ---------------------------------------------------------------------------
# Hit rate and EV
#
# _conf_to_hit_rate maps a 0-100 model confidence score to an estimated
# real-world hit rate via a piecewise-linear calibration curve. In
# production this curve is fit against graded picks (i.e. "of the picks we
# scored at confidence X, what fraction actually hit?"). The breakpoints
# below are illustrative — a smooth, monotonic curve — not the fitted
# production curve.
# ---------------------------------------------------------------------------

def _conf_to_hit_rate(conf: float) -> float:
    conf = float(np.clip(conf, 0.0, 100.0))
    if conf <= 60:  return 0.500
    if conf <= 70:  return 0.500 + (conf - 60)  / (70 - 60)  * (0.520 - 0.500)
    if conf <= 80:  return 0.520 + (conf - 70)  / (80 - 70)  * (0.540 - 0.520)
    if conf <= 90:  return 0.540 + (conf - 80)  / (90 - 80)  * (0.570 - 0.540)
    return          0.570 + (conf - 90) / (100 - 90) * (0.610 - 0.570)


def _edge_to_hit_rate(edge_for_side: float, market: str) -> float:
    """Convert a projected stat-line edge (projection minus market line)
    into an implied hit rate via a normal-CDF model, using a per-market
    standard deviation for the underlying stat."""
    if pd.isna(edge_for_side):
        return 0.500
    std_dev  = 1.8 if str(market).upper() == "K" else 2.5
    from scipy.stats import norm
    hit_rate = norm.cdf(edge_for_side / std_dev)
    return float(np.clip(hit_rate, 0.50, 0.85))


def _hybrid_hit_rate(conf: float, edge_for_side: float, market: str) -> float:
    """Blend the confidence-calibrated hit rate with the edge-implied hit
    rate. Illustrative blend weight below."""
    conf_rate = _conf_to_hit_rate(conf)
    edge_rate = _edge_to_hit_rate(edge_for_side, market)
    blended   = 0.50 * conf_rate + 0.50 * edge_rate
    return round(float(np.clip(blended, 0.50, 0.85)), 4)


def _american_to_decimal_payout(odds: float) -> float:
    if odds >= 0:
        return odds / 100.0
    return 100.0 / abs(odds)


def compute_ev(conf: float, edge_for_side: float, market: str, odds_american: float) -> float:
    hit_rate = _hybrid_hit_rate(conf, edge_for_side, market)
    payout   = _american_to_decimal_payout(odds_american)
    ev       = (hit_rate * payout) - ((1.0 - hit_rate) * 1.0)
    return round(float(ev), 4)


# ---------------------------------------------------------------------------
# Kelly sizing
# ---------------------------------------------------------------------------

def compute_kelly_bet(
    hit_rate: float,
    odds_american: float,
    kelly_fraction: float = KELLY_FRACTION,
    max_bet_pct: float    = KELLY_MAX_BET_PCT,
    bankroll: float       = BANKROLL,
) -> dict:
    """
    Fractional Kelly bet sizing.

    Full Kelly: K = (hit_rate * payout - loss_rate) / payout
    Fractional: K * kelly_fraction, capped at max_bet_pct of bankroll.

    Returns dict with kelly_pct, frac_kelly_pct, capped_pct,
    bet_units (1 unit = 1% of bankroll), bet_dollars, is_capped.
    """
    payout = _american_to_decimal_payout(odds_american)

    if payout <= 0 or hit_rate <= 0:
        return {
            "kelly_pct": 0.0, "frac_kelly_pct": 0.0,
            "capped_pct": 0.0, "bet_units": 0.0,
            "bet_dollars": 0.0, "is_capped": False,
        }

    loss_rate  = 1.0 - hit_rate
    full_kelly = max(0.0, (hit_rate * payout - loss_rate) / payout)
    frac_kelly = full_kelly * kelly_fraction
    capped     = min(frac_kelly, max_bet_pct)
    is_capped  = capped < frac_kelly

    return {
        "kelly_pct":      round(full_kelly * 100, 2),
        "frac_kelly_pct": round(frac_kelly * 100, 2),
        "capped_pct":     round(capped * 100, 2),
        "bet_units":      round(capped * 100, 3),
        "bet_dollars":    round(capped * bankroll, 2),
        "is_capped":      is_capped,
    }


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def project_pitcher_props(df: pd.DataFrame) -> pd.DataFrame:
    """Project a raw stat line (strikeouts or outs recorded) from
    percentile-ranked pitcher/opponent inputs, then compute the edge
    against the market line. Coefficients below are illustrative."""
    df = df.copy()
    projs = []
    edge_signeds   = []
    edge_for_sides = []

    for _, r in df.iterrows():
        market = r.get("market")
        side   = _norm_side(r.get("side"))
        avg_bf = _safe_num(r.get("avg_bf_per_start"), 24.0)
        avg_ip = _safe_num(r.get("avg_ip_per_start"), 5.5)
        p_k    = _safe_num(r.get("pctl_k_pct"),              50.0)
        opp_k  = _safe_num(r.get("opp_pctl_k_pct_vs_hand"),  50.0)
        p_bb   = _safe_num(r.get("pctl_bb_pct"),             50.0)

        est_pitcher_k_rate = 0.220 + (p_k   - 50.0) / 250.0
        est_opp_k_factor   = 1.0  + (opp_k - 50.0) / 200.0
        est_bb_penalty     = 1.0  - max(0.0, (p_bb - 50.0) / 300.0)

        if market in ("K", "K_ALT"):
            proj = avg_bf * est_pitcher_k_rate * est_opp_k_factor * est_bb_penalty
        elif market == "OUTS":
            proj = avg_ip * 3.0 * est_bb_penalty
        else:
            proj = 0.0

        line          = _safe_num(r.get("line"), 0.0)
        edge_signed   = proj - line
        edge_for_side = edge_signed if side == "Over" else -edge_signed

        projs.append(round(proj, 3))
        edge_signeds.append(round(edge_signed, 3))
        edge_for_sides.append(round(edge_for_side, 3))

    df["proj"]          = projs
    df["edge_signed"]   = edge_signeds
    df["edge_for_side"] = edge_for_sides
    df["edge_abs"]      = df["edge_signed"].abs().round(3)
    return df


# ---------------------------------------------------------------------------
# Early season penalties
# ---------------------------------------------------------------------------

def _apply_early_season_k_under_penalty(df: pd.DataFrame) -> pd.DataFrame:
    """Down-weight Under picks on strikeout props early in the season, when
    a pitcher's current-year performance has trended meaningfully hotter
    than their prior-year baseline (i.e. the projection is anchored to
    stale, colder data). Penalty thresholds below are illustrative."""
    out = df.copy()
    required = ["market", "side", "confidence_score"]
    for c in required:
        if c not in out.columns:
            return out

    prior_k     = pd.to_numeric(out.get("prior_pctl_k_pct"),     errors="coerce")
    curr_k      = pd.to_numeric(out.get("curr_pctl_k_pct"),      errors="coerce")
    prior_swstr = pd.to_numeric(out.get("prior_pctl_swstr_pct"), errors="coerce")
    curr_swstr  = pd.to_numeric(out.get("curr_pctl_swstr_pct"),  errors="coerce")
    prior_csw   = pd.to_numeric(out.get("prior_pctl_csw_pct"),   errors="coerce")
    curr_csw    = pd.to_numeric(out.get("curr_pctl_csw_pct"),    errors="coerce")
    starts      = pd.to_numeric(out.get("curr_starts_used"),     errors="coerce").fillna(0)

    k_delta     = (curr_k     - prior_k).fillna(0.0)
    swstr_delta = (curr_swstr - prior_swstr).fillna(0.0)
    csw_delta   = (curr_csw   - prior_csw).fillna(0.0)

    market       = out["market"].astype(str).str.upper()
    side         = out["side"].astype(str).str.lower()
    k_under_mask = market.eq("K") & side.eq("under")
    penalty      = pd.Series(0.0, index=out.index, dtype="float64")

    penalty = penalty.mask(
        k_under_mask & (starts >= 2) & (
            (k_delta >= 20) | ((k_delta >= 12) & ((swstr_delta >= 8) | (csw_delta >= 8)))
        ), 12.0
    )
    penalty = penalty.mask(
        k_under_mask & (starts >= 2) & (penalty == 0) & (
            (k_delta >= 12) | (swstr_delta >= 6) | (csw_delta >= 6)
        ), 8.0
    )
    penalty = penalty.mask(
        k_under_mask & (starts >= 1) & (penalty == 0) & (
            (k_delta >= 6) | (swstr_delta >= 4) | (csw_delta >= 4)
        ), 4.0
    )

    out["confidence_score"] = (
        pd.to_numeric(out["confidence_score"], errors="coerce").fillna(0.0) - penalty
    ).clip(lower=0.0, upper=100.0)

    if "notes" not in out.columns:
        out["notes"] = ""
    out["notes"] = out["notes"].fillna("").astype(str)
    out.loc[penalty > 0, "notes"] = (
        out.loc[penalty > 0, "notes"].str.strip() + " early-season-k-under-penalty;"
    ).str.strip()

    return out


def _apply_missing_pitcher_data_guardrail(df: pd.DataFrame) -> pd.DataFrame:
    """Cap confidence when a pitcher has no current-season signal and a
    flat (uninformative) prior-season baseline, so the model doesn't
    overstate confidence on a near-blank input."""
    out = df.copy()
    market      = out.get("market", pd.Series("", index=out.index)).astype(str).str.upper()
    starts      = pd.to_numeric(out.get("curr_starts_used"),     errors="coerce").fillna(0)
    prior_k     = pd.to_numeric(out.get("prior_pctl_k_pct"),     errors="coerce")
    curr_k      = pd.to_numeric(out.get("curr_pctl_k_pct"),      errors="coerce")
    prior_swstr = pd.to_numeric(out.get("prior_pctl_swstr_pct"), errors="coerce")
    curr_swstr  = pd.to_numeric(out.get("curr_pctl_swstr_pct"),  errors="coerce")
    prior_csw   = pd.to_numeric(out.get("prior_pctl_csw_pct"),   errors="coerce")
    curr_csw    = pd.to_numeric(out.get("curr_pctl_csw_pct"),    errors="coerce")

    no_current_signal = starts.eq(0) & curr_k.isna() & curr_swstr.isna() & curr_csw.isna()
    flat_prior = (
        prior_k.fillna(50).between(49.9, 50.1) &
        prior_swstr.fillna(50).between(49.9, 50.1) &
        prior_csw.fillna(50).between(49.9, 50.1)
    )
    weak_info_mask = market.eq("K") & no_current_signal & flat_prior

    out.loc[weak_info_mask, "confidence_score"] = pd.to_numeric(
        out.loc[weak_info_mask, "confidence_score"], errors="coerce"
    ).clip(upper=58.0)

    if "notes" not in out.columns:
        out["notes"] = ""
    out["notes"] = out["notes"].fillna("").astype(str)
    out.loc[weak_info_mask, "notes"] = (
        out.loc[weak_info_mask, "notes"].str.strip() + " missing-pitcher-signal-cap;"
    ).str.strip()

    bucket = pd.Series("none", index=out.index, dtype="object")
    bucket = bucket.mask(out["confidence_score"] >= THRESH_LOTTO,      "lotto")
    bucket = bucket.mask(out["confidence_score"] >= THRESH_PARLAY_LEG, "parlay")
    bucket = bucket.mask(out["confidence_score"] >= THRESH_FLAGSHIP,   "flagship")
    out["confidence_bucket"] = bucket

    return out


# ---------------------------------------------------------------------------
# EV + Kelly column computation
# ---------------------------------------------------------------------------

def _compute_ev_column(df: pd.DataFrame) -> pd.DataFrame:
    out  = df.copy()
    conf = pd.to_numeric(out.get("confidence_score"), errors="coerce").fillna(0.0)
    odds = pd.to_numeric(out.get("odds_american"),    errors="coerce").fillna(-110.0)
    edge = pd.to_numeric(out.get("edge_for_side"),    errors="coerce").fillna(0.0)
    mkt  = out.get("market", pd.Series("K", index=out.index)).astype(str)

    hit_rates = [
        _hybrid_hit_rate(c, e, m)
        for c, e, m in zip(conf, edge, mkt)
    ]
    out["estimated_hit_rate"] = hit_rates

    out["ev_per_unit"] = [
        round((hr * _american_to_decimal_payout(o)) - ((1.0 - hr) * 1.0), 4)
        for hr, o in zip(hit_rates, odds)
    ]

    kelly_results = [
        compute_kelly_bet(hr, o)
        for hr, o in zip(hit_rates, odds)
    ]
    out["kelly_pct"]         = [k["kelly_pct"]      for k in kelly_results]
    out["frac_kelly_pct"]    = [k["frac_kelly_pct"] for k in kelly_results]
    out["kelly_bet_units"]   = [k["bet_units"]       for k in kelly_results]
    out["kelly_bet_dollars"] = [k["bet_dollars"]     for k in kelly_results]
    out["kelly_is_capped"]   = [bool(k["is_capped"]) for k in kelly_results]

    return out


# ---------------------------------------------------------------------------
# Main scorer
#
# Combines five independently-weighted signal groups into a single 0-100
# confidence score: statistical edge percentile, matchup quality, pitcher
# "leash" (expected workload), park/umpire environment, and market
# (line-movement) agreement — plus a recent-form trend penalty for OUTS
# unders. Point allocations below are illustrative; the production model's
# exact weights are tuned against a running backtest and kept private.
# ---------------------------------------------------------------------------

def score_pitcher_props(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if len(df) == 0:
        for col in ["confidence_score", "confidence_bucket", "edge_pctl_slate",
                    "estimated_hit_rate", "ev_per_unit",
                    "kelly_bet_units", "kelly_bet_dollars", "kelly_is_capped"]:
            if col not in df.columns:
                df[col] = []
        return df

    df["edge_for_side_nonneg"] = df["edge_for_side"].clip(lower=0.0)
    df["edge_pctl_slate"] = (
        df.groupby("market")["edge_for_side_nonneg"]
          .rank(pct=True, method="average")
          .fillna(0.0) * 100.0
    )

    scores  = []
    buckets = []

    for _, r in df.iterrows():
        if not bool(r.get("hard_gate_pass", False)):
            scores.append(0.0)
            buckets.append("none")
            continue

        market        = r.get("market")
        side          = _norm_side(r.get("side"))
        edge_for_side = _safe_num(r.get("edge_for_side"), 0.0)

        # --- Edge points (max 35): how large is the projected edge relative
        # to the rest of today's slate? ---
        if edge_for_side <= 0:
            edge_pts = 0
        else:
            edge_p = _safe_num(r.get("edge_pctl_slate"), 0.0)
            if edge_p < 40:   edge_pts = 0
            elif edge_p < 55: edge_pts = 10
            elif edge_p < 70: edge_pts = 18
            elif edge_p < 80: edge_pts = 24
            elif edge_p < 90: edge_pts = 29
            else:             edge_pts = 35

        # --- Matchup points (max 25): pitcher-vs-opponent strikeout/walk
        # matchup quality for K markets; walk/contact quality for OUTS. ---
        opp_k    = _safe_num(r.get("opp_pctl_k_pct_vs_hand"),  50.0)
        opp_bb   = _safe_num(r.get("opp_pctl_bb_pct_vs_hand"), 50.0)
        opp_woba = _safe_num(r.get("opp_pctl_woba_vs_hand"),   50.0)

        if market in ("K", "K_ALT"):
            over_raw    = 0.7 * opp_k + 0.3 * (100.0 - opp_bb)
            matchup_raw = over_raw if side == "Over" else (100.0 - over_raw)
        else:
            over_raw    = 0.6 * (100.0 - opp_bb) + 0.4 * (100.0 - opp_woba)
            matchup_raw = over_raw if side == "Over" else (100.0 - over_raw)

        if matchup_raw < 10:   matchup_pts = 0
        elif matchup_raw < 25: matchup_pts = 6
        elif matchup_raw < 40: matchup_pts = 10
        elif matchup_raw < 60: matchup_pts = 14
        elif matchup_raw < 75: matchup_pts = 18
        elif matchup_raw < 90: matchup_pts = 22
        else:                  matchup_pts = 25

        # --- Leash points (max 20): expected workload (innings/batters
        # faced) — a longer expected outing supports higher confidence. ---
        avg_ip    = _safe_num(r.get("avg_ip_per_start"), 5.5)
        avg_bf    = _safe_num(r.get("avg_bf_per_start"), 24.0)
        leash_raw = min(100.0, (avg_ip / 6.5) * 60.0 + (avg_bf / 27.0) * 40.0)

        if leash_raw < 10:   leash_pts = 0
        elif leash_raw < 25: leash_pts = 4
        elif leash_raw < 40: leash_pts = 8
        elif leash_raw < 60: leash_pts = 12
        elif leash_raw < 75: leash_pts = 16
        else:                leash_pts = 20

        # --- Environment points (max 10): park factor + weather (7pts) and
        # umpire strike-zone tendency (3pts). Missing park data or a delay
        # risk flag zeroes out the park component rather than defaulting
        # to neutral, since an unknown environment isn't a neutral one. ---
        delay         = bool(r.get("delay_risk_flag", False))
        park_runs_raw = r.get("pctl_park_runs")
        has_park      = park_runs_raw is not None and not pd.isna(park_runs_raw)

        if delay or not has_park:
            park_pts = 0
        else:
            park_runs = float(park_runs_raw)
            park_raw  = 100.0 - abs(park_runs - 50.0)
            if park_raw < 10:   park_pts = 0
            elif park_raw < 25: park_pts = 1
            elif park_raw < 40: park_pts = 3
            elif park_raw < 60: park_pts = 4
            elif park_raw < 75: park_pts = 5
            elif park_raw < 90: park_pts = 6
            else:               park_pts = 7

        umpire_raw = r.get("pctl_umpire_k")
        has_umpire = umpire_raw is not None and not pd.isna(umpire_raw)

        if not has_umpire:
            ump_pts = 1  # missing umpire data defaults to neutral
        else:
            ump_pctl = float(umpire_raw)
            if side == "Over":
                if ump_pctl >= 80:   ump_pts = 3
                elif ump_pctl >= 60: ump_pts = 2
                elif ump_pctl >= 40: ump_pts = 1
                else:                ump_pts = 0
            else:
                if ump_pctl <= 20:   ump_pts = 3
                elif ump_pctl <= 40: ump_pts = 2
                elif ump_pctl <= 60: ump_pts = 1
                else:                ump_pts = 0

        env_pts = park_pts + ump_pts

        # --- Market points (max 10): does the market's line movement agree
        # with the model's side? Only awarded if real line-movement data
        # exists — missing data scores zero, not neutral. ---
        line_delta_raw = r.get("pctl_line_delta")
        has_line_delta = line_delta_raw is not None and not pd.isna(line_delta_raw)

        if not has_line_delta:
            market_pts = 0
        else:
            line_delta_p = float(line_delta_raw)
            market_raw   = 100.0 - line_delta_p
            if market_raw < 10:   market_pts = 0
            elif market_raw < 25: market_pts = 2
            elif market_raw < 40: market_pts = 4
            elif market_raw < 60: market_pts = 6
            elif market_raw < 75: market_pts = 8
            else:                 market_pts = 10

        # --- Recent-form trend penalty (OUTS unders only): if a pitcher has
        # been going meaningfully deeper into games recently, penalize an
        # Under call so the score reflects the current trend, not just the
        # season-long average. ---
        curr_starts   = int(_safe_num(r.get("curr_starts_used"), 0))
        ip_trend      = _safe_num(r.get("recent_ip_trend"), 0.0)
        trend_penalty = 0.0
        market_str = str(r.get("market","")).upper()
        side_str   = str(r.get("side","")).lower()
        if market_str == "OUTS" and side_str == "under" and curr_starts >= 3:
            if ip_trend >= 1.0:    trend_penalty = -15.0
            elif ip_trend >= 0.5:  trend_penalty = -8.0
            elif ip_trend >= 0.25: trend_penalty = -4.0

        score  = float(edge_pts + matchup_pts + leash_pts + env_pts + market_pts + trend_penalty)
        score  = _clamp(score, 0.0, 100.0)

        if score >= THRESH_FLAGSHIP:     bucket = "flagship"
        elif score >= THRESH_PARLAY_LEG: bucket = "parlay"
        elif score >= THRESH_LOTTO:      bucket = "lotto"
        else:                            bucket = "none"

        scores.append(round(score, 2))
        buckets.append(bucket)

    df["confidence_score"]  = scores
    df["confidence_bucket"] = buckets

    df = _apply_early_season_k_under_penalty(df)
    df = _apply_missing_pitcher_data_guardrail(df)

    # Compute EV and Kelly after all confidence adjustments are finalized
    df = _compute_ev_column(df)

    return df
