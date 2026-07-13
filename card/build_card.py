from __future__ import annotations

import pandas as pd

from mlb.config import (
    UNITS_FLAGSHIP,
    UNITS_PARLAY,
    UNITS_LOTTO,
    THRESH_FLAGSHIP,
    FLAGSHIP_MAX_JUICE,
    FLAGSHIP_MIN_JUICE,
    FLAGSHIP_MIN_EV,
    FLAGSHIP_USE_JUICE_WINDOW,
    FLAGSHIP_UNDER_MIN_CONF,
)
from mlb.scoring.score_pitchers import compute_ev
from mlb.card.build_also_rec_parlay import build_also_rec_parlay
from mlb.card.build_k_ladder_parlay import build_k_ladder_parlay


def _classify_flagship_tier(confidence_score: float, odds_american: float, ev_per_unit: float, side: str = "") -> str:
    if confidence_score < THRESH_FLAGSHIP:
        return "SKIP"
    # Unders require a higher confidence floor than Overs before qualifying
    # (see FLAGSHIP_UNDER_MIN_CONF in config.py).
    if str(side).strip().lower() == "under" and confidence_score < FLAGSHIP_UNDER_MIN_CONF:
        return "SKIP"
    # Juice window is optional — when disabled, EV > 0 is the only price
    # filter; a favorite qualifies only if the model's probability keeps
    # EV positive at that price.
    if FLAGSHIP_USE_JUICE_WINDOW:
        juice_ok = (odds_american >= FLAGSHIP_MAX_JUICE) and (odds_american <= FLAGSHIP_MIN_JUICE)
    else:
        juice_ok = True
    ev_ok    = ev_per_unit > FLAGSHIP_MIN_EV
    if juice_ok and ev_ok:
        return "FLAGSHIP"
    return "SKIP"


def _select_flagship_pool(df: pd.DataFrame) -> list[dict]:
    """
    Return ALL plays that clear the flagship gate.
    Pool pivot — every qualifying play is a bet recommendation.
    Gate: conf >= THRESH_FLAGSHIP, juice within window, EV > 0.
    """
    if df is None or len(df) == 0:
        return []

    d = df.copy()
    d = d[d["hard_gate_pass"] == True]
    if len(d) == 0:
        return []

    if "ev_per_unit" not in d.columns:
        d["ev_per_unit"] = 0.0
    if "estimated_hit_rate" not in d.columns:
        d["estimated_hit_rate"] = 0.0
    if "odds_american" not in d.columns:
        d["odds_american"] = -110.0

    d["odds_american"]    = pd.to_numeric(d["odds_american"],    errors="coerce").fillna(-110.0)
    d["ev_per_unit"]      = pd.to_numeric(d["ev_per_unit"],      errors="coerce").fillna(-1.0)
    d["confidence_score"] = pd.to_numeric(d["confidence_score"], errors="coerce").fillna(0.0)
    d["edge_for_side"]    = pd.to_numeric(d.get("edge_for_side"), errors="coerce").fillna(0.0)

    d["_tier"] = d.apply(
        lambda r: _classify_flagship_tier(
            r["confidence_score"], r["odds_american"], r["ev_per_unit"], r.get("side")
        ),
        axis=1,
    )

    playable = d[d["_tier"] == "FLAGSHIP"].copy()
    if len(playable) == 0:
        return []

    playable = playable.sort_values(
        by=["ev_per_unit", "edge_for_side", "confidence_score"],
        ascending=[False, False, False],
    )

    results = []
    for _, row in playable.iterrows():
        results.append({
            "pitcher_name":        row.get("pitcher_name"),
            "pitcher_id":          row.get("pitcher_id"),
            "pitcher_team":        row.get("pitcher_team"),
            "opponent_team":       row.get("opponent_team"),
            "home_away":           row.get("home_away"),
            "market":              row.get("market"),
            "side":                row.get("side"),
            "line":                row.get("line"),
            "proj":                row.get("proj"),
            "odds_american":       row.get("odds_american"),
            "edge_for_side":       row.get("edge_for_side"),
            "confidence_score":    row.get("confidence_score"),
            "confidence_bucket":   row.get("confidence_bucket"),
            "estimated_hit_rate":  row.get("estimated_hit_rate"),
            "ev_per_unit":         round(float(row.get("ev_per_unit", 0.0)), 4),
            "kelly_bet_units":     row.get("kelly_bet_units"),
            "kelly_bet_dollars":   row.get("kelly_bet_dollars"),
            "kelly_is_capped":     row.get("kelly_is_capped"),
            "frac_kelly_pct":      row.get("frac_kelly_pct"),
            "tier_label":          "FLAGSHIP",
            "units":               row.get("kelly_bet_units", UNITS_FLAGSHIP) / 100.0 if row.get("kelly_bet_units") is not None else UNITS_FLAGSHIP,
            "_pitcher_id":         row.get("pitcher_id"),
            "_game_id":            row.get("game_id"),
            "_market":             row.get("market"),
            "_side":               row.get("side"),
        })

    return results


def _select_parlay(df: pd.DataFrame, flagship_pool: list[dict] | None = None):
    from mlb.card.build_parlay import build_parlay_candidates

    if df is None or len(df) == 0:
        return None

    candidates = build_parlay_candidates(df, exclude_flagship=flagship_pool[0] if flagship_pool else None)
    if candidates is None or len(candidates) == 0:
        return None

    best = candidates.iloc[0]

    return {
        "leg1": {
            "pitcher_name":     best.get("leg1_pitcher"),
            "pitcher_id":       best.get("leg1_pitcher_id"),
            "market":           best.get("leg1_market"),
            "side":             best.get("leg1_side"),
            "line":             best.get("leg1_line"),
            "odds_american":    best.get("leg1_odds_american"),
            "confidence_score": best.get("leg1_conf"),
            "pitcher_team":     best.get("leg1_pitcher_team"),
            "opponent_team":    best.get("leg1_opponent_team"),
            "home_away":        best.get("leg1_home_away"),
        },
        "leg2": {
            "pitcher_name":     best.get("leg2_pitcher"),
            "pitcher_id":       best.get("leg2_pitcher_id"),
            "market":           best.get("leg2_market"),
            "side":             best.get("leg2_side"),
            "line":             best.get("leg2_line"),
            "confidence_score": best.get("leg2_conf"),
            "odds_american":    best.get("leg2_odds_american"),
            "pitcher_team":     best.get("leg2_pitcher_team"),
            "opponent_team":    best.get("leg2_opponent_team"),
            "home_away":        best.get("leg2_home_away"),
        },
        "corr_type":   best.get("corr_type"),
        "corr_bonus":  best.get("corr_bonus"),
        "combo_score": best.get("combo_score"),
        "units":       UNITS_PARLAY,
    }


def _select_also_recommended(df: pd.DataFrame, flagship_pool: list[dict] | None = None) -> list[dict]:
    """
    Return all gate-clearing plays that are not in the flagship pool.
    Tracked for calibration only — not bet recommendations. Uses a looser
    juice gate than the flagship tier to maximize the calibration sample.
    """
    if df is None or len(df) == 0:
        return []

    d = df.copy()
    d = d[d["hard_gate_pass"] == True]
    if len(d) == 0:
        return []

    d["odds_american"]    = pd.to_numeric(d["odds_american"],    errors="coerce").fillna(-110.0)
    d["ev_per_unit"]      = pd.to_numeric(d["ev_per_unit"],      errors="coerce").fillna(-1.0)
    d["confidence_score"] = pd.to_numeric(d["confidence_score"], errors="coerce").fillna(0.0)
    d["edge_for_side"]    = pd.to_numeric(d.get("edge_for_side"), errors="coerce").fillna(0.0)

    def _tracked_tier(conf, odds, ev):
        if conf < THRESH_FLAGSHIP:  return "SKIP"
        if ev <= FLAGSHIP_MIN_EV:   return "SKIP"
        if odds <= -118 and odds >= -170:  return "TRACKED"  # mirrors flagship juice window (illustrative)
        return "SKIP"

    d["_tier"] = d.apply(
        lambda r: _tracked_tier(r["confidence_score"], r["odds_american"], r["ev_per_unit"]),
        axis=1,
    )

    playable = d[d["_tier"] == "TRACKED"].copy()
    if len(playable) == 0:
        return []

    playable = playable.sort_values(
        by=["ev_per_unit", "edge_for_side", "confidence_score"],
        ascending=[False, False, False],
    )

    flagship_keys = set()
    for f in (flagship_pool or []):
        key = (
            str(f.get("pitcher_name", "")).strip().lower(),
            str(f.get("market", "")),
            str(f.get("side", "")).lower(),
        )
        flagship_keys.add(key)

    results = []
    for _, row in playable.iterrows():
        key = (
            str(row.get("pitcher_name", "")).strip().lower(),
            str(row.get("market", "")),
            str(row.get("side", "")).lower(),
        )
        if key in flagship_keys:
            continue

        results.append({
            "pitcher_name":       row.get("pitcher_name"),
            "pitcher_id":         row.get("pitcher_id"),
            "pitcher_team":       row.get("pitcher_team"),
            "opponent_team":      row.get("opponent_team"),
            "home_away":          row.get("home_away"),
            "market":             row.get("market"),
            "side":               row.get("side"),
            "line":               row.get("line"),
            "proj":               row.get("proj"),
            "odds_american":      row.get("odds_american"),
            "edge_for_side":      row.get("edge_for_side"),
            "confidence_score":   row.get("confidence_score"),
            "estimated_hit_rate": row.get("estimated_hit_rate"),
            "ev_per_unit":        round(float(row.get("ev_per_unit", 0.0)), 4),
            "tier_label":         "TRACKED",
        })

    return results


def build_autocard(pitcher_props: pd.DataFrame, lotto: dict | None = None) -> dict:
    flagship_pool    = _select_flagship_pool(pitcher_props)
    also_recommended = _select_also_recommended(pitcher_props, flagship_pool=flagship_pool)
    parlay           = _select_parlay(pitcher_props, flagship_pool=flagship_pool)

    flagship_pool_clean = []
    for f in flagship_pool:
        flagship_pool_clean.append({k: v for k, v in f.items() if not k.startswith("_")})

    card = {
        "flagship_pool":    flagship_pool_clean,
        "also_recommended": also_recommended,
        "parlay":           parlay,
        "lotto":            lotto,
        "units": {
            "flagship": UNITS_FLAGSHIP,
            "parlay":   UNITS_PARLAY,
            "lotto":    UNITS_LOTTO,
        },
        "notes": [],
    }

    if not flagship_pool_clean:
        card["notes"].append("No Plays Today — no play cleared juice/EV gate.")
    else:
        card["notes"].append(f"{len(flagship_pool_clean)} play(s) on today's card.")
    if parlay is None:
        card["notes"].append("No Parlay Today — no valid 2-leg cross-game K pair found.")
    if lotto is None:
        card["notes"].append("Lotto unavailable — hitter markets not returned by current feed.")

    return card
