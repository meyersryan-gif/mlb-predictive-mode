from __future__ import annotations

"""
batter_hr_gate.py — Gates lotto HR picks based on current season HR rate.

Fetches 2026 batting stats from MLB Stats API for each batter.
Applies penalties and hard gates for cold/slumping hitters:

  - 50+ PA, 0 HR  -> hard gate fail + large penalty
  - 50+ PA, HR rate < 2%  -> soft penalty
  - 30-49 PA, 0 HR -> soft penalty (small sample caution)
  - < 30 PA -> no penalty (too small to judge)

This catches hitters like Kyle Stowers who have elite Statcast
profiles but zero HRs in the current season.
"""

import time
from typing import Optional

import pandas as pd
import requests


MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

# Cache to avoid repeated API calls for same batter
_BATTER_STATS_CACHE: dict[int, dict] = {}


def _get_json(url: str, params: dict = None) -> dict:
    try:
        r = requests.get(url, params=params or {}, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def _fetch_batter_season_stats(player_id: int, season: int) -> dict:
    """Fetch season batting stats for a single batter."""
    if player_id in _BATTER_STATS_CACHE:
        return _BATTER_STATS_CACHE[player_id]

    url    = MLB_API_BASE + f"/people/{player_id}/stats"
    params = {"stats": "season", "group": "hitting", "season": season, "sportIds": 1}
    data   = _get_json(url, params)

    try:
        splits = data.get("stats", [{}])[0].get("splits", [])
        if not splits:
            return {}
        stat = splits[0].get("stat", {})
        pa   = int(stat.get("plateAppearances", 0) or 0)
        ab   = int(stat.get("atBats",           0) or 0)
        hr   = int(stat.get("homeRuns",          0) or 0)

        result = {
            "pa":      pa,
            "ab":      ab,
            "hr":      hr,
            "hr_rate": (hr / pa) if pa > 0 else None,
        }
        _BATTER_STATS_CACHE[player_id] = result
        return result
    except Exception:
        return {}


def attach_batter_hr_gate(
    df: pd.DataFrame,
    season: int = 2026,
) -> pd.DataFrame:
    """
    Attach current season HR rate context and apply gate/penalty
    to HR market rows with cold or slumping hitters.

    Adds columns:
        curr_season_hr     - home runs this season
        curr_season_pa     - plate appearances this season
        curr_season_hr_rate - HR per PA this season
        hr_gate_note       - reason for any penalty applied
    """
    out = df.copy()

    for col in ["curr_season_hr", "curr_season_pa", "curr_season_hr_rate", "hr_gate_note"]:
        if col not in out.columns:
            out[col] = None

    market = out.get("market", pd.Series("", index=out.index)).astype(str).str.upper()
    hr_mask = market.eq("HR")

    if not hr_mask.any():
        return out

    # Only fetch stats for HR market rows
    hr_rows    = out[hr_mask].copy()
    batter_ids = pd.to_numeric(hr_rows.get("batter_id"), errors="coerce").dropna().astype(int).unique()

    fetched = 0
    for pid in batter_ids:
        stats = _fetch_batter_season_stats(int(pid), season)
        if stats:
            fetched += 1
        time.sleep(0.05)  # rate limit

    print(f"[batter_hr_gate] fetched stats for {fetched}/{len(batter_ids)} batters")

    # Apply gate and penalties
    for idx, row in out[hr_mask].iterrows():
        batter_id = pd.to_numeric(row.get("batter_id"), errors="coerce")
        if pd.isna(batter_id):
            continue

        stats = _BATTER_STATS_CACHE.get(int(batter_id), {})
        if not stats:
            continue

        pa      = stats.get("pa", 0)
        hr      = stats.get("hr", 0)
        hr_rate = stats.get("hr_rate")

        out.at[idx, "curr_season_hr"]      = hr
        out.at[idx, "curr_season_pa"]      = pa
        out.at[idx, "curr_season_hr_rate"] = hr_rate

        note = ""

        if pa >= 50 and hr == 0:
            # Hard gate — 50+ PA with zero HRs is a clear cold signal
            out.at[idx, "hard_gate_pass"] = False
            out.at[idx, "lotto_score"]    = out.at[idx, "lotto_score"] - 20.0
            note = f"hr-cold-gate:{pa}PA-0HR;"

        elif pa >= 50 and hr_rate is not None and hr_rate < 0.02:
            # Soft penalty — very low HR rate (< 1 HR per 50 PA)
            out.at[idx, "lotto_score"] = out.at[idx, "lotto_score"] - 10.0
            note = f"hr-low-rate:{pa}PA-{hr}HR;"

        elif pa >= 30 and hr == 0:
            # Soft penalty — moderate sample, still hitless
            out.at[idx, "lotto_score"] = out.at[idx, "lotto_score"] - 8.0
            note = f"hr-caution:{pa}PA-0HR;"

        if note:
            existing = str(out.at[idx, "notes"] or "").strip()
            out.at[idx, "notes"] = (existing + " " + note).strip()
            out.at[idx, "hr_gate_note"] = note

    # Clip scores
    out["lotto_score"] = pd.to_numeric(out["lotto_score"], errors="coerce").clip(20.0, 85.0)

    gated   = out[hr_mask & ~out["hard_gate_pass"]].shape[0] if "hard_gate_pass" in out.columns else 0
    penalized = out[hr_mask & out["hr_gate_note"].notna()].shape[0] if "hr_gate_note" in out.columns else 0
    print(f"[batter_hr_gate] gated={gated}  penalized={penalized}  hr_rows={hr_mask.sum()}")

    return out
