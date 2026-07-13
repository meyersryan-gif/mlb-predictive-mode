from __future__ import annotations

import pandas as pd


def _num(df: pd.DataFrame, col: str, default=0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")


def select_best_lotto(df: pd.DataFrame):
    if df is None or len(df) == 0:
        return None

    cand = df.copy()

    if "hard_gate_pass" in cand.columns:
        cand = cand[cand["hard_gate_pass"].fillna(False)]

    if len(cand) == 0:
        return None

    market = cand.get("market", pd.Series("", index=cand.index)).astype(str).str.upper()
    side = cand.get("side", pd.Series("", index=cand.index)).astype(str).str.upper()
    line = pd.to_numeric(cand.get("line", pd.Series(pd.NA, index=cand.index)), errors="coerce")
    odds_american = _num(cand, "odds_american", default=0.0)

    cand = cand[side.eq("OVER")]
    if len(cand) == 0:
        return None

    market = cand["market"].astype(str).str.upper()
    line = pd.to_numeric(cand["line"], errors="coerce")
    odds_american = _num(cand, "odds_american", default=0.0)

    valid_mask = pd.Series(True, index=cand.index)

    valid_mask &= ~(market.eq("HR") & (line > 0.5))
    valid_mask &= ~(market.eq("HITS") & (line > 1.5))
    valid_mask &= ~(market.eq("TB") & (line > 2.5))
    valid_mask &= ~(market.eq("RBI") & (line > 1.5))

    valid_mask &= ~(market.eq("HR") & ((odds_american < 300) | (odds_american > 1500)))
    valid_mask &= ~(market.isin(["HITS", "TB", "RBI"]) & (odds_american > 2500))

    if "batter_team" in cand.columns:
        valid_mask &= cand["batter_team"].notna()
    if "pitcher_name" in cand.columns:
        valid_mask &= cand["pitcher_name"].notna()

    cand = cand[valid_mask].copy()
    if len(cand) == 0:
        return None

    cand["__rank_score"] = (
        _num(cand, "lotto_score", 0.0)
        + 0.15 * _num(cand, "confidence_score", 0.0)
    )
    cand = cand.sort_values(
        ["__rank_score", "lotto_score", "confidence_score"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    hr = cand[cand["market"].astype(str).str.upper().eq("HR")].copy().reset_index(drop=True)

    best_pair = None
    best_pair_score = -1e9

    if len(hr) >= 2:
        for i in range(len(hr)):
            for j in range(i + 1, len(hr)):
                a = hr.iloc[i]
                b = hr.iloc[j]

                # keep these hard bans
                if str(a.get("pitcher_name")) == str(b.get("pitcher_name")):
                    continue
                if str(a.get("batter_team")) == str(b.get("batter_team")):
                    continue

                pair_score = float(a.get("lotto_score", 0.0)) + float(b.get("lotto_score", 0.0))

                # small bonus for different games, but do not require it
                if str(a.get("game_id")) != str(b.get("game_id")):
                    pair_score += 1.0

                if pair_score > best_pair_score:
                    best_pair_score = pair_score
                    best_pair = (a.to_dict(), b.to_dict())

    if best_pair is not None:
        return {
            "type": "HR_2_LEG",
            "leg1": best_pair[0],
            "leg2": best_pair[1],
            "units": 0.10,
        }

    best_pair = None
    best_pair_score = -1e9

    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            a = cand.iloc[i]
            b = cand.iloc[j]

            if str(a.get("pitcher_name")) == str(b.get("pitcher_name")):
                continue
            if str(a.get("batter_team")) == str(b.get("batter_team")):
                continue

            pair_score = float(a.get("lotto_score", 0.0)) + float(b.get("lotto_score", 0.0))

            markets = {str(a.get("market")).upper(), str(b.get("market")).upper()}
            if "HR" in markets and len(markets) > 1:
                pair_score += 1.0

            if str(a.get("game_id")) != str(b.get("game_id")):
                pair_score += 0.5

            if pair_score > best_pair_score:
                best_pair_score = pair_score
                best_pair = (a.to_dict(), b.to_dict())

    if best_pair is not None:
        return {
            "type": "MIXED_2_LEG",
            "leg1": best_pair[0],
            "leg2": best_pair[1],
            "units": 0.10,
        }

    return None
