import numpy as np
import pandas as pd


def _to_num(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def _safe_series_mean(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.mean()) if len(s) else np.nan


def _weighted_recent_mean(series: pd.Series) -> float:
    """
    More recent starts get slightly more weight.
    Assumes incoming rows are already sorted oldest -> newest.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan
    weights = np.arange(1, len(s) + 1, dtype=float)
    return float(np.average(s.values, weights=weights))


def current_season_weight(n_starts: int) -> float:
    """
    Conservative early-season blend:
      0 starts -> 0.00
      1 start  -> 0.15
      2 starts -> 0.25
      3 starts -> 0.35
      4 starts -> 0.45
      5+       -> 0.50
    """
    if n_starts <= 0:
        return 0.00
    if n_starts == 1:
        return 0.15
    if n_starts == 2:
        return 0.25
    if n_starts == 3:
        return 0.35
    if n_starts == 4:
        return 0.45
    return 0.50


def blend_stat(prior_val, current_vals: pd.Series, n_starts: int, use_recent_weighting: bool = True) -> float:
    prior_val = _to_num(prior_val)
    cur_val = (
        _weighted_recent_mean(current_vals)
        if use_recent_weighting
        else _safe_series_mean(current_vals)
    )

    w_cur = current_season_weight(n_starts)
    w_prior = 1.0 - w_cur

    if np.isnan(prior_val) and np.isnan(cur_val):
        return np.nan
    if np.isnan(prior_val):
        return cur_val
    if np.isnan(cur_val):
        return prior_val

    return float((w_prior * prior_val) + (w_cur * cur_val))


def blend_pitcher_row(
    prior_row: pd.Series,
    current_starts_df: pd.DataFrame,
    stat_map: dict,
    start_date_col: str = "game_date",
) -> dict:
    """
    prior_row: one-row Series for baseline season pitcher values
    current_starts_df: all current-season starts for this pitcher
    stat_map: mapping of output stat -> current season column name
        example:
        {
            "k_per_ip": "k_per_ip",
            "k_per_bf": "k_per_bf",
            "ip_per_start": "ip",
            "bf_per_start": "batters_faced",
            "pitches_per_start": "pitches"
        }
    """
    cur = current_starts_df.copy() if current_starts_df is not None else pd.DataFrame()

    if len(cur) and start_date_col in cur.columns:
        cur[start_date_col] = pd.to_datetime(cur[start_date_col], errors="coerce")
        cur = cur.sort_values(start_date_col)

    n_starts = int(len(cur))
    w_cur = current_season_weight(n_starts)
    w_prior = 1.0 - w_cur

    out = {
        "curr_starts_used": n_starts,
        "w_prior": round(w_prior, 3),
        "w_current": round(w_cur, 3),
    }

    for out_col, cur_col in stat_map.items():
        prior_val = prior_row.get(out_col, np.nan)
        cur_vals = cur[cur_col] if (len(cur) and cur_col in cur.columns) else pd.Series(dtype=float)

        blended_val = blend_stat(prior_val, cur_vals, n_starts=n_starts, use_recent_weighting=True)

        out[f"prior_{out_col}"] = _to_num(prior_val)
        out[f"curr_{out_col}"] = (
            _weighted_recent_mean(cur_vals) if len(cur_vals) else np.nan
        )
        out[f"blended_{out_col}"] = blended_val

    return out


def apply_early_season_under_k_penalty(
    df: pd.DataFrame,
    side_col: str = "side",
    conf_col: str = "confidence_score",
    starts_col: str = "curr_starts_used",
    prior_col: str = "prior_k_per_ip",
    curr_col: str = "curr_k_per_ip",
) -> pd.DataFrame:
    """
    Penalize strikeout UNDER confidence if current-season K/IP is running
    materially above prior baseline.
    """
    out = df.copy()

    for c in [starts_col, prior_col, curr_col, conf_col]:
        if c not in out.columns:
            return out

    delta = pd.to_numeric(out[curr_col], errors="coerce") - pd.to_numeric(out[prior_col], errors="coerce")
    starts = pd.to_numeric(out[starts_col], errors="coerce").fillna(0)

    penalty = np.select(
        [
            (starts >= 2) & (delta >= 0.30),
            (starts >= 2) & (delta >= 0.20),
            (starts >= 2) & (delta >= 0.10),
        ],
        [12, 8, 4],
        default=0,
    )

    if side_col in out.columns:
        under_mask = out[side_col].astype(str).str.lower().eq("under")
        out.loc[under_mask, conf_col] = (
            pd.to_numeric(out.loc[under_mask, conf_col], errors="coerce").fillna(0) - penalty[under_mask]
        ).clip(lower=1, upper=99)

    return out
