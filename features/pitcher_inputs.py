from __future__ import annotations

import re
import unicodedata
import numpy as np
import pandas as pd
import requests
from pybaseball import statcast_pitcher_percentile_ranks

from mlb.data_sources.mlb_stats_api import BASE


# ---------------------------------------------------------------------------
# League-wide reference distributions for percentile ranking.
# ---------------------------------------------------------------------------
_LEAGUE_K_RATE_MEAN  = 0.220
_LEAGUE_K_RATE_STD   = 0.045
_LEAGUE_BB_RATE_MEAN = 0.085
_LEAGUE_BB_RATE_STD  = 0.025
_LEAGUE_GB_RATE_MEAN = 0.440
_LEAGUE_GB_RATE_STD  = 0.065


# ---------------------------------------------------------------------------
# Blend weight ramp — conservative, prior season dominates until ~10 starts.
# Early April has 4-6 starts which is not enough sample to trust over a
# full prior season. Aggressive weighting caused systematic under-projection
# by burying legitimate baseline K% under noisy small-sample current data.
#
# Schedule:
#   0  starts -> 0.00 (pure prior)
#   1  start  -> 0.10
#   2  starts -> 0.20
#   3  starts -> 0.30
#   4  starts -> 0.40
#   5  starts -> 0.45
#   6-7 starts-> 0.50
#   8-9 starts-> 0.55
#   10-13     -> 0.65
#   14-17     -> 0.75
#   18+       -> 0.80
# ---------------------------------------------------------------------------
def _current_season_weight(n_starts: int) -> float:
    if n_starts <= 0:  return 0.00
    if n_starts == 1:  return 0.10
    if n_starts == 2:  return 0.20
    if n_starts == 3:  return 0.30
    if n_starts == 4:  return 0.40
    if n_starts == 5:  return 0.45
    if n_starts <= 7:  return 0.50
    if n_starts <= 9:  return 0.55
    if n_starts <= 13: return 0.65
    if n_starts <= 17: return 0.75
    return 0.80


def _pick_first_existing(df: pd.DataFrame, candidates: list[str], default=None):
    for c in candidates:
        if c in df.columns:
            return df[c]
    if default is None:
        return pd.Series([pd.NA] * len(df), index=df.index)
    return pd.Series([default] * len(df), index=df.index)


def _num(df: pd.DataFrame, candidates: list[str], default=0.0) -> pd.Series:
    for c in candidates:
        if c and c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def _canon_name(s: pd.Series) -> pd.Series:
    def one(v):
        if pd.isna(v):
            return ""
        v = str(v)
        v = unicodedata.normalize("NFKD", v).encode("ascii", "ignore").decode("ascii")
        v = v.strip().lower()
        if "," in v:
            parts = [p.strip() for p in v.split(",", 1)]
            if len(parts) == 2:
                v = f"{parts[1]} {parts[0]}"
        v = re.sub(r"[^a-z0-9 ]+", "", v)
        v = re.sub(r"\s+", " ", v).strip()
        return v.replace(" ", "")
    return s.apply(one)


def _find_col(df: pd.DataFrame, patterns: list[str]) -> str | None:
    cols = list(df.columns)
    for pat in patterns:
        rx = re.compile(pat, re.I)
        for c in cols:
            if rx.search(c):
                return c
    return None


def _clip_pct(s: pd.Series, default=50.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(default).clip(0, 100)


def _pct_rank(s: pd.Series, ascending: bool = True, default=50.0) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(default, index=s.index, dtype="float64")
    pct = s.rank(pct=True, method="average", ascending=ascending) * 100.0
    return pct.fillna(default).clip(0, 100)


def _rate_to_league_pctile(rate: float, mean: float, std: float, ascending: bool = True) -> float:
    if pd.isna(rate) or std <= 0:
        return 50.0
    from scipy.stats import norm
    z      = (rate - mean) / std
    pctile = norm.cdf(z) * 100.0
    if not ascending:
        pctile = 100.0 - pctile
    return float(np.clip(pctile, 1.0, 99.0))


def _blend(prior_val, current_val, n_starts: int, default):
    prior_val   = pd.to_numeric(pd.Series([prior_val]),   errors="coerce").iloc[0]
    current_val = pd.to_numeric(pd.Series([current_val]), errors="coerce").iloc[0]
    if pd.isna(prior_val) and pd.isna(current_val):
        return float(default)
    if pd.isna(prior_val):
        return float(current_val)
    if pd.isna(current_val):
        return float(prior_val)
    w_cur   = _current_season_weight(int(n_starts))
    w_prior = 1.0 - w_cur
    return float((w_prior * prior_val) + (w_cur * current_val))


def _ip_str_to_float(ip_val) -> float:
    if ip_val is None or pd.isna(ip_val):
        return np.nan
    try:
        s = str(ip_val)
        if "." in s:
            whole, frac = s.split(".", 1)
            whole = int(whole)
            frac  = int(frac)
            if frac == 0: return float(whole)
            if frac == 1: return whole + (1.0 / 3.0)
            if frac == 2: return whole + (2.0 / 3.0)
        return float(s)
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Statcast baseline — PRIMARY name-match source (711 pitchers, full coverage)
# ---------------------------------------------------------------------------

def _load_statcast_baseline(season: int) -> pd.DataFrame:
    """
    Load full Statcast percentile ranks for a season.
    This is the PRIMARY baseline source because it has the best name coverage
    (~711 pitchers) and includes K%, BB%, SwStr, CSW, xERA, GB%.

    Falls back gracefully if FanGraphs is blocked (403).
    In that case the MLB Stats API supplement below fills the gap.
    """
    try:
        raw = statcast_pitcher_percentile_ranks(season).copy()
    except Exception as e:
        print(f"[pitcher_inputs] Statcast unavailable for {season}: {e}")
        return pd.DataFrame()

    if raw is None or raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    df["mlbID"]   = _pick_first_existing(df, ["player_id", "playerid", "mlbID", "key_mlbam"])
    df["Name"]    = _pick_first_existing(
        df, ["player_name", "name", "Name", "last_name, first_name"], default=""
    ).astype(str)
    df["NameKey"] = _canon_name(df["Name"])

    k_col     = _find_col(df, [r"(^|_)k(_|$)", r"strikeout.*percent", r"k.*percent"])
    bb_col    = _find_col(df, [r"(^|_)bb(_|$)", r"walk.*percent"])
    xera_col  = _find_col(df, [r"xera"])
    gb_col    = _find_col(df, [r"gb.*percent", r"ground.*ball"])
    csw_col   = _find_col(df, [r"csw"])
    swstr_col = _find_col(df, [r"swstr", r"swing.*strike"])
    ppa_col   = _find_col(df, [r"pitches.*pa", r"pitch.*per.*pa"])

    k_pctile    = _clip_pct(_num(df, [k_col],    50.0), 50.0)
    bb_pctile   = _clip_pct(_num(df, [bb_col],   50.0), 50.0)
    xera_pctile = _clip_pct(_num(df, [xera_col], 50.0), 50.0)
    gb_pctile   = _clip_pct(_num(df, [gb_col],   50.0), 50.0)
    csw_pctile  = _clip_pct(_num(df, [csw_col],  50.0), 50.0)
    swstr_pctile= _clip_pct(_num(df, [swstr_col],50.0), 50.0)
    ppa_pctile  = _clip_pct(_num(df, [ppa_col],  50.0), 50.0)

    df["pctl_k_pct"]          = k_pctile
    df["pctl_bb_pct"]         = 100 - bb_pctile
    df["pctl_gb_pct"]         = gb_pctile
    df["pctl_csw_pct"]        = csw_pctile
    df["pctl_swstr_pct"]      = swstr_pctile
    df["pctl_pitches_per_pa"] = 100 - ppa_pctile

    # Estimate IP/BF from Statcast percentiles using calibrated formula
    # This is a fallback — MLB API values will override where available
    df["avg_ip_per_start"] = (
        3.9
        + 0.020 * df["pctl_k_pct"]
        + 0.010 * xera_pctile
        + 0.006 * df["pctl_gb_pct"]
        - 0.010 * (100 - df["pctl_bb_pct"])
    ).clip(3.5, 7.2)
    df["avg_bf_per_start"] = (df["avg_ip_per_start"] * 4.25).clip(15.0, 30.0)

    keep = [
        "mlbID", "Name", "NameKey",
        "avg_ip_per_start", "avg_bf_per_start",
        "pctl_k_pct", "pctl_bb_pct", "pctl_gb_pct",
        "pctl_swstr_pct", "pctl_csw_pct", "pctl_pitches_per_pa",
    ]
    out = df[keep].copy()
    out = out.drop_duplicates(subset=["NameKey"], keep="last").reset_index(drop=True)

    print(f"[pitcher_inputs] Statcast baseline loaded: {len(out)} pitchers for {season}")
    return out


# ---------------------------------------------------------------------------
# MLB Stats API — real IP/BF supplement (overrides formula estimates)
# ---------------------------------------------------------------------------

def _fetch_real_ip_bf_from_mlbapi(season: int) -> pd.DataFrame:
    """
    Pull real IP and BF per start from MLB Stats API season aggregates.
    Used to OVERRIDE the formula-estimated IP/BF in the Statcast baseline.
    Only pitchers with GS >= 1 are included.
    """
    url    = BASE + "/stats"
    params = {
        "stats":    "season",
        "group":    "pitching",
        "gameType": "R",
        "season":   season,
        "sportId":  1,
        "limit":    2000,
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[pitcher_inputs] MLB API IP/BF fetch failed for {season}: {e}")
        return pd.DataFrame()

    rows = []
    for entry in (data.get("stats") or []):
        for split in (entry.get("splits") or []):
            person = split.get("player") or {}
            stat   = split.get("stat")   or {}

            pid  = person.get("id")
            name = person.get("fullName", "")
            gs   = pd.to_numeric(stat.get("gamesStarted", 0), errors="coerce") or 0
            ip   = _ip_str_to_float(stat.get("inningsPitched"))
            bf   = pd.to_numeric(stat.get("battersFaced"), errors="coerce")

            if not pid or gs < 1:
                continue

            avg_ip = (ip / gs) if pd.notna(ip) and gs > 0 else np.nan
            avg_bf = (bf / gs) if pd.notna(bf) and gs > 0 else np.nan

            rows.append({
                "mlbID":               int(pid),
                "NameKey":             _canon_name(pd.Series([name])).iloc[0],
                "real_avg_ip_per_start": np.clip(avg_ip, 3.5, 7.5) if pd.notna(avg_ip) else np.nan,
                "real_avg_bf_per_start": np.clip(avg_bf, 13.0, 32.0) if pd.notna(avg_bf) else np.nan,
            })

    if not rows:
        # Primary MLB season returned nothing — try Triple-A as fallback
        # Covers debut pitchers like JR Ritchie who were in MiLB last season
        print(f"[pitcher_inputs] MLB API returned no rows for {season} — trying Triple-A fallback")
        params["sportId"] = 11
        try:
            r2 = requests.get(url, params=params, timeout=30)
            r2.raise_for_status()
            data2 = r2.json()
            for entry in (data2.get("stats") or []):
                for split in (entry.get("splits") or []):
                    person2 = split.get("player") or {}
                    stat2   = split.get("stat")   or {}
                    pid2    = person2.get("id")
                    name2   = person2.get("fullName", "")
                    gs2     = pd.to_numeric(stat2.get("gamesStarted", 0), errors="coerce") or 0
                    ip2     = _ip_str_to_float(stat2.get("inningsPitched"))
                    bf2     = pd.to_numeric(stat2.get("battersFaced"), errors="coerce")
                    if not pid2 or gs2 < 3:
                        continue
                    avg_ip2 = (ip2 / gs2) if pd.notna(ip2) and gs2 > 0 else np.nan
                    avg_bf2 = (bf2 / gs2) if pd.notna(bf2) and gs2 > 0 else np.nan
                    rows.append({
                        "mlbID":                 int(pid2),
                        "NameKey":               _canon_name(pd.Series([name2])).iloc[0],
                        "real_avg_ip_per_start": np.clip(avg_ip2, 3.5, 7.5) if pd.notna(avg_ip2) else np.nan,
                        "real_avg_bf_per_start": np.clip(avg_bf2, 13.0, 32.0) if pd.notna(avg_bf2) else np.nan,
                        "_source": "milb",
                    })
            if rows:
                print(f"[pitcher_inputs] Triple-A fallback loaded: {len(rows)} pitchers for {season}")
        except Exception as e_milb:
            print(f"[pitcher_inputs] Triple-A fallback failed: {e_milb}")
        if not rows:
            return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=["mlbID"], keep="last").reset_index(drop=True)
    print(f"[pitcher_inputs] MLB API real IP/BF loaded: {len(out)} pitchers for {season}")
    return out


def load_pitcher_baseline_stats(
    start_season: int,
    end_season: int | None = None,
    qual: int = 0,
) -> pd.DataFrame:
    """
    Load pitcher baseline using Statcast as primary (711 pitchers, best coverage)
    with MLB Stats API real IP/BF overriding formula estimates where available.

    Data flow:
      1. Statcast percentile ranks  -> K%, BB%, SwStr, CSW, GB% + formula IP/BF
      2. MLB Stats API season aggs  -> real IP/BF overrides formula estimates
      3. Merge on NameKey + mlbID   -> best of both sources
    """
    if end_season is None:
        end_season = start_season

    all_frames = []

    for season in range(start_season, end_season + 1):
        # Step 1: Statcast — primary source, best name coverage
        sc_df = _load_statcast_baseline(season)

        if sc_df.empty:
            print(f"[pitcher_inputs] No Statcast data for {season} — skipping")
            continue

        # Step 2: MLB Stats API — real IP/BF supplement
        mlb_df = _fetch_real_ip_bf_from_mlbapi(season)

        if not mlb_df.empty:
            # Merge by NameKey first
            sc_df = sc_df.merge(
                mlb_df[["NameKey", "real_avg_ip_per_start", "real_avg_bf_per_start"]],
                on="NameKey",
                how="left",
            )
            # Also try mlbID merge for any misses
            if "mlbID" in sc_df.columns:
                sc_df["mlbID"] = pd.to_numeric(sc_df["mlbID"], errors="coerce")
                mlb_id_df = mlb_df[["mlbID", "real_avg_ip_per_start", "real_avg_bf_per_start"]].copy()
                mlb_id_df["mlbID"] = pd.to_numeric(mlb_id_df["mlbID"], errors="coerce")

                # Fill missing real values via mlbID
                missing_mask = sc_df["real_avg_ip_per_start"].isna()
                if missing_mask.any():
                    id_lookup = mlb_id_df.set_index("mlbID")
                    for idx in sc_df[missing_mask].index:
                        mid = sc_df.at[idx, "mlbID"]
                        if pd.notna(mid) and mid in id_lookup.index:
                            sc_df.at[idx, "real_avg_ip_per_start"] = id_lookup.at[mid, "real_avg_ip_per_start"]
                            sc_df.at[idx, "real_avg_bf_per_start"] = id_lookup.at[mid, "real_avg_bf_per_start"]

            # Override formula estimates with real values where available
            real_ip = pd.to_numeric(sc_df["real_avg_ip_per_start"], errors="coerce")
            real_bf = pd.to_numeric(sc_df["real_avg_bf_per_start"], errors="coerce")
            sc_df["avg_ip_per_start"] = real_ip.fillna(
                pd.to_numeric(sc_df["avg_ip_per_start"], errors="coerce")
            )
            sc_df["avg_bf_per_start"] = real_bf.fillna(
                pd.to_numeric(sc_df["avg_bf_per_start"], errors="coerce")
            )

            real_ip_count = real_ip.notna().sum()
            print(f"[pitcher_inputs] Real IP/BF applied to {real_ip_count}/{len(sc_df)} pitchers")

        sc_df["Season"] = season

        # Step 3: Triple-A supplement — covers debut pitchers not in Statcast pool
        # Pitchers like JR Ritchie who were in MiLB last season get a real prior
        # instead of neutral 50th percentile defaults
        try:
            milb_url    = BASE + "/stats"
            milb_params = {"stats": "season", "group": "pitching", "gameType": "R",
                           "season": season, "sportId": 11, "limit": 2000}
            milb_r    = requests.get(milb_url, params=milb_params, timeout=30)
            milb_r.raise_for_status()
            milb_data = milb_r.json()
            milb_rows = []
            for entry in (milb_data.get("stats") or []):
                for split in (entry.get("splits") or []):
                    p2   = split.get("player") or {}
                    s2   = split.get("stat")   or {}
                    pid2 = p2.get("id")
                    nm2  = p2.get("fullName", "")
                    gs2  = pd.to_numeric(s2.get("gamesStarted", 0), errors="coerce") or 0
                    ip2  = _ip_str_to_float(s2.get("inningsPitched"))
                    bf2  = pd.to_numeric(s2.get("battersFaced"),  errors="coerce")
                    k2   = pd.to_numeric(s2.get("strikeOuts"),    errors="coerce")
                    bb2  = pd.to_numeric(s2.get("baseOnBalls"),   errors="coerce")
                    if not pid2 or gs2 < 3:
                        continue
                    k_rate2  = (k2  / bf2) if pd.notna(k2)  and pd.notna(bf2) and bf2 > 0 else np.nan
                    bb_rate2 = (bb2 / bf2) if pd.notna(bb2) and pd.notna(bf2) and bf2 > 0 else np.nan
                    avg_ip2  = (ip2 / gs2) if pd.notna(ip2) and gs2 > 0 else np.nan
                    avg_bf2  = (bf2 / gs2) if pd.notna(bf2) and gs2 > 0 else np.nan
                    nk2 = _canon_name(pd.Series([nm2])).iloc[0]
                    milb_rows.append({"mlbID": int(pid2), "Name": nm2, "NameKey": nk2,
                        "avg_ip_per_start": np.clip(avg_ip2, 3.5, 6.5) if pd.notna(avg_ip2) else 5.0,
                        "avg_bf_per_start": np.clip(avg_bf2, 13.0, 28.0) if pd.notna(avg_bf2) else 22.0,
                        "pctl_k_pct":  _rate_to_league_pctile(k_rate2  * 0.92, _LEAGUE_K_RATE_MEAN,  _LEAGUE_K_RATE_STD,  True)  if pd.notna(k_rate2)  else 50.0,
                        "pctl_bb_pct": _rate_to_league_pctile(bb_rate2 * 0.92, _LEAGUE_BB_RATE_MEAN, _LEAGUE_BB_RATE_STD, False) if pd.notna(bb_rate2) else 50.0,
                        "pctl_gb_pct": 50.0, "pctl_swstr_pct": 50.0, "pctl_csw_pct": 50.0,
                        "pctl_pitches_per_pa": 50.0, "Season": season})
            if milb_rows:
                milb_df2      = pd.DataFrame(milb_rows)
                existing_keys = set(sc_df["NameKey"].tolist())
                new_milb      = milb_df2[~milb_df2["NameKey"].isin(existing_keys)].copy()
                if not new_milb.empty:
                    for col in sc_df.columns:
                        if col not in new_milb.columns:
                            new_milb[col] = np.nan
                    sc_df = pd.concat([sc_df, new_milb[sc_df.columns]], ignore_index=True)
                    print(f"[pitcher_inputs] Triple-A supplement: {len(new_milb)} debut pitchers added for {season}")
        except Exception as e_milb:
            print(f"[pitcher_inputs] Triple-A supplement failed: {e_milb}")
        all_frames.append(sc_df)

    if not all_frames:
        return pd.DataFrame(columns=[
            "mlbID", "Name", "NameKey", "Season",
            "avg_ip_per_start", "avg_bf_per_start",
            "pctl_k_pct", "pctl_bb_pct", "pctl_gb_pct",
            "pctl_swstr_pct", "pctl_csw_pct", "pctl_pitches_per_pa",
        ])

    out = (
        pd.concat(all_frames, ignore_index=True)
        .sort_values(["NameKey", "Season"])
        .drop_duplicates(subset=["NameKey"], keep="last")
        .reset_index(drop=True)
    )

    for col, default in [
        ("avg_ip_per_start",    5.1),
        ("avg_bf_per_start",    22.5),
        ("pctl_k_pct",          50.0),
        ("pctl_bb_pct",         50.0),
        ("pctl_gb_pct",         50.0),
        ("pctl_swstr_pct",      50.0),
        ("pctl_csw_pct",        50.0),
        ("pctl_pitches_per_pa", 50.0),
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(default)

    return out


# ---------------------------------------------------------------------------
# Current season live game logs
# ---------------------------------------------------------------------------

def _fetch_pitcher_game_logs(player_id: int, season: int) -> list[dict]:
    url    = BASE + f"/people/{player_id}/stats"
    params = {"stats": "gameLog", "group": "pitching", "season": season, "sportIds": 1}
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []
    splits = (((data or {}).get("stats") or [{}])[0].get("splits") or [])
    return splits if isinstance(splits, list) else []


def load_pitcher_current_stats_for_ids(player_ids: list[int], season: int) -> pd.DataFrame:
    """
    Pull live game logs for current season per-start averages.
    SwStr/CSW remain NaN — not available from game log endpoint.
    """
    rows = []

    for pid in sorted({int(x) for x in player_ids if pd.notna(x)}):
        splits = _fetch_pitcher_game_logs(pid, season)
        if not splits:
            continue

        g_rows = []
        for sp in splits:
            stat = (sp or {}).get("stat") or {}
            gs   = pd.to_numeric(pd.Series([stat.get("gamesStarted")]), errors="coerce").iloc[0]
            ip   = _ip_str_to_float(stat.get("inningsPitched"))
            bf   = pd.to_numeric(pd.Series([stat.get("battersFaced")]),  errors="coerce").iloc[0]
            k    = pd.to_numeric(pd.Series([stat.get("strikeOuts")]),    errors="coerce").iloc[0]
            bb   = pd.to_numeric(pd.Series([stat.get("baseOnBalls")]),   errors="coerce").iloc[0]

            is_start = (not pd.isna(gs) and float(gs) >= 1.0)
            if not is_start:
                if pd.notna(ip) and ip >= 2.0:  is_start = True
                elif pd.notna(bf) and bf >= 9:  is_start = True
            if not is_start:
                continue
            g_rows.append({"ip": ip, "bf": bf, "k": k, "bb": bb})

        if not g_rows:
            continue

        gdf      = pd.DataFrame(g_rows)
        starts   = len(gdf)
        total_bf = pd.to_numeric(gdf["bf"], errors="coerce").sum(min_count=1)
        total_k  = pd.to_numeric(gdf["k"],  errors="coerce").sum(min_count=1)
        total_bb = pd.to_numeric(gdf["bb"], errors="coerce").sum(min_count=1)
        total_ip = pd.to_numeric(gdf["ip"], errors="coerce").sum(min_count=1)

        curr_k_rate  = (total_k  / total_bf) if pd.notna(total_k)  and pd.notna(total_bf) and total_bf > 0 else np.nan
        curr_bb_rate = (total_bb / total_bf) if pd.notna(total_bb) and pd.notna(total_bf) and total_bf > 0 else np.nan

        curr_pctl_k  = _rate_to_league_pctile(curr_k_rate,  _LEAGUE_K_RATE_MEAN,  _LEAGUE_K_RATE_STD,  ascending=True)  if pd.notna(curr_k_rate)  else np.nan
        curr_pctl_bb = _rate_to_league_pctile(curr_bb_rate, _LEAGUE_BB_RATE_MEAN, _LEAGUE_BB_RATE_STD, ascending=False) if pd.notna(curr_bb_rate) else np.nan

        # Recent form — last 3 starts IP average and trend vs season avg
        recent_n      = min(3, len(gdf))
        recent_df     = gdf.tail(recent_n)
        recent_ip_raw = pd.to_numeric(recent_df["ip"], errors="coerce")
        recent_avg_ip = float(recent_ip_raw.mean()) if recent_n >= 2 and recent_ip_raw.notna().sum() >= 2 else np.nan
        season_avg_ip = float(total_ip / starts) if starts > 0 and pd.notna(total_ip) else np.nan
        ip_trend      = float(recent_avg_ip - season_avg_ip) if pd.notna(recent_avg_ip) and pd.notna(season_avg_ip) else np.nan


        rows.append({
            "pitcher_id":               pid,
            "curr_starts_used":         int(starts),
            "curr_avg_ip_per_start":    (total_ip / starts) if starts > 0 and pd.notna(total_ip) else np.nan,
            "curr_avg_bf_per_start":    (total_bf / starts) if starts > 0 and pd.notna(total_bf) else np.nan,
            "curr_k_rate_raw":          curr_k_rate,
            "curr_bb_rate_raw":         curr_bb_rate,
            "curr_pctl_k_pct":          curr_pctl_k,
            "curr_pctl_bb_pct":         curr_pctl_bb,
            "curr_pctl_swstr_pct":      np.nan,
            "curr_pctl_csw_pct":        np.nan,
            "curr_pctl_pitches_per_pa": np.nan,
            "curr_pctl_gb_pct":         np.nan,
            "recent_avg_ip_per_start":  float(np.clip(recent_avg_ip, 3.5, 7.5)) if pd.notna(recent_avg_ip) else np.nan,
            "recent_ip_trend":          round(float(ip_trend), 3) if pd.notna(ip_trend) else 0.0,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=[
            "pitcher_id", "curr_starts_used",
            "curr_avg_ip_per_start", "curr_avg_bf_per_start",
            "curr_pctl_k_pct", "curr_pctl_bb_pct",
            "curr_pctl_swstr_pct", "curr_pctl_csw_pct",
            "curr_pctl_gb_pct", "recent_avg_ip_per_start", "recent_ip_trend",
        ])

    out["curr_avg_ip_per_start"] = pd.to_numeric(out["curr_avg_ip_per_start"], errors="coerce").clip(3.5, 7.5)
    out["curr_avg_bf_per_start"] = pd.to_numeric(out["curr_avg_bf_per_start"], errors="coerce").clip(13.0, 32.0)
    if "recent_avg_ip_per_start" in out.columns:
        out["recent_avg_ip_per_start"] = pd.to_numeric(out["recent_avg_ip_per_start"], errors="coerce").clip(3.5, 7.5)
    if "recent_ip_trend" in out.columns:
        out["recent_ip_trend"] = pd.to_numeric(out["recent_ip_trend"], errors="coerce").fillna(0.0)

    keep = [
        "pitcher_id", "curr_starts_used",
        "curr_avg_ip_per_start", "curr_avg_bf_per_start",
        "curr_pctl_k_pct", "curr_pctl_bb_pct",
        "curr_pctl_swstr_pct", "curr_pctl_csw_pct",
        "curr_pctl_gb_pct", "recent_avg_ip_per_start", "recent_ip_trend",
    ]
    return out[keep].copy()


# ---------------------------------------------------------------------------
# Attach baseline to pitcher props DataFrame
# ---------------------------------------------------------------------------

def attach_pitcher_baseline_stats(
    df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    current_season: int = 2026,
) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out

    pitcher_col = None
    for c in ["pitcher_name", "Pitcher", "pitcher", "PitcherName", "player_name", "Name"]:
        if c in out.columns:
            pitcher_col = c
            break

    out["NameKey"] = _canon_name(out[pitcher_col]) if pitcher_col else ""

    if baseline_df is None or baseline_df.empty:
        baseline_df = pd.DataFrame(columns=[
            "NameKey", "mlbID",
            "avg_ip_per_start", "avg_bf_per_start",
            "pctl_k_pct", "pctl_bb_pct", "pctl_gb_pct",
            "pctl_swstr_pct", "pctl_csw_pct", "pctl_pitches_per_pa",
        ])

    base = baseline_df.copy().rename(columns={
        "mlbID":               "base_mlbID",
        "avg_ip_per_start":    "base_avg_ip_per_start",
        "avg_bf_per_start":    "base_avg_bf_per_start",
        "pctl_k_pct":          "base_pctl_k_pct",
        "pctl_bb_pct":         "base_pctl_bb_pct",
        "pctl_gb_pct":         "base_pctl_gb_pct",
        "pctl_swstr_pct":      "base_pctl_swstr_pct",
        "pctl_csw_pct":        "base_pctl_csw_pct",
        "pctl_pitches_per_pa": "base_pctl_pitches_per_pa",
    })

    keep = [c for c in [
        "NameKey", "base_mlbID",
        "base_avg_ip_per_start", "base_avg_bf_per_start",
        "base_pctl_k_pct", "base_pctl_bb_pct", "base_pctl_gb_pct",
        "base_pctl_swstr_pct", "base_pctl_csw_pct", "base_pctl_pitches_per_pa",
    ] if c in base.columns]
    base = base[keep].copy()

    # Primary merge — by NameKey
    merged = out.merge(base, on="NameKey", how="left")

    # Secondary merge — by mlbID for any unmatched rows
    if "base_mlbID" in merged.columns and "pitcher_id" in merged.columns:
        merged["pitcher_id"] = pd.to_numeric(merged["pitcher_id"], errors="coerce")
        merged["base_mlbID"] = pd.to_numeric(merged["base_mlbID"], errors="coerce")
        unmatched_mask = merged["base_avg_ip_per_start"].isna()

        if unmatched_mask.any():
            id_base = base.copy()
            if "base_mlbID" in id_base.columns:
                id_base = id_base.rename(columns={"base_mlbID": "pitcher_id"})
                id_base["pitcher_id"] = pd.to_numeric(id_base["pitcher_id"], errors="coerce")
                id_base = id_base.dropna(subset=["pitcher_id"])

                base_cols = [c for c in id_base.columns if c.startswith("base_")]

                for _, id_row in id_base.iterrows():
                    pid  = id_row["pitcher_id"]
                    mask = unmatched_mask & (merged["pitcher_id"] == pid)
                    if mask.any():
                        for col in base_cols:
                            merged.loc[mask, col] = id_row.get(col)

    # Fetch current season game logs
    player_ids = pd.to_numeric(out.get("pitcher_id"), errors="coerce").dropna().astype(int).unique().tolist()
    curr = load_pitcher_current_stats_for_ids(player_ids, current_season)

    if not curr.empty and "pitcher_id" in merged.columns:
        merged["pitcher_id"] = pd.to_numeric(merged["pitcher_id"], errors="coerce")
        curr["pitcher_id"]   = pd.to_numeric(curr["pitcher_id"],   errors="coerce")
        merged = merged.merge(curr, on="pitcher_id", how="left")

    merged["curr_starts_used"] = pd.to_numeric(
        merged.get("curr_starts_used"), errors="coerce"
    ).fillna(0).astype(int)
    merged["w_current"] = merged["curr_starts_used"].apply(_current_season_weight)
    merged["w_prior"]   = 1.0 - merged["w_current"]

    mapping = {
        "avg_ip_per_start":    ("base_avg_ip_per_start",    "curr_avg_ip_per_start",    5.1),
        "avg_bf_per_start":    ("base_avg_bf_per_start",    "curr_avg_bf_per_start",    22.5),
        "pctl_k_pct":          ("base_pctl_k_pct",          "curr_pctl_k_pct",          50.0),
        "pctl_bb_pct":         ("base_pctl_bb_pct",         "curr_pctl_bb_pct",         50.0),
        "pctl_swstr_pct":      ("base_pctl_swstr_pct",      "curr_pctl_swstr_pct",      50.0),
        "pctl_csw_pct":        ("base_pctl_csw_pct",        "curr_pctl_csw_pct",        50.0),
        "pctl_pitches_per_pa": ("base_pctl_pitches_per_pa", "curr_pctl_pitches_per_pa", 50.0),
        "pctl_gb_pct":         ("base_pctl_gb_pct",         "curr_pctl_gb_pct",         50.0),
    }

    for live_col, (base_col, curr_col, default) in mapping.items():
        merged[base_col] = pd.to_numeric(merged.get(base_col), errors="coerce")
        merged[curr_col] = pd.to_numeric(merged.get(curr_col), errors="coerce")
        merged[f"prior_{live_col}"]   = merged[base_col]
        merged[f"blended_{live_col}"] = [
            _blend(pv, cv, ns, default)
            for pv, cv, ns in zip(
                merged[base_col],
                merged[curr_col],
                merged["curr_starts_used"],
            )
        ]
        merged[live_col] = merged[f"blended_{live_col}"]

    # Pass recent form columns through directly — no blending, raw values
    if "recent_avg_ip_per_start" in merged.columns:
        merged["recent_avg_ip_per_start"] = pd.to_numeric(merged["recent_avg_ip_per_start"], errors="coerce")
    else:
        merged["recent_avg_ip_per_start"] = np.nan
    if "recent_ip_trend" in merged.columns:
        merged["recent_ip_trend"] = pd.to_numeric(merged["recent_ip_trend"], errors="coerce").fillna(0.0)
    else:
        merged["recent_ip_trend"] = 0.0

    matches      = merged["base_avg_ip_per_start"].notna().sum() if "base_avg_ip_per_start" in merged.columns else 0
    curr_matches = merged["curr_starts_used"].gt(0).sum()
    print(f"[pitcher_attach] name_matches={matches}/{len(merged)} using {pitcher_col}")
    print(f"[pitcher_attach] current_season_matches={curr_matches}/{len(merged)} season={current_season}")

    return merged







