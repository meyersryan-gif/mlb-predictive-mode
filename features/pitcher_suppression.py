from __future__ import annotations

import re
import unicodedata
import pandas as pd
from pybaseball import pitching_stats, statcast_pitcher_percentile_ranks


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


def _num(df: pd.DataFrame, candidates: list[str], default=0.0) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")


def _pct_rank(s: pd.Series, ascending: bool = True, default=50.0) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(default, index=s.index, dtype="float64")
    pct = s.rank(pct=True, method="average", ascending=ascending) * 100.0
    return pct.fillna(default).clip(0, 100)


FG_FALLBACK_PATHS = [
    f"fg_pitching_{{season}}.csv",
    f"mlb/data/fg_pitching_{{season}}.csv",
]


def _load_fg_fallback(season: int) -> pd.DataFrame:
    """
    Load FanGraphs pitching CSV exported manually from fangraphs.com.
    Place file at project root as fg_pitching_2025.csv (or mlb/data/).
    Columns needed: Name, HR/9, FIP, xFIP
    """
    from pathlib import Path
    for template in FG_FALLBACK_PATHS:
        path = Path(template.format(season=season))
        if path.exists():
            try:
                df = pd.read_csv(path)
                print(f"[pitcher_supp] FanGraphs fallback loaded: {path} ({len(df)} rows)")
                return df
            except Exception as e:
                print(f"[pitcher_supp] FanGraphs fallback read error: {e}")
    return pd.DataFrame()


def _build_one_season(season: int) -> pd.DataFrame:
    frames = []

    # FanGraphs pitching stats — try live pybaseball first, then local CSV fallback
    fg = pd.DataFrame()
    try:
        fg = pitching_stats(season, season, qual=0).copy()
        if fg is not None and not fg.empty:
            print(f"[pitcher_supp] FanGraphs live pull OK: {len(fg)} rows")
    except Exception as e:
        print(f"[pitcher_supp] FanGraphs live pull failed ({e}) — trying local fallback")
        fg = _load_fg_fallback(season)

    # If FanGraphs unavailable, build FIP/xFIP from Statcast via pybaseball
    if fg is None or fg.empty:
        try:
            from pybaseball import pitching_stats_bref
            bref = pitching_stats_bref(season).copy()
            if bref is not None and not bref.empty:
                print(f"[pitcher_supp] Baseball Reference fallback OK: {len(bref)} rows")
                # B-Ref columns: Name, FIP (if available)
                if "FIP" in bref.columns:
                    bref["Name"] = bref.get("Name", bref.index)
                    fg = bref
        except Exception as e2:
            print(f"[pitcher_supp] Baseball Reference fallback failed ({e2})")

    # Last resort — compute FIP proxy from Statcast expected stats
    if fg is None or fg.empty:
        try:
            from pybaseball import statcast_pitcher_expected_stats
            sc_exp = statcast_pitcher_expected_stats(season).copy()
            if sc_exp is not None and not sc_exp.empty:
                print(f"[pitcher_supp] Statcast expected stats fallback OK: {len(sc_exp)} rows")
                # Map to FIP proxy using xwOBA and xERA as proxies
                name_col = next((c for c in ["player_name","last_name, first_name","name","Name"] if c in sc_exp.columns), None)
                if name_col is None:
                    print(f"[pitcher_supp] Statcast expected stats: no name column found. Cols={list(sc_exp.columns[:10])}")
                else:
                    sc_exp["pitcher_name"] = sc_exp[name_col].astype(str)
                    sc_exp["pitcher_name_norm"] = _canon_name(sc_exp["pitcher_name"])
                    xera = _num(sc_exp, ["xera", "xERA", "est_era"], 4.0)
                    sc_exp["FIP"]  = xera
                    sc_exp["xFIP"] = xera
                    sc_exp["Name"] = sc_exp["pitcher_name"]
                    fg = sc_exp
                    print(f"[pitcher_supp] Using xERA as FIP/xFIP proxy — name_col={name_col} rows={len(fg)}")
        except Exception as e3:
            print(f"[pitcher_supp] Statcast expected stats failed ({e3})")

    if fg is not None and not fg.empty:
        fg["pitcher_name"] = fg["Name"].astype(str)
        fg["pitcher_name_norm"] = _canon_name(fg["pitcher_name"])

        hr9 = _num(fg, ["HR/9", "HR9"], 0.0)
        fip = _num(fg, ["FIP"], 0.0)
        xfip = _num(fg, ["xFIP"], 0.0)

        fg_out = pd.DataFrame({
            "Season": season,
            "pitcher_name_norm": fg["pitcher_name_norm"],
            "pctl_pitcher_hr9": _pct_rank(hr9, ascending=True, default=50.0),
            "pctl_pitcher_fip": _pct_rank(fip, ascending=True, default=50.0),
            "pctl_pitcher_xfip": _pct_rank(xfip, ascending=True, default=50.0),
        })
        frames.append(fg_out)

    # Statcast percentile ranks
    try:
        sc = statcast_pitcher_percentile_ranks(season).copy()
    except Exception:
        sc = pd.DataFrame()

    if sc is not None and not sc.empty:
        sc["pitcher_name"] = sc["player_name"].astype(str)
        sc["pitcher_name_norm"] = _canon_name(sc["pitcher_name"])

        k_pct = _num(sc, ["k_percent", "K%", "k_pct"], 0.0)
        bb_pct = _num(sc, ["bb_percent", "BB%", "bb_pct"], 0.0)

        # If direct rates are missing, try percentile columns
        if (k_pct == 0).all():
            k_pct = _num(sc, ["strikeout_percentile", "k_percentile"], 50.0)
        else:
            k_pct = _pct_rank(k_pct, ascending=True, default=50.0)

        if (bb_pct == 0).all():
            bb_pct = _num(sc, ["walk_percentile", "bb_percentile"], 50.0)
            bb_pct = 100.0 - bb_pct
        else:
            bb_pct = _pct_rank(bb_pct, ascending=False, default=50.0)

        sc_out = pd.DataFrame({
            "Season": season,
            "pitcher_name_norm": sc["pitcher_name_norm"],
            "pctl_pitcher_k_pct": pd.to_numeric(k_pct, errors="coerce").fillna(50.0).clip(0, 100),
            "pctl_pitcher_bb_pct": pd.to_numeric(bb_pct, errors="coerce").fillna(50.0).clip(0, 100),
        })
        frames.append(sc_out)

    if not frames:
        return pd.DataFrame(columns=[
            "Season",
            "pitcher_name_norm",
            "pctl_pitcher_k_pct",
            "pctl_pitcher_bb_pct",
            "pctl_pitcher_hr9",
            "pctl_pitcher_fip",
            "pctl_pitcher_xfip",
        ])

    merged = frames[0]
    for nxt in frames[1:]:
        merged = merged.merge(nxt, on=["Season", "pitcher_name_norm"], how="outer")

    for col in [
        "pctl_pitcher_k_pct",
        "pctl_pitcher_bb_pct",
        "pctl_pitcher_hr9",
        "pctl_pitcher_fip",
        "pctl_pitcher_xfip",
    ]:
        if col not in merged.columns:
            merged[col] = 50.0
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(50.0).clip(0, 100)

    return merged


def load_pitcher_suppression_stats(start_season: int, end_season: int | None = None, qual: int = 0) -> pd.DataFrame:
    if end_season is None:
        end_season = start_season

    frames = []
    for season in range(start_season, end_season + 1):
        one = _build_one_season(season)
        if one is not None and not one.empty:
            frames.append(one)

    if not frames:
        return pd.DataFrame(columns=[
            "pitcher_name_norm",
            "pctl_pitcher_k_pct",
            "pctl_pitcher_bb_pct",
            "pctl_pitcher_hr9",
            "pctl_pitcher_fip",
            "pctl_pitcher_xfip",
        ])

    out = pd.concat(frames, ignore_index=True)

    # Prefer the newest season if duplicates exist
    out = (
        out.sort_values(["pitcher_name_norm", "Season"])
           .drop_duplicates(subset=["pitcher_name_norm"], keep="last")
           .reset_index(drop=True)
    )

    return out[[
        "pitcher_name_norm",
        "pctl_pitcher_k_pct",
        "pctl_pitcher_bb_pct",
        "pctl_pitcher_hr9",
        "pctl_pitcher_fip",
        "pctl_pitcher_xfip",
    ]]