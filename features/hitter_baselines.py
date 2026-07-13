from __future__ import annotations

import re
import unicodedata
import pandas as pd
from pybaseball import statcast_batter_expected_stats, statcast_batter_percentile_ranks


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
            return pd.to_numeric(df[c], errors="coerce").fillna(default)
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


def _pct_rank(s: pd.Series, ascending: bool = True, default=50.0) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(default, index=s.index, dtype="float64")
    pct = s.rank(pct=True, method="average", ascending=ascending) * 100.0
    return pct.fillna(default).clip(0, 100)


def load_hitter_baseline_stats(start_season: int, end_season: int | None = None, qual: int = 0) -> pd.DataFrame:
    if end_season is None:
        end_season = start_season

    frames = []

    for season in range(start_season, end_season + 1):
        try:
            exp = statcast_batter_expected_stats(season, 1).copy()
        except TypeError:
            exp = statcast_batter_expected_stats(season, minPA=1).copy()
        except Exception:
            exp = pd.DataFrame()

        try:
            pct = statcast_batter_percentile_ranks(season).copy()
        except Exception:
            pct = pd.DataFrame()

        if exp is None or exp.empty:
            continue

        df = exp.copy()
        df["Season"] = season

        df["batter_id"] = _pick_first_existing(df, ["player_id", "playerid", "mlbID", "key_mlbam", "batter"])
        df["batter_name"] = _pick_first_existing(
            df,
            ["last_name, first_name", "player_name", "name", "Name"],
            default=""
        ).astype(str)
        df["batter_name_norm"] = _canon_name(df["batter_name"])

        # expected stats source
        pa = _num(df, ["pa"], 0.0)
        ba = _num(df, ["ba"], 0.0)
        est_ba = _num(df, ["est_ba"], 0.0)
        slg = _num(df, ["slg"], 0.0)
        est_slg = _num(df, ["est_slg"], 0.0)
        woba = _num(df, ["woba"], 0.0)
        est_woba = _num(df, ["est_woba"], 0.0)

        df["iso_raw"] = (slg - ba).clip(lower=0.0)
        df["ops_raw"] = ba + slg
        df["avg_raw"] = ba
        df["xba_raw"] = est_ba.where(est_ba > 0, ba)
        df["woba_raw"] = est_woba.where(est_woba > 0, woba)

        # percentile source
        if pct is not None and not pct.empty:
            p = pct.copy()
            p["batter_id"] = _pick_first_existing(p, ["player_id", "playerid", "mlbID", "key_mlbam"])
            p["batter_name"] = _pick_first_existing(
                p,
                ["player_name", "name", "Name", "last_name, first_name"],
                default=""
            ).astype(str)
            p["batter_name_norm"] = _canon_name(p["batter_name"])

            p["xiso_raw"] = _num(p, ["xiso"], 0.0)
            p["xwoba_raw"] = _num(p, ["xwoba"], 0.0)
            p["xba_pct_src"] = _num(p, ["xba"], 50.0)
            p["barrel_pct_src"] = _num(p, ["brl_percent"], 50.0)
            p["hardhit_pct_src"] = _num(p, ["hard_hit_percent"], 50.0)
            p["k_pct_src"] = _num(p, ["k_percent"], 50.0)
            p["bb_pct_src"] = _num(p, ["bb_percent"], 50.0)
            p["xslg_raw"] = _num(p, ["xslg"], 0.0)

            p_small = p[[
                "batter_id", "batter_name_norm",
                "xiso_raw", "xwoba_raw", "xba_pct_src", "xslg_raw",
                "barrel_pct_src", "hardhit_pct_src", "k_pct_src", "bb_pct_src"
            ]].copy()

            if df["batter_id"].notna().sum() > 0 and p_small["batter_id"].notna().sum() > 0:
                df = df.merge(
                    p_small.drop(columns=["batter_name_norm"]),
                    on="batter_id",
                    how="left"
                )
            else:
                df = df.merge(
                    p_small.drop(columns=["batter_id"]),
                    on="batter_name_norm",
                    how="left"
                )
        else:
            df["xiso_raw"] = 0.0
            df["xwoba_raw"] = 0.0
            df["xba_pct_src"] = 50.0
            df["xslg_raw"] = 0.0
            df["barrel_pct_src"] = 50.0
            df["hardhit_pct_src"] = 50.0
            df["k_pct_src"] = 50.0
            df["bb_pct_src"] = 50.0

        hr_proxy = (
            0.45 * _num(df, ["xiso_raw"], 0.0) +
            0.35 * _num(df, ["barrel_pct_src"], 50.0) / 100.0 +
            0.20 * _num(df, ["hardhit_pct_src"], 50.0) / 100.0
        )

        df["pctl_iso_vs_hand"] = _pct_rank(df["iso_raw"], ascending=True, default=50.0)
        df["pctl_woba_vs_hand"] = _pct_rank(df["woba_raw"], ascending=True, default=50.0)
        df["pctl_ops_vs_hand"] = _pct_rank(df["ops_raw"], ascending=True, default=50.0)
        df["pctl_hr_rate"] = _pct_rank(hr_proxy, ascending=True, default=50.0)
        df["pctl_barrel_pct"] = pd.to_numeric(df["barrel_pct_src"], errors="coerce").fillna(50.0).clip(0, 100)
        df["pctl_hardhit_pct"] = pd.to_numeric(df["hardhit_pct_src"], errors="coerce").fillna(50.0).clip(0, 100)
        df["pctl_k_pct"] = 100.0 - pd.to_numeric(df["k_pct_src"], errors="coerce").fillna(50.0).clip(0, 100)
        df["pctl_bb_pct"] = pd.to_numeric(df["bb_pct_src"], errors="coerce").fillna(50.0).clip(0, 100)
        df["pctl_avg_vs_hand"] = _pct_rank(df["avg_raw"], ascending=True, default=50.0)
        df["pctl_xba_vs_hand"] = _pct_rank(df["xba_raw"], ascending=True, default=50.0)

        keep = [
            "Season",
            "batter_id",
            "batter_name",
            "batter_name_norm",
            "pctl_iso_vs_hand",
            "pctl_woba_vs_hand",
            "pctl_ops_vs_hand",
            "pctl_hr_rate",
            "pctl_barrel_pct",
            "pctl_hardhit_pct",
            "pctl_k_pct",
            "pctl_bb_pct",
            "pctl_avg_vs_hand",
            "pctl_xba_vs_hand",
        ]
        frames.append(df[keep].copy())

    if not frames:
        return pd.DataFrame(columns=[
            "Season", "batter_id", "batter_name", "batter_name_norm",
            "pctl_iso_vs_hand", "pctl_woba_vs_hand", "pctl_ops_vs_hand",
            "pctl_hr_rate", "pctl_barrel_pct", "pctl_hardhit_pct",
            "pctl_k_pct", "pctl_bb_pct", "pctl_avg_vs_hand", "pctl_xba_vs_hand",
        ])

    out = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["batter_name_norm"], keep="first")
        .reset_index(drop=True)
    )

    for col in [
        "pctl_iso_vs_hand", "pctl_woba_vs_hand", "pctl_ops_vs_hand",
        "pctl_hr_rate", "pctl_barrel_pct", "pctl_hardhit_pct",
        "pctl_k_pct", "pctl_bb_pct", "pctl_avg_vs_hand", "pctl_xba_vs_hand",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(50.0).clip(0, 100)

    return out


def attach_hitter_baseline_stats(df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out

    batter_col = None
    for c in ["batter_name", "Batter", "batter", "player_name", "Name"]:
        if c in out.columns:
            batter_col = c
            break

    out["batter_name_norm"] = _canon_name(out[batter_col]) if batter_col else ""

    if baseline_df is None or baseline_df.empty:
        for col in [
            "pctl_iso_vs_hand", "pctl_woba_vs_hand", "pctl_ops_vs_hand",
            "pctl_hr_rate", "pctl_barrel_pct", "pctl_hardhit_pct",
            "pctl_k_pct", "pctl_bb_pct", "pctl_avg_vs_hand", "pctl_xba_vs_hand",
        ]:
            out[col] = pd.to_numeric(out.get(col), errors="coerce").fillna(50.0)
        print(f"[hitter_attach] name_matches=0/{len(out)} using {batter_col}")
        return out

    base = baseline_df.copy().rename(columns={
        "pctl_iso_vs_hand": "base_pctl_iso_vs_hand",
        "pctl_woba_vs_hand": "base_pctl_woba_vs_hand",
        "pctl_ops_vs_hand": "base_pctl_ops_vs_hand",
        "pctl_hr_rate": "base_pctl_hr_rate",
        "pctl_barrel_pct": "base_pctl_barrel_pct",
        "pctl_hardhit_pct": "base_pctl_hardhit_pct",
        "pctl_k_pct": "base_pctl_k_pct",
        "pctl_bb_pct": "base_pctl_bb_pct",
        "pctl_avg_vs_hand": "base_pctl_avg_vs_hand",
        "pctl_xba_vs_hand": "base_pctl_xba_vs_hand",
    })

    keep = [
        "batter_name_norm",
        "base_pctl_iso_vs_hand",
        "base_pctl_woba_vs_hand",
        "base_pctl_ops_vs_hand",
        "base_pctl_hr_rate",
        "base_pctl_barrel_pct",
        "base_pctl_hardhit_pct",
        "base_pctl_k_pct",
        "base_pctl_bb_pct",
        "base_pctl_avg_vs_hand",
        "base_pctl_xba_vs_hand",
    ]
    base = base[[c for c in keep if c in base.columns]].copy()

    merged = out.merge(base, on="batter_name_norm", how="left")

    mapping = {
        "pctl_iso_vs_hand": "base_pctl_iso_vs_hand",
        "pctl_woba_vs_hand": "base_pctl_woba_vs_hand",
        "pctl_ops_vs_hand": "base_pctl_ops_vs_hand",
        "pctl_hr_rate": "base_pctl_hr_rate",
        "pctl_barrel_pct": "base_pctl_barrel_pct",
        "pctl_hardhit_pct": "base_pctl_hardhit_pct",
        "pctl_k_pct": "base_pctl_k_pct",
        "pctl_bb_pct": "base_pctl_bb_pct",
        "pctl_avg_vs_hand": "base_pctl_avg_vs_hand",
        "pctl_xba_vs_hand": "base_pctl_xba_vs_hand",
    }

    for live_col, base_col in mapping.items():
        merged[base_col] = pd.to_numeric(merged.get(base_col), errors="coerce")
        if live_col in merged.columns:
            merged[live_col] = pd.to_numeric(merged[live_col], errors="coerce").fillna(merged[base_col]).fillna(50.0)
        else:
            merged[live_col] = merged[base_col].fillna(50.0)

    matches = merged["base_pctl_iso_vs_hand"].notna().sum() if "base_pctl_iso_vs_hand" in merged.columns else 0
    print(f"[hitter_attach] name_matches={matches}/{len(merged)} using {batter_col}")

    return merged
