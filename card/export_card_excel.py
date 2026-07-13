from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def _safe_sheet_name(name: str) -> str:
    bad = ['\\', '/', '*', '?', ':', '[', ']']
    for ch in bad:
        name = name.replace(ch, "-")
    return name[:31]


def _autofit_worksheet(ws) -> None:
    for col_cells in ws.columns:
        max_len = 0
        col_letter = col_cells[0].column_letter
        for cell in col_cells:
            try:
                val = "" if cell.value is None else str(cell.value)
                max_len = max(max_len, len(val))
            except Exception:
                pass
        ws.column_dimensions[col_letter].width = min(max(max_len + 2, 10), 40)


def _excel_safe_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame([])
    out = df.copy()
    if len(out) == 0:
        return out

    for col in out.columns:
        s = out[col]
        # Handle already-timezone-aware datetime columns
        if hasattr(s.dtype, "tz") and s.dtype.tz is not None:
            out[col] = s.dt.tz_localize(None)
            continue
        # For object columns, attempt datetime conversion safely
        if s.dtype == "object":
            try:
                converted = pd.to_datetime(s, errors="coerce", format="mixed")
                if converted.notna().any():
                    if hasattr(converted.dtype, "tz") and converted.dtype.tz is not None:
                        out[col] = converted.dt.tz_localize(None)
            except Exception:
                pass
    return out


def _fmt_num(x, ndigits: int = 3):
    try:
        if pd.isna(x):
            return None
        return round(float(x), ndigits)
    except Exception:
        return x


def _build_card_table(card: Dict[str, Any]) -> pd.DataFrame:
    rows = []

    flagship = card.get("flagship")
    if flagship:
        rows.append({
            "CardType": "FLAGSHIP",
            "Units": flagship.get("units"),
            "Player": flagship.get("pitcher_name"),
            "Market": flagship.get("market"),
            "Side": flagship.get("side"),
            "Line": flagship.get("line"),
            "Proj": _fmt_num(flagship.get("proj"), 3),
            "Price": None,
            "ImpliedProb": None,
            "EdgeForSide": _fmt_num(flagship.get("edge_for_side"), 3),
            "Confidence": flagship.get("confidence_score"),
            "Team": flagship.get("pitcher_team"),
            "Opponent": flagship.get("opponent_team"),
            "HomeAway": flagship.get("home_away"),
            "Book": "FanDuel",
            "Notes": None,
        })

    parlay = card.get("parlay")
    if parlay:
        leg1 = parlay.get("leg1", {})
        leg2 = parlay.get("leg2", {})
        rows.append({
            "CardType": "PARLAY",
            "Units": parlay.get("units"),
            "Player": leg1.get("pitcher_name"),
            "Market": leg1.get("market"),
            "Side": leg1.get("side"),
            "Line": leg1.get("line"),
            "Proj": None,
            "Price": None,
            "ImpliedProb": None,
            "EdgeForSide": None,
            "Confidence": leg1.get("confidence_score"),
            "Team": leg1.get("pitcher_team"),
            "Opponent": leg1.get("opponent_team"),
            "HomeAway": leg1.get("home_away"),
            "Book": "FanDuel",
            "Notes": f"Leg1 | {parlay.get('corr_type')} | Bonus={parlay.get('corr_bonus')}",
        })
        rows.append({
            "CardType": "PARLAY",
            "Units": parlay.get("units"),
            "Player": leg2.get("pitcher_name"),
            "Market": leg2.get("market"),
            "Side": leg2.get("side"),
            "Line": leg2.get("line"),
            "Proj": None,
            "Price": None,
            "ImpliedProb": None,
            "EdgeForSide": None,
            "Confidence": leg2.get("confidence_score"),
            "Team": leg2.get("pitcher_team"),
            "Opponent": leg2.get("opponent_team"),
            "HomeAway": leg2.get("home_away"),
            "Book": "FanDuel",
            "Notes": f"Leg2 | ComboScore={parlay.get('combo_score')}",
        })

    lotto = card.get("lotto")
    if lotto:
        leg1 = lotto.get("leg1", {})
        leg2 = lotto.get("leg2", {})
        ltype = lotto.get("type", "")
        rows.append({
            "CardType": "LOTTO",
            "Units": lotto.get("units"),
            "Player": leg1.get("batter_name"),
            "Market": leg1.get("market"),
            "Side": leg1.get("side"),
            "Line": leg1.get("line"),
            "Proj": None,
            "Price": leg1.get("odds_american"),
            "ImpliedProb": _fmt_num(leg1.get("implied_prob"), 4),
            "EdgeForSide": None,
            "Confidence": leg1.get("confidence_score"),
            "Team": leg1.get("batter_team"),
            "Opponent": leg1.get("opponent_team"),
            "HomeAway": leg1.get("home_away"),
            "Book": "FanDuel",
            "Notes": f"Leg1 | {ltype} | vs {leg1.get('pitcher_name')}",
        })
        rows.append({
            "CardType": "LOTTO",
            "Units": lotto.get("units"),
            "Player": leg2.get("batter_name"),
            "Market": leg2.get("market"),
            "Side": leg2.get("side"),
            "Line": leg2.get("line"),
            "Proj": None,
            "Price": leg2.get("odds_american"),
            "ImpliedProb": _fmt_num(leg2.get("implied_prob"), 4),
            "EdgeForSide": None,
            "Confidence": leg2.get("confidence_score"),
            "Team": leg2.get("batter_team"),
            "Opponent": leg2.get("opponent_team"),
            "HomeAway": leg2.get("home_away"),
            "Book": "FanDuel",
            "Notes": f"Leg2 | {ltype} | vs {leg2.get('pitcher_name')}",
        })

    if not rows:
        rows.append({
            "CardType": "NONE", "Units": None, "Player": None,
            "Market": None, "Side": None, "Line": None,
            "Proj": None, "Price": None, "ImpliedProb": None,
            "EdgeForSide": None, "Confidence": None,
            "Team": None, "Opponent": None, "HomeAway": None,
            "Book": "FanDuel", "Notes": "No card generated",
        })

    return pd.DataFrame(rows)


def _build_flagship_pool(scored_pitchers: pd.DataFrame, min_conf: float, top_n: int) -> pd.DataFrame:
    if scored_pitchers is None or len(scored_pitchers) == 0:
        return pd.DataFrame([])

    pool = scored_pitchers.copy()
    pool = pool[
        (pool["hard_gate_pass"] == True) &
        (pd.to_numeric(pool["confidence_score"], errors="coerce") >= min_conf)
    ].copy()

    if len(pool) == 0:
        pool = scored_pitchers.copy()

    sort_cols = [c for c in ["confidence_score", "edge_for_side"] if c in pool.columns]
    if sort_cols:
        pool = pool.sort_values(by=sort_cols, ascending=[False] * len(sort_cols), na_position="last")

    keep_cols = [c for c in [
        "pitcher_name", "pitcher_team", "opponent_team", "home_away",
        "market", "side", "line", "proj", "odds_american", "implied_prob",
        "edge_for_side", "confidence_score", "confidence_bucket",
        "pitcher_hand", "notes",
    ] if c in pool.columns]

    pool = pool[keep_cols].head(top_n).copy()
    return pool.rename(columns={
        "pitcher_name": "Player", "pitcher_team": "Team",
        "opponent_team": "Opponent", "home_away": "HomeAway",
        "market": "Market", "side": "Side", "line": "Line",
        "proj": "Proj", "odds_american": "Price",
        "implied_prob": "ImpliedProb", "edge_for_side": "EdgeForSide",
        "confidence_score": "Confidence", "confidence_bucket": "Bucket",
        "pitcher_hand": "PitcherHand", "notes": "Notes",
    })


def _build_scored_props_view(scored_pitchers: pd.DataFrame) -> pd.DataFrame:
    if scored_pitchers is None or len(scored_pitchers) == 0:
        return pd.DataFrame([])

    df = scored_pitchers.copy()
    sort_cols = [c for c in ["confidence_score", "edge_for_side"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(by=sort_cols, ascending=[False] * len(sort_cols), na_position="last")

    keep_cols = [c for c in [
        "pitcher_name", "pitcher_team", "opponent_team", "home_away", "pitcher_hand",
        "market", "side", "line", "proj", "odds_american", "implied_prob",
        "edge_for_side", "confidence_score", "confidence_bucket",
        "hard_gate_pass", "notes", "game_time",
    ] if c in df.columns]

    return df[keep_cols].copy().rename(columns={
        "pitcher_name": "Player", "pitcher_team": "Team",
        "opponent_team": "Opponent", "home_away": "HomeAway",
        "pitcher_hand": "PitcherHand", "market": "Market",
        "side": "Side", "line": "Line", "proj": "Proj",
        "odds_american": "Price", "implied_prob": "ImpliedProb",
        "edge_for_side": "EdgeForSide", "confidence_score": "Confidence",
        "confidence_bucket": "Bucket", "hard_gate_pass": "HardGatePass",
        "notes": "Notes", "game_time": "GameTime",
    })


def _build_parlay_candidates_view(parlay_candidates: pd.DataFrame) -> pd.DataFrame:
    if parlay_candidates is None or len(parlay_candidates) == 0:
        return pd.DataFrame([])

    df = parlay_candidates.copy()
    df = df.sort_values(by=["combo_score", "corr_bonus"], ascending=[False, False], na_position="last")
    return df.rename(columns={
        "leg1_pitcher": "Leg1Player", "leg1_market": "Leg1Market",
        "leg1_side": "Leg1Side", "leg1_line": "Leg1Line",
        "leg1_conf": "Leg1Confidence", "leg2_pitcher": "Leg2Player",
        "leg2_market": "Leg2Market", "leg2_side": "Leg2Side",
        "leg2_line": "Leg2Line", "leg2_conf": "Leg2Confidence",
        "corr_type": "CorrType", "corr_bonus": "CorrBonus",
        "combo_score": "ComboScore",
    })


def export_mlb_card_excel(
    outpath: str | Path,
    date_iso: str,
    card: Dict[str, Any],
    scored_pitchers: pd.DataFrame,
    parlay_candidates: Optional[pd.DataFrame] = None,
    flagship_pool_min_conf: float = 74.0,
    flagship_pool_top_n: int = 15,
) -> Path:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    card_df        = _excel_safe_df(_build_card_table(card))
    flagship_pool  = _excel_safe_df(_build_flagship_pool(scored_pitchers, flagship_pool_min_conf, flagship_pool_top_n))
    scored_view    = _excel_safe_df(_build_scored_props_view(scored_pitchers))
    parlay_view    = _excel_safe_df(_build_parlay_candidates_view(parlay_candidates))

    with pd.ExcelWriter(outpath, engine="openpyxl") as writer:
        card_df.to_excel(writer,       sheet_name="Card",              index=False)
        flagship_pool.to_excel(writer, sheet_name="FlagshipPool",      index=False)
        scored_view.to_excel(writer,   sheet_name="ScoredProps",       index=False)
        parlay_view.to_excel(writer,   sheet_name="ParlayCandidates",  index=False)

        wb = writer.book
        for ws in wb.worksheets:
            ws.freeze_panes = "A2"
            _autofit_worksheet(ws)

    return outpath
