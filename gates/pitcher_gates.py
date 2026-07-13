from __future__ import annotations

import pandas as pd

from mlb.config import PITCHER_MARKETS


def apply_pitcher_hard_gates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply v1 hard gates for pitcher props.

    Rules (v1):
      1. Must have mapped pitcher_id (probable starter known)
      2. Market must be allowed (K or OUTS)
      3. Must have valid line and odds
    """

    df = df.copy()

    hard_pass = []
    notes = []

    for _, r in df.iterrows():

        row_notes = []
        ok = True

        # 1) Starter must be mapped
        if pd.isna(r.get("pitcher_id")):
            ok = False
            row_notes.append("no_pitcher_id")

        # 2) Market filter
        # K_ALT rows pass the hard gate but are excluded from card selection
        # They are only used for the K Ladder Parlay tier
        if r.get("market") == "K_ALT":
            pass  # K_ALT rows always pass hard gate — filtered in card selection
        elif r.get("market") not in PITCHER_MARKETS:
            ok = False
            row_notes.append("invalid_market")

        # 3) Line + odds sanity
        if pd.isna(r.get("line")):
            ok = False
            row_notes.append("missing_line")

        if pd.isna(r.get("odds_american")):
            ok = False
            row_notes.append("missing_odds")

        hard_pass.append(ok)

        if row_notes:
            notes.append(",".join(row_notes))
        else:
            notes.append(None)

    df["hard_gate_pass"] = hard_pass

    # Merge with existing notes
    if "notes" not in df.columns:
        df["notes"] = None

    df["notes"] = df["notes"].fillna("")
    df["notes"] = df["notes"].astype(str)

    new_notes = pd.Series(notes).fillna("").astype(str)

    df["notes"] = (
        df["notes"].str.strip()
        + (";" + new_notes).where(new_notes != "", "")
    ).str.strip(";")

    df.loc[df["notes"] == "", "notes"] = None

    return df

