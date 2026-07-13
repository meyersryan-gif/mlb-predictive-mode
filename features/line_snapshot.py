from __future__ import annotations

"""
line_snapshot.py — Saves and retrieves intraday line snapshots.

First run of the day saves current lines as the opening snapshot.
Subsequent runs load the snapshot as line_open for movement calculation.

This gives us real intraday line movement without needing the paid
Odds API historical endpoint.
"""

import json
from datetime import datetime
from pathlib import Path

OUTPUTS_DIR = Path("mlb") / "outputs"


def _snapshot_path(date_iso: str) -> Path:
    return OUTPUTS_DIR / f"lines_snapshot_{date_iso.replace('-', '')}.json"


def save_line_snapshot(df, date_iso: str) -> None:
    """
    Save current lines as opening snapshot for the day.
    Only saves if no snapshot exists yet — first run of the day wins.
    """
    path = _snapshot_path(date_iso)
    if path.exists():
        return  # Already have a snapshot today — don't overwrite

    snapshot = {}
    for _, row in df.iterrows():
        if str(row.get("market","")).upper() == "K_ALT":
            continue
        key  = f"{row.get('pitcher_name')}_{row.get('market')}_{row.get('side')}"
        # Save ODDS not line value — books move odds on K/OUTS props, not the point total
        odds = row.get("odds_american")
        line = row.get("line_latest") or row.get("line")
        if odds is not None and line is not None:
            snapshot[key] = {"odds": float(odds), "line": float(line)}

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f)
    print(f"[line_snapshot] Saved opening snapshot: {len(snapshot)} lines for {date_iso}")


def load_line_snapshot(date_iso: str) -> dict:
    """
    Load the opening snapshot for the day.
    Returns empty dict if no snapshot exists (first run).
    """
    path = _snapshot_path(date_iso)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[line_snapshot] Loaded opening snapshot: {len(data)} lines for {date_iso}")
    return data


def attach_opening_lines(df, date_iso: str):
    """
    Attach line_open from snapshot to props DataFrame.
    If no snapshot exists, line_open = line_latest (no movement yet).
    On first run: saves current lines as snapshot, line_open = line_latest.
    On subsequent runs: loads snapshot as line_open, enabling real delta.
    """
    import pandas as pd
    out = df.copy()

    snapshot = load_line_snapshot(date_iso)

    if not snapshot:
        # First run — save snapshot and set line_open = line_latest
        save_line_snapshot(out, date_iso)
        out["line_open"] = out.get("line_latest", out.get("line"))
        print("[line_snapshot] First run today — snapshot saved, no movement yet")
        return out

    # Subsequent run — attach opening lines from snapshot
    def _get_open(row):
        key   = f"{row.get('pitcher_name')}_{row.get('market')}_{row.get('side')}"
        entry = snapshot.get(key)
        if entry is None:
            return row.get("line_latest") or row.get("line")
        if isinstance(entry, dict):
            return entry.get("line", row.get("line"))
        return entry  # legacy float format

    out["line_open"] = out.apply(_get_open, axis=1)

    # Attach opening odds from snapshot for comparison
    def _get_open_odds(row):
        key   = f"{row.get('pitcher_name')}_{row.get('market')}_{row.get('side')}"
        entry = snapshot.get(key)
        if entry is None:
            return row.get("odds_american")
        if isinstance(entry, dict):
            return entry.get("odds")
        return None

    out["odds_open"] = out.apply(_get_open_odds, axis=1)

    # Count meaningful moves — odds shifted by 5+ points
    odds_now  = pd.to_numeric(out.get("odds_american"), errors="coerce")
    odds_open = pd.to_numeric(out.get("odds_open"),     errors="coerce")
    delta     = (odds_now - odds_open).abs()
    moved     = (delta >= 5).sum()
    print(f"[line_snapshot] Odds moved since morning snapshot: {moved}/{len(out)}")

    return out



