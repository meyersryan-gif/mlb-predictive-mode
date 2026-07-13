from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from mlb.schemas import PITCHER_PROPS_SCHEMA, ensure_schema_columns


BOOK_TARGET = "FanDuel"

MARKET_MAP = {
    "pitcher_strikeouts":           "K",
    "pitcher_outs":                 "OUTS",
    "pitcher_strikeouts_alternate": "K_ALT",
}


def american_to_decimal(odds: int) -> float:
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def implied_prob_from_american(odds: int) -> float:
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def _parse_commence_time_iso(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _norm_side_token(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s == "over":
        return "Over"
    if s == "under":
        return "Under"
    return None


def build_pitcher_props_from_event_odds(date_iso: str, event_odds: Dict[str, Any]) -> pd.DataFrame:
    """
    Build rows for FanDuel pitcher K and OUTS main lines from a single event odds payload.

    IMPORTANT:
    Odds API MLB player props commonly come through as:
      - outcome.name = "Over"/"Under"
      - outcome.description = player name

    So we explicitly detect that shape.
    """
    rows: List[Dict[str, Any]] = []

    event_id = event_odds.get("id")
    home_team = event_odds.get("home_team")
    away_team = event_odds.get("away_team")
    game_time = _parse_commence_time_iso(event_odds.get("commence_time", ""))

    bookmakers = event_odds.get("bookmakers", []) or []
    fd = None
    for b in bookmakers:
        if (b.get("title") == BOOK_TARGET) or (b.get("key") == "fanduel"):
            fd = b
            break
    if not fd:
        return ensure_schema_columns(pd.DataFrame([]), PITCHER_PROPS_SCHEMA)

    for m in fd.get("markets", []) or []:
        market_key = m.get("key")
        if market_key not in MARKET_MAP:
            continue

        our_market = MARKET_MAP[market_key]

        for o in m.get("outcomes", []) or []:
            name_val = o.get("name")
            desc_val = o.get("description")
            point_val = o.get("point")
            price_val = o.get("price")

            if point_val is None or price_val is None:
                continue

            # MLB player props from Odds API are often:
            #   name="Over"/"Under"
            #   description="<Player Name>"
            #
            # We support both possible shapes just in case.
            side_from_name = _norm_side_token(name_val)
            side_from_desc = _norm_side_token(desc_val)

            if side_from_name is not None:
                side_norm = side_from_name
                player = desc_val
            elif side_from_desc is not None:
                side_norm = side_from_desc
                player = name_val
            else:
                # Unknown shape; skip
                continue

            if player is None:
                continue

            try:
                price_int = int(price_val)
                line_float = float(point_val)
            except Exception:
                continue

            rows.append({
                "date": date_iso,
                "game_id": str(event_id),
                "game_time": game_time,
                "home_team": home_team,
                "away_team": away_team,
                "pitcher_id": None,
                "pitcher_name": str(player).strip(),
                "pitcher_team": None,
                "opponent_team": None,
                "home_away": None,

                "market": our_market,
                "side": side_norm,
                "line": line_float,
                "odds_american": price_int,
                "odds_decimal": float(american_to_decimal(price_int)),
                "implied_prob": float(implied_prob_from_american(price_int)),

                "line_open": line_float,
                "line_latest": line_float,
                "line_move_abs": 0.0,
            })

    df = pd.DataFrame(rows)
    df = ensure_schema_columns(df, PITCHER_PROPS_SCHEMA)
    return df

