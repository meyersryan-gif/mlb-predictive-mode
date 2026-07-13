from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from mlb.schemas import HITTER_LOTTO_SCHEMA, ensure_schema_columns


BOOK_TARGET = "FanDuel"

HITTER_MARKET_MAP = {
    "batter_home_runs": "HR",
    "batter_home_runs_alternate": "HR",
    "player_home_runs": "HR",

    "batter_hits": "HITS",
    "batter_hits_alternate": "HITS",
    "player_hits": "HITS",

    "batter_total_bases": "TB",
    "batter_total_bases_alternate": "TB",
    "player_total_bases": "TB",

    "batter_rbis": "RBI",
    "batter_rbis_alternate": "RBI",
    "player_rbis": "RBI",
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


def build_hitter_lotto_from_event_odds(date_iso: str, event_odds: Dict[str, Any]) -> pd.DataFrame:
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
        return ensure_schema_columns(pd.DataFrame([]), HITTER_LOTTO_SCHEMA)

    for m in fd.get("markets", []) or []:
        market_key = m.get("key")
        if market_key not in HITTER_MARKET_MAP:
            continue

        our_market = HITTER_MARKET_MAP[market_key]

        for o in m.get("outcomes", []) or []:
            name_val = o.get("name")
            desc_val = o.get("description")
            point_val = o.get("point")
            price_val = o.get("price")

            if price_val is None:
                continue

            side_from_name = _norm_side_token(name_val)
            side_from_desc = _norm_side_token(desc_val)

            if side_from_name is not None:
                side_norm = side_from_name
                player = desc_val
            elif side_from_desc is not None:
                side_norm = side_from_desc
                player = name_val
            else:
                side_norm = "Over"
                player = name_val if name_val is not None else desc_val

            if player is None:
                continue

            try:
                price_int = int(price_val)
            except Exception:
                continue

            line_float = None
            if point_val is not None:
                try:
                    line_float = float(point_val)
                except Exception:
                    line_float = None

            rows.append({
                "date": date_iso,
                "game_id": str(event_id),
                "game_time": game_time,
                "home_team": home_team,
                "away_team": away_team,

                "batter_id": None,
                "batter_name": str(player).strip(),
                "batter_team": None,
                "opponent_team": None,
                "home_away": None,
                "batter_hand": None,
                "pitcher_id": None,
                "pitcher_name": None,
                "pitcher_hand": None,

                "market": our_market,
                "side": side_norm,
                "line": line_float,
                "odds_american": price_int,
                "odds_decimal": float(american_to_decimal(price_int)),
                "implied_prob": float(implied_prob_from_american(price_int)),
            })

    df = pd.DataFrame(rows)
    df = ensure_schema_columns(df, HITTER_LOTTO_SCHEMA)
    return df


def list_fd_market_keys(event_odds: Dict[str, Any]) -> List[str]:
    bookmakers = event_odds.get("bookmakers", []) or []
    fd = None
    for b in bookmakers:
        if (b.get("title") == BOOK_TARGET) or (b.get("key") == "fanduel"):
            fd = b
            break
    if not fd:
        return []

    return sorted({str(m.get("key")) for m in fd.get("markets", []) or [] if m.get("key")})
