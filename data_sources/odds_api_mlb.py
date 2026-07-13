from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

import requests


ODDS_API_BASE = "https://api.the-odds-api.com/v4"
MLB_LOCAL_TZ = ZoneInfo("America/New_York")

PITCHER_MARKETS = "pitcher_strikeouts,pitcher_outs,pitcher_strikeouts_alternate"
HITTER_MARKETS = ",".join([
    "batter_home_runs",
    "batter_hits",
    "batter_total_bases",
    "batter_rbis",
    "batter_home_runs_alternate",
    "batter_hits_alternate",
    "batter_total_bases_alternate",
    "batter_rbis_alternate",
])
ALL_MLB_MARKETS = f"{PITCHER_MARKETS},{HITTER_MARKETS}"


@dataclass(frozen=True)
class OddsApiConfig:
    api_key: str
    regions: str = "us"
    markets: str = PITCHER_MARKETS
    odds_format: str = "american"
    date_format: str = "iso"
    sports_key: str = "baseball_mlb"


def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def default_config_from_env(markets: str = PITCHER_MARKETS) -> OddsApiConfig:
    return OddsApiConfig(
        api_key=_require_env("ODDS_API_KEY"),
        markets=markets,
    )


def _parse_iso_utc(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def _local_date_from_commence(ts: str) -> str | None:
    dt = _parse_iso_utc(ts)
    if dt is None:
        return None
    return dt.astimezone(MLB_LOCAL_TZ).strftime("%Y-%m-%d")


def get_mlb_events(cfg: OddsApiConfig, date_iso: str) -> List[Dict[str, Any]]:
    url = f"{ODDS_API_BASE}/sports/{cfg.sports_key}/events"
    params = {
        "apiKey": cfg.api_key,
        "dateFormat": cfg.date_format,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    events = r.json()

    out = []
    for e in events:
        commence = e.get("commence_time", "")
        local_date = _local_date_from_commence(commence)
        if local_date == date_iso:
            out.append(e)
    return out


def get_event_odds(cfg: OddsApiConfig, event_id: str) -> Dict[str, Any]:
    url = f"{ODDS_API_BASE}/sports/{cfg.sports_key}/events/{event_id}/odds"
    params = {
        "apiKey": cfg.api_key,
        "regions": cfg.regions,
        "markets": cfg.markets,
        "oddsFormat": cfg.odds_format,
        "dateFormat": cfg.date_format,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

