from __future__ import annotations

import time
from datetime import datetime, timezone, date as date_type
from typing import Optional

import pandas as pd
import requests

STADIUM_COORDS = {
    "arizona diamondbacks":   (33.4455, -112.0667, True),
    "atlanta braves":         (33.8908,  -84.4678, False),
    "baltimore orioles":      (39.2839,  -76.6216, False),
    "boston red sox":         (42.3467,  -71.0972, False),
    "chicago cubs":           (41.9484,  -87.6553, False),
    "chicago white sox":      (41.8300,  -87.6339, False),
    "cincinnati reds":        (39.0979,  -84.5082, False),
    "cleveland guardians":    (41.4962,  -81.6852, False),
    "colorado rockies":       (39.7559, -104.9942, False),
    "detroit tigers":         (42.3390,  -83.0485, False),
    "houston astros":         (29.7573,  -95.3555, True),
    "kansas city royals":     (39.0517,  -94.4803, False),
    "los angeles angels":     (33.8003, -117.8827, False),
    "los angeles dodgers":    (34.0739, -118.2400, False),
    "miami marlins":          (25.7781,  -80.2197, True),
    "milwaukee brewers":      (43.0280,  -87.9712, True),
    "minnesota twins":        (44.9817,  -93.2778, False),
    "new york mets":          (40.7571,  -73.8458, False),
    "new york yankees":       (40.8296,  -73.9262, False),
    "athletics":              (37.7516, -122.2005, False),
    "oakland athletics":      (37.7516, -122.2005, False),
    "philadelphia phillies":  (39.9061,  -75.1665, False),
    "pittsburgh pirates":     (40.4469,  -80.0057, False),
    "san diego padres":       (32.7076, -117.1570, False),
    "san francisco giants":   (37.7786, -122.3893, False),
    "seattle mariners":       (47.5914, -122.3325, True),
    "st. louis cardinals":    (38.6226,  -90.1928, False),
    "tampa bay rays":         (27.7683,  -82.6534, True),
    "texas rangers":          (32.7473,  -97.0822, True),
    "toronto blue jays":      (43.6414,  -79.3894, True),
    "washington nationals":   (38.8730,  -77.0074, False),
}

PRECIP_RISK_THRESH = 40
WIND_RISK_THRESH   = 20


def _normalize_team(team: str) -> str:
    return str(team).strip().lower().replace(".", "")


def _get_stadium(team: str):
    key = _normalize_team(team)
    if key in STADIUM_COORDS:
        return STADIUM_COORDS[key]
    for k, v in STADIUM_COORDS.items():
        if k in key or key in k:
            return v
    return None


def _is_past_date(date_iso: str) -> bool:
    try:
        game_date = datetime.strptime(date_iso, "%Y-%m-%d").date()
        return game_date < date_type.today()
    except Exception:
        return False


def _fetch_weather(lat: float, lon: float, game_hour_utc: int, date_iso: str) -> dict:
    is_past = _is_past_date(date_iso)

    if is_past:
        url = "https://archive-api.open-meteo.com/v1/archive"
    else:
        url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude":         lat,
        "longitude":        lon,
        "hourly":           "temperature_2m,wind_speed_10m,wind_direction_10m,precipitation_probability",
        "wind_speed_unit":  "mph",
        "temperature_unit": "fahrenheit",
        "timezone":         "UTC",
        "start_date":       date_iso,
        "end_date":         date_iso,
    }

    # Note: do not pass forecast_days when using start_date/end_date — API rejects it

    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return {"error": str(e)}

    hourly = data.get("hourly", {})
    times  = hourly.get("time", [])
    temps  = hourly.get("temperature_2m", [])
    winds  = hourly.get("wind_speed_10m", [])
    dirs   = hourly.get("wind_direction_10m", [])
    precip = hourly.get("precipitation_probability", [])

    if not times:
        return {"error": "no hourly data"}

    best_idx  = 0
    best_diff = 999
    for i, t in enumerate(times):
        try:
            hour = int(t.split("T")[1].split(":")[0])
            diff = abs(hour - game_hour_utc)
            if diff < best_diff:
                best_diff = diff
                best_idx  = i
        except Exception:
            continue

    def _safe_get(lst, idx, default=None):
        try:
            return lst[idx]
        except (IndexError, TypeError):
            return default

    temp_val   = _safe_get(temps,  best_idx)
    wind_val   = _safe_get(winds,  best_idx)
    dir_val    = _safe_get(dirs,   best_idx)
    precip_val = _safe_get(precip, best_idx)

    # Archive API returns None or string "undefined" for precip — treat as 0
    try:
        precip_val = float(precip_val) if precip_val is not None else 0.0
    except (ValueError, TypeError):
        precip_val = 0.0

    return {
        "temp_f":      temp_val,
        "wind_mph":    wind_val,
        "wind_dir":    dir_val,
        "precip_prob": precip_val,
    }


def _indoor_defaults() -> dict:
    return {
        "temp_f":          72.0,
        "wind_mph":        0.0,
        "wind_dir":        0.0,
        "precip_prob":     0.0,
        "delay_risk_flag": False,
    }


def attach_weather(df: pd.DataFrame, date_iso: str) -> pd.DataFrame:
    out = df.copy()

    for col in ["temp_f", "wind_mph", "wind_dir", "precip_prob", "delay_risk_flag"]:
        if col not in out.columns:
            out[col] = None

    weather_cache: dict = {}
    fetched = failed = indoor = 0

    for idx, row in out.iterrows():
        game_id   = str(row.get("game_id", "") or "")
        home_team = str(row.get("home_team", "") or "")
        game_time = row.get("game_time")

        if game_id and game_id in weather_cache:
            w = weather_cache[game_id]
        else:
            stadium = _get_stadium(home_team)

            if stadium is None:
                w = {"temp_f": 72.0, "wind_mph": 5.0, "wind_dir": 0.0,
                     "precip_prob": 0.0, "delay_risk_flag": False}
                failed += 1
            elif stadium[2]:
                w = _indoor_defaults()
                indoor += 1
            else:
                lat, lon, _ = stadium

                game_hour_utc = 23
                if game_time is not None:
                    try:
                        if isinstance(game_time, str):
                            dt = datetime.fromisoformat(game_time.replace("Z", "+00:00"))
                        elif isinstance(game_time, pd.Timestamp):
                            dt = game_time.to_pydatetime()
                        else:
                            dt = game_time
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        game_hour_utc = dt.hour
                    except Exception:
                        pass

                raw = _fetch_weather(lat, lon, game_hour_utc, date_iso)

                if "error" in raw:
                    w = {"temp_f": 70.0, "wind_mph": 5.0, "wind_dir": 0.0,
                         "precip_prob": 5.0, "delay_risk_flag": False}
                    failed += 1
                else:
                    precip     = float(raw.get("precip_prob") or 0)
                    wind       = float(raw.get("wind_mph")    or 0)
                    delay_risk = (precip >= PRECIP_RISK_THRESH or wind >= WIND_RISK_THRESH)
                    w = {
                        "temp_f":          float(raw.get("temp_f")   or 70.0),
                        "wind_mph":        wind,
                        "wind_dir":        float(raw.get("wind_dir") or 0.0),
                        "precip_prob":     precip,
                        "delay_risk_flag": delay_risk,
                    }
                    fetched += 1

                time.sleep(0.15)

            if game_id:
                weather_cache[game_id] = w

        out.at[idx, "temp_f"]          = w.get("temp_f")
        out.at[idx, "wind_mph"]        = w.get("wind_mph")
        out.at[idx, "wind_dir"]        = w.get("wind_dir")
        out.at[idx, "precip_prob"]     = w.get("precip_prob")
        out.at[idx, "delay_risk_flag"] = w.get("delay_risk_flag")

    print(f"[weather] fetched={fetched}  indoor={indoor}  failed={failed}  total_games={len(weather_cache)}")
    delay_games = int(out["delay_risk_flag"].eq(True).sum())
    if delay_games:
        print(f"[weather] delay_risk_flag=True for {delay_games} rows")
    return out


