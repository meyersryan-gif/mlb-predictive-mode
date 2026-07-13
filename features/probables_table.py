from __future__ import annotations

from typing import List
import pandas as pd

from mlb.data_sources.mlb_stats_api import ScheduleGame


def probables_to_df(games: List[ScheduleGame]) -> pd.DataFrame:
    rows = []
    for g in games:
        rows.append({
            "game_pk": g.game_pk,
            "game_date": g.game_date,
            "game_time_utc": g.game_time_utc,
            "home_team": g.home_team,
            "away_team": g.away_team,
            "home_probable_id": g.home_probable_id,
            "home_probable_name": g.home_probable_name,
            "away_probable_id": g.away_probable_id,
            "away_probable_name": g.away_probable_name,
        })
    return pd.DataFrame(rows)
