from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Schema:
    name: str
    columns: List[str]


PITCHER_PROPS_SCHEMA = Schema(
    name="pitcher_props_df",
    columns=[
        "date", "game_id", "game_time",
        "home_team", "away_team",
        "pitcher_id", "pitcher_name", "pitcher_team",
        "opponent_team", "home_away",

        "market", "side", "line",
        "odds_american", "odds_decimal", "implied_prob",

        "is_confirmed_starter", "starter_type",
        "days_rest", "injury_flag", "pitch_limit_flag",

        "pitch_count_l5_median", "pitch_count_l5_std",
        "ip_l5_median", "ip_l5_std",
        "avg_ip_per_start", "avg_bf_per_start",

        "pctl_k_pct", "pctl_bb_pct", "pctl_swstr_pct",
        "pctl_csw_pct", "pctl_pitches_per_pa", "pctl_gb_pct",

        "opp_hand",
        "opp_pctl_k_pct_vs_hand", "opp_pctl_bb_pct_vs_hand",
        "opp_pctl_woba_vs_hand", "opp_pctl_iso_vs_hand",
        "opp_pctl_pitches_pa_vs_hand",

        "lineup_status",
        "lineup_k_pct_avg", "lineup_high_k_count", "lineup_low_k_count",

        "park_id", "pctl_park_runs", "pctl_park_hr",
        "temp_f", "wind_mph", "wind_dir", "precip_prob", "delay_risk_flag",

        "line_consensus", "line_delta", "pctl_line_delta",
        "line_open", "line_latest", "line_move_abs", "pctl_line_move",

        "proj", "edge_abs", "edge_pctl_slate",
        "confidence_score", "confidence_bucket",
        "hard_gate_pass", "notes",
    ],
)

HITTER_LOTTO_SCHEMA = Schema(
    name="hitter_lotto_df",
    columns=[
        "date", "game_id", "game_time",
        "home_team", "away_team",
        "batter_id", "batter_name", "batter_team", "opponent_team",
        "home_away", "batter_hand",
        "pitcher_id", "pitcher_name", "pitcher_hand",

        "market", "side", "line",
        "odds_american", "odds_decimal", "implied_prob",

        "lineup_status", "is_starting", "batting_order_spot", "platoon_risk_flag",

        "pctl_iso_vs_hand", "pctl_woba_vs_hand", "pctl_ops_vs_hand",
        "pctl_hr_rate", "pctl_barrel_pct", "pctl_hardhit_pct",
        "pctl_k_pct", "pctl_bb_pct",
        "pctl_avg_vs_hand", "pctl_xba_vs_hand",

        "pctl_pitcher_k_pct", "pctl_pitcher_bb_pct",
        "pctl_pitcher_hr_per_9", "pctl_pitcher_barrel_allowed",
        "pctl_pitcher_hardhit_allowed", "pctl_pitcher_gb_pct",
        "pctl_pitcher_woba_allowed",

        "team_implied_runs", "pctl_team_implied_runs",

        "park_id", "pctl_park_runs", "pctl_park_hr",
        "temp_f", "wind_mph", "wind_dir", "precip_prob", "delay_risk_flag",

        "implied_prob_consensus", "implied_prob_delta", "pctl_implied_prob_delta",

        "confidence_score", "hard_gate_pass", "caps_applied", "notes",
    ],
)


def ensure_schema_columns(df, schema: Schema):
    for c in schema.columns:
        if c not in df.columns:
            df[c] = None
    return df[schema.columns]
