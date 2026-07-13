from __future__ import annotations

import pandas as pd

PARK_FACTORS = {
    "coors_field":          (117, 121),
    "great_american":       (108, 114),
    "citizens_bank":        (107, 113),
    "fenway":               (106, 108),
    "yankee_stadium":       (105, 112),
    "camden_yards":         (104, 107),
    "angel_stadium":        (103, 105),
    "kauffman":             (103, 104),
    "pnc_park":             (101, 102),
    "oracle_park":          (100,  98),
    "busch_stadium":        ( 99,  97),
    "progressive_field":    ( 99,  96),
    "petco_park":           ( 98,  95),
    "wrigley_field":        ( 98,  97),
    "nationals_park":       ( 97,  96),
    "truist_park":          ( 97,  95),
    "citi_field":           ( 97,  94),
    "dodger_stadium":       ( 96,  95),
    "comerica_park":        ( 96,  92),
    "globe_life":           (100, 102),
    "minute_maid":          (100, 101),
    "chase_field":          (101, 104),
    "american_family":      ( 99,  97),
    "t_mobile":             ( 97,  95),
    "loandepot":            ( 96,  93),
    "guaranteed_rate":      ( 99, 101),
    "target_field":         ( 98,  96),
    "rogers_centre":        (103, 107),
    "tropicana":            ( 97,  94),
    "oakland_coliseum":     ( 95,  91),
}

TEAM_TO_STADIUM = {
    "colorado rockies":       "coors_field",
    "cincinnati reds":        "great_american",
    "philadelphia phillies":  "citizens_bank",
    "boston red sox":         "fenway",
    "new york yankees":       "yankee_stadium",
    "baltimore orioles":      "camden_yards",
    "los angeles angels":     "angel_stadium",
    "kansas city royals":     "kauffman",
    "pittsburgh pirates":     "pnc_park",
    "san francisco giants":   "oracle_park",
    "st. louis cardinals":    "busch_stadium",
    "cleveland guardians":    "progressive_field",
    "san diego padres":       "petco_park",
    "chicago cubs":           "wrigley_field",
    "washington nationals":   "nationals_park",
    "atlanta braves":         "truist_park",
    "new york mets":          "citi_field",
    "los angeles dodgers":    "dodger_stadium",
    "detroit tigers":         "comerica_park",
    "texas rangers":          "globe_life",
    "houston astros":         "minute_maid",
    "arizona diamondbacks":   "chase_field",
    "milwaukee brewers":      "american_family",
    "seattle mariners":       "t_mobile",
    "miami marlins":          "loandepot",
    "chicago white sox":      "guaranteed_rate",
    "minnesota twins":        "target_field",
    "toronto blue jays":      "rogers_centre",
    "tampa bay rays":         "tropicana",
    "athletics":              "oakland_coliseum",
    "oakland athletics":      "oakland_coliseum",
}


def _normalize_team(team: str) -> str:
    return str(team).strip().lower().replace(".", "")


def _build_percentile_lookup() -> dict:
    stadium_ids = list(PARK_FACTORS.keys())
    run_factors = [PARK_FACTORS[s][0] for s in stadium_ids]
    hr_factors  = [PARK_FACTORS[s][1] for s in stadium_ids]

    run_series = pd.Series(run_factors, index=stadium_ids)
    hr_series  = pd.Series(hr_factors,  index=stadium_ids)

    run_pctls = run_series.rank(pct=True) * 100.0
    hr_pctls  = hr_series.rank(pct=True)  * 100.0

    return {
        sid: (round(run_pctls[sid], 1), round(hr_pctls[sid], 1))
        for sid in stadium_ids
    }


_PCTL_LOOKUP = _build_percentile_lookup()


def attach_park_factors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in ["park_id", "pctl_park_runs", "pctl_park_hr"]:
        if col not in out.columns:
            out[col] = None

    matched = unmatched = 0

    for idx, row in out.iterrows():
        home_team  = _normalize_team(row.get("home_team", "") or "")
        stadium_id = TEAM_TO_STADIUM.get(home_team)

        if stadium_id is None:
            for k, v in TEAM_TO_STADIUM.items():
                if k in home_team or home_team in k:
                    stadium_id = v
                    break

        if stadium_id and stadium_id in _PCTL_LOOKUP:
            pctl_runs, pctl_hr = _PCTL_LOOKUP[stadium_id]
            out.at[idx, "park_id"]        = stadium_id
            out.at[idx, "pctl_park_runs"] = pctl_runs
            out.at[idx, "pctl_park_hr"]   = pctl_hr
            matched += 1
        else:
            out.at[idx, "park_id"]        = "unknown"
            out.at[idx, "pctl_park_runs"] = 50.0
            out.at[idx, "pctl_park_hr"]   = 50.0
            unmatched += 1

    print(f"[park_factors] matched={matched}  unmatched={unmatched}")
    return out
