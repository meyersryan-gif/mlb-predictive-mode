from __future__ import annotations

import requests
import pandas as pd

from mlb.data_sources.mlb_stats_api import BASE

UMPIRE_K_TENDENCIES: dict[str, float] = {
    "Angel Hernandez":      1.18,
    "CB Bucknor":           1.15,
    "Doug Eddings":         1.14,
    "Dan Iassogna":         1.13,
    "Laz Diaz":             1.12,
    "Joe West":             1.11,
    "Marvin Hudson":        1.11,
    "Jerry Meals":          1.10,
    "Jim Reynolds":         1.10,
    "Mike Everitt":         1.09,
    "Mark Wegner":          1.09,
    "Lance Barksdale":      1.08,
    "Greg Gibson":          1.08,
    "Toby Basner":          1.08,
    "Mike DiMuro":          1.07,
    "Ron Kulpa":            1.07,
    "Alfonso Marquez":      1.07,
    "Chris Guccione":       1.06,
    "Hunter Wendelstedt":   1.06,
    "Fieldin Culbreth":     1.06,
    "Ed Hickox":            1.05,
    "Tripp Gibson":         1.05,
    "Carlos Torres":        1.05,
    "Cory Blaser":          1.05,
    "Scott Barry":          1.04,
    "Nic Lentz":            1.04,
    "Manny Gonzalez":       1.04,
    "Chad Fairchild":       1.04,
    "Jeremie Rehak":        1.03,
    "Sean Barber":          1.03,
    "John Tumpane":         1.03,
    "Nate Tomlinson":       1.03,
    "Brennan Miller":       1.02,
    "Shane Livensparger":   1.02,
    "Jansen Visconti":      1.02,
    "Derek Thomas":         1.02,
    "Bill Miller":          1.01,
    "Tim Timmons":          1.01,
    "Paul Emmel":           1.01,
    "Mike Muchlinski":      1.01,
    "Vic Carapazza":        1.00,
    "Mark Carlson":         1.00,
    "David Rackley":        1.00,
    "Clint Vondrak":        1.00,
    "Nestor Ceja":          1.00,
    "Alex Tosi":            0.99,
    "Chris Conroy":         0.99,
    "Gabe Morales":         0.99,
    "Adam Beck":            0.99,
    "Brian Knight":         0.99,
    "Dan Bellino":          0.98,
    "Phil Cuzzi":           0.98,
    "Chris Segal":          0.98,
    "Ted Barrett":          0.98,
    "Mike Winters":         0.97,
    "James Hoye":           0.97,
    "Andy Fletcher":        0.97,
    "Pat Hoberg":           0.97,
    "Kerwin Danley":        0.96,
    "Adrian Johnson":       0.96,
    "Tom Hallion":          0.96,
    "Gary Cederstrom":      0.95,
    "Wally Bell":           0.95,
    "Paul Nauert":          0.95,
    "Dale Scott":           0.94,
    "John Hirschbeck":      0.94,
    "Tim McClelland":       0.93,
    "Bruce Dreckman":       0.93,
    "Larry Vanover":        0.92,
    "Jeff Nelson":          0.91,
    "Tim Welke":            0.90,
    "Willie Traynor":       1.03,
    "Jonathan Parra":       1.01,
    "Will Little":          0.98,
    "John Bacon":           0.99,
    "Ramon De Jesus":       1.02,
    "Ryan Blakney":         1.00,
    "Ben May":              1.01,
    "Junior Valentine":     0.97,
    "Mike Estabrook":       1.04,
    "Roberto Ortiz":        0.99,
    "Malachi Moore":        1.00,
    "Takahito Matsuda":     1.00,
}


def _build_percentile_lookup() -> dict[str, float]:
    if not UMPIRE_K_TENDENCIES:
        return {}
    s = pd.Series(UMPIRE_K_TENDENCIES)
    pctls = s.rank(pct=True) * 100.0
    return pctls.to_dict()


_PCTL_LOOKUP: dict[str, float] = _build_percentile_lookup()
_DEFAULT_PCTL = 50.0


def _normalize_team(team: str) -> str:
    return str(team).strip().lower().replace(".", "")


def _fetch_hp_umpires(game_date: str) -> dict[str, str]:
    url = BASE + "/schedule"
    params = {
        "sportId":  1,
        "date":     game_date,
        "hydrate":  "officials",
        "gameType": "R",
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[umpire] MLB API fetch failed: {e}")
        return {}

    assignments: dict[str, str] = {}
    for date_block in data.get("dates", []):
        for game in date_block.get("games", []):
            home_team = game.get("teams", {}).get("home", {}).get("team", {}).get("name", "")
            officials = game.get("officials", [])
            for official in officials:
                if official.get("officialType", "") == "Home Plate":
                    name = official.get("official", {}).get("fullName", "")
                    if home_team and name:
                        assignments[_normalize_team(home_team)] = name
                    break
    return assignments


def attach_umpire_tendencies(
    df: pd.DataFrame,
    game_date: str,
) -> pd.DataFrame:
    out = df.copy()

    if "pctl_umpire_k" not in out.columns:
        out["pctl_umpire_k"] = _DEFAULT_PCTL

    hp_umpires = _fetch_hp_umpires(game_date)
    n_assigned = len(hp_umpires)

    if n_assigned == 0:
        print(f"[umpire] MLB API returned 0 assignments — using neutral default for all rows")
        out["pctl_umpire_k"]  = _DEFAULT_PCTL
        out["hp_umpire_name"] = "unknown"
        return out

    print(f"[umpire] HP assignments loaded: {n_assigned} games for {game_date}")

    matched = unmatched = unknown_ump = 0

    for idx, row in out.iterrows():
        home_team   = _normalize_team(str(row.get("home_team", "") or ""))
        umpire_name = hp_umpires.get(home_team, "")

        if not umpire_name:
            out.at[idx, "pctl_umpire_k"]  = _DEFAULT_PCTL
            out.at[idx, "hp_umpire_name"] = "unknown"
            unmatched += 1
            continue

        out.at[idx, "hp_umpire_name"] = umpire_name
        pctl = _PCTL_LOOKUP.get(umpire_name)
        if pctl is not None:
            out.at[idx, "pctl_umpire_k"] = round(pctl, 1)
            matched += 1
        else:
            out.at[idx, "pctl_umpire_k"] = _DEFAULT_PCTL
            unknown_ump += 1

    print(
        f"[umpire] matched={matched}/{len(out)}  "
        f"unmatched_game={unmatched}  unknown_umpire={unknown_ump}"
    )
    return out

