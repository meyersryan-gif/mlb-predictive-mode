from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from mlb.card.build_card import build_autocard
from mlb.card.build_parlay import build_parlay_candidates
from mlb.card.export_card_excel import export_mlb_card_excel
from mlb.card.select_lotto import select_best_lotto
from mlb.data_sources.odds_api_mlb import (
    default_config_from_env,
    get_mlb_events,
    get_event_odds,
    PITCHER_MARKETS,
    HITTER_MARKETS,
)
from mlb.data_sources.mlb_stats_api import get_schedule_with_probables
from mlb.features.build_hitter_lotto import build_hitter_lotto_from_event_odds
from mlb.features.build_pitcher_props import build_pitcher_props_from_event_odds
from mlb.features.map_hitters import attach_hitter_context
from mlb.features.map_probables import attach_probables_to_pitcher_props, attach_pitcher_hand
from mlb.features.opponent_splits import load_team_opponent_tendencies, attach_opponent_tendencies
from mlb.features.pitcher_inputs import load_pitcher_baseline_stats, attach_pitcher_baseline_stats
from mlb.features.probables_table import probables_to_df
from mlb.gates.pitcher_gates import apply_pitcher_hard_gates
from mlb.scoring.score_hitters import score_hitter_lotto
from mlb.scoring.score_pitchers import project_pitcher_props, score_pitcher_props
from mlb.schemas import PITCHER_PROPS_SCHEMA, ensure_schema_columns


def _json_default(obj):
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
    except Exception:
        pass

    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:
            pass
    return str(obj)


def _render_card_text(card: dict, date_iso: str) -> str:
    lines = []
    lines.append(f"MLB CARD — {date_iso}")
    lines.append("")

    flagship_pool = card.get("flagship_pool", [])
    lines.append("TODAY'S CARD")
    if flagship_pool:
        for play in flagship_pool:
            lines.append(
                f"  {play.get('pitcher_name')} {play.get('side')} {play.get('line')} {play.get('market')} "
                f"| Conf {play.get('confidence_score')} | Proj {play.get('proj')} | {play.get('units')}u"
            )
    else:
        lines.append("  No Plays Today — gate found no qualifying edges")

    lines.append("")

    parlay = card.get("parlay")
    if parlay:
        lines.append("PARLAY")
        leg1 = parlay.get("leg1", {})
        leg2 = parlay.get("leg2", {})
        lines.append(
            f"  Leg 1: {leg1.get('pitcher_name')} {leg1.get('side')} {leg1.get('line')} {leg1.get('market')} "
            f"| Conf {leg1.get('confidence_score')}"
        )
        lines.append(
            f"  Leg 2: {leg2.get('pitcher_name')} {leg2.get('side')} {leg2.get('line')} {leg2.get('market')} "
            f"| Conf {leg2.get('confidence_score')}"
        )
        lines.append(
            f"  Type: {parlay.get('corr_type')} | ComboScore {parlay.get('combo_score')} | {parlay.get('units')}u"
        )
    else:
        lines.append("PARLAY")
        lines.append("  No Parlay Today")

    lines.append("")

    lotto = card.get("lotto")
    if lotto:
        leg1 = lotto.get("leg1", {})
        leg2 = lotto.get("leg2", {})
        lines.append("LOTTO")
        lines.append(
            f"  Leg 1: {leg1.get('batter_name')} {leg1.get('side')} {leg1.get('line')} {leg1.get('market')} "
            f"| Price {leg1.get('odds_american')} | Conf {leg1.get('confidence_score')}"
        )
        lines.append(
            f"  Leg 2: {leg2.get('batter_name')} {leg2.get('side')} {leg2.get('line')} {leg2.get('market')} "
            f"| Price {leg2.get('odds_american')} | Conf {leg2.get('confidence_score')}"
        )
        lines.append(f"  Type: {lotto.get('type')} | {lotto.get('units')}u")
    else:
        lines.append("LOTTO")
        lines.append("  Unavailable")

    notes = card.get("notes", []) or []
    if notes:
        lines.append("")
        lines.append("NOTES")
        for n in notes:
            lines.append(f"  - {n}")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--outdir", default=str(Path("mlb") / "outputs"))
    parser.add_argument("--game-types", default="R")
    parser.add_argument("--baseline-season", type=int, default=2025)
    parser.add_argument("--opponent-season", type=int, default=2025)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[mlb] date={args.date}")

    baseline_df = load_pitcher_baseline_stats(args.baseline_season, args.baseline_season, qual=0)
    print(f"[stage0] pitcher_baselines rows={len(baseline_df)} season={args.baseline_season}")

    opp_df = load_team_opponent_tendencies(args.opponent_season, args.opponent_season, qual=0)
    print(f"[stage0] opponent_tendencies rows={len(opp_df)} season={args.opponent_season}")

    games = get_schedule_with_probables(args.date, game_types=args.game_types)
    print(f"[stage1b] statsapi_games_found={len(games)}")

    prob_df = probables_to_df(games)
    prob_path = outdir / f"mlb_probables_{args.date.replace('-', '')}.csv"
    prob_df.to_csv(prob_path, index=False)

    cfg_pitch = default_config_from_env(markets=PITCHER_MARKETS)
    pitch_events = get_mlb_events(cfg_pitch, args.date)
    print(f"[stage1] pitcher_events_found={len(pitch_events)}")

    pitcher_frames = []
    pitcher_ok = 0
    pitcher_fail = 0
    for e in pitch_events:
        event_id = e.get("id")
        if not event_id:
            continue
        try:
            odds = get_event_odds(cfg_pitch, event_id)
            df_event = build_pitcher_props_from_event_odds(args.date, odds)
            if len(df_event):
                pitcher_frames.append(df_event)
            pitcher_ok += 1
        except Exception as exc:
            pitcher_fail += 1
            print(f"[stage1] event_odds_failed event_id={event_id} err={exc}")
    print(f"[stage1b] events_ok={pitcher_ok}  events_failed={pitcher_fail}")

    if pitcher_frames:
        pitcher_props = pd.concat(pitcher_frames, ignore_index=True)
    else:
        pitcher_props = ensure_schema_columns(pd.DataFrame([]), PITCHER_PROPS_SCHEMA)
        pitcher_props["date"] = args.date

    if len(pitcher_props):
        pitcher_props = attach_probables_to_pitcher_props(pitcher_props, games)
        pitcher_props = attach_pitcher_hand(pitcher_props)
        pitcher_props["opp_hand"] = pitcher_props["pitcher_hand"]

        pitcher_props = attach_pitcher_baseline_stats(pitcher_props, baseline_df)
        filled_ip = pitcher_props["avg_ip_per_start"].notna().sum()
        filled_bf = pitcher_props["avg_bf_per_start"].notna().sum()
        filled_k = pitcher_props["pctl_k_pct"].notna().sum()
        filled_bb = pitcher_props["pctl_bb_pct"].notna().sum()
        print(f"[stage1c] baseline_fill avg_ip={filled_ip} avg_bf={filled_bf} k_pct={filled_k} bb_pct={filled_bb}")

        pitcher_props = attach_opponent_tendencies(pitcher_props, opp_df)
        filled_opp_k = pitcher_props["opp_pctl_k_pct_vs_hand"].notna().sum()
        filled_opp_bb = pitcher_props["opp_pctl_bb_pct_vs_hand"].notna().sum()
        filled_opp_woba = pitcher_props["opp_pctl_woba_vs_hand"].notna().sum()
        print(f"[stage1d] opponent_fill k_pct={filled_opp_k} bb_pct={filled_opp_bb} woba={filled_opp_woba}")

    if len(pitcher_props):
        pitcher_props = apply_pitcher_hard_gates(pitcher_props)
        eligible = pitcher_props["hard_gate_pass"].sum()
        print(f"[stage2] eligible_pitcher_rows={eligible}/{len(pitcher_props)}")
    else:
        print("[stage2] no pitcher props to gate")

    if len(pitcher_props):
        pitcher_props = project_pitcher_props(pitcher_props)
        pitcher_props = score_pitcher_props(pitcher_props)
        scored = (pitcher_props["confidence_score"] > 0).sum()
        print(f"[stage3] scored_pitcher_rows={scored}/{len(pitcher_props)}")
    else:
        print("[stage3] no pitcher props to score")

    stage3_path = outdir / f"mlb_pitcher_props_{args.date.replace('-', '')}_STAGE3.csv"
    pitcher_props.to_csv(stage3_path, index=False)
    print(f"[export] {stage3_path}")

    cfg_hit = default_config_from_env(markets=HITTER_MARKETS)
    hit_events = get_mlb_events(cfg_hit, args.date)
    print(f"[stage4] hitter_events_found={len(hit_events)}")

    hitter_frames = []
    hitter_ok = 0
    hitter_fail = 0
    for e in hit_events:
        event_id = e.get("id")
        if not event_id:
            continue
        try:
            odds = get_event_odds(cfg_hit, event_id)
            df_event = build_hitter_lotto_from_event_odds(args.date, odds)
            if len(df_event):
                hitter_frames.append(df_event)
            hitter_ok += 1
        except Exception as exc:
            hitter_fail += 1
            print(f"[stage4] event_odds_failed event_id={event_id} err={exc}")
    print(f"[stage4b] events_ok={hitter_ok}  events_failed={hitter_fail}  frames_with_data={len(hitter_frames)}")

    if hitter_frames:
        hitter_props = pd.concat(hitter_frames, ignore_index=True)
    else:
        hitter_props = pd.DataFrame([])

    if len(hitter_props):
        hitter_props = attach_hitter_context(
            hitter_props,
            games,
            date_iso=args.date,
            season=args.baseline_season,
            game_types=args.game_types,
        )

        if "pctl_pitcher_k_pct" not in hitter_props.columns:
            hitter_props["pctl_pitcher_k_pct"] = 50.0
        if "pctl_pitcher_bb_pct" not in hitter_props.columns:
            hitter_props["pctl_pitcher_bb_pct"] = 50.0

        hitter_props = score_hitter_lotto(hitter_props)
        lotto = select_best_lotto(hitter_props)
        print(f"[stage5] scored_hitter_rows={len(hitter_props)}")
    else:
        hitter_props = pd.DataFrame([])
        lotto = None
        print("[stage5] no hitter props scored")

    hitter_path = outdir / f"mlb_hitter_lotto_{args.date.replace('-', '')}_SCORED.csv"
    if len(hitter_props):
        hitter_props.to_csv(hitter_path, index=False)
        print(f"[export] {hitter_path}")

    card = build_autocard(pitcher_props, lotto=lotto)
    parlay_candidates = build_parlay_candidates(pitcher_props)

    card_json_path = outdir / f"mlb_card_{args.date.replace('-', '')}.json"
    with card_json_path.open("w", encoding="utf-8") as f:
        json.dump(card, f, indent=2, default=_json_default)

    card_text = _render_card_text(card, args.date)
    card_txt_path = outdir / f"mlb_card_{args.date.replace('-', '')}.txt"
    card_txt_path.write_text(card_text, encoding="utf-8")

    card_xlsx_path = outdir / f"mlb_card_{args.date.replace('-', '')}.xlsx"
    export_mlb_card_excel(
        outpath=card_xlsx_path,
        date_iso=args.date,
        card=card,
        scored_pitchers=pitcher_props,
        parlay_candidates=parlay_candidates,
        flagship_pool_min_conf=74.0,
        flagship_pool_top_n=15,
    )

    print(f"[export] {card_json_path}")
    print(f"[export] {card_txt_path}")
    print(f"[export] {card_xlsx_path}")
    print("")
    print(card_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

