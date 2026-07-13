# MLB Predictive Modeling Pipeline

A live, daily-running predictive model for MLB player-prop markets (pitcher strikeouts/outs, hitter home runs/hits/total bases/RBI), built and independently maintained by directing AI coding tools rather than hand-writing the implementation.

**This is a portfolio/showcase version of a live production system.** The core architecture, data pipeline, and gating logic are unchanged from production. The specific tuned model weights, confidence thresholds, and backtest results that constitute the actual predictive edge have been replaced with illustrative placeholders — see [Note on redactions](#note-on-redactions) below.

## What this demonstrates

- **End-to-end system ownership**: live data ingestion → feature engineering → scoring → gating → output generation, running daily without hand-holding.
- **Directing AI tools to build real software**: I specify the problem precisely (grounded in domain knowledge I've built up watching these markets), direct AI tools to implement and debug it, and evaluate whether the output is actually correct — the same workflow described in my [job search materials](../Application%20Materials).
- **Production-minded engineering habits**: explicit hard-gates for missing/unreliable data rather than silent defaults, config-driven thresholds instead of magic numbers scattered through the code, and a single source of truth for tunable parameters (`config.py`).

## Architecture

```
data_sources/     Live data ingestion — MLB Stats API (rosters, probable
                   starters, live stats) and The Odds API (betting lines
                   across markets)

features/         ~15 feature-engineering modules: park factors, weather,
                   umpire tendencies, opponent splits, lineup quality,
                   pitcher workload/suppression, line movement, early-
                   season blending of prior/current-year data, etc.

gates/             Hard data-quality gates — a pick can't reach scoring
                   without a mapped player ID, a valid market, and valid
                   line/odds data.

scoring/            Multi-factor scoring: percentile-ranked inputs across
                   skill, matchup, workload, environment, and market
                   agreement combine into a 0-100 confidence score, which
                   is then calibrated to an estimated hit rate and run
                   through fractional Kelly bet sizing.

card/               Assembles the day's output: a flagship pick pool, a
                   cross-game parlay, a "lotto" tier for hitter props, and
                   a lower-confidence tracked pool used for ongoing model
                   calibration.

config.py           Single source of truth for every tunable threshold,
                   gate, and sizing parameter used across the pipeline.

schemas.py, main.py  Data contracts and the daily entry point.
```

## How it runs

Each day: pull probable starters and live odds → build features per pitcher/hitter → apply hard gates → score → apply Kelly sizing → assemble the card → export.

## Note on redactions

`config.py` and the two `scoring/` modules contain the pipeline's tunable parameters and weighted-scoring formulas. In this portfolio version:

- Exact confidence thresholds, odds-price gating windows, and Kelly sizing parameters are illustrative placeholders, not production values.
- The specific coefficients in the multi-factor scoring formulas (e.g. how much weight goes to matchup quality vs. workload vs. environment) are illustrative placeholders that preserve the *shape* of the production model, not the tuned weights themselves.
- Backtest-derived commentary (win rates, ROI figures) has been removed from code comments.

Everything else — the data pipeline, feature engineering, gating logic, and system architecture — is unchanged from what actually runs daily.

## Stack

Python, pandas, numpy, scipy — live API integrations (MLB Stats API, The Odds API) — no ML framework dependency; the scoring model is a hand-specified multi-factor system rather than a trained black-box model, which is a deliberate choice for interpretability (every score is traceable to its inputs).
