from __future__ import annotations

# ---------------------------------------------------------------------------
# NOTE: This is a portfolio/showcase version of the configuration module.
# The tuning constants below (thresholds, odds windows, Kelly parameters)
# are illustrative placeholders, not the production values. The original
# values are derived from an ongoing backtest against graded picks and are
# kept private. The structure, gating logic, and architecture are unchanged
# from production.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Unit sizing — single source of truth for all card tiers
# ---------------------------------------------------------------------------
UNITS_FLAGSHIP = 1.00
UNITS_PARLAY   = 0.25
UNITS_LOTTO    = 0.10

# ---------------------------------------------------------------------------
# Confidence thresholds — single source of truth
# Each tier (flagship / parlay-leg / lotto) requires a minimum model
# confidence score (0-100) before a pick is eligible for that tier.
# Illustrative values below; production thresholds are tuned via backtest.
# ---------------------------------------------------------------------------
THRESH_FLAGSHIP   = 80
THRESH_PARLAY_LEG = 70
THRESH_LOTTO      = 60

# ---------------------------------------------------------------------------
# Flagship juice and EV gates
#
# The model restricts which market prices (American odds) a flagship pick
# is allowed to be priced at, on top of the confidence-score gate. This
# window is periodically re-tuned against graded results — the bounds here
# are illustrative, not the live production values.
#
# FLAGSHIP_MAX_JUICE: floor — reject anything juicier than this.
# FLAGSHIP_MIN_JUICE: ceiling — reject anything lighter than this.
# FLAGSHIP_MIN_EV: EV must still be strictly positive (price-sanity filter).
# ---------------------------------------------------------------------------
FLAGSHIP_MAX_JUICE = -200   # American odds floor
FLAGSHIP_MIN_JUICE = -130   # American odds ceiling
FLAGSHIP_MIN_EV    = 0.0    # EV must be strictly positive

# ---------------------------------------------------------------------------
# Flagship juice-window switch
#
# True  = enforce the [FLAGSHIP_MAX_JUICE, FLAGSHIP_MIN_JUICE] window above.
# False = ignore the window entirely; gate on conf>=THRESH_FLAGSHIP and EV>0
#         only. This switch lets the price window be A/B tested against the
#         unrestricted baseline as new graded data comes in.
# ---------------------------------------------------------------------------
FLAGSHIP_USE_JUICE_WINDOW = True

# ---------------------------------------------------------------------------
# Side-specific confidence floor
#
# Overs and Unders on the same market don't always carry equal signal
# strength — in this model, Unders require a higher confidence floor than
# Overs before qualifying for the flagship tier. Set equal to
# THRESH_FLAGSHIP to disable the asymmetry.
# ---------------------------------------------------------------------------
FLAGSHIP_UNDER_MIN_CONF = 84

# ---------------------------------------------------------------------------
# Track-record launch date
# Picks graded on/after this date are reported as LIVE (actually bet under
# the current gate); picks before it are reported as BACKTEST (modeled
# restatement under the current gate, for track-record continuity).
# ---------------------------------------------------------------------------
LAUNCH_DATE = "2026-06-15"

# ---------------------------------------------------------------------------
# Parlay settings
# Only K props are parlay-eligible on FanDuel.
# OUTS is a standalone market only and cannot be combined with anything.
# Both legs must be different pitchers from different games.
# PARLAY_MAX_JUICE: hardest odds allowed per leg — beyond this the book is
# more confident than the model, making the leg a negative-EV inclusion.
# ---------------------------------------------------------------------------
PARLAY_NUM_LEGS = 2
PARLAY_MAX_JUICE = -130
PARLAY_CORR_BONUS = {
    "two_pitcher_k_same_side": 4,
}

# Cross-game is enforced (different games) as a pricing rule — same-game
# parlays get correlation-adjusted (shortened) by the book, so same-game
# pairs are excluded regardless of the setting below.
PARLAY_REQUIRE_SAME_SIDE = False

# ---------------------------------------------------------------------------
# Market filters
# ---------------------------------------------------------------------------
PITCHER_MARKETS      = {"K", "OUTS", "K_ALT"}
HITTER_LOTTO_MARKETS = {"HR", "TB", "HITS_2PLUS", "RBI"}

# ---------------------------------------------------------------------------
# Scoring caps
# ---------------------------------------------------------------------------
CAP_NO_FLAGSHIP          = True
CAP_IF_MATCHUP_POINTS_LT = (10, 79)
CAP_IF_LEASH_POINTS_LT   = (12, 79)
CAP_IF_MARKET_POINTS_LT  = (4, 79)

HITTER_LOTTO_MAX_CONF = 85

# ---------------------------------------------------------------------------
# Kelly Criterion bankroll management
#
# KELLY_FRACTION: fraction of full Kelly to use (1.00 = full Kelly,
#   0.50 = half Kelly, 0.25 = quarter Kelly for conservative sizing).
# KELLY_MAX_BET_PCT: hard cap as % of bankroll per play regardless of Kelly,
#   to bound variance on any single pick.
# BANKROLL: placeholder value — update to reflect actual bankroll in a real
#   deployment. Used only to show illustrative dollar sizing on the card.
# ---------------------------------------------------------------------------
KELLY_FRACTION    = 0.50
KELLY_MAX_BET_PCT = 0.02
BANKROLL = 10000.0
