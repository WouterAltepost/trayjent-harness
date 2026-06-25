"""TBH harness configuration. Mac-only, offline. No live secrets required.

Only the Claude scoring step (Phase 4) needs ANTHROPIC_API_KEY, and it is read
lazily there, not at import time, so the rest of the harness imports cleanly
without any environment setup.
"""
import os

# ── Paths ───────────────────────────────────────────────────────────────
HARNESS_ROOT = os.path.dirname(os.path.abspath(__file__))
TRADING_AGENT_DIR = os.path.normpath(os.path.join(HARNESS_ROOT, "..", "trading-agent"))
DATA_DIR = os.path.join(HARNESS_ROOT, "data")     # Parquet bars
CACHE_DIR = os.path.join(HARNESS_ROOT, "cache")   # SQLite Claude cache

# ── Scoring / cost (Phase 4+) ───────────────────────────────────────────
# The prompt is a versioned cache-key input. Bump when the agent prompt
# changes so the Claude cache invalidates. Mirror trading-agent prompt edits.
PROMPT_VERSION = "v9.5"
# Hard ceiling for a single backtest's Claude spend, enforced before a run
# that would call the API (Phase 5). Rough full-Pulse estimate is $200-400.
COST_CEILING_USD = 50.0
# Per-model token prices ($/Mtok), confirmed PF-1 against current Anthropic
# pricing: claude-opus-4-7 is the Opus 4.x tier at $5 in / $25 out. The cost
# tracker (Phase 4) computes spend from this table; a wrong rate makes the
# ceiling lie, so this is a confirmed external value, not a guess.
PRICES = {
    "claude-opus-4-7": {"input_per_mtok": 5.00, "output_per_mtok": 25.00},
}

# ── Indicator fidelity windows (see discovery §4) ───────────────────────
# EMA seeds from the first `period` values of the close list, so indicator
# values depend on how far back the slice starts. To reproduce production
# scoring, feed a trailing window matching what the live system fetches.
# Live Steady: ~300-420 daily bars. Live Pulse: ~440 hourly bars over 60d.
STEADY_BARS = 300        # trailing daily closes per decision
PULSE_BARS = 440         # trailing hourly closes per decision
MIN_ROWS = 300           # mirrors fetch_intraday_price_data fail-closed floor

# ── Position sizing (Phase 2): frozen mirror of live POSITION_SIZING ─────
# tools.sizing.compute_position_size is config-free (L5) and takes this dict as
# a required arg. Mirrors trading-agent/config.py POSITION_SIZING; NOT imported
# from live config (it hard-fails on missing env at import). Frozen backtest
# input — update deliberately when live sizing changes. Pinned by
# tests/test_reuse_import.py.
POSITION_SIZING = {
    "base_pct_per_score": 0.05,
    "cash_safety_pct": 0.80,
    "min_trade_dollars": 500,
    "multiplier_cap": 4,
}

# ── First exit model (D6) ───────────────────────────────────────────────
# "soft" = check unrealized_pct vs threshold at bar close, fill at that close
# (current production). "bracket" reserved for post-v10.
EXIT_MODEL = "soft"

# ── Portfolio simulator (Phase 3) ───────────────────────────────────────
# Starting cash for a backtest portfolio. Mirrors the Alpaca paper accounts;
# a deliberate backtest input, settable per run later (L9).
INITIAL_CAPITAL = 100_000.0
# Idealized fills (L5): v1 fills at the raw decision-bar close with ZERO costs.
# These two seams exist so non-zero fees/slippage is a one-line change later
# (applied through Portfolio._fill_price); no non-zero behavior this phase.
# Any result built on these is gross, not net-of-costs — Phase 5 output must
# flag "idealized fills (0 fees, 0 slippage, fill at close)".
FEES_BPS = 0.0
SLIPPAGE_BPS = 0.0

# ── Data layer (Phase 1): watchlists + pull universe ────────────────────
# Mirrors trading-agent/config.py v9.5. Frozen backtest input; update
# deliberately. NOT imported from live config.py (it hard-fails on missing
# env vars at import time, per phase0_discovery §3).
STEADY_WATCHLIST = ["SPY", "MSFT", "NVDA", "AMD", "AMZN", "GOOGL", "GLD"]
PULSE_WATCHLIST = ["NVDA", "AMD", "TSLA", "META", "AMZN", "GOOGL"]

# Per-timeframe pull universe (brief L4). Daily = Steady watchlist + ^VIX
# (the regime signal is daily even when Pulse runs hourly). Hourly = Pulse
# watchlist + SPY (market/breadth context). SPY is held on both timeframes.
DAILY_UNIVERSE = STEADY_WATCHLIST + ["^VIX"]
HOURLY_UNIVERSE = ["SPY"] + PULSE_WATCHLIST
# 30-min universe (Phase 5 L3): Pulse watchlist + SPY, same as hourly. The
# 30m bars feed Pulse-30min's decision/exit *prices*; indicators still come
# from the 1h bars (L2). No ^VIX — the regime signal is daily (A4).
THIRTYMIN_UNIVERSE = ["SPY"] + PULSE_WATCHLIST

# timeframe label -> {yfinance interval, pull universe}. The label is also the
# Parquet subdir: data/{timeframe}/{ticker}.parquet (L3).
TIMEFRAMES = {
    "1d": {"interval": "1d", "universe": DAILY_UNIVERSE},
    "1h": {"interval": "1h", "universe": HOURLY_UNIVERSE},
    "30m": {"interval": "30m", "universe": THIRTYMIN_UNIVERSE},
}

# ── Data layer: history depth + adjustment (brief L1, L5) ───────────────
DAILY_HISTORY_YEARS = 10        # yfinance period for daily pulls ("10y")
# Hourly is hard-capped ~730 days by Yahoo (verified PF-3). "2y" cleanly
# returns the true ~729-day max; the literal "730d" is not a valid yfinance
# period string and returns cache-polluted data, so it is deliberately avoided.
HOURLY_PERIOD = "2y"
# 30-min is hard-capped at ~60 trading days by Yahoo (verified Phase 5 PF-1:
# "60d" returns ~779 SPY bars ≈ 60 sessions; "90d"/"730d"/"2y" all return
# EMPTY with "must be within the last 60 days"). "60d" is the true max.
THIRTYMIN_PERIOD = "60d"
# Store split/dividend-adjusted OHLCV (L1). Set explicitly so the harness is
# deterministic regardless of the installed yfinance default.
AUTO_ADJUST = True

# ── Sealed out-of-sample windows (Phase 5 L11) ──────────────────────────
# NEVER pass these ranges to a development run. The runner takes an explicit
# [start, end]; these slices are simply never requested until the one
# post-Phase-6 read. Each is (start, end) inclusive, ISO date.
#   Steady (daily):       a 2020-2021 slab.
#   Pulse-hourly (edge):  the most recent quarter.
# Pulse-30min is mechanical fidelity only (≈60d, overlaps the hourly quarter
# by nature) — do not tune any strategy parameter from it, and keep the hourly
# OOS quarter untouched.
OOS_STEADY = ("2020-01-01", "2021-12-31")
OOS_PULSE_HOURLY = ("2026-04-01", "2026-06-24")
