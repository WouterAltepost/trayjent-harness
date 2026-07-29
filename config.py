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
# Hard ceiling for a single backtest's Claude spend, enforced PER RUN before a
# call that would breach it (Phase 5; fails closed via a rolling per-call avg).
# Set from Stage B's measured cost: ~$0.036/call (Opus 4.7, ~3,830 in / ~675
# out tok). Largest single in-sample run (Pulse-hourly ~2,640 pts) ~$95, Steady
# ~$41, so $250 clears the worst legitimate run ~2.6x while halting a runaway
# well short of $1k. Per-run, so it covers the largest single run, not the sum.
COST_CEILING_USD = 250.0
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

# ── Strategy configs (Phase 5): frozen mirror of live STRATEGY_* ─────────
# Decision-relevant keys only — the live I/O keys (alpaca_*, sheet_name,
# interval, lookback) are dropped: the backtest never trades, fetches, or logs
# to Sheets. A frozen backtest input carried through the runner as RunConfig
# .strategy; NOT imported from trading-agent/config.py (it hard-fails on missing
# env at import). Mirrors STRATEGY_STEADY / STRATEGY_PULSE — update deliberately
# when live strategy params change.
STRATEGY_STEADY = {
    "name": "steady",
    "buy_threshold": 7,
    # take_profit is UNUSED in the backtest once use_trailing_stop governs the
    # exit pass below (live Steady still trades the 5% TP until go-live); kept
    # so this dict stays a faithful live mirror.
    "take_profit": 0.05,
    # stop_loss now serves as the trailing exit's hard-stop floor during the
    # fresh phase (pre-breakeven handoff) and still feeds the sizing 2% clamp.
    "stop_loss": 0.03,
    "ma_short": 50,
    "ma_long": 200,
    "use_obv": False,
    "use_ema_short": False,
    "cash_safety_pct": 0.80,
    "watchlist": STEADY_WATCHLIST,
    # Steady redesign Step 1: ATR trailing stop with hard-stop handoff replaces
    # the fixed TP in the backtest exit pass. mult and period are the sweep
    # dials; these are the pre-sweep defaults.
    "use_trailing_stop": True,
    "trailing_atr_mult": 3.0,
    "atr_period": 22,
}
STRATEGY_PULSE = {
    "name": "pulse",
    "buy_threshold": 6,
    "take_profit": 0.02,
    "stop_loss": 0.01,
    "ma_short": 20,
    "ma_long": 50,
    "use_obv": True,
    "use_ema_short": True,
    "max_hold_hours": 48,
    "cash_safety_pct": 0.90,
    "watchlist": PULSE_WATCHLIST,
    "rsi_buy_ceiling": 75,
    "late_day_block_utc_hours": [19],
}

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

# ── Shortvol sleeve: data foundation (data-only, no strategy logic yet) ──
# The shortvol series live in their own directory, fully separate from the
# frozen Steady/Pulse windows in data/{1d,1h,30m} — the shortvol pull never
# reads or rewrites those, and nothing here touches cache/.
SHORTVOL_DATA_DIR = os.path.join(DATA_DIR, "shortvol")
# ETP/equity legs, pulled from yfinance like the rest of the harness but with
# period="max" — full available history is the point of this sleeve, not the
# 10y window the frozen daily data uses.
SHORTVOL_EQUITY_UNIVERSE = ["SVXY", "VXX", "SPY"]
# Index legs come from CBOE's free published histories, not yfinance: full
# depth (VIX from 1990-01-02, VIX3M from 2009-09-18 — the CBOE file starts
# there, not at the index's 2007 launch; verified 2026-07-29). Stored WITHOUT
# the caret ("VIX", not "^VIX") to keep them visibly distinct from the
# yfinance-sourced ^VIX in the frozen data/1d.
SHORTVOL_CBOE_URLS = {
    "VIX": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
    "VIX3M": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv",
}
# NYSE sessions genuinely absent from CBOE's published VIX history (verified
# against the raw file 2026-07-29: each date is missing while its adjacent
# sessions are present — a quirk of the CBOE file, not a parse error). The
# cross-check treats these as expected; any OTHER index gap vs SPY trading
# days is a hard failure.
CBOE_KNOWN_MISSING_DAYS = {"1997-01-31", "1997-11-26", "1999-12-31"}
# SVXY changed target leverage from -1x to -0.5x effective 2018-02-27 (the
# post-Volmageddon prospectus change). Pre/post are DIFFERENT INSTRUMENTS
# sharing a ticker; any backtest whose window crosses this date must treat the
# two regimes separately, never as one continuous -0.5x series.
SVXY_LEVERAGE_CHANGE_DATE = "2018-02-27"
# VXX coverage: the original ETN (series A, inception 2009-01-30) matured
# 2019-01-30; the current VXX is series B (launched Jan 2018 as VXXB, renamed
# VXX in 2019). yfinance "VXX" carries series B ONLY, split-adjusted across
# its 1:4 reverse splits (verified: history begins 2018-01-25 and the largest
# daily moves are genuine vol spikes, not ~75% split artifacts). The 2009-2018
# series-A history is NOT available under this ticker; if pre-2018 short-vol
# ETN history is ever needed it must come from another source.
VXX_HISTORY_START = "2018-01-25"

# ── Calm (shortvol sleeve) backtest inputs ──────────────────────────────
# Cost model: bps per side on EVERY fill — entry, exit, and the daily
# resize while long. A deliberate backtest input, not a live mirror.
CALM_COST_BPS_PER_SIDE = 5.0
# Tuning window: SVXY inception through the last session before the sealed
# OOS start (OOS_SHORTVOL below). ALL parameter selection happens here.
CALM_TUNING = ("2011-10-04", "2021-12-31")

# ── Rotor sleeve: crypto data foundation (data-only, no strategy yet) ───
# Weekly crypto momentum rotation with a BTC trend gate. Data lives in its
# own directory; the frozen Steady/Pulse windows and cache/ are untouched.
ROTOR_DATA_DIR = os.path.join(DATA_DIR, "rotor")
# Source: Alpaca crypto market data v1beta3 — DELIBERATE: Alpaca is the venue
# Rotor would trade, so its bars embed the venue's pricing. The bars endpoint
# is free and keyless (verified 2026-07-29: HTTP 200 with no auth headers).
ROTOR_BARS_URL = "https://data.alpaca.markets/v1beta3/crypto/us"
# 24/7 BAR CONVENTION (verified 2026-07-29): Alpaca daily crypto bars are
# keyed at 00:00:00Z and cover the UTC calendar day [00:00, 24:00). There is
# no exchange calendar — a Rotor "day" is that UTC bar, never an NYSE
# session. Stored timestamps are the bar's CLOSE instant (key + 24h), same
# no-lookahead rule as every other sleeve: a bar exists only after its
# window completes. A missing day is ALWAYS a data problem, never a holiday
# — sanity demands full 7-day weeks.
# UNIVERSE DISCOVERY, keyless: the trading-API assets endpoint needs keys
# (verified: 401 keyless), so the current tradable set is discovered by
# querying latest/bars with the hand-maintained candidate superset below —
# Alpaca silently omits symbols it does not serve. LIMITATION: a coin Alpaca
# adds that is not in this list is invisible until the list is extended;
# revisit when Alpaca announces additions. All pairs are {COIN}/USD.
ROTOR_CANDIDATES = [
    "AAVE", "ADA", "ALGO", "APT", "ARB", "ATOM", "AVAX", "BAT", "BCH", "BONK",
    "BTC", "COMP", "CRV", "DAI", "DOGE", "DOT", "ETC", "ETH", "FIL", "GRT",
    "HBAR", "ICP", "INJ", "JUP", "LDO", "LINK", "LTC", "MANA", "MATIC", "MKR",
    "NEAR", "ONDO", "OP", "PAXG", "PEPE", "POL", "PYUSD", "RENDER", "SAND",
    "SEI", "SHIB", "SKL", "SNX", "SOL", "SUI", "SUSHI", "TIA", "TON", "TRUMP",
    "TRX", "TUSD", "UNI", "USDC", "USDG", "USDT", "WIF", "XLM", "XRP", "XTZ",
    "YFI",
]
# Stablecoins are excluded from the universe (momentum on a peg is noise).
ROTOR_STABLECOINS = {"USDT", "USDC", "DAI", "TUSD", "BUSD", "PYUSD", "USDG",
                     "USDP", "GUSD", "EURC"}
# Excluded by DECISION, not by rule: PAXG is a gold-backed token — a
# commodity proxy, not crypto momentum — and carries a 968-day
# delist->relist gap on the venue.
ROTOR_EXCLUDED = {"PAXG"}
# Delisted pairs pulled DELIBERATELY alongside the live universe (Alpaca
# serves their full history keylessly — see the survivorship note above):
# these five died on the venue (the June-2023 SEC wave; MKR 2025-09). They
# exist to KILL survivorship bias: the backtest's point-in-time universe
# sees them as tradable during their listing windows and gone afterward.
# The listing windows are DERIVED from the stored bars
# (data_layer.rotor.listing_windows) so the map can never drift from data.
ROTOR_DELISTED = ("ALGO", "MATIC", "MKR", "NEAR", "TRX")
# SURVIVORSHIP BIAS, stated honestly: the universe is Alpaca's CURRENT list,
# so coins they delisted are excluded from any backtest built on this data.
# That is a KNOWN UPWARD BIAS on historical performance. Accepted for v1
# because Rotor only ever holds top-liquidity names — but any backtest result
# on this data must carry this caveat. Two facts learned at the first pull
# (2026-07-29) that a v2 could use to shrink the bias: (a) latest/bars serves
# STALE bars for delisted pairs (observed: ALGO, MATIC, NEAR, TRX ended
# 2023; MKR 2025-09), which is why discovery filters by bar recency; (b)
# Alpaca serves delisted pairs' full history keylessly, so their series could
# be re-added deliberately later. Relatedly, currently-listed coins can carry
# DELIST->RELIST gaps (SOL 2023-06..2024-08, PAXG 2023-06..2026-02): venue
# tradability windows, not data corruption — the strategy step must never
# hold a coin across one or treat a cross-gap return as a daily return.
# Fees, pinned NOW so the backtest step cannot forget them: Alpaca crypto
# tier-1 (lowest volume tier) per-side rates.
ROTOR_ALPACA_CRYPTO_FEES = {"taker": 0.0025, "maker": 0.0015}

# ── Sealed out-of-sample windows (Phase 5 L11) ──────────────────────────
# NEVER pass these ranges to a development run. The runner takes an explicit
# [start, end]; each sealed slice is read exactly once, deliberately, via
# --allow-sealed. Each is (start, end) inclusive, ISO date.
#   Steady (daily):       a 2020-2021 slab (consumed by the Stage E read —
#                         stays fenced) + a fresh 2016-2019 slab reserved for
#                         the grower redesign's one validation read (Step 1).
#   Pulse-hourly (edge):  the most recent quarter.
# Pulse-30min is mechanical fidelity only (≈60d, overlaps the hourly quarter
# by nature) — do not tune any strategy parameter from it, and keep the hourly
# OOS quarter untouched.
#   Calm (shortvol, daily): EVERYTHING from 2022-01-01 onward is sealed —
#                           registered at sleeve creation (2026-07-29) before
#                           any strategy metric was computed on it, so the
#                           2022 bear and the 2024-08 / 2025-04 vol spikes
#                           stay a clean validation set. The end date below is
#                           only the data horizon at sealing: the seal is
#                           START-anchored and covers all data ever pulled
#                           after it. calm.backtest.build_inputs fails closed
#                           on any window touching it.
OOS_STEADY = ("2020-01-01", "2021-12-31")
OOS_STEADY_2016 = ("2016-01-01", "2019-12-31")
OOS_PULSE_HOURLY = ("2026-04-01", "2026-06-24")
OOS_SHORTVOL = ("2022-01-01", "2026-07-29")
