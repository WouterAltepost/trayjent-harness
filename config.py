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

# ── Indicator fidelity windows (see discovery §4) ───────────────────────
# EMA seeds from the first `period` values of the close list, so indicator
# values depend on how far back the slice starts. To reproduce production
# scoring, feed a trailing window matching what the live system fetches.
# Live Steady: ~300-420 daily bars. Live Pulse: ~440 hourly bars over 60d.
STEADY_BARS = 300        # trailing daily closes per decision
PULSE_BARS = 440         # trailing hourly closes per decision
MIN_ROWS = 300           # mirrors fetch_intraday_price_data fail-closed floor

# ── First exit model (D6) ───────────────────────────────────────────────
# "soft" = check unrealized_pct vs threshold at bar close, fill at that close
# (current production). "bracket" reserved for post-v10.
EXIT_MODEL = "soft"
