"""Shared-code shim (Phase 0, decision D1).

Path-imports the pure indicator functions from the live trading-agent repo so
backtested indicators are byte-for-byte identical to production. Only
`tools/indicators.py` is imported: it is pure stdlib with no `config` import
and no I/O, so importing it triggers none of the live stack's side effects
(yfinance, anthropic, gspread, or the env-var hard-fail in trading-agent's
config.py).

`tools/__init__.py` is empty, so `from tools.indicators import ...` is safe.

Phase 2 extracts the live strategy logic into config-free `tools/` modules and
re-exports it here alongside the indicators. Re-exported so far: breadth, VIX,
and the exit primitives (sizing, scoring core, and the BUY cascade land in
later Phase 2 commits). The heavy-side-effect modules (workflow, agent,
market_data) are still never imported here.
"""
import os
import sys

import config as harness_config

_TRADING_AGENT = harness_config.TRADING_AGENT_DIR

if not os.path.isdir(_TRADING_AGENT):
    raise RuntimeError(
        f"trading-agent not found at {_TRADING_AGENT}. The harness expects to "
        f"sit as a sibling of trading-agent/ under trayjent/."
    )

if _TRADING_AGENT not in sys.path:
    sys.path.insert(0, _TRADING_AGENT)

from tools.indicators import (  # noqa: E402
    compute_indicators,
    compute_indicators_pulse,
)
from tools.breadth import trend_anchor, compute_breadth  # noqa: E402
from tools.vix import classify_vix_regime  # noqa: E402
from tools.exit_rules import (  # noqa: E402
    evaluate_price_exit,
    should_force_close_for_max_hold,
)

__all__ = [
    "compute_indicators",
    "compute_indicators_pulse",
    "trend_anchor",
    "compute_breadth",
    "classify_vix_regime",
    "evaluate_price_exit",
    "should_force_close_for_max_hold",
]
