"""Shared-code shim (Phase 0, decision D1).

Path-imports the pure indicator functions from the live trading-agent repo so
backtested indicators are byte-for-byte identical to production. Only
`tools/indicators.py` is imported: it is pure stdlib with no `config` import
and no I/O, so importing it triggers none of the live stack's side effects
(yfinance, anthropic, gspread, or the env-var hard-fail in trading-agent's
config.py).

`tools/__init__.py` is empty, so `from tools.indicators import ...` is safe.

Sizing, gates, and scoring are deliberately NOT imported here. They live in
modules with heavy import-time side effects and are deferred to Phase 2, where
they get extracted into pure functions on the live side.
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

__all__ = ["compute_indicators", "compute_indicators_pulse"]
