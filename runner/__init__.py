"""TBH Phase 5 backtest runner.

Ties Phases 1-4 into a deterministic replay: generate decision points over
historical bars (``schedule``), slice indicator windows + context, score
(cached Claude or rules-only), run the exit pass and BUY cascade against the
simulated portfolio (``run``), and emit a versioned JSON with trades, an equity
curve, and metrics. This package is orchestration + state threading only — every
decision primitive already exists via the reuse shim and the portfolio.
"""
from runner.configs import (
    RunConfig,
    build_run_config,
    PRESET_NAMES,
    MODES,
)
from runner.schedule import decision_points
from runner.run import run_backtest, RunResult

__all__ = [
    "RunConfig",
    "build_run_config",
    "PRESET_NAMES",
    "MODES",
    "decision_points",
    "run_backtest",
    "RunResult",
]
