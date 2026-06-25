"""Backtest metrics + SPY benchmark (Phase 5 brief Step 5 / L9).

Computed from the equity curve + closed-trade ledger of a :class:`RunResult`.
Every metric has an explicit definition (§Metrics); the non-obvious ones:

- **Sharpe / beta are sampled on the run's own cadence** (L9). Mixing daily and
  intraday returns is meaningless. The annualization factor is derived
  EMPIRICALLY from the equity curve — ``(n_points - 1) / years_elapsed``, the
  true sampling frequency — rather than a fixed per-cadence constant. That is
  exact for all three cadences and robust to half-days, holidays, and the 15:30
  hourly stub bar (the curve already encodes how often the run actually ticks).
  Risk-free rate = 0.
- **The SPY benchmark is sampled on that same cadence** — a buy-and-hold equity
  series read at each decision ``as_of`` — so beta and the benchmark comparison
  line up bar-for-bar. It doubles as the Phase-6 comparison baseline.

The pure scalar helpers (``_max_drawdown``, ``_sharpe``, ``_beta``,
``_profit_factor``, ``_cagr``) take plain number lists so they can be
hand-checked in isolation; :func:`compute_metrics` wires them to a run + the
SPY series (the one place this module reads stored bars via the slicer).
"""
import statistics
from collections import Counter

from data_layer.slice import get_price_asof

_SECONDS_PER_YEAR = 365.25 * 24 * 3600


# ── Pure scalar helpers (hand-checkable) ────────────────────────────────
def _years_between(ts_first, ts_last) -> float:
    return (ts_last - ts_first).total_seconds() / _SECONDS_PER_YEAR


def _periods_per_year(ts_list: list):
    """Empirical sampling frequency: intervals (n-1) per elapsed year. None for
    a degenerate (< 2 points or zero-span) curve. Supersedes fixed per-cadence
    constants — exact and robust to half-days / holidays / the stub bar."""
    if len(ts_list) < 2:
        return None
    years = _years_between(ts_list[0], ts_list[-1])
    return (len(ts_list) - 1) / years if years > 0 else None


def _period_returns(series: list) -> list:
    """Simple per-period returns of a value series (len-1 of them)."""
    return [series[i] / series[i - 1] - 1.0
            for i in range(1, len(series)) if series[i - 1]]


def _max_drawdown(series: list):
    """Max peak-to-trough fractional decline (>= 0). None on an empty series."""
    if not series:
        return None
    peak = series[0]
    mdd = 0.0
    for v in series:
        if v > peak:
            peak = v
        if peak > 0:
            mdd = max(mdd, (peak - v) / peak)
    return mdd


def _sharpe(returns: list, periods_per_year: float):
    """mean/std(returns) * sqrt(periods_per_year), rf=0. None if < 2 points,
    zero variance, or no sampling frequency (std uses sample estimator, ddof=1)."""
    if periods_per_year is None or len(returns) < 2:
        return None
    sd = statistics.stdev(returns)
    if sd == 0:
        return None
    return (statistics.mean(returns) / sd) * (periods_per_year ** 0.5)


def _sample_cov(xs: list, ys: list) -> float:
    n = len(xs)
    mx, my = statistics.mean(xs), statistics.mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n - 1)


def _beta(returns: list, bench_returns: list):
    """cov(r, r_spy) / var(r_spy), both sample estimators on the same cadence.
    None if < 2 aligned points or the benchmark has zero variance."""
    if len(returns) < 2 or len(returns) != len(bench_returns):
        return None
    var = statistics.variance(bench_returns)
    if var == 0:
        return None
    return _sample_cov(returns, bench_returns) / var


def _profit_factor(closed: list):
    """gross_profit / |gross_loss| over realized dollars. None when there are no
    losing trades (the ∞-guard) or no trades at all."""
    gross_profit = sum(t.realized_dollars for t in closed if t.realized_dollars > 0)
    gross_loss = -sum(t.realized_dollars for t in closed if t.realized_dollars < 0)
    if gross_loss == 0:
        return None
    return gross_profit / gross_loss


def _cagr(initial: float, final: float, years: float):
    """(final/initial)^(1/years) - 1. None when the horizon or base is degenerate."""
    if years <= 0 or initial <= 0:
        return None
    return (final / initial) ** (1.0 / years) - 1.0


def _hold_hours(trade) -> float:
    return (trade.exit_ts - trade.entry_ts).total_seconds() / 3600.0


# ── Integration: a full metrics block for a RunResult ───────────────────
def spy_benchmark_prices(ts_list: list, decision_tf: str) -> list:
    """SPY decision-tf price at each decision ``as_of`` — the buy-and-hold
    series, sampled on the run cadence (L9). Reads stored bars via the slicer."""
    return [get_price_asof("SPY", decision_tf, ts) for ts in ts_list]


def compute_metrics(run_result) -> dict:
    """Return the metrics block (incl. nested ``benchmark_spy``) for a run.

    Empty/degenerate inputs yield ``None`` for the affected metric rather than
    raising, so a no-trade or single-point run still produces a valid block.
    """
    rc = run_result.run_config
    curve = run_result.equity_curve
    closed = run_result.closed_trades
    initial = run_result.initial_capital

    equity = [pt["portfolio_value"] for pt in curve]
    ts_list = [pt["ts"] for pt in curve]
    final = equity[-1] if equity else initial
    years = _years_between(ts_list[0], ts_list[-1]) if len(ts_list) >= 2 else 0.0
    # Annualization frequency derived from the curve's own sampling (L9).
    ppy = _periods_per_year(ts_list)

    strat_returns = _period_returns(equity)
    spy_prices = spy_benchmark_prices(ts_list, rc.decision_tf)
    spy_returns = _period_returns(spy_prices)

    # Exposure + holds.
    n_points = len(curve)
    holds = [_hold_hours(t) for t in closed]
    wins = sum(1 for t in closed if t.realized_dollars > 0)

    benchmark = None
    if spy_prices:
        bench_equity = [initial * (p / spy_prices[0]) for p in spy_prices]
        benchmark = {
            "total_return": bench_equity[-1] / initial - 1.0,
            "cagr": _cagr(initial, bench_equity[-1], years),
            "sharpe": _sharpe(spy_returns, ppy),
            "max_drawdown": _max_drawdown(bench_equity),
        }

    return {
        "total_return": final / initial - 1.0 if initial else None,
        "cagr": _cagr(initial, final, years),
        "win_rate": wins / len(closed) if closed else None,
        "profit_factor": _profit_factor(closed),
        "max_drawdown": _max_drawdown(equity),
        "sharpe": _sharpe(strat_returns, ppy),
        "beta_spy": _beta(strat_returns, spy_returns),
        "periods_per_year": ppy,
        "exposure": {
            "pct_decisions_with_position": (
                sum(1 for pt in curve if pt["n_positions"] >= 1) / n_points
                if n_points else None),
            "avg_positions": (
                sum(pt["n_positions"] for pt in curve) / n_points
                if n_points else None),
            "avg_hold_hours": statistics.mean(holds) if holds else None,
            "median_hold_hours": statistics.median(holds) if holds else None,
        },
        "trades": {
            "total": len(closed),
            "by_exit_reason": dict(Counter(t.exit_reason for t in closed)),
        },
        "benchmark_spy": benchmark,
    }
