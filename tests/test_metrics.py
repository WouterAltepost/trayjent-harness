"""Hand-checked metrics tests (brief Step 5).

The scalar helpers are pinned against arithmetic oracles; the integration
``compute_metrics`` runs over a tiny synthetic equity curve + closed-trade
ledger + a 3-bar SPY Parquet, with every expected value computed by hand in the
comments. Offline: a tmp DATA_DIR holds the SPY benchmark series.

    python tests/test_metrics.py
"""
import math
import os
import sys
import tempfile
from types import SimpleNamespace

import pandas as pd

_HARNESS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HARNESS_ROOT not in sys.path:
    sys.path.insert(0, _HARNESS_ROOT)

import config
from simulator.portfolio import ClosedTrade
from runner.configs import RunConfig
from runner.run import RunResult
from runner import metrics as m


# ── Pure scalar oracles ─────────────────────────────────────────────────
def test_max_drawdown_oracle():
    # peak 120 then trough 90 -> (120-90)/120 = 0.25; later 130 is a new peak.
    assert m._max_drawdown([100, 120, 90, 130]) == 0.25
    assert m._max_drawdown([]) is None


def test_profit_factor_and_no_loss_guard():
    closed = [SimpleNamespace(realized_dollars=x) for x in (500, -200, 300)]
    assert m._profit_factor(closed) == 800 / 200          # 4.0
    only_wins = [SimpleNamespace(realized_dollars=x) for x in (10, 20)]
    assert m._profit_factor(only_wins) is None            # ∞-guard, no losses


def test_sharpe_guards_and_annualization_scaling():
    returns = [0.01, 0.02, 0.01, 0.03]
    assert m._sharpe([0.01], 252) is None                 # < 2 points
    assert m._sharpe([0.01, 0.01], 252) is None           # zero variance
    assert m._sharpe(returns, None) is None               # no sampling frequency
    # The cadence enters ONLY through sqrt(periods_per_year): scaling the
    # frequency by 13 must scale Sharpe by sqrt(13).
    ratio = m._sharpe(returns, 13 * 252.0) / m._sharpe(returns, 252.0)
    assert math.isclose(ratio, math.sqrt(13.0), rel_tol=1e-12)


def test_periods_per_year_derived_from_curve():
    base = pd.Timestamp("2024-01-01", tz="UTC")
    # 2 points exactly one year apart -> 1 interval / 1 year = 1.0.
    one_year = [base, base + pd.Timedelta(seconds=m._SECONDS_PER_YEAR)]
    assert math.isclose(m._periods_per_year(one_year), 1.0, rel_tol=1e-9)
    # 3 points over half a year -> 2 intervals / 0.5 year = 4.0.
    half = m._SECONDS_PER_YEAR / 2
    half_year = [base, base + pd.Timedelta(seconds=half / 2), base + pd.Timedelta(seconds=half)]
    assert math.isclose(m._periods_per_year(half_year), 4.0, rel_tol=1e-9)
    assert m._periods_per_year([base]) is None            # degenerate


def test_beta_identical_and_zero_variance():
    r = [0.1, -0.1, 0.05]
    assert math.isclose(m._beta(r, r), 1.0, rel_tol=1e-12)   # vs itself -> 1
    assert m._beta([0.1, -0.1], [0.02, 0.02]) is None        # benchmark flat


# ── Integration over a synthetic run ────────────────────────────────────
def _trade(realized, entry, exit_, reason):
    return ClosedTrade(
        ticker="X", entry_ts=entry, entry_price=100.0, exit_ts=exit_,
        exit_price=100.0 + realized / 10, qty=10.0, notional=1000.0,
        realized_dollars=realized, realized_pct=realized / 10, exit_reason=reason,
        meta={},
    )


def test_compute_metrics_integration():
    data_dir = tempfile.mkdtemp(prefix="tbh_metrics_")
    config.DATA_DIR = data_dir
    ts = pd.date_range("2024-03-01 21:00", periods=3, freq="B", tz="UTC")
    # SPY benchmark series: 100 -> 105 -> 102 (close-stamped daily bars).
    spy = pd.DataFrame({
        "timestamp": ts, "open": [100, 105, 102], "high": [100, 105, 102],
        "low": [100, 105, 102], "close": [100.0, 105.0, 102.0],
        "volume": [0, 0, 0], "ticker": "SPY", "timeframe": "1d",
    })
    os.makedirs(os.path.join(data_dir, "1d"), exist_ok=True)
    spy.to_parquet(os.path.join(data_dir, "1d", "SPY.parquet"), index=False)

    # Strategy equity 100k -> 110k -> 99k; n_positions 1,2,0.
    curve = [
        {"ts": ts[0], "portfolio_value": 100_000.0, "cash": 0.0, "n_positions": 1},
        {"ts": ts[1], "portfolio_value": 110_000.0, "cash": 0.0, "n_positions": 2},
        {"ts": ts[2], "portfolio_value": 99_000.0, "cash": 0.0, "n_positions": 0},
    ]
    closed = [
        _trade(500, ts[0], ts[0] + pd.Timedelta(hours=24), "SELL (TAKE PROFIT)"),
        _trade(-200, ts[0], ts[0] + pd.Timedelta(hours=48), "SELL (STOP LOSS)"),
        _trade(300, ts[0], ts[0] + pd.Timedelta(hours=72), "SELL (TAKE PROFIT)"),
    ]
    rc = RunConfig(name="steady", strategy={"name": "steady"}, indicator_tf="1d",
                   indicator_bars=300, decision_tf="1d", cadence="daily",
                   start=ts[0], end=ts[2], mode="rules_only")
    res = RunResult(run_config=rc, portfolio=None, closed_trades=closed,
                    equity_curve=curve, n_decision_points=3, initial_capital=100_000.0)

    out = m.compute_metrics(res)

    # total_return = 99000/100000 - 1 = -0.01.
    assert math.isclose(out["total_return"], -0.01, abs_tol=1e-12)
    # max_drawdown: peak 110k, trough 99k -> 11000/110000 = 0.10.
    assert math.isclose(out["max_drawdown"], 0.10, abs_tol=1e-12)
    # strat returns [+0.10, -0.10]: mean 0 -> Sharpe 0.
    assert math.isclose(out["sharpe"], 0.0, abs_tol=1e-12)
    # win_rate 2/3; profit_factor 800/200 = 4.0.
    assert math.isclose(out["win_rate"], 2 / 3, rel_tol=1e-12)
    assert math.isclose(out["profit_factor"], 4.0, rel_tol=1e-12)
    # Exposure: 2 of 3 points hold a position; avg positions (1+2+0)/3 = 1.0.
    assert math.isclose(out["exposure"]["pct_decisions_with_position"], 2 / 3, rel_tol=1e-12)
    assert math.isclose(out["exposure"]["avg_positions"], 1.0, rel_tol=1e-12)
    assert out["exposure"]["avg_hold_hours"] == 48.0 and out["exposure"]["median_hold_hours"] == 48.0
    assert out["trades"] == {"total": 3, "by_exit_reason": {"SELL (TAKE PROFIT)": 2, "SELL (STOP LOSS)": 1}}

    # Benchmark SPY 100->105->102: total_return 0.02; drawdown 3000/105000.
    b = out["benchmark_spy"]
    assert math.isclose(b["total_return"], 0.02, rel_tol=1e-12)
    assert math.isclose(b["max_drawdown"], 3000 / 105000, rel_tol=1e-12)
    # beta = cov(strat, spy)/var(spy) with strat [0.1,-0.1], spy [0.05,-0.0285714].
    assert math.isclose(out["beta_spy"], 2.54545, abs_tol=1e-4)


if __name__ == "__main__":
    test_max_drawdown_oracle()
    test_profit_factor_and_no_loss_guard()
    test_sharpe_guards_and_annualization_scaling()
    test_periods_per_year_derived_from_curve()
    test_beta_identical_and_zero_variance()
    test_compute_metrics_integration()
    print("test_metrics OK: drawdown, profit factor, Sharpe guards + annualization "
          "scaling, periods-from-curve, beta, full integration block.")
