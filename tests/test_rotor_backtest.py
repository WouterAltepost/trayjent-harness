"""Offline tests for the Rotor backtest engine.

Synthetic Parquet stores in a tmp dir, engine constants patched small so the
arithmetic is hand-computable. The core mechanics each get a dedicated test:
sealed guard, no-cross-gap returns, weekly rebalance cost math, forced
delisting exits, the BTC gate, the per-coin MA filter (slots stay cash), the
90-day history rule, the degenerated plane-selection rule, determinism.

Runnable via pytest or directly:

    python tests/test_rotor_backtest.py
"""
import contextlib
import os
import sys
import tempfile

import pandas as pd

_HARNESS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HARNESS_ROOT not in sys.path:
    sys.path.insert(0, _HARNESS_ROOT)

import config
from data_layer import rotor as rotor_data
from rotor import backtest as bt

_COST = config.ROTOR_ALPACA_CRYPTO_FEES["taker"] + config.ROTOR_SLIPPAGE_BPS / 1e4


@contextlib.contextmanager
def _tmp_store(**patches):
    """Tmp ROTOR_DATA_DIR plus small engine constants for hand-math.

    Defaults shrink the liquidity/history/filter windows; individual tests
    override via kwargs (e.g. MIN_HISTORY_DAYS=6).
    """
    saved_dir = config.ROTOR_DATA_DIR
    consts = {"VOLUME_WINDOW": 3, "MIN_HISTORY_DAYS": 0, "COIN_MA": 3}
    consts.update(patches)
    saved = {k: getattr(bt, k) for k in consts}
    with tempfile.TemporaryDirectory() as td:
        config.ROTOR_DATA_DIR = td
        for k, v in consts.items():
            setattr(bt, k, v)
        try:
            yield td
        finally:
            config.ROTOR_DATA_DIR = saved_dir
            for k, v in saved.items():
                setattr(bt, k, v)


def _write(coin, first_day, closes, dvol=1000.0, skip_days=()):
    """Daily bars from first_day, one per close; vwap=1 so volume==dollar vol."""
    days = pd.date_range(first_day, periods=len(closes), freq="D", tz="UTC")
    rows = [(d, c) for d, c in zip(days, closes)
            if d.strftime("%Y-%m-%d") not in skip_days]
    df = pd.DataFrame({
        "timestamp": [d + pd.Timedelta(hours=24) for d, _ in rows],
        "open": [c for _, c in rows], "high": [c for _, c in rows],
        "low": [c for _, c in rows], "close": [c for _, c in rows],
        "volume": [float(dvol)] * len(rows), "vwap": [1.0] * len(rows),
        "trades": [10] * len(rows), "ticker": coin, "timeframe": "1d",
    })
    df.to_parquet(os.path.join(config.ROTOR_DATA_DIR, f"{coin}.parquet"),
                  engine="pyarrow", index=False)


def _geo(start, rate, n):
    return [start * (1.0 + rate) ** i for i in range(n)]


# 2021-01-01 is a Friday: bar days Jan1..Jan10 -> close stamps Jan2..Jan11;
# Monday close instants are Jan 4 (covers Sun Jan 3) and Jan 11.
_N = 10
_START, _END = "2021-01-01", "2021-01-10"


def test_sealed_guard_fails_closed():
    try:
        bt.build_inputs("2021-01-01", "2025-06-01")
        assert False, "expected ValueError on a window touching the sealed OOS"
    except ValueError as e:
        assert "sealed" in str(e).lower()
    try:
        bt.build_inputs("2021-01-01", config.OOS_ROTOR[0])
        assert False, "expected ValueError on end == sealed start"
    except ValueError as e:
        assert "sealed" in str(e).lower()


def test_no_returns_across_venue_gaps():
    with _tmp_store():
        _write("BTC", _START, _geo(100, 0.0, _N))
        # X has a 3-day gap: the relist bar's return must be NaN, and a
        # formation window spanning the gap must disqualify, not bridge.
        _write("X", _START, _geo(100, 0.10, _N),
               skip_days=("2021-01-05", "2021-01-06", "2021-01-07"))
        inputs = bt.build_inputs(_START, _END)
        ret = inputs["ret"]["X"]
        relist_stamp = pd.Timestamp("2021-01-09", tz="UTC")  # bar covering Jan 8
        assert pd.isna(ret.loc[relist_stamp])                # no cross-gap return
        assert not pd.isna(ret.loc[pd.Timestamp("2021-01-10", tz="UTC")])
        # X's listing map has two segments and the first ends Jan 4.
        assert len(inputs["segment_ends"]["X"]) == 2


def test_weekly_rebalance_cost_arithmetic():
    with _tmp_store():
        _write("BTC", _START, _geo(100, 0.0, _N), dvol=1.0)   # flat, low volume
        _write("X", _START, _geo(100, 0.10, _N), dvol=1000)   # strong momentum
        _write("Y", _START, _geo(100, 0.01, _N), dvol=1000)   # weak momentum
        # Window ends before the second Monday: exactly ONE rebalance, so
        # the turnover/cost assertions cover the entry pass alone. (Data
        # extends past the window end, so no coin ends a segment inside it.)
        inputs = bt.build_inputs(_START, "2021-01-09")
        res = bt.run_backtest(inputs, formation=2, n_held=2, gate_ma=None,
                              ma_filter=False)
        # Entry at the Jan-4 Monday close: two buys at (1-c)/2 each.
        target = (1.0 - _COST) / 2
        fees = 2 * target * _COST
        e0 = 1.0 - fees
        assert abs(res["equity"][0] - e0) < 1e-12
        assert res["rebalances"][0]["held"] == ["X", "Y"]     # formation order
        assert abs(res["traded_frac"] - 2 * target) < 1e-9    # vs pre-cost equity
        # Next day: X +10%, Y +1% on the held units, cash unchanged.
        cash0 = 1.0 - 2 * target - fees
        e1 = cash0 + target * 1.10 + target * 1.01
        assert abs(res["equity"][1] - e1) < 1e-12
        assert abs(res["day_returns"][0] - (e0 - 1.0)) < 1e-12  # entry cost day
        assert res["invested"][0] > 0.99


def test_forced_exit_at_delisting():
    with _tmp_store():
        _write("BTC", _START, _geo(100, 0.0, _N), dvol=1.0)
        _write("X", _START, _geo(100, 0.10, _N), dvol=1000)
        # Y delists after its Wednesday Jan 6 bar (close stamp Jan 7).
        _write("Y", _START, _geo(100, 0.01, 6), dvol=1000)
        inputs = bt.build_inputs(_START, "2021-01-09")
        res = bt.run_backtest(inputs, formation=2, n_held=2, gate_ma=None,
                              ma_filter=False)
        ts = res["timestamps"]
        end_stamp = pd.Timestamp("2021-01-07", tz="UTC")
        i = ts.index(end_stamp)
        # Exact post-forced-exit book: X rode +10%/day for 3 days, Y's
        # proceeds (3 days of +1%) landed in cash minus the exit fee.
        target = (1.0 - _COST) / 2
        cash0 = 1.0 - 2 * target - 2 * target * _COST
        x_val = target * 1.10 ** 3
        y_proceeds = target * 1.01 ** 3 * (1.0 - _COST)
        equity_i = cash0 + y_proceeds + x_val
        assert abs(res["equity"][i] - equity_i) < 1e-12
        assert abs(res["invested"][i] - x_val / equity_i) < 1e-12
        # Nothing NaN afterwards — the mark-to-market invariant held.
        assert all(e == e for e in res["equity"])             # no NaN
        # The forced sell paid fees: cost_frac exceeds the two entry legs.
        entry_cost_frac = 2 * ((1.0 - _COST) / 2) * _COST
        assert res["cost_frac"] > entry_cost_frac + 1e-6


def test_btc_gate_goes_to_cash_and_reenters():
    with _tmp_store():
        # BTC: above its 2d MA at the Jan-4 Monday (rising), below at Jan 11
        # (falling tail) -> invested week 1, gated (all cash) at week 2.
        btc = [100, 101, 102, 103, 104, 103, 101, 99, 96, 92]
        _write("BTC", _START, btc, dvol=1.0)
        _write("X", _START, _geo(100, 0.05, _N), dvol=1000)
        inputs = bt.build_inputs(_START, _END)
        res = bt.run_backtest(inputs, formation=2, n_held=1, gate_ma=2,
                              ma_filter=False)
        assert res["rebalances"][0]["gated"] is False
        assert res["rebalances"][1]["gated"] is True
        assert res["rebalances"][1]["held"] == []
        assert res["invested"][-1] == 0.0                     # sold everything
        # The gated sell paid a fee on the full position notional.
        assert res["cost_frac"] > 0


def test_ma_filter_slots_stay_cash():
    with _tmp_store():
        _write("BTC", _START, _geo(100, 0.0, _N), dvol=1.0)
        _write("X", _START, _geo(100, 0.10, _N), dvol=1000)   # above its 3d MA
        _write("Y", _START, _geo(100, -0.05, _N), dvol=1000)  # below its 3d MA
        inputs = bt.build_inputs(_START, _END)
        res = bt.run_backtest(inputs, formation=2, n_held=2, gate_ma=None,
                              ma_filter=True)
        # Y is selected by rank (top-2 of 3) but fails its own MA: the slot
        # stays cash, X still gets only 1/N of equity.
        assert res["rebalances"][0]["held"] == ["X"]
        assert 0.45 < res["invested"][0] < 0.55


def test_min_history_rule():
    with _tmp_store(MIN_HISTORY_DAYS=6):
        # 13 days of data so the Jan-11 Monday is NOT any coin's final bar
        # (the never-buy-at-segment-end rule would otherwise mask this test).
        _write("BTC", _START, _geo(100, 0.0, 13), dvol=1.0)
        _write("X", _START, _geo(100, 0.10, 13), dvol=1000)
        inputs = bt.build_inputs(_START, _END)
        res = bt.run_backtest(inputs, formation=2, n_held=1, gate_ma=None,
                              ma_filter=False)
        # Jan-4 Monday: X is 2 days old -> too young, nothing held.
        assert res["rebalances"][0]["held"] == []
        # Jan-11 Monday: 9 days old -> eligible.
        assert res["rebalances"][1]["held"] == ["X"]


def test_select_robust_picks_most_consistent_plane():
    cells = []
    for gate in bt.GATE_GRID:
        for filt in bt.FILTER_GRID:
            for i, formation in enumerate(bt.FORMATION_GRID):
                for j, n in enumerate(bt.N_GRID):
                    if (gate, filt) == (100, False):
                        s = 3.0 if (i, j) == (0, 0) else 0.0   # spiky plane
                    elif (gate, filt) == (200, True):
                        s = 1.0 + 0.01 * i                     # flat plane
                    else:
                        s = 0.5
                    cells.append({"formation": formation, "n_held": n,
                                  "gate_ma": gate, "ma_filter": filt,
                                  "metrics": {"sharpe": s}})
    chosen = bt.select_robust(cells)
    # The flat (200, True) plane wins on mean-std; within it, tie-breaks
    # land on the highest own Sharpe (formation index 1).
    assert (chosen["gate_ma"], chosen["ma_filter"]) == (200, True)
    assert chosen["formation"] == bt.FORMATION_GRID[1]
    assert chosen["nbhd_std"] < 0.01


def test_determinism():
    with _tmp_store():
        _write("BTC", _START, _geo(100, 0.0, _N), dvol=1.0)
        _write("X", _START, _geo(100, 0.10, _N), dvol=1000)
        _write("Y", _START, _geo(100, 0.01, _N), dvol=1000)
        inputs = bt.build_inputs(_START, _END)
        a = bt.run_backtest(inputs, 2, 2, None, False)
        b = bt.run_backtest(inputs, 2, 2, None, False)
        assert a["equity"] == b["equity"]
        assert a["rebalances"] == b["rebalances"]


if __name__ == "__main__":
    test_sealed_guard_fails_closed()
    test_no_returns_across_venue_gaps()
    test_weekly_rebalance_cost_arithmetic()
    test_forced_exit_at_delisting()
    test_btc_gate_goes_to_cash_and_reenters()
    test_ma_filter_slots_stay_cash()
    test_min_history_rule()
    test_select_robust_picks_most_consistent_plane()
    test_determinism()
    print("test_rotor_backtest OK: sealed guard, no-cross-gap returns, cost "
          "arithmetic, forced delisting exit, BTC gate, MA-filter cash slots, "
          "history rule, plane selection, determinism.")
