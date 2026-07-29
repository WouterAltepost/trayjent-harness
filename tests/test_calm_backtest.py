"""Offline tests for the Calm (shortvol) backtest engine.

Everything is synthetic and deterministic — no network, no stored data. The
engine arithmetic tests hand-compute the expected equity step by step so a
silent change to cost timing, sizing cadence, or fill placement fails loudly.

Runnable via pytest or directly:

    python tests/test_calm_backtest.py
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
from calm import backtest as bt

_C = 5.0 / 1e4  # 5 bps per side, as a fraction


# ── Synthetic engine frame (no files) ───────────────────────────────────
def _frame(rows):
    """rows: list of (date, ret_cc, ret_on, ret_id, ratio, vix)."""
    ts = pd.to_datetime([r[0] for r in rows]).tz_localize("UTC") + pd.Timedelta(hours=21)
    ratio = [r[4] for r in rows]
    return pd.DataFrame({
        "timestamp": ts,
        "ret_cc": [r[1] for r in rows],
        "ret_on": [r[2] for r in rows],
        "ret_id": [r[3] for r in rows],
        "ratio_raw": ratio,
        "ratio_3d": ratio,   # engine tests drive smoothing="raw"
        "vix": [r[5] for r in rows],
        "spy_ret": [0.0] * len(rows),
    })


# ── State machine + sizing ──────────────────────────────────────────────
def test_hysteresis_state_machine():
    seq = [
        (bt.FLAT, 0.97, bt.FLAT, False),   # between bands from FLAT: no entry
        (bt.FLAT, 0.94, bt.LONG, False),   # < entry: enter
        (bt.LONG, 0.97, bt.LONG, False),   # between bands from LONG: hold
        (bt.LONG, 1.00, bt.LONG, False),   # ratio == exit is NOT > exit: hold
        (bt.LONG, 1.01, bt.FLAT, True),    # > exit: exit, one round trip
        (bt.FLAT, 0.95, bt.FLAT, False),   # ratio == entry is NOT < entry
    ]
    for state, ratio, want_state, want_trip in seq:
        got_state, got_trip = bt.next_state(state, ratio, entry=0.95, exit_=1.00)
        assert (got_state, got_trip) == (want_state, want_trip), (state, ratio)


def test_sizing():
    assert bt._target_weight(bt.FLAT, 20, 12.0) == 0.0
    assert bt._target_weight(bt.LONG, None, 80.0) == 1.0     # K=None: full size
    assert bt._target_weight(bt.LONG, 20, 25.0) == 0.8       # K/VIX
    assert bt._target_weight(bt.LONG, 20, 10.0) == 1.0       # capped at 1


# ── Engine arithmetic, close mode ───────────────────────────────────────
def test_close_mode_equity_hand_computed():
    df = _frame([
        ("2024-01-02", 0.00, 0.0, 0.0, 0.94, 10.0),   # enter at close, w->1.0
        ("2024-01-03", 0.02, 0.0, 0.0, 0.97, 25.0),   # +2% at w=1; resize to 0.8
        ("2024-01-04", -0.01, 0.0, 0.0, 1.01, 25.0),  # -1% at w=0.8; exit
    ])
    res = bt.run_backtest(df, entry=0.95, exit_=1.00, k=20, smoothing="raw",
                          mode="close", cost_bps=5.0)
    e0 = 1.0 * (1 - _C * 1.0)                 # day 1: entry fill 0 -> 1.0
    e1 = e0 * 1.02 * (1 - _C * 0.2)           # day 2: +2% at w=1, resize 1->0.8
    e2 = e1 * (1 - 0.8 * 0.01) * (1 - _C * 0.8)   # day 3: -1% at 0.8, exit 0.8->0
    for got, want in zip(res["equity"], [e0, e1, e2]):
        assert abs(got - want) < 1e-12
    assert res["weights"] == [0.0, 1.0, 0.8]  # exposure held INTO each close
    assert res["round_trips"] == 1
    assert res["final_state"] == bt.FLAT


def test_next_open_mode_equity_hand_computed():
    df = _frame([
        ("2024-01-02", 0.0, 0.00, 0.00, 0.90, 15.0),  # signal long, fills tomorrow
        ("2024-01-03", 0.0, 0.01, 0.02, 1.10, 15.0),  # fill at open; exit signal
        ("2024-01-04", 0.0, -0.03, 0.05, 1.10, 15.0), # overnight loss at OLD w; flat at open
    ])
    res = bt.run_backtest(df, entry=0.95, exit_=1.00, k=None, smoothing="raw",
                          mode="next_open", cost_bps=5.0)
    e0 = 1.0                                   # day 1: no fill yet
    e1 = 1.0 * (1 - _C) * 1.02                 # day 2: fill at open, intraday at w=1
    e2 = e1 * (1 - 0.03) * (1 - _C)            # day 3: overnight at w=1, exit at open
    for got, want in zip(res["equity"], [e0, e1, e2]):
        assert abs(got - want) < 1e-12, (got, want)
    assert res["weights"] == [0.0, 1.0, 0.0]
    assert res["round_trips"] == 1


def test_determinism():
    df = _frame([
        ("2024-01-02", 0.00, 0.0, 0.0, 0.92, 18.0),
        ("2024-01-03", 0.015, 0.0, 0.0, 0.96, 22.0),
        ("2024-01-04", -0.02, 0.0, 0.0, 1.02, 30.0),
        ("2024-01-05", 0.01, 0.0, 0.0, 0.93, 19.0),
    ])
    a = bt.run_backtest(df, 0.95, 1.00, 20)
    b = bt.run_backtest(df, 0.95, 1.00, 20)
    assert a["equity"] == b["equity"]
    assert a["weights"] == b["weights"]


# ── Metrics ─────────────────────────────────────────────────────────────
def test_cell_metrics_definitions():
    # Mon 2024-01-01 .. Mon 2024-01-08: two W-FRI weeks, one month.
    days = ["2024-01-01", "2024-01-02", "2024-01-03",
            "2024-01-04", "2024-01-05", "2024-01-08"]
    ts = list(pd.to_datetime(days).tz_localize("UTC") + pd.Timedelta(hours=21))
    rets = [0.0, 0.01, -0.02, 0.0, 0.03, 0.01]
    eq, curve = 1.0, []
    for r in rets:
        eq *= 1.0 + r
        curve.append(eq)
    res = {"equity": curve, "day_returns": rets,
           "weights": [0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
           "round_trips": 2, "final_state": bt.FLAT}
    m = bt.compute_cell_metrics(res, ts)
    assert m["daily_win"] == 3 / 4            # exposed days 1,2,4,5; wins 1,4,5
    assert m["time_in_market"] == 4 / 6
    assert m["worst_day"] == -0.02
    assert m["round_trips"] == 2
    # Week 1 (01-01..01-05): (1.01*0.98*1.03)-1 > 0; week 2 (01-08): +1% > 0.
    assert m["pos_weeks"] == 1.0
    # One month, positive overall, had exposure.
    assert m["pos_months"] == 1.0
    assert abs(m["max_drawdown"] - 0.02) < 1e-12   # the single -2% day


def test_episode_stats_slices_by_date():
    days = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    ts = list(pd.to_datetime(days).tz_localize("UTC") + pd.Timedelta(hours=21))
    res = {"day_returns": [0.10, -0.10, 0.05, 0.20]}
    ep = bt.episode_stats(res, ts, "2024-01-03", "2024-01-04")
    assert ep["days"] == 2
    assert abs(ep["pnl"] - (0.9 * 1.05 - 1.0)) < 1e-12
    assert abs(ep["max_drawdown"] - 0.10) < 1e-12  # from the base, day one -10%


# ── Robust selection ────────────────────────────────────────────────────
def test_select_robust_prefers_plateau_over_spike():
    # Build all 64 cells with crafted Sharpes: one isolated spike in the
    # (raw, 1.00) plane, a uniform plateau in the (3d, 1.05) plane. The
    # plateau must win despite the spike's higher peak.
    cells = []
    for smoothing in bt.SMOOTHING_GRID:
        for exit_ in bt.EXIT_GRID:
            for entry in bt.ENTRY_GRID:
                for k in bt.K_GRID:
                    if smoothing == "raw" and exit_ == 1.00:
                        s = 5.0 if (entry == 0.925 and k == 20) else 0.0
                    elif smoothing == "3d" and exit_ == 1.05:
                        s = 1.0
                    else:
                        s = 0.2
                    cells.append({"entry": entry, "exit": exit_, "k": k,
                                  "smoothing": smoothing,
                                  "metrics": {"sharpe": s}})
    chosen = bt.select_robust(cells)
    assert chosen["smoothing"] == "3d" and chosen["exit"] == 1.05
    assert chosen["nbhd_std"] == 0.0 and chosen["nbhd_mean"] == 1.0
    # Interior-only rule: the chosen cell must have a FULL 3x3 neighborhood —
    # corner/edge cells (clipped neighborhoods) are never eligible.
    assert chosen["nbhd_n"] == 9
    assert chosen["entry"] in bt.ENTRY_GRID[1:-1]
    assert chosen["k"] in bt.K_GRID[1:-1]


# ── build_inputs: sealed guard + assembly on synthetic Parquet ──────────
def test_sealed_window_guard_fails_closed():
    # The guard fires BEFORE any data is read, so no fixture is needed.
    try:
        bt.build_inputs("2011-10-04", "2022-06-01")
        assert False, "expected ValueError on a window touching the sealed OOS"
    except ValueError as e:
        assert "sealed" in str(e).lower()
    # Boundary: the sealed start itself is sealed.
    try:
        bt.build_inputs("2011-10-04", config.OOS_SHORTVOL[0])
        assert False, "expected ValueError on end == sealed start"
    except ValueError as e:
        assert "sealed" in str(e).lower()


@contextlib.contextmanager
def _tmp_shortvol_dir():
    saved = config.SHORTVOL_DATA_DIR
    with tempfile.TemporaryDirectory() as td:
        config.SHORTVOL_DATA_DIR = td
        try:
            yield td
        finally:
            config.SHORTVOL_DATA_DIR = saved


def _write(ticker, dates, closes, opens=None):
    ts = pd.to_datetime(dates).tz_localize("UTC") + pd.Timedelta(hours=21)
    df = pd.DataFrame({
        "timestamp": ts,
        "open": opens if opens is not None else closes,
        "high": closes, "low": closes, "close": closes,
        "volume": [0] * len(closes), "ticker": ticker, "timeframe": "1d",
    })
    df.to_parquet(os.path.join(config.SHORTVOL_DATA_DIR, f"{ticker}.parquet"),
                  engine="pyarrow", index=False)


def test_build_inputs_leverage_scaling_and_smoothing_seed():
    # Days straddling the real flip date 2018-02-27. SVXY +10% every day:
    # pre-flip days must scale to +5%, the flip date itself and later stay +10%.
    days = ["2018-02-21", "2018-02-22", "2018-02-23",
            "2018-02-26", "2018-02-27", "2018-02-28"]
    with _tmp_shortvol_dir():
        closes = [100.0 * 1.1 ** i for i in range(len(days))]
        _write("SVXY", days, closes, opens=[c / 1.05 for c in closes])
        _write("SPY", days, [400.0 + i for i in range(len(days))])
        # VIX3M fixed at 20; VIX chosen so the ratio walks 0.90..0.95.
        ratios = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95]
        _write("VIX", days, [r * 20.0 for r in ratios])
        _write("VIX3M", days, [20.0] * len(days))

        win = bt.build_inputs("2018-02-26", "2018-02-28")
        assert len(win) == 3
        got = win["ret_cc"].tolist()
        assert abs(got[0] - 0.05) < 1e-12    # 2018-02-26: pre-flip, scaled
        assert abs(got[1] - 0.10) < 1e-12    # 2018-02-27: flip date, unscaled
        assert abs(got[2] - 0.10) < 1e-12
        # 3-day mean ENDING at the window's first row (2018-02-26) is seeded
        # from pre-window history — no truncated seed, no NaN.
        assert abs(win["ratio_3d"].iloc[0] - (0.91 + 0.92 + 0.93) / 3) < 1e-12
        assert abs(win["ratio_3d"].iloc[-1] - (0.93 + 0.94 + 0.95) / 3) < 1e-12


def test_build_inputs_fails_closed_on_signal_hole():
    # Pre-2022 dates so the sealed guard (correctly) does not fire first.
    days = ["2021-06-01", "2021-06-02", "2021-06-03"]
    with _tmp_shortvol_dir():
        _write("SVXY", days, [100.0, 101.0, 102.0])
        _write("SPY", days, [400.0, 401.0, 402.0])
        _write("VIX", days, [18.0, 18.5, 19.0])
        _write("VIX3M", days[:2], [20.0, 20.0])   # missing the last day
        try:
            bt.build_inputs("2021-06-01", "2021-06-03")
            assert False, "expected ValueError on missing signal data"
        except ValueError as e:
            assert "signal" in str(e).lower() or "missing" in str(e).lower()


if __name__ == "__main__":
    test_hysteresis_state_machine()
    test_sizing()
    test_close_mode_equity_hand_computed()
    test_next_open_mode_equity_hand_computed()
    test_determinism()
    test_cell_metrics_definitions()
    test_episode_stats_slices_by_date()
    test_select_robust_prefers_plateau_over_spike()
    test_sealed_window_guard_fails_closed()
    test_build_inputs_leverage_scaling_and_smoothing_seed()
    test_build_inputs_fails_closed_on_signal_hole()
    print("test_calm_backtest OK: hysteresis, sizing, close/next-open arithmetic, "
          "determinism, metric definitions, episode slicing, plateau-over-spike "
          "selection, sealed guard, leverage scaling + smoothing seed, "
          "fail-closed on signal holes.")
