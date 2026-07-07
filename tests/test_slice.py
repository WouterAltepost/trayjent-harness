"""Offline regression tests for the no-lookahead slicer (brief Step 6).

The slicer is the one module that can silently fake a profitable backtest via
lookahead, so it does not land untested. Everything here runs on a synthetic
Parquet built in a tmp dir — no network, no stored data required.

Runnable via pytest or directly:

    python tests/test_slice.py
"""
import os
import sys
import tempfile

import pandas as pd

# Make `config` and the `data_layer` package importable when run from anywhere.
_HARNESS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HARNESS_ROOT not in sys.path:
    sys.path.insert(0, _HARNESS_ROOT)

import config
from data_layer import slice as sl


# ── Synthetic fixture ───────────────────────────────────────────────────
# Built once, into a tmp DATA_DIR. Closes are strictly increasing so
# oldest->newest ordering and "which bar is last" are unambiguous.
_FIXTURE = {}


def _write_bars(data_dir, ticker, timeframe, closes, start, freq, volumes=None):
    ts = pd.date_range(start, periods=len(closes), freq=freq, tz="UTC")
    if volumes is None:
        volumes = [1000 + i for i in range(len(closes))]
    df = pd.DataFrame({
        "timestamp": ts,
        # high/low straddle close by ±1.0 so an OHLC column mix-up in a reader
        # is detectable (degenerate high==low==close would hide it). Nothing
        # asserts on `open`.
        "open": closes,
        "high": [c + 1.0 for c in closes],
        "low": [c - 1.0 for c in closes],
        "close": closes,
        "volume": volumes, "ticker": ticker, "timeframe": timeframe,
    })
    path = os.path.join(data_dir, timeframe, f"{ticker}.parquet")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)
    return df


def _setup():
    """Idempotently build the fixture and point config.DATA_DIR at it."""
    if _FIXTURE:
        config.DATA_DIR = _FIXTURE["dir"]
        return _FIXTURE

    data_dir = tempfile.mkdtemp(prefix="tbh_slice_")
    config.DATA_DIR = data_dir

    # SPY daily: 700 bars, well above STEADY_BARS(300) and MIN_ROWS(300).
    spy_d = _write_bars(data_dir, "SPY", "1d",
                        [100.0 + i for i in range(700)], "2022-01-03 21:00", "B")
    # SPY hourly: 900 bars, well above PULSE_BARS(440). Contiguous (freq "h")
    # and START-stamped, so close_time = start + 1h — the no-lookahead surface.
    spy_h = _write_bars(data_dir, "SPY", "1h",
                        [50.0 + i for i in range(900)], "2024-01-02 14:30", "h")
    # SPY 30-min: a short contiguous series — get_price_asof has no MIN_ROWS
    # floor, so 12 bars is enough to pin its close-time + no-lookahead rule.
    spy_30 = _write_bars(data_dir, "SPY", "30m",
                         [70.0 + i for i in range(12)], "2024-01-02 14:30", "30min")

    # ^VIX daily: 30 bars, mostly 12.0, with boundary closes planted at known
    # indices so the regime ladder can be pinned. Indices >= 20 also exercise
    # the 20-bar MA (>=20 closes -> not None).
    vix_closes = [12.0] * 30
    vix_closes[20] = 14.99   # low
    vix_closes[21] = 15.00   # normal
    vix_closes[22] = 19.99   # normal
    vix_closes[23] = 20.00   # elevated
    vix_closes[24] = 29.99   # elevated
    vix_closes[25] = 30.00   # stressed
    vix = _write_bars(data_dir, "^VIX", "1d", vix_closes, "2024-01-01 21:00", "B",
                      volumes=[0] * 30)

    _FIXTURE.update({"dir": data_dir, "spy_d": spy_d, "spy_h": spy_h,
                     "spy_30": spy_30, "vix": vix})
    return _FIXTURE


# ── Lookahead boundary (the one that matters most) ──────────────────────
def test_lookahead_boundary_both_sides():
    f = _setup()
    df = f["spy_d"]
    k = 400  # well above the 300 floor

    # A bar stamped exactly at as_of IS included (it is the last bar).
    at = sl.get_window("SPY", "1d", df["timestamp"].iloc[k], n_bars=300)
    assert at["close_prices"][-1] == float(df["close"].iloc[k]), "bar at as_of must be included"

    # A bar one unit (1s) after as_of is EXCLUDED: querying just before bar k's
    # stamp must end on bar k-1, never reach into the future bar k.
    before = sl.get_window("SPY", "1d", df["timestamp"].iloc[k] - pd.Timedelta(seconds=1), n_bars=300)
    assert before["close_prices"][-1] == float(df["close"].iloc[k - 1]), \
        "a bar stamped after as_of must be excluded"


# ── Intraday close-time no-lookahead (PF-2) ─────────────────────────────
# yfinance START-stamps intraday bars and consecutive bars are contiguous, so a
# bar whose START <= as_of may still be FORMING. Eligibility must test the bar's
# CLOSE time (start + tf_duration); the forming bar is never included.
def test_intraday_close_time_no_lookahead():
    f = _setup()
    df = f["spy_h"]
    k = 500  # well above the 300 floor
    start_k = df["timestamp"].iloc[k]

    # as_of exactly at bar k's START: bar k spans [start, start+1h) and has NOT
    # closed -> its FINAL close must be excluded; window ends on bar k-1.
    at_start = sl.get_window("SPY", "1h", start_k, n_bars=300)
    assert at_start["close_prices"][-1] == float(df["close"].iloc[k - 1]), \
        "a still-forming intraday bar (start <= as_of < close) must be excluded"

    # as_of at bar k's CLOSE (start + 1h): now bar k is complete and included.
    at_close = sl.get_window("SPY", "1h", start_k + pd.Timedelta(hours=1), n_bars=300)
    assert at_close["close_prices"][-1] == float(df["close"].iloc[k]), \
        "an intraday bar is included once its close time <= as_of"


# ── get_price_asof: latest closed price, no floor, no lookahead ──────────
def test_get_price_asof_close_time_and_no_floor():
    f = _setup()
    df = f["spy_30"]            # only 12 bars — below MIN_ROWS, on purpose
    k = 5
    start_k = df["timestamp"].iloc[k]

    # No MIN_ROWS floor: returns even with a handful of bars.
    # At bar k's start, bar k is forming -> latest CLOSED 30m bar is k-1.
    assert sl.get_price_asof("SPY", "30m", start_k) == float(df["close"].iloc[k - 1]), \
        "price is the latest bar whose close time <= as_of (forming bar excluded)"
    # At bar k's close (start + 30m), bar k is the latest closed price.
    assert sl.get_price_asof("SPY", "30m", start_k + pd.Timedelta(minutes=30)) \
        == float(df["close"].iloc[k]), "bar included once its close time <= as_of"

    # Before the first bar has closed -> nothing eligible -> raise.
    first_start = df["timestamp"].iloc[0]
    try:
        sl.get_price_asof("SPY", "30m", first_start)  # first bar still forming
        assert False, "expected ValueError when no 30m bar has closed yet"
    except ValueError:
        pass


# ── Window length, recency, ordering ────────────────────────────────────
def test_window_length_recency_and_order():
    f = _setup()
    df = f["spy_h"]              # 900 bars, eligible well above n_bars
    n = config.PULSE_BARS       # 440
    # as_of at the last hourly bar's CLOSE (start + 1h) so that bar is eligible;
    # at its raw start it would still be forming and correctly excluded.
    last_close = df["timestamp"].iloc[-1] + pd.Timedelta(hours=1)
    w = sl.get_window("SPY", "1h", last_close, n_bars=n)

    assert len(w["close_prices"]) == n, "exact n_bars when eligible exceeds n_bars"
    # Most recent n_bars, oldest->newest.
    expected = [float(c) for c in df["close"].iloc[-n:].tolist()]
    assert w["close_prices"] == expected, "must be the most recent n_bars, oldest->newest"
    assert w["close_prices"] == sorted(w["close_prices"]), "ordered oldest->newest"
    assert all(isinstance(v, int) for v in w["volumes"]) and isinstance(w["close_prices"][0], float)


# ── Fail-closed floor ───────────────────────────────────────────────────
def test_floor_below_min_rows_raises():
    f = _setup()
    df = f["spy_d"]
    # as_of at index 250 -> 251 eligible bars, below MIN_ROWS(300).
    early = df["timestamp"].iloc[250]
    try:
        sl.get_window("SPY", "1d", early, n_bars=300)
        assert False, "expected ValueError below MIN_ROWS"
    except ValueError:
        pass


# ── Default window resolves per timeframe (L7) ──────────────────────────
def test_default_window_per_timeframe():
    f = _setup()
    last_d = f["spy_d"]["timestamp"].iloc[-1]              # daily: close-stamped
    last_h = f["spy_h"]["timestamp"].iloc[-1] + pd.Timedelta(hours=1)  # 1h close
    assert len(sl.get_window("SPY", "1d", last_d)["close_prices"]) == config.STEADY_BARS
    assert len(sl.get_window("SPY", "1h", last_h)["close_prices"]) == config.PULSE_BARS


# ── get_ohlc_window: aligned OHLC for the ATR (Steady redesign Step 1) ──
def test_ohlc_window_aligned_and_agrees_with_get_window():
    f = _setup()
    df = f["spy_h"]
    last_close = df["timestamp"].iloc[-1] + pd.Timedelta(hours=1)
    n = 15
    w = sl.get_ohlc_window("SPY", "1h", last_close, n_bars=n)
    assert w["ticker"] == "SPY"
    assert len(w["highs"]) == len(w["lows"]) == len(w["closes"]) == n
    assert w["closes"][-1] == float(df["close"].iloc[-1]), "ends at the decision bar"
    # Fixture straddle: high = close + 1, low = close - 1. A reader that
    # grabbed the wrong column collapses one of these.
    assert w["highs"] == [c + 1.0 for c in w["closes"]]
    assert w["lows"] == [c - 1.0 for c in w["closes"]]
    # Cross-check: the two slicers agree on identical args.
    gw = sl.get_window("SPY", "1h", last_close, n_bars=n)
    assert w["closes"] == gw["close_prices"], "ohlc closes must match get_window's"


def test_ohlc_window_no_lookahead():
    f = _setup()
    df = f["spy_h"]
    k = 500
    start_k = df["timestamp"].iloc[k]
    # At bar k's START it is still forming -> the window ends on bar k-1.
    at_start = sl.get_ohlc_window("SPY", "1h", start_k, n_bars=15)
    assert at_start["closes"][-1] == float(df["close"].iloc[k - 1]), \
        "a still-forming bar must be excluded from the OHLC window"
    assert at_start["highs"][-1] == float(df["high"].iloc[k - 1])
    # At bar k's CLOSE (start + 1h) it is complete and included.
    at_close = sl.get_ohlc_window("SPY", "1h", start_k + pd.Timedelta(hours=1), n_bars=15)
    assert at_close["closes"][-1] == float(df["close"].iloc[k])


def test_ohlc_window_floor_is_n_bars_not_min_rows():
    f = _setup()
    df = f["spy_30"]            # 12 stored bars, far below MIN_ROWS(300)
    last_close = df["timestamp"].iloc[-1] + pd.Timedelta(minutes=30)
    # 12 eligible bars cannot fill a 15-bar ATR window -> fail closed.
    try:
        sl.get_ohlc_window("SPY", "30m", last_close, n_bars=15)
        assert False, "expected ValueError when eligible bars < n_bars"
    except ValueError:
        pass
    # But the floor is n_bars itself, not MIN_ROWS: 12 bars fill a 12-bar ask.
    w = sl.get_ohlc_window("SPY", "30m", last_close, n_bars=12)
    assert len(w["closes"]) == 12


# ── VIX as-of: latest close, 20MA None rule, regime ladder ──────────────
def test_vix_latest_close_and_20ma_none():
    f = _setup()
    df = f["vix"]
    # Fewer than 20 eligible closes -> vix_20ma is None.
    r6 = sl.get_vix_asof(df["timestamp"].iloc[5])     # 6 closes eligible
    assert r6["vix_20ma"] is None, "20MA must be None below 20 closes"
    assert r6["vix_close"] == 12.0

    # Most recent close at or before T, even when T falls between bars.
    between = df["timestamp"].iloc[25] + pd.Timedelta(hours=1)
    assert sl.get_vix_asof(between)["vix_close"] == 30.0
    # >=20 eligible closes -> 20MA computed (not None).
    assert sl.get_vix_asof(df["timestamp"].iloc[25])["vix_20ma"] is not None


def test_vix_regime_ladder_boundaries():
    f = _setup()
    df = f["vix"]
    # (planted index, expected regime) — pins the copied classifier to live's
    # <15 / <20 / <30 / 30+ ladder. Drift here means the harness regime diverges
    # from production scoring.
    cases = [
        (20, 14.99, "low"),
        (21, 15.00, "normal"),
        (22, 19.99, "normal"),
        (23, 20.00, "elevated"),
        (24, 29.99, "elevated"),
        (25, 30.00, "stressed"),
    ]
    for idx, expected_close, expected_regime in cases:
        r = sl.get_vix_asof(df["timestamp"].iloc[idx])
        assert r["vix_close"] == expected_close, f"idx {idx}: close {r['vix_close']} != {expected_close}"
        assert r["vix_regime"] == expected_regime, \
            f"close {expected_close} -> {r['vix_regime']}, expected {expected_regime}"


if __name__ == "__main__":
    test_lookahead_boundary_both_sides()
    test_intraday_close_time_no_lookahead()
    test_get_price_asof_close_time_and_no_floor()
    test_window_length_recency_and_order()
    test_floor_below_min_rows_raises()
    test_default_window_per_timeframe()
    test_ohlc_window_aligned_and_agrees_with_get_window()
    test_ohlc_window_no_lookahead()
    test_ohlc_window_floor_is_n_bars_not_min_rows()
    test_vix_latest_close_and_20ma_none()
    test_vix_regime_ladder_boundaries()
    print("test_slice OK: lookahead (daily + intraday close-time), get_price_asof, "
          "window length/recency, floor, default window, get_ohlc_window, "
          "VIX as-of + regime ladder.")
