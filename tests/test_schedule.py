"""Offline tests for the decision calendar + as_of generator (brief Step 3).

The schedule decides WHEN the runner ticks, so a lookahead or an off-by-one in
the warm-up gate would silently shift every decision. All synthetic: SPY daily /
1h / 30m Parquet in a tmp DATA_DIR — no network, no stored data.

    python tests/test_schedule.py
"""
import os
import sys
import tempfile

import pandas as pd

_HARNESS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HARNESS_ROOT not in sys.path:
    sys.path.insert(0, _HARNESS_ROOT)

import config
from data_layer import slice as sl
from runner.configs import build_run_config
from runner.schedule import decision_points


# ── Synthetic fixture ───────────────────────────────────────────────────
# Strictly increasing closes so "which bar" is unambiguous. Daily bars are
# close-stamped (21:00 UTC) like store writes them; intraday bars are
# START-stamped and contiguous, like yfinance.
_FIXTURE = {}


def _write(data_dir, timeframe, n, start, freq):
    ts = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    closes = [50.0 + i for i in range(n)]
    df = pd.DataFrame({
        "timestamp": ts, "open": closes, "high": closes, "low": closes,
        "close": closes, "volume": [1000 + i for i in range(n)],
        "ticker": "SPY", "timeframe": timeframe,
    })
    path = os.path.join(data_dir, timeframe, "SPY.parquet")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)
    return ts


def _setup():
    if _FIXTURE:
        config.DATA_DIR = _FIXTURE["dir"]
        return _FIXTURE
    data_dir = tempfile.mkdtemp(prefix="tbh_sched_")
    config.DATA_DIR = data_dir
    # 400 daily (close-stamped 21:00 UTC) and 400 contiguous 1h bars.
    d = _write(data_dir, "1d", 400, "2022-01-03 21:00", "B")
    h = _write(data_dir, "1h", 400, "2024-01-02 14:30", "h")
    # 40 contiguous 30m bars placed well after the 1h warm-up point (the 300th
    # 1h bar closes ~2024-01-15 02:30 UTC), so every 30m mark is 1h-warm — this
    # is the Pulse-30min split-timeframe case (indicators 1h, prices 30m).
    h30 = _write(data_dir, "30m", 40, "2024-01-15 14:30", "30min")
    _FIXTURE.update({"dir": data_dir, "d": d, "h": h, "h30": h30})
    return _FIXTURE


_WIDE = ("2000-01-01", "2035-01-01")  # window wider than any fixture span


# ── as_of marks are bar CLOSE instants, warm-gated, ascending ───────────
def test_steady_marks_are_daily_closes_after_warmup():
    f = _setup()
    d = f["d"]
    rc = build_run_config("steady", *_WIDE, mode="rules_only")
    pts = decision_points(rc)

    # Daily close instant == the stored 21:00 stamp (TF_DURATION 0, L14). The
    # k-th bar has k+1 bars closed at its close, so warm (>= MIN_ROWS=300)
    # starts at index 299 -> marks are indices 299..399 (101 of them).
    assert config.MIN_ROWS == 300
    assert len(pts) == 101
    assert pts[0] == d[299], "first mark = the 300th daily bar's close"
    assert pts[-1] == d[399]
    assert d[298] not in pts, "pre-warm decision must be skipped"
    assert pts == sorted(pts), "ascending"


# ── No-lookahead: a mark sees exactly its own decision bar, not the next ─
def test_hourly_mark_sees_its_own_bar_only():
    f = _setup()
    h = f["h"]
    rc = build_run_config("pulse_hourly", *_WIDE, mode="rules_only")
    pts = decision_points(rc)

    assert len(pts) == 101                       # same warm arithmetic as daily
    assert pts[0] == h[299] + pd.Timedelta(hours=1), "mark = bar close (start+1h)"

    # The decision bar's price is visible at its mark; the next bar is not.
    k = 350
    mark = h[k] + pd.Timedelta(hours=1)
    assert mark in pts
    assert sl.get_price_asof("SPY", "1h", mark) == 50.0 + k, \
        "mark sees its own decision bar's close"
    assert mark < h[k + 1] + pd.Timedelta(hours=1), "strictly before the next bar's close"


# ── Split timeframe: 30m calendar, 1h warm gate (Pulse-30min) ───────────
def test_pulse_30min_split_timeframe_all_warm():
    f = _setup()
    h30 = f["h30"]
    rc = build_run_config("pulse_30min", *_WIDE, mode="rules_only")
    pts = decision_points(rc)

    # decision_tf=30m, indicator_tf=1h. The 30m bars sit past the 1h warm-up,
    # so all 40 fire; marks are the 30m close instants (start + 30m).
    assert rc.indicator_tf == "1h" and rc.decision_tf == "30m"
    assert len(pts) == 40
    assert pts[0] == h30[0] + pd.Timedelta(minutes=30)
    assert pts[-1] == h30[-1] + pd.Timedelta(minutes=30)


# ── Cadence counts differ per preset over the same data ─────────────────
def test_cadence_counts_per_preset():
    _setup()
    counts = {
        name: len(decision_points(build_run_config(name, *_WIDE, mode="rules_only")))
        for name in ("steady", "pulse_hourly", "pulse_30min")
    }
    assert counts == {"steady": 101, "pulse_hourly": 101, "pulse_30min": 40}


# ── Range filter is inclusive, end-of-day coerced ───────────────────────
def test_range_inclusive_end_of_day():
    f = _setup()
    d = f["d"]
    # Bare-date bounds spanning warm indices 300..350; daily decisions close at
    # 21:00 UTC, so end-of-day coercion must include index 350's 21:00 mark.
    start_date = d[300].strftime("%Y-%m-%d")
    end_date = d[350].strftime("%Y-%m-%d")
    rc = build_run_config("steady", start_date, end_date, mode="rules_only")
    pts = decision_points(rc)

    assert d[300] in pts, "start day's close included"
    assert d[350] in pts, "end day's 21:00 close included (end coerced to day end)"
    assert d[299] not in pts and d[351] not in pts, "bounds are exclusive beyond the range"
    assert len(pts) == 51


if __name__ == "__main__":
    test_steady_marks_are_daily_closes_after_warmup()
    test_hourly_mark_sees_its_own_bar_only()
    test_pulse_30min_split_timeframe_all_warm()
    test_cadence_counts_per_preset()
    test_range_inclusive_end_of_day()
    print("test_schedule OK: daily-close marks + warm-up, hourly no-lookahead, "
          "30m split-tf gate, cadence counts, inclusive range.")
