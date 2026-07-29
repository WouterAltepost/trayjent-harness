"""Offline tests for the Rotor sleeve's data foundation.

Pure seams only — no network: the Alpaca bar parser (schema, close
stamping, boundary-drift rejection, partial-bar guard), keyless universe
discovery filtering (stablecoin exclusion, BTC/ETH guard) via a stubbed
HTTP layer, and the cross-check report (strict 24/7 no-gap rule, majors
depth flag). Synthetic Parquet in a tmp dir, same pattern as the other
sleeves' tests.

Runnable via pytest or directly:

    python tests/test_rotor.py
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
from data_layer import rotor, rotor_store


def _bar(t, c=100.0):
    return {"t": t, "o": c, "h": c, "l": c, "c": c,
            "v": 12.345678, "vw": c, "n": 42}


# ── Parser ──────────────────────────────────────────────────────────────
def test_parse_bars_schema_and_close_stamp():
    df = rotor_store.parse_bars(
        [_bar("2024-01-02T00:00:00Z", 100.0), _bar("2024-01-03T00:00:00Z", 101.0)],
        "BTC")
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close",
                                "volume", "vwap", "trades", "ticker", "timeframe"]
    # Close instant = key + 24h: the Jan-2 UTC bar becomes knowable Jan-3 00:00Z.
    assert str(df["timestamp"].iloc[0]) == "2024-01-03 00:00:00+00:00"
    assert rotor.bar_day(df["timestamp"].iloc[0]) == "2024-01-02"
    # Crypto volume stays fractional — never cast to int64.
    assert df["volume"].dtype == "float64"
    assert abs(df["volume"].iloc[0] - 12.345678) < 1e-12
    assert df["trades"].iloc[0] == 42


def test_parse_bars_rejects_payload_drift():
    try:
        rotor_store.parse_bars([{"t": "2024-01-02T00:00:00Z", "c": 1.0}], "BTC")
        assert False, "expected ValueError on missing bar fields"
    except ValueError as e:
        assert "missing field" in str(e)


def test_parse_bars_rejects_boundary_drift():
    # A daily bar keyed off-midnight means Alpaca changed its bar boundary —
    # the pinned convention would silently be wrong, so the parser must raise.
    try:
        rotor_store.parse_bars([_bar("2024-01-02T05:00:00Z")], "BTC")
        assert False, "expected ValueError on a non-00:00Z daily key"
    except ValueError as e:
        assert "boundary" in str(e).lower() or "00:00Z" in str(e)


def test_parse_bars_drops_forming_bar():
    today_key = pd.Timestamp.now(tz="UTC").normalize()   # today's bar: closes tomorrow
    done_key = today_key - pd.Timedelta(days=1)
    df = rotor_store.parse_bars(
        [_bar(done_key.isoformat()), _bar(today_key.isoformat())], "BTC")
    assert len(df) == 1                                  # forming bar never persists
    assert df["timestamp"].iloc[0] == done_key + pd.Timedelta(hours=24)


# ── Universe discovery (stubbed HTTP) ───────────────────────────────────
def _stub_get_json(pairs, stale=()):
    """latest/bars stub: fresh pairs keyed today, stale pairs keyed in 2023
    (the delisted-pair signature latest/bars really serves)."""
    fresh_key = pd.Timestamp.now(tz="UTC").normalize().isoformat()
    bars = {p: _bar(fresh_key) for p in pairs}
    bars.update({p: _bar("2023-06-23T00:00:00Z") for p in stale})
    return lambda url: {"bars": bars}


def test_fetch_universe_excludes_stablecoins_and_decided_exclusions():
    saved = rotor_store._get_json
    rotor_store._get_json = _stub_get_json(
        ["BTC/USD", "ETH/USD", "SOL/USD", "USDT/USD", "USDC/USD", "DAI/USD",
         "PAXG/USD"])
    try:
        # PAXG responds live but is excluded by decision (gold proxy).
        assert rotor_store.fetch_universe() == ["BTC", "ETH", "SOL"]
    finally:
        rotor_store._get_json = saved


def test_listing_windows_derives_segments():
    with _tmp_rotor_dir():
        _write("BTC", "2021-01-01", 200)
        # Delist->relist: two segments; a dead coin: one segment ending early.
        _write("SOL", "2021-01-01", 100,
               skip_days=[f"2021-02-{d:02d}" for d in range(10, 20)])
        _write("NEAR", "2021-01-01", 50)
        wins = rotor.listing_windows()
        assert len(wins["BTC"]) == 1
        assert len(wins["SOL"]) == 2
        assert rotor.bar_day(wins["SOL"][0][1]) == "2021-02-09"
        assert rotor.bar_day(wins["SOL"][1][0]) == "2021-02-20"
        assert rotor.bar_day(wins["NEAR"][0][1]) == "2021-02-19"


def test_fetch_universe_excludes_stale_delisted_pairs():
    # latest/bars answers for delisted pairs with a YEARS-old bar (verified
    # against the live endpoint) — responding must not mean tradable.
    saved = rotor_store._get_json
    rotor_store._get_json = _stub_get_json(
        ["BTC/USD", "ETH/USD"], stale=["ALGO/USD", "NEAR/USD"])
    try:
        assert rotor_store.fetch_universe() == ["BTC", "ETH"]
    finally:
        rotor_store._get_json = saved


def test_fetch_universe_fails_closed_without_majors():
    saved = rotor_store._get_json
    rotor_store._get_json = _stub_get_json(["SOL/USD", "DOGE/USD"])
    try:
        rotor_store.fetch_universe()
        assert False, "expected ValueError when discovery lacks BTC/ETH"
    except ValueError as e:
        assert "BTC/ETH" in str(e)
    finally:
        rotor_store._get_json = saved


# ── Cross-check, on synthetic Parquet ───────────────────────────────────
@contextlib.contextmanager
def _tmp_rotor_dir():
    saved = config.ROTOR_DATA_DIR
    with tempfile.TemporaryDirectory() as td:
        config.ROTOR_DATA_DIR = td
        try:
            yield td
        finally:
            config.ROTOR_DATA_DIR = saved


def _write(coin, first_day, n_days, skip_days=()):
    """Close-stamped daily bars covering n_days UTC days from first_day."""
    days = pd.date_range(first_day, periods=n_days, freq="D", tz="UTC")
    days = [d for d in days if d.strftime("%Y-%m-%d") not in skip_days]
    df = pd.DataFrame({
        "timestamp": [d + pd.Timedelta(hours=24) for d in days],
        "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
        "volume": 1.5, "vwap": 100.0, "trades": 10,
        "ticker": coin, "timeframe": "1d",
    })
    df.to_parquet(os.path.join(config.ROTOR_DATA_DIR, f"{coin}.parquet"),
                  engine="pyarrow", index=False)


def test_cross_check_clean_universe_passes():
    with _tmp_rotor_dir():
        _write("BTC", "2021-01-01", 1900)
        _write("ETH", "2021-01-01", 1900)
        _write("SOL", "2022-06-15", 900)
        report = rotor.cross_check()
        assert report["failures"] == []
        assert all(c["missing_days"] == 0 for c in report["coins"])
        assert report["majors"]["BTC"]["years"] > 4.0
        assert report["majors_flag"] is False


def test_cross_check_fails_on_single_day_hole():
    with _tmp_rotor_dir():
        _write("BTC", "2021-01-01", 1900)
        _write("ETH", "2021-01-01", 1900)
        # A 24/7 market: one absent Tuesday is a feed hole, never a holiday.
        _write("SOL", "2022-06-15", 900, skip_days=("2022-08-16",))
        report = rotor.cross_check()
        assert any("SOL" in f and "2022-08-16" in f and "hole" in f
                   for f in report["failures"])
        sol = next(c for c in report["coins"] if c["coin"] == "SOL")
        assert sol["missing_days"] == 1
        assert sol["hole_dates"] == ["2022-08-16"]
        assert sol["segments"] == 2
        assert report["venue_gaps"] == []


def test_cross_check_classifies_venue_gap_not_failure():
    with _tmp_rotor_dir():
        _write("BTC", "2021-01-01", 1900)
        _write("ETH", "2021-01-01", 1900)
        # A multi-day missing run is a delist->relist window (venue history):
        # reported as a venue gap, NOT a failure.
        _write("SOL", "2022-06-15", 900,
               skip_days=("2022-08-16", "2022-08-17", "2022-08-18"))
        report = rotor.cross_check()
        assert report["failures"] == []
        assert report["venue_gaps"] == [{
            "coin": "SOL", "gap_start": "2022-08-16",
            "gap_end": "2022-08-18", "days": 3,
        }]
        sol = next(c for c in report["coins"] if c["coin"] == "SOL")
        assert sol["missing_days"] == 3
        assert sol["segments"] == 2
        assert sol["hole_dates"] == []


def test_cross_check_flags_short_major_history():
    with _tmp_rotor_dir():
        _write("BTC", "2021-01-01", 1900)
        _write("ETH", "2024-01-01", 400)      # under the ~4y bar
        report = rotor.cross_check()
        assert report["majors_flag"] is True
        assert report["majors"]["ETH"]["years"] < 4.0
        # Short history is a FLAG for window design, not a data failure.
        assert report["failures"] == []


def test_cross_check_fails_on_missing_major():
    with _tmp_rotor_dir():
        _write("BTC", "2021-01-01", 1900)
        report = rotor.cross_check()
        assert any("ETH" in f and "missing" in f for f in report["failures"])
        assert report["majors_flag"] is True


if __name__ == "__main__":
    test_parse_bars_schema_and_close_stamp()
    test_parse_bars_rejects_payload_drift()
    test_parse_bars_rejects_boundary_drift()
    test_parse_bars_drops_forming_bar()
    test_fetch_universe_excludes_stablecoins_and_decided_exclusions()
    test_fetch_universe_excludes_stale_delisted_pairs()
    test_fetch_universe_fails_closed_without_majors()
    test_listing_windows_derives_segments()
    test_cross_check_clean_universe_passes()
    test_cross_check_fails_on_single_day_hole()
    test_cross_check_classifies_venue_gap_not_failure()
    test_cross_check_flags_short_major_history()
    test_cross_check_fails_on_missing_major()
    print("test_rotor OK: parse + drift/boundary rejection, forming-bar guard, "
          "universe filter incl. stale-delisted exclusion + majors guard, "
          "hole-vs-venue-gap classification, majors depth flag + "
          "missing-major failure.")
