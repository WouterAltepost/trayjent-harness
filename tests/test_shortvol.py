"""Offline tests for the shortvol sleeve's data foundation.

Covers the pure seams only — no network: the CBOE CSV parser (schema, close
stamping, format-drift failure), the partial-bar guard, the term-structure
join/ratio, and the cross-check report (gap vs SPY days, VIX3M positivity,
backwardation counts). Synthetic Parquet in a tmp dir, same pattern as
test_slice.py.

Runnable via pytest or directly:

    python tests/test_shortvol.py
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
from data_layer import shortvol, shortvol_store


# ── CBOE parser ─────────────────────────────────────────────────────────

_CBOE_PAYLOAD = (
    b"DATE,OPEN,HIGH,LOW,CLOSE\n"
    b"01/02/2024,14.20,14.50,13.90,14.10\n"
    b"01/03/2024,14.30,15.00,14.10,14.80\n"
)


def test_parse_cboe_csv_schema_and_stamp():
    df = shortvol_store.parse_cboe_csv(_CBOE_PAYLOAD, "VIX")
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close",
                                "volume", "ticker", "timeframe"]
    assert len(df) == 2
    # Daily close stamp: 21:00 UTC, tz-aware — same convention as store.py.
    assert str(df["timestamp"].iloc[0]) == "2024-01-02 21:00:00+00:00"
    assert df["close"].iloc[1] == 14.80
    assert (df["volume"] == 0).all()          # indices don't trade
    assert (df["ticker"] == "VIX").all()


def test_parse_cboe_csv_rejects_schema_drift():
    bad = b"Date,Px\n01/02/2024,14.10\n"
    try:
        shortvol_store.parse_cboe_csv(bad, "VIX")
        assert False, "expected ValueError on a changed CBOE header"
    except ValueError as e:
        assert "missing expected column" in str(e)


def test_finalize_drops_future_bars():
    future = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=2)
    df = pd.DataFrame({
        "timestamp": [pd.Timestamp("2024-01-02 21:00", tz="UTC"), future],
        "open": [1.0, 2.0], "high": [1.0, 2.0], "low": [1.0, 2.0],
        "close": [1.0, 2.0], "volume": [0, 0],
        "ticker": "TEST", "timeframe": "1d",
    })
    out = shortvol_store._finalize(df)
    assert len(out) == 1                      # the forming bar never persists
    assert out["close"].iloc[0] == 1.0


# ── Loader + cross-check, on synthetic Parquet ──────────────────────────

@contextlib.contextmanager
def _tmp_shortvol_dir():
    """Point config.SHORTVOL_DATA_DIR at a tmp dir for the test's duration."""
    saved = config.SHORTVOL_DATA_DIR
    with tempfile.TemporaryDirectory() as td:
        config.SHORTVOL_DATA_DIR = td
        try:
            yield td
        finally:
            config.SHORTVOL_DATA_DIR = saved


def _write(ticker, dates, closes):
    ts = pd.to_datetime(dates).tz_localize("UTC") + pd.Timedelta(hours=21)
    df = pd.DataFrame({
        "timestamp": ts,
        "open": closes, "high": closes, "low": closes, "close": closes,
        "volume": [0] * len(closes), "ticker": ticker, "timeframe": "1d",
    })
    df.to_parquet(os.path.join(config.SHORTVOL_DATA_DIR, f"{ticker}.parquet"),
                  engine="pyarrow", index=False)


_DAYS = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]


def _write_clean_universe(vix=(15.0, 16.0, 25.0, 18.0),
                          vix3m=(17.0, 17.5, 20.0, 19.0),
                          vix_days=_DAYS):
    _write("VIX", vix_days, list(vix)[: len(vix_days)])
    _write("VIX3M", _DAYS, list(vix3m))
    _write("SVXY", _DAYS, [50.0, 51.0, 45.0, 48.0])
    _write("VXX", _DAYS, [20.0, 19.5, 24.0, 22.0])
    _write("SPY", _DAYS, [470.0, 471.0, 465.0, 468.0])


def test_term_structure_join_and_ratio():
    with _tmp_shortvol_dir():
        # VIX has one extra early day VIX3M lacks — inner join must drop it.
        _write("VIX", ["2023-12-29"] + _DAYS, [14.0, 15.0, 16.0, 25.0, 18.0])
        _write("VIX3M", _DAYS, [17.0, 17.5, 20.0, 19.0])
        ts = shortvol.load_term_structure()
        assert len(ts) == 4
        assert list(ts.columns) == ["timestamp", "vix", "vix3m", "vix_ratio"]
        assert abs(ts["vix_ratio"].iloc[0] - 15.0 / 17.0) < 1e-12
        # Day 3 is the planted backwardation day (25 / 20 = 1.25).
        assert ts["vix_ratio"].iloc[2] == 1.25


def test_cross_check_clean_universe_passes():
    with _tmp_shortvol_dir():
        _write_clean_universe()
        report = shortvol.cross_check()
        assert report["failures"] == []
        assert report["warnings"] == []
        assert report["cboe_only_days"] == []
        assert all(s["missing_days"] == 0 for s in report["series"])
        assert report["term"]["inversion_days"] == 1      # the planted 25/20 day
        assert report["term"]["inversion_by_year"] == {2024: 1}
        assert report["term"]["vix3m_min"] == 17.0


def test_cross_check_flags_vix_gap_vs_spy_days():
    with _tmp_shortvol_dir():
        # VIX is missing 2024-01-04, an SPY trading day inside its own range.
        _write_clean_universe(vix=(15.0, 16.0, 18.0),
                              vix_days=["2024-01-02", "2024-01-03", "2024-01-05"])
        report = shortvol.cross_check()
        assert any("VIX:" in f and "2024-01-04" in f for f in report["failures"])
        vix_row = next(s for s in report["series"] if s["ticker"] == "VIX")
        assert vix_row["missing_days"] == 1
        assert vix_row["known_gaps"] == 0

        # Allowlisting the date turns the failure into a documented known gap.
        saved = config.CBOE_KNOWN_MISSING_DAYS
        config.CBOE_KNOWN_MISSING_DAYS = saved | {"2024-01-04"}
        try:
            report = shortvol.cross_check()
            assert report["failures"] == []
            vix_row = next(s for s in report["series"] if s["ticker"] == "VIX")
            assert vix_row["known_gaps"] == 1
        finally:
            config.CBOE_KNOWN_MISSING_DAYS = saved


def test_cross_check_warns_on_stale_equity_tail():
    with _tmp_shortvol_dir():
        # SPY's last Yahoo bar is 2024-01-04 but the CBOE calendar runs
        # through 01-05 — the "Yahoo hasn't materialized yesterday" case. A
        # SPY-only calendar can't see its own tail hole; the staleness check
        # must warn (not fail — a re-pull upsert-fills it).
        _write("VIX", _DAYS, [15.0, 16.0, 25.0, 18.0])
        _write("VIX3M", _DAYS, [17.0, 17.5, 20.0, 19.0])
        _write("SVXY", _DAYS, [50.0, 51.0, 45.0, 48.0])
        _write("VXX", _DAYS, [20.0, 19.5, 24.0, 22.0])
        _write("SPY", _DAYS[:3], [470.0, 471.0, 465.0])
        report = shortvol.cross_check()
        assert report["failures"] == []
        assert any("SPY:" in w and "2024-01-05" in w and "trails" in w
                   for w in report["warnings"])
        assert not any(w.startswith(("SVXY:", "VXX:")) for w in report["warnings"])


def test_cross_check_warns_on_etp_hole_vs_spy():
    with _tmp_shortvol_dir():
        # SVXY lacks 2024-01-04, a day SPY traded — an interior Yahoo hole.
        _write("VIX", _DAYS, [15.0, 16.0, 25.0, 18.0])
        _write("VIX3M", _DAYS, [17.0, 17.5, 20.0, 19.0])
        _write("SVXY", ["2024-01-02", "2024-01-03", "2024-01-05"],
               [50.0, 51.0, 48.0])
        _write("VXX", _DAYS, [20.0, 19.5, 24.0, 22.0])
        _write("SPY", _DAYS, [470.0, 471.0, 465.0, 468.0])
        report = shortvol.cross_check()
        assert report["failures"] == []
        assert any(w.startswith("SVXY:") and "2024-01-04" in w
                   for w in report["warnings"])


def test_cross_check_reports_cboe_only_days():
    with _tmp_shortvol_dir():
        # The CBOE file has 2024-01-04 but SPY doesn't trade it (the
        # holiday-row pattern CBOE introduced in 2022). Informational only:
        # no failure, no warning, listed under cboe_only_days.
        _write("VIX", _DAYS, [15.0, 16.0, 25.0, 18.0])
        _write("VIX3M", _DAYS, [17.0, 17.5, 20.0, 19.0])
        _write("SVXY", ["2024-01-02", "2024-01-03", "2024-01-05"],
               [50.0, 51.0, 48.0])
        _write("VXX", ["2024-01-02", "2024-01-03", "2024-01-05"],
               [20.0, 19.5, 22.0])
        _write("SPY", ["2024-01-02", "2024-01-03", "2024-01-05"],
               [470.0, 471.0, 468.0])
        report = shortvol.cross_check()
        assert report["failures"] == []
        assert report["warnings"] == []
        assert report["cboe_only_days"] == ["2024-01-04"]


def test_cross_check_flags_nonpositive_vix3m():
    with _tmp_shortvol_dir():
        _write_clean_universe(vix3m=(17.0, -1.0, 20.0, 19.0))
        report = shortvol.cross_check()
        assert any("non-positive" in f for f in report["failures"])


if __name__ == "__main__":
    test_parse_cboe_csv_schema_and_stamp()
    test_parse_cboe_csv_rejects_schema_drift()
    test_finalize_drops_future_bars()
    test_term_structure_join_and_ratio()
    test_cross_check_clean_universe_passes()
    test_cross_check_flags_vix_gap_vs_spy_days()
    test_cross_check_warns_on_stale_equity_tail()
    test_cross_check_warns_on_etp_hole_vs_spy()
    test_cross_check_reports_cboe_only_days()
    test_cross_check_flags_nonpositive_vix3m()
    print("test_shortvol OK: CBOE parse + drift rejection, partial-bar guard, "
          "term-structure join/ratio, cross-check pass + gap allowlist + "
          "staleness/hole warnings + CBOE-only days + positivity flags.")
