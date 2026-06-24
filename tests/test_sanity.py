"""Offline tests for the ingest sanity check (brief Step 6).

Planted bad bars must be caught: a duplicate timestamp and a non-monotonic
timestamp hard-fail (they corrupt the slicer); an 80% single-bar jump produces
a warning (possible unadjusted split / bad data). No network, no stored data.

Runnable via pytest or directly:

    python tests/test_sanity.py
"""
import os
import sys

import pandas as pd

_HARNESS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HARNESS_ROOT not in sys.path:
    sys.path.insert(0, _HARNESS_ROOT)

from data_layer import sanity


def _frame(stamps, closes):
    ts = pd.to_datetime(stamps, utc=True)
    return pd.DataFrame({
        "timestamp": ts,
        "open": closes, "high": closes, "low": closes, "close": closes,
        "volume": [1000] * len(closes), "ticker": "TEST", "timeframe": "1d",
    })


def test_clean_frame_has_no_hard_fail():
    df = _frame(["2024-01-02 21:00", "2024-01-03 21:00", "2024-01-04 21:00"],
                [100.0, 101.0, 102.0])
    # Returns a list (may carry a row-band warning); the point is it does not raise.
    assert isinstance(sanity.check_integrity(df, "TEST", "1d"), list)


def test_duplicate_timestamp_raises():
    df = _frame(["2024-01-02 21:00", "2024-01-02 21:00"], [100.0, 101.0])
    try:
        sanity.check_integrity(df, "TEST", "1d")
        assert False, "expected ValueError on duplicate timestamp"
    except ValueError as e:
        assert "duplicate" in str(e).lower()


def test_non_monotonic_timestamp_raises():
    df = _frame(["2024-01-03 21:00", "2024-01-02 21:00"], [100.0, 101.0])  # out of order
    try:
        sanity.check_integrity(df, "TEST", "1d")
        assert False, "expected ValueError on non-monotonic timestamps"
    except ValueError as e:
        assert "monoton" in str(e).lower()


def test_eighty_percent_jump_warns():
    df = _frame(["2024-01-02 21:00", "2024-01-03 21:00"], [100.0, 180.0])  # +80%
    warnings = sanity.check_integrity(df, "TEST", "1d")
    assert any("return" in w for w in warnings), f"expected a large-move warning, got {warnings}"


if __name__ == "__main__":
    test_clean_frame_has_no_hard_fail()
    test_duplicate_timestamp_raises()
    test_non_monotonic_timestamp_raises()
    test_eighty_percent_jump_warns()
    print("test_sanity OK: duplicate + non-monotonic raise, 80% jump warns, clean frame passes.")
