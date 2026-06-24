"""Phase 0 acceptance: prove the path-import shim works and the live indicator
code computes a known value. Runnable via pytest or directly:

    python tests/test_reuse_import.py
"""
import os
import sys

# Make `config` and the `harness` package importable when run from anywhere.
_HARNESS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HARNESS_ROOT not in sys.path:
    sys.path.insert(0, _HARNESS_ROOT)

from harness.reuse import compute_indicators, compute_indicators_pulse


def _strictly_increasing(n=319):
    """A strictly increasing close series -> all gains -> RSI must be 100.0."""
    closes = [float(x) for x in range(1, n + 1)]
    volumes = [1000 for _ in range(n)]
    return {"ticker": "TEST", "close_prices": closes, "volumes": volumes}


def test_indicators_import_and_known_value():
    pd = _strictly_increasing()
    out = compute_indicators(pd)
    # Known-value checks on a deterministic monotonic series.
    assert out["ticker"] == "TEST"
    assert out["rsi"] == 100.0, f"expected RSI 100.0 on all-gains series, got {out['rsi']}"
    assert out["latest_close"] == 319.0
    assert out["ma_50"] is not None and out["ma_200"] is not None
    # Shape contract: the keys the agent scores must be present.
    for key in ("macd_histogram", "ema_20", "obv_trending_up"):
        assert key in out, f"missing indicator key {key}"


def test_pulse_indicators_shape():
    pd = _strictly_increasing()
    out = compute_indicators_pulse(pd)
    assert out["rsi"] == 100.0
    for key in ("ema_9", "ema_21", "ema_50", "price_above_ema21", "obv_trending_up"):
        assert key in out, f"missing pulse indicator key {key}"


if __name__ == "__main__":
    test_indicators_import_and_known_value()
    test_pulse_indicators_shape()
    print("Phase 0 reuse shim OK: indicators imported from trading-agent and computed known values.")
