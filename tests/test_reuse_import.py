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


def test_phase2_pure_modules_import_and_known_values():
    """Phase 2: the new pure modules import green via the shim and compute
    known values, proving harness == live for breadth, VIX, and exits."""
    from datetime import datetime, timedelta, timezone

    from harness.reuse import (
        trend_anchor,
        compute_breadth,
        classify_vix_regime,
        evaluate_price_exit,
        should_force_close_for_max_hold,
    )

    # VIX ladder boundaries (<15 / <20 / <30 / 30+).
    assert classify_vix_regime(14.99) == "low"
    assert classify_vix_regime(15.0) == "normal"
    assert classify_vix_regime(19.99) == "normal"
    assert classify_vix_regime(20.0) == "elevated"
    assert classify_vix_regime(29.99) == "elevated"
    assert classify_vix_regime(30.0) == "stressed"

    # Exit ladder (percent units).
    assert evaluate_price_exit(6.0, 5.0, 3.0) == "SELL (TAKE PROFIT)"
    assert evaluate_price_exit(5.0, 5.0, 3.0) == "SELL (TAKE PROFIT)"   # >= boundary
    assert evaluate_price_exit(-3.0, 5.0, 3.0) == "SELL (STOP LOSS)"    # <= boundary
    assert evaluate_price_exit(1.0, 5.0, 3.0) is None

    # Max-hold age check, fail-soft.
    now = datetime(2026, 6, 24, tzinfo=timezone.utc)
    assert should_force_close_for_max_hold(now - timedelta(hours=50), 48, now=now) is True
    assert should_force_close_for_max_hold(now, 48, now=now) is False
    assert should_force_close_for_max_hold(None, 48, now=now) is False
    assert should_force_close_for_max_hold(now - timedelta(hours=50), None, now=now) is False

    # Breadth: SPY excluded; % above trend anchor, 1dp; None on empty universe.
    assert trend_anchor({"ma_50": 5, "ema_50": 9}) == 5      # ma_50 wins
    assert trend_anchor({"ema_50": 9}) == 9                   # ema_50 fallback
    assert trend_anchor({}) is None
    assert compute_breadth([]) is None
    sigs = [
        {"ticker": "SPY", "ma_50": 1, "latest_close": 99},   # excluded
        {"ticker": "AAA", "ma_50": 10, "latest_close": 11},  # above
        {"ticker": "BBB", "ma_50": 10, "latest_close": 9},   # below
    ]
    assert compute_breadth(sigs) == 50.0


def test_phase2_sizing_matches_live():
    """Phase 2: sizing is re-exported from live via the shim, and the harness's
    frozen POSITION_SIZING reproduces the live sizing battery — so harness
    sizing == live sizing on shared inputs (Phase 0 §5)."""
    import config as harness_config
    from harness.reuse import compute_position_size

    sizing = harness_config.POSITION_SIZING
    # Minimal strategy dicts — compute_position_size reads only these three keys.
    steady = {"buy_threshold": 7, "cash_safety_pct": 0.80, "stop_loss": 0.03}
    pulse = {"buy_threshold": 6, "cash_safety_pct": 0.90, "stop_loss": 0.01}

    assert compute_position_size(9, 100_000, 100_000, steady, sizing) == (15000.0, "score_ladder")
    assert compute_position_size(8, 100_000, 10_000, pulse, sizing) == (9000.0, "cash_cap")
    assert compute_position_size(6, 100_000, 100_000, steady, sizing) == (None, "multiplier_zero")
    assert compute_position_size(7, 8_000, 100_000, steady, sizing) == (None, "insufficient_capital")

    # The frozen harness sizing config matches the live constants exactly.
    assert sizing == {
        "base_pct_per_score": 0.05,
        "cash_safety_pct": 0.80,
        "min_trade_dollars": 500,
        "multiplier_cap": 4,
    }


if __name__ == "__main__":
    test_indicators_import_and_known_value()
    test_pulse_indicators_shape()
    test_phase2_pure_modules_import_and_known_values()
    test_phase2_sizing_matches_live()
    print("Phase 0 reuse shim OK: indicators imported from trading-agent and computed known values.")
    print("Phase 2 reuse shim OK: breadth, vix, exit_rules, sizing imported and known values verified.")
