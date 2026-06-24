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


def test_phase2_scoring_core_matches_live():
    """Phase 2: scoring core re-exported via the shim. The assembled prompt is
    byte-identical to live (same golden hashes as the trading-agent
    characterization), and parse applies the score>=threshold action fallback."""
    import hashlib

    from harness.reuse import (
        build_scoring_prompt,
        parse_scoring_response,
        SCORING_MODEL,
    )

    signals = [
        {"ticker": "NVDA", "rsi": 45.0, "macd_histogram": 0.5, "latest_close": 100.0,
         "ma_50": 95.0, "ma_200": 90.0, "previous_score": 6},
        {"ticker": "AMD", "rsi": 62.0, "macd_histogram": -0.2, "latest_close": 50.0,
         "ma_50": 51.0, "ma_200": 48.0},
    ]
    mc = {"vix_close": 18.5, "vix_20ma": 17.2, "vix_regime": "normal",
          "market_breadth_pct": 62.5}
    gold = {
        "steady": "24664d20806526b8efe70e4ceddc1b835ae98b985e64872d40c48a29ea5bebb7",
        "pulse": "2cb2218b891fc3eea3f2dd174b24caf111f726edfb1c973027e1f3ca4a5581b8",
    }
    for name, thr in (("steady", 7), ("pulse", 6)):
        prompt = build_scoring_prompt(signals, {"name": name, "buy_threshold": thr}, mc)
        assert hashlib.sha256(prompt.encode()).hexdigest() == gold[name], \
            f"{name} prompt diverged from live"

    assert SCORING_MODEL == "claude-opus-4-7"

    class _Block:
        def __init__(self, type, name=None, input=None):
            self.type, self.name, self.input = type, name, input

    content = [_Block("tool_use", "submit_ticker_scores", {
        "market_assessment": "",
        "scores": [
            {"ticker": "NVDA", "score": 8, "reasoning": "r"},
            {"ticker": "AMD", "score": 4, "reasoning": "r"},
        ],
    })]
    decisions = parse_scoring_response(content, {"name": "steady", "buy_threshold": 7})
    assert decisions == [
        {"ticker": "NVDA", "score": 8, "reasoning": "r", "action": "BUY"},
        {"ticker": "AMD", "score": 4, "reasoning": "r", "action": "SKIP"},
    ]


if __name__ == "__main__":
    test_indicators_import_and_known_value()
    test_pulse_indicators_shape()
    test_phase2_pure_modules_import_and_known_values()
    test_phase2_sizing_matches_live()
    test_phase2_scoring_core_matches_live()
    print("Phase 0 reuse shim OK: indicators imported from trading-agent and computed known values.")
    print("Phase 2 reuse shim OK: breadth, vix, exit_rules, sizing, scoring_core imported and known values verified.")
