"""Offline tests for the Phase-4 rules-only scorer (brief Commit 4).

Each point rule fires on a crafted single-rule signal (isolated so score-5 ==
that rule's points), for both strategies; the L11a mapping clamps to 0-10;
action = BUY iff score >= buy_threshold; decision shape matches
parse_scoring_response; None/missing values never fire and never crash; the
confirmed omissions (MACD turning/deeply-negative, previous_score nudge) do not
contribute. Pure — no network.

Runnable via pytest or directly:
    python tests/test_rules_only.py
"""
import os
import sys

_HARNESS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HARNESS_ROOT not in sys.path:
    sys.path.insert(0, _HARNESS_ROOT)

from scoring.rules_only import score_rules_only

STEADY = {"name": "steady", "buy_threshold": 7}
PULSE = {"name": "pulse", "buy_threshold": 6}


def _score(signal, strategy):
    return score_rules_only([{"ticker": "X", **signal}], strategy)[0]["score"]


# (label, single-rule signal, strategy, expected delta from baseline 5)
_STEADY_RULES = [
    ("RSI<30 +2", {"rsi": 25.0}, 2),
    ("RSI 30-40 +1", {"rsi": 35.0}, 1),
    ("RSI>70 -1", {"rsi": 80.0}, -1),
    ("MACD hist>0 +2", {"macd_histogram": 0.5}, 2),
    ("close>50MA +1", {"latest_close": 100.0, "ma_50": 90.0}, 1),
    ("close>200MA +1", {"latest_close": 100.0, "ma_200": 90.0}, 1),
    ("golden cross +1", {"ma_50": 95.0, "ma_200": 90.0}, 1),
    ("death cross -2", {"ma_50": 90.0, "ma_200": 95.0}, -2),
]

_PULSE_RULES = [
    ("price>EMA21 +1", {"price_above_ema21": True}, 1),
    ("bounced EMA9 +1", {"price_bounced_ema9": True}, 1),
    ("close>EMA50 +1", {"latest_close": 100.0, "ema_50": 90.0}, 1),
    ("EMA9 x-up +2", {"ema9_cross_above_ema21": True}, 2),
    ("EMA9 x-down -2", {"ema9_cross_below_ema21": True}, -2),
    ("RSI<30 +2", {"rsi": 25.0}, 2),
    ("RSI 30-50 +1", {"rsi": 45.0}, 1),
    ("RSI>70 -1", {"rsi": 80.0}, -1),
    ("MACD hist>0 & line>signal +2", {"macd_histogram": 0.3, "macd_line": 1.0, "macd_signal": 0.7}, 2),
    ("OBV up +1", {"obv_trending_up": True}, 1),
    ("OBV bear-div -1", {"obv_bearish_divergence": True}, -1),
]


def test_each_rule_fires():
    for label, sig, delta in _STEADY_RULES:
        assert _score(sig, STEADY) == 5 + delta, ("steady", label)
    for label, sig, delta in _PULSE_RULES:
        assert _score(sig, PULSE) == 5 + delta, ("pulse", label)


def test_mapping_clamps_to_0_10():
    # Steady all-bullish: +2+2+1+1+1 = +7 -> 5+7=12 -> clamps to 10.
    steady_max = {"rsi": 25.0, "macd_histogram": 1.0, "latest_close": 100.0,
                  "ma_50": 95.0, "ma_200": 90.0}
    assert _score(steady_max, STEADY) == 10
    # Pulse all-bullish: +1+1+1+2+2+2+1 = +10 -> 5+10=15 -> clamps to 10.
    pulse_max = {"price_above_ema21": True, "price_bounced_ema9": True,
                 "latest_close": 100.0, "ema_50": 90.0, "ema9_cross_above_ema21": True,
                 "rsi": 25.0, "macd_histogram": 0.3, "macd_line": 1.0, "macd_signal": 0.7,
                 "obv_trending_up": True}
    assert _score(pulse_max, PULSE) == 10


def test_action_threshold():
    # Steady threshold 7: point_sum 2 -> 7 -> BUY; point_sum 1 -> 6 -> SKIP.
    # (RSI 35 +1 and close>50MA +1; ma_200 omitted so no 200MA/cross points.)
    buy = score_rules_only([{"ticker": "A", "rsi": 35.0, "latest_close": 100.0, "ma_50": 90.0}], STEADY)[0]
    assert buy["score"] == 7 and buy["action"] == "BUY"
    skip = score_rules_only([{"ticker": "A", "latest_close": 100.0, "ma_50": 90.0}], STEADY)[0]
    assert skip["score"] == 6 and skip["action"] == "SKIP"
    # Pulse threshold 6: point_sum 1 -> 6 -> BUY; point_sum 0 -> 5 -> SKIP.
    pbuy = score_rules_only([{"ticker": "A", "price_above_ema21": True}], PULSE)[0]
    assert pbuy["score"] == 6 and pbuy["action"] == "BUY"
    pskip = score_rules_only([{"ticker": "A"}], PULSE)[0]
    assert pskip["score"] == 5 and pskip["action"] == "SKIP"


def test_decision_shape_matches_parse():
    d = score_rules_only([{"ticker": "NVDA", "rsi": 25.0}], STEADY)[0]
    assert set(d.keys()) == {"ticker", "score", "reasoning", "action"}
    assert d["ticker"] == "NVDA"
    assert "rules-only" in d["reasoning"]   # self-labelled, not a Claude result


def test_none_and_missing_never_fire():
    # All-None / empty signal: no rule fires -> baseline 5, SKIP, no crash.
    none_sig = {"ticker": "Z", "rsi": None, "macd_histogram": None, "macd_line": None,
                "macd_signal": None, "latest_close": None, "ma_50": None, "ma_200": None,
                "ema_50": None}
    d = score_rules_only([none_sig], STEADY)[0]
    assert d["score"] == 5 and d["action"] == "SKIP"
    assert score_rules_only([{"ticker": "Z"}], PULSE)[0]["score"] == 5


def test_confirmed_omissions_do_not_contribute():
    # Negative histogram alone must NOT add a point (deeply-negative omitted).
    assert _score({"macd_histogram": -5.0}, STEADY) == 5
    assert _score({"macd_histogram": -5.0, "macd_line": -2.0, "macd_signal": -1.0}, PULSE) == 5
    # previous_score is ignored (nudge omitted) — score unchanged with it present.
    assert _score({"rsi": 25.0, "previous_score": 9}, STEADY) == 7


def test_unknown_strategy_raises():
    try:
        score_rules_only([{"ticker": "X"}], {"name": "mystery", "buy_threshold": 7})
        assert False, "expected ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    test_each_rule_fires()
    test_mapping_clamps_to_0_10()
    test_action_threshold()
    test_decision_shape_matches_parse()
    test_none_and_missing_never_fire()
    test_confirmed_omissions_do_not_contribute()
    test_unknown_strategy_raises()
    print("test_rules_only OK: every rule fires, clamp 0-10, action threshold, "
          "decision shape, None-safe, confirmed omissions, unknown-strategy guard.")
