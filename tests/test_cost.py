"""Offline tests for the Phase-4 fail-closed cost tracker (brief Commit 3).

Cost math from a known PRICES table, accumulation, the fail-closed guard
(raises before a breaching spend, never mutates on the check), and the rolling
forward estimate. Pure — fake usage objects, no network.

Runnable via pytest or directly:
    python tests/test_cost.py
"""
import os
import sys

_HARNESS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HARNESS_ROOT not in sys.path:
    sys.path.insert(0, _HARNESS_ROOT)

from scoring.cost import CostTracker, CostCeilingExceeded


PRICES = {"m": {"input_per_mtok": 10.0, "output_per_mtok": 30.0}}


class _Usage:
    def __init__(self, input_tokens, output_tokens):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


def test_estimate_math():
    t = CostTracker(50.0, PRICES, "m")
    assert t.estimate(_Usage(1_000_000, 1_000_000)) == 40.0   # 10 + 30
    assert t.estimate(_Usage(500_000, 100_000)) == 8.0        # 5 + 3
    assert t.spent_usd == 0.0                                  # estimate is pure


def test_add_accumulates():
    t = CostTracker(50.0, PRICES, "m")
    t.add(_Usage(1_000_000, 0))   # +10
    t.add(_Usage(0, 1_000_000))   # +30
    assert t.spent_usd == 40.0
    assert t.n_calls == 2


def test_would_exceed_strict():
    t = CostTracker(50.0, PRICES, "m")
    t.add(_Usage(1_000_000, 1_000_000))   # 40 spent
    assert t.would_exceed(8.0) is False    # 48 <= 50
    assert t.would_exceed(10.0) is False   # 50 == ceiling, not strictly over
    assert t.would_exceed(11.0) is True    # 51 > 50


def test_check_raises_before_breach_without_mutation():
    t = CostTracker(50.0, PRICES, "m")
    t.add(_Usage(1_000_000, 1_000_000))   # 40 spent
    t.check(8.0)                           # under ceiling -> no raise
    try:
        t.check(11.0)
        assert False, "expected CostCeilingExceeded"
    except CostCeilingExceeded:
        pass
    assert t.spent_usd == 40.0             # check() never accrues


def test_rolling_estimate():
    t = CostTracker(50.0, PRICES, "m")
    assert t.rolling_estimate() == 0.0     # no history -> first call not pre-empted
    t.add(_Usage(1_000_000, 0))            # +10
    t.add(_Usage(0, 1_000_000))            # +30 -> 40 over 2 calls
    assert t.rolling_estimate() == 20.0


if __name__ == "__main__":
    test_estimate_math()
    test_add_accumulates()
    test_would_exceed_strict()
    test_check_raises_before_breach_without_mutation()
    test_rolling_estimate()
    print("test_cost OK: pricing math, accumulation, fail-closed guard, rolling estimate.")
