"""Offline tests for the pure ATR helper (Steady redesign Step 1).

Hand-calculated true ranges pin the formula — one case per branch of the max
(intrabar range, gap-up term, gap-down term) — plus the legit-zero output the
exit primitive's guard consumes, and the period+1 input floor. Pure
arithmetic — no Parquet, no network.

Runnable via pytest or directly:

    python tests/test_atr.py
"""
import math
import os
import sys

# Make the `runner` package importable when run from anywhere.
_HARNESS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HARNESS_ROOT not in sys.path:
    sys.path.insert(0, _HARNESS_ROOT)

from runner.atr import compute_atr


def _close(a, b):
    return math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9)


def test_atr_hand_calc_each_max_branch():
    """Four bars, period 3 — each TR decided by a different branch of the max.

    i  high  low  close  prev_c | h-l  |h-pc|  |l-pc| -> TR
    1   11    9    10      10   | 2.0   1.0     1.0      2.0  (intrabar range)
    2   13   12    12.5    10   | 1.0   3.0     2.0      3.0  (gap-up term)
    3   11   10    10.5    12.5 | 1.0   1.5     2.5      2.5  (gap-down term)
    ATR(3) = (2.0 + 3.0 + 2.5) / 3 = 2.5
    """
    highs = [10.0, 11.0, 13.0, 11.0]
    lows = [9.0, 9.0, 12.0, 10.0]
    closes = [10.0, 10.0, 12.5, 10.5]
    assert _close(compute_atr(highs, lows, closes, period=3), 2.5)


def test_atr_gap_term_beats_intrabar_range():
    # Prior close 9.5 sits below the bar's [11.5, 12.0] range: the true range
    # is the full gap |12.0 - 9.5| = 2.5, not the 0.5 intrabar range.
    assert _close(compute_atr([10.0, 12.0], [9.0, 11.5], [9.5, 11.8], period=1), 2.5)


def test_atr_flat_bars_zero():
    # All-equal bars -> every TR is 0 -> ATR 0.0. A legitimate output, not an
    # error: evaluate_trailing_exit's atr<=0 guard owns that case, so this
    # pins that compute_atr returns 0 rather than raising.
    assert compute_atr([100.0] * 5, [100.0] * 5, [100.0] * 5, period=3) == 0.0


def test_atr_insufficient_bars_raises():
    # period+1 bars are required (a prior close for the first TR): 3 bars
    # cannot feed ATR(3)...
    try:
        compute_atr([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], period=3)
        assert False, "expected ValueError below period+1 bars"
    except ValueError:
        pass
    # ...and exactly period+1 is the boundary that works.
    assert compute_atr([1.0] * 4, [1.0] * 4, [1.0] * 4, period=3) == 0.0


if __name__ == "__main__":
    test_atr_hand_calc_each_max_branch()
    test_atr_gap_term_beats_intrabar_range()
    test_atr_flat_bars_zero()
    test_atr_insufficient_bars_raises()
    print("test_atr OK: hand-calc branches, gap term, flat-zero, period+1 floor.")
