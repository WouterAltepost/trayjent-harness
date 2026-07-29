"""The one registered test for the Drift sealed-read script (spec §6):
synthetic inputs only, no rotor data of any kind is read — the refusal path
exits before any data access, and the verdict/round-trip logic is pure.

Runnable via pytest or directly:

    python tests/test_drift_oos.py
"""
import os
import subprocess
import sys

_HARNESS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HARNESS_ROOT not in sys.path:
    sys.path.insert(0, _HARNESS_ROOT)

from scripts.backtest_drift_oos import count_round_trips, verdict


def test_drift_verdict_rule_round_trips_and_refusal():
    # Registered bar, bull case (B&H > 0): needs maxDD <= AND ret >= 0.75x.
    assert verdict(0.80, 0.30, 1.00, 0.40) == (True, True, True)
    assert verdict(0.74, 0.30, 1.00, 0.40) == (False, True, False)   # ret < 0.75x
    assert verdict(0.80, 0.41, 1.00, 0.40) == (False, False, True)   # deeper DD
    assert verdict(0.75, 0.40, 1.00, 0.40) == (True, True, True)     # both exact
    # Bear case (B&H <= 0): must beat B&H return outright AND draw down less.
    assert verdict(-0.10, 0.20, -0.30, 0.50) == (True, True, True)
    assert verdict(-0.30, 0.20, -0.30, 0.50) == (False, True, False) # ties lose
    assert verdict(0.05, 0.60, -0.30, 0.50) == (False, False, True)

    # Round trips from a synthetic invested series: two completed cycles,
    # the trailing open position does not count.
    assert count_round_trips([0, 1, 1, 0, 0, 0.9, 0, 1, 1]) == 2
    assert count_round_trips([1, 1, 1]) == 0
    assert count_round_trips([]) == 0

    # Refusal: without --spend-the-window the script exits non-zero having
    # read nothing (the flag check precedes all data access).
    proc = subprocess.run(
        [sys.executable, os.path.join(_HARNESS_ROOT, "scripts", "backtest_drift_oos.py")],
        capture_output=True, text=True, timeout=120)
    assert proc.returncode == 1
    assert "REFUSED" in proc.stdout
    assert "spend-the-window" in proc.stdout


if __name__ == "__main__":
    test_drift_verdict_rule_round_trips_and_refusal()
    print("test_drift_oos OK: registered verdict bar (bull/bear/edge cases), "
          "round-trip counting, refusal without the flag.")
