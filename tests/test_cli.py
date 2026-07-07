"""Tests for the CLI's sealed-OOS guard (Steady redesign Step 1, Commit 5).

The fence is the only thing standing between a careless dev run and a burned
out-of-sample window, so both Steady slabs (consumed 2020-2021, fresh
2016-2019) and the Pulse quarter are pinned here. Guard-only tests — no
backtest is run.

Runnable via pytest or directly:

    python tests/test_cli.py
"""
import os
import sys

_HARNESS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HARNESS_ROOT not in sys.path:
    sys.path.insert(0, _HARNESS_ROOT)

from runner.cli import _check_oos
from runner.configs import build_run_config


def _rc(name, start, end):
    return build_run_config(name, start, end, "rules_only")


def _assert_refused(rc, *expected_in_message):
    try:
        _check_oos(rc, allow_sealed=False)
        assert False, f"expected SystemExit for {rc.name} {rc.start.date()}..{rc.end.date()}"
    except SystemExit as e:
        for fragment in expected_in_message:
            assert fragment in str(e), f"refusal must name the window: {fragment!r}"


def test_fresh_2016_window_fenced():
    # The full fresh slab is refused without the override...
    _assert_refused(_rc("steady", "2016-01-01", "2019-12-31"),
                    "2016-01-01", "2019-12-31")
    # ...a partial overlap (poking two weeks into the slab) is refused too...
    _assert_refused(_rc("steady", "2015-06-01", "2016-01-15"), "2016-01-01")
    # ...and --allow-sealed opens it for the one deliberate read.
    _check_oos(_rc("steady", "2016-01-01", "2019-12-31"), allow_sealed=True)


def test_old_2020_window_still_fenced():
    # No regression on the consumed slab: still refused without the override.
    _assert_refused(_rc("steady", "2020-06-01", "2021-06-01"),
                    "2020-01-01", "2021-12-31")


def test_in_sample_window_passes():
    # The dev window (post-2021, clear of both slabs) sails through.
    _check_oos(_rc("steady", "2022-01-01", "2026-06-30"), allow_sealed=False)
    # Pre-2016 history is also unfenced.
    _check_oos(_rc("steady", "2012-01-01", "2015-12-31"), allow_sealed=False)


def test_pulse_fences_unchanged():
    # Pulse-hourly's quarter: still refused / still opened by the flag.
    _assert_refused(_rc("pulse_hourly", "2026-04-01", "2026-06-24"), "2026-04-01")
    _check_oos(_rc("pulse_hourly", "2026-04-01", "2026-06-24"), allow_sealed=True)
    _check_oos(_rc("pulse_hourly", "2025-01-01", "2025-12-31"), allow_sealed=False)
    # Steady's slabs must not leak onto Pulse: its 2016-2019 runs pass.
    _check_oos(_rc("pulse_hourly", "2016-01-01", "2019-12-31"), allow_sealed=False)
    # pulse_30min has no fence at all (nothing to guard, per the _SEALED note).
    _check_oos(_rc("pulse_30min", "2026-04-01", "2026-06-24"), allow_sealed=False)


if __name__ == "__main__":
    test_fresh_2016_window_fenced()
    test_old_2020_window_still_fenced()
    test_in_sample_window_passes()
    test_pulse_fences_unchanged()
    print("test_cli OK: fresh 2016-2019 fence, consumed 2020-2021 fence, "
          "in-sample pass-through, pulse fences unchanged.")
