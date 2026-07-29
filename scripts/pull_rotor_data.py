"""Mac-side entry point for the Rotor sleeve: discover, pull, cross-check.

    python scripts/pull_rotor_data.py

Same shape as the other sleeves' pulls — a thin orchestrator over
``data_layer.rotor_store.pull_all_rotor`` plus the sleeve's cross-checks
(``data_layer.rotor.cross_check``). Network runs here (Alpaca crypto data,
keyless), so it is run on the Mac, not in CI/offline.

Writes ONLY ``data/rotor/`` — never the frozen Steady/Pulse windows, never
the other sleeves' data, never ``cache/``.

Exits non-zero if any coin failed to pull OR any cross-check failed (in a
24/7 market every gap is a data problem), so a broken foundation is visible
rather than silently "green".
"""
import os
import sys

# Make `config` and the `data_layer` package importable when run from anywhere.
_HARNESS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HARNESS_ROOT not in sys.path:
    sys.path.insert(0, _HARNESS_ROOT)

import config
from data_layer import rotor, rotor_store


def main() -> int:
    results = rotor_store.pull_all_rotor()

    print(f"\n{'coin':<8} {'rows':>6}  {'first (UTC day)':<16} {'last (UTC day)':<16} status")
    print("-" * 62)
    for r in results:
        status = "ERROR" if r["error"] else (f"{len(r['warnings'])} warn" if r["warnings"] else "ok")
        first = rotor.bar_day(r["first"]) if r["first"] is not None else "-"
        last = rotor.bar_day(r["last"]) if r["last"] is not None else "-"
        print(f"{r['ticker']:<8} {r['rows']:>6}  {first:<16} {last:<16} {status}")

    warned = [r for r in results if r["warnings"]]
    failed = [r for r in results if r["error"]]

    if warned:
        print("\nWarnings (crypto single-day moves beyond 35% are plausible for the")
        print("small caps — listed for eyeballing, not auto-excused):")
        for r in warned:
            for w in r["warnings"]:
                print(f"  [{r['ticker']}] {w}")

    if failed:
        print("\nFailures (not written):")
        for r in failed:
            print(f"  [{r['ticker']}] {r['error']}")
        print(f"\n{len(failed)} coin(s) failed to pull — cross-checks still run "
              "on what was stored.")

    # ── Cross-checks (offline, on what was just stored) ─────────────────
    report = rotor.cross_check()

    print(f"\nCoverage — {len(report['coins'])} coins stored "
          f"(universe: Alpaca CURRENTLY TRADABLE minus stablecoins/exclusions,")
    print("PLUS the deliberately-pulled delisted series (ROTOR_DELISTED) whose")
    print("listing windows kill survivorship bias — see config):")
    print(f"{'coin':<8} {'first day':<12} {'last day':<12} {'rows':>6} "
          f"{'missing':>8} {'segs':>5}")
    print("-" * 56)
    for c in report["coins"]:
        print(f"{c['coin']:<8} {c['first_day']:<12} {c['last_day']:<12} "
              f"{c['rows']:>6} {c['missing_days']:>8} {c['segments']:>5}")

    if report["venue_gaps"]:
        print("\nVENUE LISTING GAPS (delist->relist windows — real Alpaca history,")
        print("not data corruption; the strategy step must treat these as")
        print("not-tradable windows and never compute a return across one):")
        for g in report["venue_gaps"]:
            print(f"  {g['coin']:<8} {g['gap_start']} .. {g['gap_end']}  "
                  f"({g['days']} days)")

    print("\nMAJORS HISTORY DEPTH (window design next step depends on this):")
    for coin, m in report["majors"].items():
        print(f"  {coin}: {m['first_day']} .. {m['last_day']}  ({m['years']:.1f} years)")
    if report["majors_flag"]:
        print("  *** FLAG: a major has under ~4 years of usable history — "
              "tuning-vs-sealed window design must account for this. ***")
    else:
        print("  Both majors clear the ~4-year bar for a tuning + sealed-OOS split.")

    print(f"\nBar convention: UTC day bars keyed 00:00Z, stored at close instant "
          f"(+24h); fees pinned: taker {config.ROTOR_ALPACA_CRYPTO_FEES['taker']:.2%} "
          f"/ maker {config.ROTOR_ALPACA_CRYPTO_FEES['maker']:.2%} (tier 1).")

    if report["failures"]:
        print("\nCROSS-CHECK FAILURES:")
        for f in report["failures"]:
            print(f"  {f}")
        return 1
    if failed:
        return 1

    print("\nAll cross-checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
