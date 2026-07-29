"""Mac-side entry point for the shortvol sleeve: pull, store, cross-check.

    python scripts/pull_shortvol_data.py

Same shape as ``pull_data.py`` — a thin orchestrator over
``data_layer.shortvol_store.pull_all_shortvol`` plus the sleeve's
cross-checks (``data_layer.shortvol.cross_check``). Network runs here (CBOE
CSVs + yfinance), so it is run on the Mac, not in CI/offline.

Writes ONLY ``data/shortvol/`` — never the frozen Steady/Pulse windows under
``data/{1d,1h,30m}``, never ``cache/``.

Exits non-zero if any series failed to pull OR any cross-check failed, so a
broken data foundation is visible rather than silently "green".
"""
import os
import sys

# Make `config` and the `data_layer` package importable when run from anywhere.
_HARNESS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HARNESS_ROOT not in sys.path:
    sys.path.insert(0, _HARNESS_ROOT)

import config
from data_layer import shortvol, shortvol_store


def _fmt_ts(ts) -> str:
    return ts.strftime("%Y-%m-%d") if ts is not None else "-"


def main() -> int:
    results = shortvol_store.pull_all_shortvol()

    print(f"\n{'series':<8} {'rows':>6}  {'first':<12}  {'last':<12}  status")
    print("-" * 56)
    for r in results:
        status = "ERROR" if r["error"] else (f"{len(r['warnings'])} warn" if r["warnings"] else "ok")
        print(f"{r['ticker']:<8} {r['rows']:>6}  "
              f"{_fmt_ts(r['first']):<12}  {_fmt_ts(r['last']):<12}  {status}")

    warned = [r for r in results if r["warnings"]]
    failed = [r for r in results if r["error"]]

    if warned:
        print("\nWarnings (large single-bar moves are EXPECTED on the vol ETPs —")
        print("Volmageddon 2018-02, COVID 2020-03, the 2024-08 spike are genuine):")
        for r in warned:
            for w in r["warnings"]:
                print(f"  [{r['ticker']}] {w}")

    if failed:
        print("\nFailures (not written):")
        for r in failed:
            print(f"  [{r['ticker']}] {r['error']}")
        print(f"\n{len(failed)} series failed to pull — cross-checks skipped.")
        return 1

    # ── Cross-checks (offline, on what was just stored) ─────────────────
    report = shortvol.cross_check()

    print("\nCoverage (missing = SPY trading days the series lacks, within its")
    print("own [first, last] overlap with SPY; known CBOE-file gaps annotated):")
    print(f"{'series':<8} {'first':<12} {'last':<12} {'rows':>6}  missing days")
    print("-" * 56)
    for s in report["series"]:
        note = f" ({s['known_gaps']} known CBOE gaps)" if s["known_gaps"] else ""
        print(f"{s['ticker']:<8} {s['first']:<12} {s['last']:<12} {s['rows']:>6}  "
              f"{s['missing_days']}{note}")

    if report["warnings"]:
        print("\nCross-source warnings (non-fatal — Yahoo lags/drops bars at times;")
        print("re-running the pull upsert-fills them once Yahoo has the data):")
        for w in report["warnings"]:
            print(f"  {w}")

    if report["cboe_only_days"]:
        co = report["cboe_only_days"]
        print(f"\nCBOE-only dates ({len(co)}): sessions the CBOE file publishes that SPY")
        print("does not trade — NYSE holidays CBOE includes since 2022, plus special")
        print("closures. Expected and informational; a normal weekday here would")
        print("instead suggest an interior SPY hole on Yahoo.")
        print(f"  {co[:12]}{'...' if len(co) > 12 else ''}")

    t = report["term"]
    print(f"\nVIX/VIX3M term structure: {t['rows']} joined days "
          f"[{t['first']} .. {t['last']}], VIX3M min {t['vix3m_min']:.2f} (must be > 0)")
    print(f"Backwardation days (VIX/VIX3M >= 1.0): {t['inversion_days']} "
          f"({t['inversion_pct']:.1f}% of joined days) — should be a small minority,")
    print("clustered in the known stress years (2015, 2018, 2020, 2024, 2025):")
    for year, n in sorted(t["inversion_by_year"].items()):
        print(f"  {year}: {n}")

    print(f"\nInstrument notes: SVXY leverage flip {config.SVXY_LEVERAGE_CHANGE_DATE} "
          f"(-1x before, -0.5x after); VXX is series B only, from {config.VXX_HISTORY_START}.")

    if report["failures"]:
        print("\nCROSS-CHECK FAILURES:")
        for f in report["failures"]:
            print(f"  {f}")
        return 1

    print("\nAll cross-checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
