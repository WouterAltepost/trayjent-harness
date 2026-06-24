"""Mac-side entry point: pull the full historical universe and summarize.

Thin orchestrator over ``data_layer.store.pull_all`` (brief Step 5) — it owns
no data logic, only invocation and a printed summary. Network runs here (this
is the one place the harness hits yfinance), so it is run on the Mac, not in
CI/offline.

    python scripts/pull_data.py

Exits non-zero if any ticker failed (download error or hard sanity failure),
so a partial pull is visible rather than silently "green".
"""
import os
import sys

# Make `config` and the `data_layer` package importable when run from anywhere.
_HARNESS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HARNESS_ROOT not in sys.path:
    sys.path.insert(0, _HARNESS_ROOT)

from data_layer import store


def _fmt_ts(ts) -> str:
    return ts.strftime("%Y-%m-%d %H:%M") if ts is not None else "-"


def main() -> int:
    results = store.pull_all()

    print(f"\n{'ticker':<8} {'tf':<4} {'rows':>6}  {'first':<16}  {'last':<16}  status")
    print("-" * 72)
    for r in results:
        status = "ERROR" if r["error"] else (f"{len(r['warnings'])} warn" if r["warnings"] else "ok")
        print(f"{r['ticker']:<8} {r['timeframe']:<4} {r['rows']:>6}  "
              f"{_fmt_ts(r['first']):<16}  {_fmt_ts(r['last']):<16}  {status}")

    warned = [r for r in results if r["warnings"]]
    failed = [r for r in results if r["error"]]

    if warned:
        print("\nWarnings:")
        for r in warned:
            for w in r["warnings"]:
                print(f"  [{r['ticker']} {r['timeframe']}] {w}")

    if failed:
        print("\nFailures (not written):")
        for r in failed:
            print(f"  [{r['ticker']} {r['timeframe']}] {r['error']}")

    print(f"\n{len(results)} pulled, {len(warned)} with warnings, {len(failed)} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
