"""Results-summary generator (Phase 6 Code Commit 1, harness-only).

Read one or more schema-v1 backtest JSONs (the files `runner/output.py` emits)
and print a deterministic markdown summary: a strategy-vs-SPY metrics table, the
omission/deviation notes verbatim, and an EMPTY ``## Verdict`` heading for a
hand-written edge read. The generator never auto-writes a verdict (L7 is a
judgement, not a formula).

Pure stdlib, offline, deterministic. Metric keys are read straight from the JSON
and mirror `runner/metrics.py` / `runner/output.py`; missing keys render blank.

    python scripts/summarize_results.py results/A_*.json
    python scripts/summarize_results.py results/C_steady_insample.json --out results/phase6_summary.md
"""
import argparse
import json
import sys


def _pct(x):
    return None if x is None else f"{x * 100:.2f}%"


def _num(x, dp):
    return None if x is None else f"{x:.{dp}f}"


def _hours(x):
    return None if x is None else f"{x:.1f} h"


def _cell(v):
    """A table cell; None -> blank (strategy-only metric or absent value)."""
    return "" if v is None else str(v)


# (label, formatter, in_benchmark) for the flat metrics block. Benchmark carries
# only total_return/cagr/sharpe/max_drawdown (see metrics.compute_metrics); the
# rest are strategy-only and leave the benchmark cell blank.
_FLAT_ROWS = [
    ("Total return", "total_return", lambda x: _pct(x), True),
    ("CAGR", "cagr", lambda x: _pct(x), True),
    ("Sharpe", "sharpe", lambda x: _num(x, 3), True),
    ("Max drawdown", "max_drawdown", lambda x: _pct(x), True),
    ("Profit factor", "profit_factor", lambda x: _num(x, 3), False),
    ("Beta vs SPY", "beta_spy", lambda x: _num(x, 3), False),
    ("Win rate", "win_rate", lambda x: _pct(x), False),
]


def _table(metrics):
    """The strategy-vs-benchmark markdown table for one run's metrics block."""
    bench = metrics.get("benchmark_spy") or {}
    lines = [
        "| Metric | Strategy | SPY benchmark |",
        "|---|---|---|",
    ]

    def row(label, strat_val, bench_val=None):
        lines.append(f"| {label} | {_cell(strat_val)} | {_cell(bench_val)} |")

    for label, key, fmt, in_bench in _FLAT_ROWS:
        sval = fmt(metrics.get(key))
        bval = fmt(bench.get(key)) if in_bench else None
        row(label, sval, bval)

    exposure = metrics.get("exposure") or {}
    row("Exposure (% decisions w/ position)",
        _pct(exposure.get("pct_decisions_with_position")))
    row("Avg positions", _num(exposure.get("avg_positions"), 2))
    row("Avg hold", _hours(exposure.get("avg_hold_hours")))
    row("Median hold", _hours(exposure.get("median_hold_hours")))

    trades = metrics.get("trades") or {}
    row("Closed trades", trades.get("total"))
    for reason in sorted((trades.get("by_exit_reason") or {}).keys()):
        row(f"&nbsp;&nbsp;{reason}", trades["by_exit_reason"][reason])

    return "\n".join(lines)


def summarize_one(doc):
    """Markdown section for a single schema-v1 document."""
    meta = doc.get("meta", {})
    period = meta.get("period", {})
    out = []
    out.append(f"# {meta.get('strategy')} — {meta.get('run_config')} ({meta.get('mode')})")
    out.append("")
    out.append(
        f"**Period:** {period.get('start')} → {period.get('end')}  ·  "
        f"**Cadence:** {period.get('cadence')}")
    out.append("")
    out.append(_table(doc.get("metrics", {})))
    out.append("")
    out.append("**Notes:**")
    out.append("")
    for note in doc.get("notes", []):
        out.append(f"- {note}")
    out.append("")
    out.append("## Verdict")
    out.append("")
    return "\n".join(out)


def summarize(paths):
    sections = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        sections.append(summarize_one(doc))
    return "\n\n---\n\n".join(sections) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize schema-v1 backtest JSON(s) to markdown.")
    parser.add_argument("paths", nargs="+", help="schema-v1 result JSON file(s)")
    parser.add_argument("--out", help="write markdown here instead of stdout")
    args = parser.parse_args(argv)

    md = summarize(args.paths)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(md)
    else:
        sys.stdout.write(md)


if __name__ == "__main__":
    main()
