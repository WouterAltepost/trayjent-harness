"""Offline test for the results-summary generator (Phase 6 Code Commit 1).

A synthetic schema-v1 doc in -> markdown out. Asserts the strategy-vs-benchmark
table rows render, the benchmark cells populate only where the benchmark carries
the metric, the notes pass through verbatim, and the ``## Verdict`` section is
left empty (the generator never auto-writes an edge call, L7).

    python tests/test_summarize_results.py
"""
import os
import sys

_HARNESS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HARNESS_ROOT not in sys.path:
    sys.path.insert(0, _HARNESS_ROOT)

from scripts.summarize_results import summarize_one


_NOTES = [
    "breaker not modeled (never tripped)",
    "earnings filter not modeled (no-earnings stub)",
    "idealized (0 fees, 0 slippage, fill at decision-bar close)",
    "market always open (no SKIP (MARKET CLOSED) path)",
]

_DOC = {
    "schema_version": 1,
    "meta": {
        "strategy": "pulse",
        "run_config": "pulse_hourly",
        "mode": "rules_only",
        "period": {"start": "2024-06-24T13:30:00+00:00",
                   "end": "2026-03-31T19:30:00+00:00", "cadence": "hourly"},
    },
    "metrics": {
        "total_return": 0.25,
        "cagr": 0.12,
        "win_rate": 0.44,
        "profit_factor": 1.077,
        "max_drawdown": 0.2567,
        "sharpe": 0.809,
        "beta_spy": 1.082,
        "exposure": {
            "pct_decisions_with_position": 0.9,
            "avg_positions": 3.5,
            "avg_hold_hours": 12.0,
            "median_hold_hours": 8.0,
        },
        "trades": {
            "total": 3,
            "by_exit_reason": {"SELL (TAKE PROFIT)": 1, "SELL (STOP LOSS)": 2},
        },
        "benchmark_spy": {
            "total_return": 0.16,
            "cagr": 0.0995,
            "sharpe": 0.654,
            "max_drawdown": 0.1943,
        },
    },
    "notes": _NOTES,
}


def _build():
    return summarize_one(_DOC)


def test_header_renders():
    md = _build()
    assert "# pulse — pulse_hourly (rules_only)" in md
    assert "**Cadence:** hourly" in md
    assert "2024-06-24T13:30:00+00:00 → 2026-03-31T19:30:00+00:00" in md


def test_table_rows_present():
    md = _build()
    # strategy values formatted (returns/dd as %, sharpe/beta/pf as decimals)
    assert "| Total return | 25.00% | 16.00% |" in md
    assert "| Sharpe | 0.809 | 0.654 |" in md
    assert "| Max drawdown | 25.67% | 19.43% |" in md
    # exposure + trade rows
    assert "| Exposure (% decisions w/ position) | 90.00% |  |" in md
    assert "| Avg positions | 3.50 |  |" in md
    assert "| Avg hold | 12.0 h |  |" in md
    assert "| Median hold | 8.0 h |  |" in md
    assert "| Closed trades | 3 |  |" in md
    assert "SELL (TAKE PROFIT) | 1 |" in md
    assert "SELL (STOP LOSS) | 2 |" in md


def test_benchmark_blank_for_strategy_only_metrics():
    md = _build()
    # profit_factor / beta / win_rate are strategy-only -> blank benchmark cell
    assert "| Profit factor | 1.077 |  |" in md
    assert "| Beta vs SPY | 1.082 |  |" in md
    assert "| Win rate | 44.00% |  |" in md


def test_notes_passthrough_verbatim():
    md = _build()
    for note in _NOTES:
        assert f"- {note}" in md


def test_verdict_section_empty():
    md = _build()
    assert "## Verdict" in md
    # nothing auto-written after the Verdict heading
    after = md.split("## Verdict", 1)[1].strip()
    assert after == ""


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name} OK")
    print("test_summarize_results OK")
