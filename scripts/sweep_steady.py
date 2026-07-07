"""Free in-sample sweep of the Steady grower dials (redesign Step 1, S1-9).

Sweeps the deterministic exit/concentration knobs over the cached 2022-2026
in-sample window in --mode claude. The scoring INPUTS are unchanged from the
Stage E runs that built cache/scoring.db (PROMPT_VERSION v9.5, same watchlist,
SCORING_MODEL claude-opus-4-7, and none of the swept dials enter the prompt),
so every Claude call is a cache hit and the sweep costs $0. That claim is
ENFORCED, not assumed: a short canary run must show zero API calls before the
grid starts, and every combo re-asserts zero spend — any miss aborts the sweep
loudly (cache drift; see the Step 1 plan's PF-4).

Two-stage plan (S1-9): Stage 1 (this file's default) fixes concentration at a
moderate more-invested default and sweeps the exit dials; Stage 2 (commented
block below) fixes the exit at the Stage 1 winner and sweeps concentration.
The script ranks and reports; it does NOT pick a winner — that verdict is
written by hand.

Run from the harness root (client construction wants a key even though an
all-hit run makes no billable call):

    export ANTHROPIC_API_KEY=...
    python3.13 scripts/sweep_steady.py

Writes results/steady_sweep_stage1.md + .csv (results/ is gitignored).
"""
import csv
import dataclasses
import itertools
import os
import sys

_HARNESS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HARNESS_ROOT not in sys.path:
    sys.path.insert(0, _HARNESS_ROOT)

import config
from runner.cli import _check_oos
from runner.configs import build_run_config
from runner.metrics import compute_metrics
from runner.run import run_backtest

# ── Windows ──────────────────────────────────────────────────────────────
# The cached in-sample window — exactly the Stage E Steady period, so the
# scoring cache covers every decision tick. Both sealed Steady slabs
# (2016-2019, 2020-2021) lie outside it; _check_oos re-verifies below.
START, END = "2022-01-01", "2026-06-24"
# One cheap month for the canary run that gates the whole grid.
CANARY_START, CANARY_END = "2023-01-01", "2023-02-01"

# ── Preserver baseline (in-sample), results/phase6_stageE_summary.md ─────
# The validated capital preserver the grower must beat on growth while
# staying under the drawdown line. Fractions (compute_metrics units).
BASELINE_TOTAL_RETURN = 0.2350   # 23.50%
BASELINE_CAGR = 0.0483           # 4.83%
BASELINE_SHARPE = 1.139
BASELINE_MAX_DD = 0.0450         # 4.50%
# PASS = grows past the preserver AND keeps the drawdown tolerable.
PASS_RETURN_FLOOR = BASELINE_TOTAL_RETURN
PASS_DD_CEILING = 0.20           # 20.0%

# ── Stage 1: sweep the exit at a fixed moderate concentration ────────────
# cash_safety_pct lives in the STRATEGY dict; base_pct_per_score and
# multiplier_cap live in sizing_config — a concentration setting overrides
# both dicts (sizing_config also carries a cash_safety_pct key; set it too so
# the two can never disagree).
STAGE1_CONC_STRATEGY = {"cash_safety_pct": 0.50}
STAGE1_CONC_SIZING = {"base_pct_per_score": 0.08, "multiplier_cap": 4,
                      "cash_safety_pct": 0.50}
# Exit grid: 4 x 2 x 2 = 16 combos. stop_loss is the hard-stop floor (fresh
# phase) and the sizing 2% clamp input, in fraction units like the strategy.
STAGE1_EXIT_GRID = {
    "trailing_atr_mult": [2.0, 2.5, 3.0, 3.5],
    "atr_period": [14, 22],
    "stop_loss": [0.03, 0.05],
}

# ── Stage 2 (fill in AFTER the Stage 1 read; do not run both blind) ──────
# Fix the exit at the Stage 1 winner, sweep concentration instead, e.g.:
#   STAGE2_EXIT = {"trailing_atr_mult": <w>, "atr_period": <w>,
#                  "stop_loss": <w>}                       # strategy dict
#   STAGE2_CONC_GRID = {
#       "base_pct_per_score": [0.05, 0.08, 0.10],          # sizing_config
#       "multiplier_cap": [4, 5],                          # sizing_config
#       "cash_safety_pct": [0.50, 0.65, 0.80],             # BOTH dicts
#   }
# Same run/assert/report loop; output name steady_sweep_stage2.*.

OUT_MD = os.path.join(config.HARNESS_ROOT, "results", "steady_sweep_stage1.md")
OUT_CSV = os.path.join(config.HARNESS_ROOT, "results", "steady_sweep_stage1.csv")

COLUMNS = ["trailing_atr_mult", "atr_period", "stop_loss",
           "total_return", "cagr", "sharpe", "max_drawdown", "beta_spy",
           "profit_factor", "trades", "n_trailing", "n_stop", "PASS"]


def _assert_free(res, label: str) -> None:
    """The zero-cost invariant, enforced. A cache-backed claude run must make
    ZERO API calls; any spend means the cache drifted and the sweep is NOT
    free — abort before the next combo can spend more."""
    t = res.cost_tracker
    if t is None:
        sys.exit(f"ABORT [{label}]: no cost tracker on a claude-mode run "
                 f"(mode misconfigured?)")
    if t.n_calls != 0 or t.spent_usd != 0.0 or res.halted_on_cost:
        sys.exit(
            f"ABORT [{label}]: cache miss / drift — sweep is not free. "
            f"n_calls={t.n_calls}, spent=${t.spent_usd:.4f}, "
            f"halted_on_cost={res.halted_on_cost}. Scoring inputs no longer "
            f"match the cache (see Step 1 plan PF-4: PROMPT_VERSION, "
            f"watchlist, SCORING_MODEL). Fix the drift before re-running."
        )


def _combo_config(base, mult, period, sl):
    strategy = {**base.strategy,
                "trailing_atr_mult": mult, "atr_period": period,
                "stop_loss": sl, **STAGE1_CONC_STRATEGY}
    sizing = {**config.POSITION_SIZING, **STAGE1_CONC_SIZING}
    return dataclasses.replace(base, strategy=strategy, sizing_config=sizing)


def _row(mult, period, sl, met):
    by_reason = met["trades"]["by_exit_reason"]
    ret, mdd = met["total_return"], met["max_drawdown"]
    passed = (ret is not None and mdd is not None
              and ret > PASS_RETURN_FLOOR and mdd <= PASS_DD_CEILING)
    return {
        "trailing_atr_mult": mult, "atr_period": period, "stop_loss": sl,
        "total_return": ret, "cagr": met["cagr"], "sharpe": met["sharpe"],
        "max_drawdown": mdd, "beta_spy": met["beta_spy"],
        "profit_factor": met["profit_factor"],
        "trades": met["trades"]["total"],
        "n_trailing": by_reason.get("SELL (TRAILING STOP)", 0),
        "n_stop": by_reason.get("SELL (STOP LOSS)", 0),
        "PASS": passed,
    }


def _fmt(key, val):
    """Human table cell. Percent-style fractions rendered as %; None as —."""
    if val is None:
        return "—"
    if key in ("total_return", "cagr", "max_drawdown"):
        return f"{val * 100:.2f}%"
    if key in ("sharpe", "beta_spy", "profit_factor"):
        return f"{val:.3f}"
    if key == "stop_loss":
        return f"{val:.2f}"
    if key == "PASS":
        return "PASS" if val else "—"
    return str(val)


def _write_outputs(rows) -> None:
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    lines = [
        "# Steady grower sweep — Stage 1 (exit dials), in-sample",
        "",
        f"Window {START}..{END} (cached, $0 verified per combo) · mode claude · "
        f"concentration fixed: base_pct_per_score "
        f"{STAGE1_CONC_SIZING['base_pct_per_score']}, multiplier_cap "
        f"{STAGE1_CONC_SIZING['multiplier_cap']}, cash_safety_pct "
        f"{STAGE1_CONC_STRATEGY['cash_safety_pct']}.",
        "",
        f"Preserver baseline (phase6_stageE_summary.md, in-sample): total return "
        f"{BASELINE_TOTAL_RETURN * 100:.2f}%, CAGR {BASELINE_CAGR * 100:.2f}%, "
        f"Sharpe {BASELINE_SHARPE}, max drawdown {BASELINE_MAX_DD * 100:.2f}%. "
        f"PASS = total return > {PASS_RETURN_FLOOR * 100:.2f}% AND max drawdown "
        f"<= {PASS_DD_CEILING * 100:.1f}%.",
        "",
        "Caveats: idealized fills (0 fees/slippage), breaker + earnings not "
        "modeled — same as all harness results. Backtest-Steady exits via the "
        "ATR trailing stop; live Steady still trades the 5% TP until go-live. "
        "No winner is auto-picked; the verdict is written by hand.",
        "",
        "| " + " | ".join(COLUMNS) + " |",
        "|" + "|".join("---" for _ in COLUMNS) + "|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(c, r[c]) for c in COLUMNS) + " |")
    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    base = build_run_config("steady", START, END, "claude")
    # Belt-and-suspenders: the sweep window must not touch a sealed slab, even
    # though this script bypasses the CLI. SystemExit if it ever does.
    _check_oos(base, allow_sealed=False)

    # ── Canary: one combo, one month — the gate for the whole grid ──────
    canary_rc = _combo_config(
        build_run_config("steady", CANARY_START, CANARY_END, "claude"),
        STAGE1_EXIT_GRID["trailing_atr_mult"][0],
        STAGE1_EXIT_GRID["atr_period"][0],
        STAGE1_EXIT_GRID["stop_loss"][0])
    _check_oos(canary_rc, allow_sealed=False)
    print("canary: one-month run, expecting 0 API calls...", file=sys.stderr)
    _assert_free(run_backtest(canary_rc), "canary")
    print("canary clean: 0 calls, $0.00 — grid is safe to run", file=sys.stderr)

    # ── The grid ─────────────────────────────────────────────────────────
    rows = []
    combos = list(itertools.product(STAGE1_EXIT_GRID["trailing_atr_mult"],
                                    STAGE1_EXIT_GRID["atr_period"],
                                    STAGE1_EXIT_GRID["stop_loss"]))
    for i, (mult, period, sl) in enumerate(combos, 1):
        label = f"mult={mult} period={period} sl={sl}"
        print(f"[{i:2d}/{len(combos)}] {label}", file=sys.stderr)
        res = run_backtest(_combo_config(base, mult, period, sl))
        _assert_free(res, label)          # every combo re-proves $0
        rows.append(_row(mult, period, sl, compute_metrics(res)))

    rows.sort(key=lambda r: (r["total_return"] is not None,
                             r["total_return"] or 0.0), reverse=True)
    _write_outputs(rows)

    print(f"\nwrote {OUT_MD} and {OUT_CSV}; top 5 by total return:", file=sys.stderr)
    header = " | ".join(COLUMNS)
    print(header, file=sys.stderr)
    for r in rows[:5]:
        print(" | ".join(_fmt(c, r[c]) for c in COLUMNS), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
