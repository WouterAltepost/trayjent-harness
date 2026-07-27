"""The FREE crash-tail diagnostic: the grower config over the consumed
2020-2021 COVID window (readiness review §5 / gate G-1).

This is a DIAGNOSTIC, not a validation. The 2020-2021 slab
(`config.OOS_STEADY`) was consumed by the Stage E preserver read — its outcome
is known and already shaped decisions — so a GOOD grower result here is NOT
admissible as sealed evidence. The asymmetry that makes this worth running: a
BAD result is fully admissible as a kill. If the trailing-stop design blows
through the 20% backtest line in a COVID-speed gap-down, the grower's
crash-tail question is answered and Option C dies for $0.

Why it is free: scoring is holdings- and dial-independent (none of the exit /
sizing / cash dials enter the prompt — proven empirically by the Stage 1 sweep
and by the 2016-2019 grower->preserver free leg), and the Stage E OOS read
cached all 505 scoring calls for this exact window. Entries are identical
between configs; only the exit/sizing replay differs. The one alignment
requirement is the run start: previous_score threading (L6) makes prompts
start-dependent, so this run starts 2020-01-01 exactly like Stage E.

$0 is ENFORCED fail-closed, not asserted after the fact: the cached scorer is
built with a live_scorer that REFUSES — a cache miss raises immediately,
before any client construction or API call. No ANTHROPIC_API_KEY is needed.
(This is deliberately stronger than `oos_steady_grower.py`'s post-hoc $0
assert, which could only detect drift after spending on the misses.)

Two legs, both free:
  1. PRESERVER control — must reproduce the published Stage E OOS numbers
     exactly (results/phase6_stageE_summary.md). A mismatch means the replay
     is not byte-aligned with Stage E (code drift since), and the grower leg
     cannot be trusted -> hard abort.
  2. GROWER diagnostic — the frozen §8 config (trailing 2.5 / 22 / 0.03,
     base_pct 0.08, mult_cap 4, cash_safety 0.50) through the COVID crash.

Run from the harness root (his Mac; needs the local parquet + scoring.db):

    python3.13 scripts/diag_steady_grower_covid.py      # dry run
    python3.13 scripts/diag_steady_grower_covid.py \
        --this-is-a-diagnostic-not-a-validation         # the read

Writes results/steady_grower_covid_diag.json,
results/steady_preserver_covid_control.json, and
results/steady_grower_covid_diag.md (results/ is gitignored). The Verdict in
the .md is written by hand, not by this script.
"""
import argparse
import dataclasses
import hashlib
import importlib.util
import os
import sys
from datetime import datetime, timezone

_HARNESS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HARNESS_ROOT not in sys.path:
    sys.path.insert(0, _HARNESS_ROOT)

import config
from harness.reuse import SCORING_MODEL
from runner.configs import build_run_config
from runner.metrics import compute_metrics
from runner.output import build_output, to_json
from runner.run import run_backtest, _score_claude
from runner.schedule import decision_points
from scoring.cache import ScoringCache
from scoring.cached_scorer import make_cached_scorer
from scoring.cost import CostTracker

# ── The consumed window (locked; assert we never point elsewhere) ────────
START, END = config.OOS_STEADY                 # ("2020-01-01", "2021-12-31")
# Stage E cached exactly this many scoring calls for the window; anything else
# means the schedule no longer matches the cache-populating run.
STAGE_E_N_POINTS = 505

# ── Frozen configs (readiness review §8) — loaded from the OOS script, NOT
# copy-pasted (brief Part 1 item 2): oos_steady_grower.py is the single source
# of the locked dicts, so the two scripts cannot drift apart silently.
_OOS_SPEC = importlib.util.spec_from_file_location(
    "oos_steady_grower",
    os.path.join(_HARNESS_ROOT, "scripts", "oos_steady_grower.py"))
_OOS = importlib.util.module_from_spec(_OOS_SPEC)
_OOS_SPEC.loader.exec_module(_OOS)
GROWER_STRATEGY = _OOS.GROWER_STRATEGY
GROWER_SIZING = _OOS.GROWER_SIZING
PRESERVER_STRATEGY = _OOS.PRESERVER_STRATEGY
# preserver sizing = frozen live mirror POSITION_SIZING, unchanged.

# ── Stage E published OOS numbers (results/phase6_stageE_summary.md) ─────
# The control leg must reproduce these to the summary's own rounding, or the
# replay is not the run the readiness review cites. Fractions.
STAGE_E_CONTROL = {
    "total_return": (0.1177, 5e-5),
    "cagr": (0.0573, 5e-5),
    "sharpe": (1.145, 5e-4),
    "max_drawdown": (0.0497, 5e-5),
    "beta_spy": (0.077, 5e-4),
    "profit_factor": (1.532, 5e-4),
}
STAGE_E_TRADES = 174

# ── The G-1 line (readiness review §9): grower backtest maxDD must be under
# the 20% ceiling pre-registered by the S1-10 bar; 1.5-2x live inflation is
# reported alongside, not applied to the pass line.
DD_CEILING = 0.20

OUT_GROWER_JSON = os.path.join(config.HARNESS_ROOT, "results", "steady_grower_covid_diag.json")
OUT_CONTROL_JSON = os.path.join(config.HARNESS_ROOT, "results", "steady_preserver_covid_control.json")
OUT_MD = os.path.join(config.HARNESS_ROOT, "results", "steady_grower_covid_diag.md")


def _grower_config(base):
    strategy = {**base.strategy, **GROWER_STRATEGY}
    sizing = {**config.POSITION_SIZING, **GROWER_SIZING}
    return dataclasses.replace(base, strategy=strategy, sizing_config=sizing)


def _preserver_config(base):
    strategy = {**base.strategy, **PRESERVER_STRATEGY}
    sizing = {**config.POSITION_SIZING}
    return dataclasses.replace(base, strategy=strategy, sizing_config=sizing)


def _make_cached_only_score_batch(cache, label):
    """A score_batch that can only ever hit the cache. A miss raises before
    any spend (there is no client to call); the tracker exists purely to
    prove n_calls stayed 0. Returns (score_batch, tracker, hit_counter)."""
    tracker = CostTracker(config.COST_CEILING_USD, config.PRICES, SCORING_MODEL)

    def _refuse(prompt):
        sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        raise SystemExit(
            f"ABORT [{label}]: cache MISS after {counter['hits']} hits — the "
            f"diagnostic is not free. prompt_sha={sha}. $0 spent (fail-closed "
            f"refusal, no API call was made). Scoring inputs no longer match "
            f"the Stage E cache (PROMPT_VERSION, watchlist, SCORING_MODEL, or "
            f"the 2020-01-01 previous_score chain). Fix the drift; do not "
            f"switch to a paid re-score — the window is consumed."
        )

    scorer = make_cached_scorer(
        cache, _refuse, tracker,
        model=SCORING_MODEL, prompt_version=config.PROMPT_VERSION,
        strategy="steady",
    )
    counter = {"hits": 0}

    def counting_scorer(prompt):
        content = scorer(prompt)      # a miss never returns (refusal above)
        counter["hits"] += 1
        return content

    def score_batch(signals, strategy, market_context):
        return _score_claude(signals, strategy, market_context, counting_scorer)

    return score_batch, tracker, counter


def _assert_free(tracker, label):
    """Belt-and-suspenders: unreachable if the refusal gate works, but a $0
    claim in a cited result gets asserted, not narrated."""
    if tracker.n_calls != 0 or tracker.spent_usd != 0.0:
        sys.exit(f"ABORT [{label}]: tracker shows spend on a cached-only run "
                 f"(n_calls={tracker.n_calls}, ${tracker.spent_usd:.4f}) — "
                 f"the refusal gate was bypassed. Do not trust this run.")


def _assert_control_matches_stage_e(met):
    """The preserver replay must BE the Stage E OOS run. Any drift means the
    shared code path changed since Stage E and the grower leg inherits an
    unknown divergence -> the diagnostic aborts rather than reports."""
    problems = []
    for key, (want, tol) in STAGE_E_CONTROL.items():
        got = met[key]
        if got is None or abs(got - want) > tol:
            problems.append(f"{key}: got {got!r}, Stage E published {want}")
    if met["trades"]["total"] != STAGE_E_TRADES:
        problems.append(f"trades: got {met['trades']['total']}, Stage E published {STAGE_E_TRADES}")
    if problems:
        sys.exit(
            "ABORT [control]: preserver replay does NOT reproduce the Stage E "
            "OOS numbers — the replay is not byte-aligned with the run the "
            "readiness review cites (code drift in shared exit/sizing/metrics "
            "since Stage E?). Grower leg untrustworthy. Diffs:\n  "
            + "\n  ".join(problems)
        )


def _pct(x):
    return "—" if x is None else f"{x * 100:.2f}%"


def _num(x):
    return "—" if x is None else f"{x:.3f}"


def _metric_block_md(title, met, cost_line):
    b = met["benchmark_spy"] or {}
    exit_reasons = met["trades"]["by_exit_reason"]
    reason_lines = "\n".join(
        f"| &nbsp;&nbsp;{r} | {n} | |" for r, n in sorted(exit_reasons.items())
    )
    return f"""## {title}

{cost_line}

| Metric | Strategy | SPY benchmark |
|---|---|---|
| Total return | {_pct(met['total_return'])} | {_pct(b.get('total_return'))} |
| CAGR | {_pct(met['cagr'])} | {_pct(b.get('cagr'))} |
| Sharpe | {_num(met['sharpe'])} | {_num(b.get('sharpe'))} |
| Max drawdown | {_pct(met['max_drawdown'])} | {_pct(b.get('max_drawdown'))} |
| Profit factor | {_num(met['profit_factor'])} | |
| Beta vs SPY | {_num(met['beta_spy'])} | |
| Win rate | {_pct(met['win_rate'])} | |
| Closed trades | {met['trades']['total']} | |
{reason_lines}
"""


def _comparison_md(g_met, p_met):
    gb = g_met["benchmark_spy"] or {}
    rows = [
        ("Total return", _pct(g_met["total_return"]), _pct(p_met["total_return"]), _pct(gb.get("total_return"))),
        ("CAGR", _pct(g_met["cagr"]), _pct(p_met["cagr"]), _pct(gb.get("cagr"))),
        ("Sharpe", _num(g_met["sharpe"]), _num(p_met["sharpe"]), _num(gb.get("sharpe"))),
        ("Max drawdown", _pct(g_met["max_drawdown"]), _pct(p_met["max_drawdown"]), _pct(gb.get("max_drawdown"))),
        ("Beta vs SPY", _num(g_met["beta_spy"]), _num(p_met["beta_spy"]), "1.000"),
        ("Profit factor", _num(g_met["profit_factor"]), _num(p_met["profit_factor"]), "—"),
        ("Closed trades", str(g_met["trades"]["total"]), str(p_met["trades"]["total"]), "—"),
    ]
    body = "\n".join(f"| {m} | {g} | {p} | {s} |" for m, g, p, s in rows)
    return ("| Metric | Grower | Preserver (Stage E control) | SPY |\n"
            "|---|---|---|---|\n" + body)


def _write_md(g_met, p_met, n_hits_g, n_hits_p):
    g_dd = g_met["max_drawdown"]
    under_line = g_dd is not None and g_dd <= DD_CEILING
    g1_line = (
        f"**G-1 mechanical check:** grower max drawdown {_pct(g_dd)} vs the "
        f"pre-registered {DD_CEILING * 100:.0f}% backtest line — "
        f"**{'UNDER the line' if under_line else 'THROUGH the line (kill condition)'}**. "
        f"At the readiness review's 1.5-2x live inflation, {_pct(g_dd)} implies "
        f"roughly {_pct(g_dd * 1.5)}-{_pct(g_dd * 2.0)} live."
        if g_dd is not None else
        "**G-1 mechanical check:** grower max drawdown unavailable — no check."
    )
    header = f"""# Steady grower — COVID crash-tail DIAGNOSTIC (2020-2021, consumed window)

**This is a diagnostic, not a validation.** The 2020-2021 slab was consumed by
the Stage E preserver read; its outcome was known before this run. A good
result here is NOT sealed evidence for the grower. A bad result IS admissible
as a kill (readiness review §5). Window {START}..{END} (`config.OOS_STEADY`),
deliberate consumed-window acknowledgment flag, run start 2020-01-01 for
previous_score cache alignment. Entries identical between configs by
construction (shared cache).

**$0 enforced fail-closed:** cached-only scorer, a miss aborts before any API
call. Grower leg: {n_hits_g} scoring prompts, all cache hits. Preserver
control: {n_hits_p} prompts, all hits. No API key loaded.

**Control leg:** the preserver replay reproduced the published Stage E OOS
numbers exactly (total return 11.77%, CAGR 5.73%, Sharpe 1.145, max drawdown
4.97%, beta 0.077, PF 1.532, 174 trades) — asserted in-script, so the grower
leg runs on a replay byte-aligned with the run the readiness review cites.

Grower exit 2.5 / 22 / 0.03, concentration base_pct 0.08 / mult_cap 4 /
cash_safety 0.50. Preserver 5% TP / 3% SL, base_pct 0.05 / cash_safety 0.80.
Caveats: idealized fills (0 fees/slippage), breaker + earnings not modeled,
decisions on the completed daily close — same as all harness results.

{g1_line}

## Head-to-head through COVID (2020-2021)

{_comparison_md(g_met, p_met)}

{_metric_block_md("Grower — steady (claude), diagnostic leg", g_met, f"**FREE run** — {n_hits_g} cache hits, $0 (fail-closed).")}
{_metric_block_md("Preserver — steady (claude), Stage E control leg", p_met, f"**FREE run** — {n_hits_p} cache hits, $0 (fail-closed). Reproduces Stage E.")}
## Verdict

_(hand-written)_
"""
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w") as f:
        f.write(header)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="diag_steady_grower_covid")
    p.add_argument("--this-is-a-diagnostic-not-a-validation",
                   dest="acknowledged", action="store_true",
                   help="deliberately replay the CONSUMED 2020-2021 window "
                        "(still $0). The flag name is the acknowledgment: a "
                        "good result here is never sealed evidence. Without "
                        "it this is a dry run: counts + configs, no backtest.")
    args = p.parse_args(argv)

    base = build_run_config("steady", START, END, "claude")

    # Belt-and-suspenders: this script must NEVER point at anything but the
    # consumed 2020-2021 slab. Guard against a future edit repointing it.
    assert (base.start.date().isoformat() == START
            and END == config.OOS_STEADY[1]), \
        f"window drift: expected {config.OOS_STEADY}, got {base.start}..{base.end}"

    # Pre-flight: the schedule must match the cache-populating Stage E run.
    marks = decision_points(base)
    n = len(marks)
    print(f"Consumed window {START}..{END} (config.OOS_STEADY)", file=sys.stderr)
    print(f"decision points (calendar-only): {n}  (Stage E cached {STAGE_E_N_POINTS})",
          file=sys.stderr)
    print(f"effective window: {marks[0]} .. {marks[-1]}", file=sys.stderr)
    print(f"grower config: strategy={{**STEADY, {GROWER_STRATEGY}}}, "
          f"sizing={{**POSITION_SIZING, {GROWER_SIZING}}}", file=sys.stderr)
    print(f"preserver control: strategy={{**STEADY, {PRESERVER_STRATEGY}}}, "
          f"sizing=POSITION_SIZING (base_pct 0.05, cash_safety 0.80)", file=sys.stderr)
    if n != STAGE_E_N_POINTS:
        sys.exit(f"ABORT: {n} decision points != Stage E's {STAGE_E_N_POINTS} — "
                 f"the schedule no longer matches the cache-populating run "
                 f"(data re-pulled? schedule change?). A run would only miss.")

    if not args.acknowledged:
        print("\nDRY RUN — nothing executed. Re-run with "
              "--this-is-a-diagnostic-not-a-validation for the deliberate "
              "consumed-window replay (still $0, fail-closed).",
              file=sys.stderr)
        return 0

    cache = ScoringCache(os.path.join(config.CACHE_DIR, "scoring.db"))

    # ── Leg 1: preserver CONTROL — must reproduce Stage E exactly ────────
    print("\n[1/2] preserver control — replaying the Stage E OOS run...",
          file=sys.stderr)
    p_batch, p_tracker, p_counter = _make_cached_only_score_batch(cache, "control")
    p_res = run_backtest(_preserver_config(base), score_batch=p_batch)
    _assert_free(p_tracker, "control")
    p_met = compute_metrics(p_res)
    _assert_control_matches_stage_e(p_met)
    print(f"control clean: {p_counter['hits']} hits, $0.00, reproduces Stage E "
          f"(ret {_pct(p_met['total_return'])}, maxDD {_pct(p_met['max_drawdown'])}, "
          f"{p_met['trades']['total']} trades)", file=sys.stderr)

    # ── Leg 2: grower DIAGNOSTIC through the COVID crash ─────────────────
    print("\n[2/2] grower diagnostic — trailing exit through COVID...",
          file=sys.stderr)
    g_batch, g_tracker, g_counter = _make_cached_only_score_batch(cache, "grower")
    g_res = run_backtest(_grower_config(base), score_batch=g_batch)
    _assert_free(g_tracker, "grower")
    g_met = compute_metrics(g_res)
    print(f"grower done: {g_counter['hits']} hits, $0.00", file=sys.stderr)

    # ── Artifacts ────────────────────────────────────────────────────────
    generated_at = datetime.now(timezone.utc).isoformat()
    os.makedirs(os.path.dirname(OUT_GROWER_JSON), exist_ok=True)
    with open(OUT_GROWER_JSON, "w") as f:
        f.write(to_json(build_output(g_res, g_met, generated_at=generated_at)) + "\n")
    with open(OUT_CONTROL_JSON, "w") as f:
        f.write(to_json(build_output(p_res, p_met, generated_at=generated_at)) + "\n")
    _write_md(g_met, p_met, g_counter["hits"], p_counter["hits"])

    g_dd = g_met["max_drawdown"]
    print(f"\nwrote:\n  {OUT_GROWER_JSON}\n  {OUT_CONTROL_JSON}\n  {OUT_MD}",
          file=sys.stderr)
    print(f"total spend: $0.00 (enforced)", file=sys.stderr)
    print(f"G-1 mechanical: grower maxDD {_pct(g_dd)} vs {DD_CEILING * 100:.0f}% line -> "
          f"{'UNDER' if g_dd is not None and g_dd <= DD_CEILING else 'THROUGH (kill)'}",
          file=sys.stderr)
    print("NEXT: hand-write the Verdict in the .md; fold the reading into the "
          "readiness review (G-1).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
