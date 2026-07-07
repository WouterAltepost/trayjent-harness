"""The one paid out-of-sample read for the Steady grower (redesign Step 1).

This is the deliberate, one-shot, sealed validation read described in
`briefs/steady-redesign/step1_implementation_brief.md` §"The one paid
out-of-sample read". It runs ONE locked grower config over the sealed
2016-2019 window (`config.OOS_STEADY_2016`), which was never scored, so this
run PAYS to score ~587 decision points (~$21 at the measured $0.036/call, far
under the $250 per-run ceiling). Then it runs the validated PRESERVER config
over the SAME window for FREE: the scoring prompt depends only on the indicator
signals + market context + strategy NAME (all identical), never on the exit /
sizing / cash dials, so every preserver call is a cache hit the grower just
populated. That gives grower vs preserver vs SPY on the same 2016-2019 regime
for the single grower spend.

After this read the window is consumed. There is no second sealed read for a
tuned Steady, ever.

The CLI cannot inject the sweep's exit + concentration overrides, so this
script mirrors `scripts/sweep_steady.py`'s `dataclasses.replace` machinery to
build the two configs. Unlike the sweep this run is PAID (fresh window), so
there is no zero-spend assertion on the grower — only the $250 ceiling as the
backstop. The preserver DOES assert $0 (any spend there means a cache-key
drift bug, not an expected cost).

Safety gate: the paid grower scoring only fires with the explicit
`--i-understand-this-spends` flag. Without it the script is a dry run: it
prints the calendar-only decision count, the cost projection, and both locked
configs, then exits having spent nothing. Run from the harness root:

    export ANTHROPIC_API_KEY=...
    python3.13 scripts/oos_steady_grower.py                        # dry run
    python3.13 scripts/oos_steady_grower.py --i-understand-this-spends  # PAID

Writes results/steady_grower_oos.json, results/steady_preserver_oos.json, and
results/steady_grower_oos.md (results/ is gitignored). The Verdict in the .md
is written by hand, not by this script.
"""
import argparse
import dataclasses
import os
import sys
from datetime import datetime, timezone

_HARNESS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HARNESS_ROOT not in sys.path:
    sys.path.insert(0, _HARNESS_ROOT)

import config
from runner.configs import build_run_config
from runner.metrics import compute_metrics
from runner.output import build_output, to_json
from runner.run import run_backtest
from runner.schedule import decision_points

# ── The sealed window (locked; assert we never point elsewhere) ──────────
START, END = config.OOS_STEADY_2016            # ("2016-01-01", "2019-12-31")

# ── Locked grower config (steady_redesign memory + Stage 1 sweep pick) ───
# Exit: trailing_atr_mult 2.5, atr_period 22, hard-stop floor 0.03. Stage 1
# concentration: base_pct_per_score 0.08, multiplier_cap 4, cash_safety 0.50
# (set in BOTH the strategy dict and sizing_config so they cannot disagree).
GROWER_STRATEGY = {
    "use_trailing_stop": True,
    "trailing_atr_mult": 2.5,
    "atr_period": 22,
    "stop_loss": 0.03,
    "cash_safety_pct": 0.50,
}
GROWER_SIZING = {
    "base_pct_per_score": 0.08,
    "multiplier_cap": 4,
    "cash_safety_pct": 0.50,
}

# ── Validated preserver config (Stage E baseline, the thing to beat) ─────
# 5% take-profit / 3% stop, NO trailing (use_trailing_stop False routes the
# runner's exit pass into evaluate_price_exit), original concentration
# (default POSITION_SIZING: base_pct 0.05, cash_safety 0.80, multiplier_cap 4).
PRESERVER_STRATEGY = {
    "use_trailing_stop": False,
    "take_profit": 0.05,
    "stop_loss": 0.03,
    "cash_safety_pct": 0.80,
}
# sizing = frozen live mirror POSITION_SIZING, unchanged (base_pct 0.05,
# cash_safety 0.80, multiplier_cap 4) — the original preserver concentration.

OUT_GROWER_JSON = os.path.join(config.HARNESS_ROOT, "results", "steady_grower_oos.json")
OUT_PRESERVER_JSON = os.path.join(config.HARNESS_ROOT, "results", "steady_preserver_oos.json")
OUT_MD = os.path.join(config.HARNESS_ROOT, "results", "steady_grower_oos.md")


def _grower_config(base):
    strategy = {**base.strategy, **GROWER_STRATEGY}
    sizing = {**config.POSITION_SIZING, **GROWER_SIZING}
    return dataclasses.replace(base, strategy=strategy, sizing_config=sizing)


def _preserver_config(base):
    strategy = {**base.strategy, **PRESERVER_STRATEGY}
    sizing = {**config.POSITION_SIZING}
    return dataclasses.replace(base, strategy=strategy, sizing_config=sizing)


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
    return ("| Metric | Grower | Preserver | SPY |\n"
            "|---|---|---|---|\n" + body)


def _write_md(g_res, g_met, p_res, p_met):
    gt = g_res.cost_tracker
    g_cost = (f"**PAID run.** spent ${gt.spent_usd:.4f} over {gt.n_calls} Claude "
              f"calls (ceiling ${gt.ceiling_usd:.0f}, halted={g_res.halted_on_cost}).")
    pt = p_res.cost_tracker
    p_cost = (f"**FREE run** (cache hits from the grower). spent ${pt.spent_usd:.4f} "
              f"over {pt.n_calls} calls — asserted $0.")
    header = f"""# Steady grower — the one paid OOS read (2016-2019 sealed)

Window {START}..{END} (`config.OOS_STEADY_2016`), effective clean window after
the 300-bar warm-up starts ~2017-09. Mode claude, `--allow-sealed` semantics.
Read once, then consumed forever.

Grower exit 2.5 / 22 / 0.03, concentration base_pct 0.08 / mult_cap 4 /
cash_safety 0.50. Preserver 5% TP / 3% SL, no trailing, base_pct 0.05 /
cash_safety 0.80.

Caveats: idealized fills (0 fees/slippage), breaker + earnings not modeled.
This window has NO COVID-speed crash (only the late-2018 selloff), so the
grower's behavior in a true panic is still untested versus the preserver's
validated 4.97% COVID drawdown.

## Head-to-head (2016-2019)

{_comparison_md(g_met, p_met)}

{_metric_block_md("Grower — steady (claude)", g_met, g_cost)}
{_metric_block_md("Preserver — steady (claude)", p_met, p_cost)}
## Verdict

_(hand-written)_
"""
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w") as f:
        f.write(header)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="oos_steady_grower")
    p.add_argument("--i-understand-this-spends", action="store_true",
                   help="actually run the PAID grower scoring (~$21). Without "
                        "it this is a dry run that spends nothing.")
    args = p.parse_args(argv)

    base = build_run_config("steady", START, END, "claude")

    # Belt-and-suspenders: this script must NEVER point at anything but the one
    # reserved 2016-2019 sealed slab. Guard against a future edit repointing it.
    assert (base.start.date().isoformat() == START
            and END == config.OOS_STEADY_2016[1]), \
        f"window drift: expected {config.OOS_STEADY_2016}, got {base.start}..{base.end}"

    # Calendar-only decision count (pure parquet, zero spend) — the PF-S1-7 /
    # Stage E discipline: confirm the count and cost BEFORE spending.
    marks = decision_points(base)
    n = len(marks)
    proj = n * 0.036
    print(f"Sealed window {START}..{END} (config.OOS_STEADY_2016)", file=sys.stderr)
    print(f"decision points (calendar-only, == Claude calls): {n}", file=sys.stderr)
    print(f"effective window: {marks[0]} .. {marks[-1]}", file=sys.stderr)
    print(f"projected grower cost @ $0.036/call: ${proj:.2f}  "
          f"(ceiling ${config.COST_CEILING_USD:.0f}, {config.COST_CEILING_USD / proj:.1f}x headroom)",
          file=sys.stderr)
    print(f"grower config: strategy={{**STEADY, {GROWER_STRATEGY}}}, "
          f"sizing={{**POSITION_SIZING, {GROWER_SIZING}}}", file=sys.stderr)
    print(f"preserver config: strategy={{**STEADY, {PRESERVER_STRATEGY}}}, "
          f"sizing=POSITION_SIZING (base_pct 0.05, cash_safety 0.80)", file=sys.stderr)

    if not args.i_understand_this_spends:
        print("\nDRY RUN — no spend. Re-run with --i-understand-this-spends to "
              "execute the paid grower read.", file=sys.stderr)
        return 0

    # Shared on-disk cache so the preserver reads exactly what the grower wrote.
    from scoring.cache import ScoringCache
    cache = ScoringCache(os.path.join(config.CACHE_DIR, "scoring.db"))

    # ── Grower: the one PAID run ─────────────────────────────────────────
    print("\n[1/2] grower — PAID run over 2016-2019...", file=sys.stderr)
    g_res = run_backtest(_grower_config(base), cache=cache)
    gt = g_res.cost_tracker
    print(f"grower done: spent ${gt.spent_usd:.4f}, {gt.n_calls} calls, "
          f"halted_on_cost={g_res.halted_on_cost}", file=sys.stderr)
    if g_res.halted_on_cost:
        sys.exit("ABORT: grower halted on the cost ceiling — investigate before "
                 "trusting any result.")

    # ── Preserver: same window, must be FREE (cache hits) ────────────────
    print("\n[2/2] preserver — FREE run over 2016-2019 (expecting 0 calls)...",
          file=sys.stderr)
    p_res = run_backtest(_preserver_config(base), cache=cache)
    pt = p_res.cost_tracker
    if pt.n_calls != 0 or pt.spent_usd != 0.0:
        sys.exit(
            f"ABORT: preserver was NOT free — n_calls={pt.n_calls}, "
            f"spent=${pt.spent_usd:.4f}. The preserver must hit the grower's "
            f"cache (scoring is holdings-independent). A miss means a cache-key "
            f"drift bug; do not trust the comparison."
        )
    print(f"preserver done: $0.00, 0 calls — free as expected", file=sys.stderr)

    # ── Metrics + artifacts ──────────────────────────────────────────────
    g_met = compute_metrics(g_res)
    p_met = compute_metrics(p_res)
    generated_at = datetime.now(timezone.utc).isoformat()
    os.makedirs(os.path.dirname(OUT_GROWER_JSON), exist_ok=True)
    with open(OUT_GROWER_JSON, "w") as f:
        f.write(to_json(build_output(g_res, g_met, generated_at=generated_at)) + "\n")
    with open(OUT_PRESERVER_JSON, "w") as f:
        f.write(to_json(build_output(p_res, p_met, generated_at=generated_at)) + "\n")
    _write_md(g_res, g_met, p_res, p_met)

    print(f"\nwrote:\n  {OUT_GROWER_JSON}\n  {OUT_PRESERVER_JSON}\n  {OUT_MD}",
          file=sys.stderr)
    print(f"total spend this session: ${gt.spent_usd:.4f}", file=sys.stderr)
    print("NEXT: back up cache/scoring.db off-machine, then hand-write the "
          "Verdict in the .md.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
