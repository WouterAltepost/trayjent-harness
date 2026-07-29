"""Calm (shortvol sleeve) TUNING run: 64-cell grid, robust selection,
baselines, stress table, equity plot.

    python scripts/backtest_shortvol_tuning.py

Reads ONLY the tuning window (config.CALM_TUNING, 2011-10-04..2021-12-31).
The sealed OOS (config.OOS_SHORTVOL, 2022-01-01 onward) is never loaded —
build_inputs fails closed on it, and this script never passes allow_sealed.

LEVERAGE APPROXIMATION on every output: pre-2018-02-27 SVXY daily returns
are scaled by 0.5 to approximate a continuous -0.5x instrument (config
SVXY_LEVERAGE_CHANGE_DATE). The post-flip sub-window check uses no scaling.

Outputs (results/ is local-only, not tracked):
    results/calm_tuning_grid.csv       all 64 cells, full precision
    results/calm_tuning_summary.md     chosen config, baselines, stress, deltas
    results/calm_tuning_equity.png     chosen vs baselines, log scale
"""
import os
import sys

_HARNESS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HARNESS_ROOT not in sys.path:
    sys.path.insert(0, _HARNESS_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config
from calm import backtest as bt

RESULTS_DIR = os.path.join(_HARNESS_ROOT, "results")

# Validated categorical palette, first three slots (dataviz reference, light
# mode, all-pairs clean): strategy / SVXY B&H / SPY B&H. Text wears ink
# tokens, never series color.
C_STRAT, C_SVXY, C_SPY = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID_C, SURFACE = "#0b0b0b", "#52514e", "#e3e2de", "#fcfcfb"

APPROX_LABEL = ("SVXY leverage-approximated: pre-2018-02-27 daily returns x 0.5 "
                "(continuous -0.5x approximation). Same-day-close fills, "
                f"{config.CALM_COST_BPS_PER_SIDE:g} bps/side.")


def _pct(x, digits=1):
    return "-" if x is None else f"{100.0 * x:.{digits}f}%"


def _num(x, digits=2):
    return "-" if x is None else f"{x:.{digits}f}"


def _kstr(k):
    return "none" if k is None else f"{k:g}"


def _cell_row(c, chosen):
    m = c["metrics"]
    mark = "*" if c is chosen else " "
    return (f"{mark} {c['smoothing']:<4} {c['entry']:<6g} {c['exit']:<5g} "
            f"{_kstr(c['k']):<5} {_pct(m['cagr']):>7} {_num(m['sharpe']):>6} "
            f"{_pct(m['max_drawdown']):>7} {_pct(m['daily_win']):>6} "
            f"{_pct(m['pos_weeks'], 0):>5} {_pct(m['pos_months'], 0):>5} "
            f"{_pct(m['time_in_market'], 0):>5} {m['round_trips']:>4} "
            f"{_pct(m['worst_day']):>7} {_num(c['nbhd_score']):>7}")


def print_grid(cells, chosen):
    print(f"\n{'=' * 100}")
    print(f"TUNING GRID — 64 cells, {config.CALM_TUNING[0]}..{config.CALM_TUNING[1]}")
    print(APPROX_LABEL)
    print("nbhd = 3x3 ENTRY x K neighborhood Sharpe (mean - std); "
          "* = chosen (robustness over peak;")
    print("only interior cells with a FULL 3x3 neighborhood are eligible — "
          "edge/corner stats shown are clipped)")
    print(f"{'=' * 100}")
    hdr = (f"  {'sm':<4} {'entry':<6} {'exit':<5} {'K':<5} {'CAGR':>7} "
           f"{'Sharpe':>6} {'maxDD':>7} {'dWin':>6} {'+wk':>5} {'+mo':>5} "
           f"{'TIM':>5} {'RT':>4} {'worst':>7} {'nbhd':>7}")
    for smoothing in bt.SMOOTHING_GRID:
        for exit_ in bt.EXIT_GRID:
            print(f"\n-- smoothing={smoothing}  EXIT={exit_:g} " + "-" * 72)
            print(hdr)
            for c in cells:
                if c["smoothing"] == smoothing and c["exit"] == exit_:
                    print(_cell_row(c, chosen))


def summarize(res, timestamps, label):
    m = bt.compute_cell_metrics(res, timestamps)
    return (f"{label:<34} CAGR {_pct(m['cagr']):>7}  Sharpe {_num(m['sharpe']):>5}  "
            f"maxDD {_pct(m['max_drawdown']):>6}  worst day {_pct(m['worst_day']):>6}  "
            f"TIM {_pct(m['time_in_market'], 0):>4}")


def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    start, end = config.CALM_TUNING
    df = bt.build_inputs(start, end)          # fails closed on the sealed OOS
    timestamps = df["timestamp"].tolist()
    print(f"Tuning window: {len(df)} sessions {start}..{end} "
          f"(sealed OOS from {config.OOS_SHORTVOL[0]} untouched)")

    # ── Grid + robust selection ─────────────────────────────────────────
    cells = bt.run_grid(df)
    chosen = bt.select_robust(cells)
    print_grid(cells, chosen)

    cm = chosen["metrics"]
    best = max(cells, key=lambda c: c["metrics"]["sharpe"])
    print(f"\nCHOSEN (robustness over peak): ENTRY {chosen['entry']:g}  "
          f"EXIT {chosen['exit']:g}  K {_kstr(chosen['k'])}  smoothing {chosen['smoothing']}")
    print(f"  own Sharpe {_num(cm['sharpe'])}  |  3x3 neighborhood: "
          f"mean {_num(chosen['nbhd_mean'])}  std {_num(chosen['nbhd_std'])}  "
          f"min {_num(chosen['nbhd_min'])}  score {_num(chosen['nbhd_score'])}")
    print(f"  best single cell for reference (NOT chosen): ENTRY {best['entry']:g}  "
          f"EXIT {best['exit']:g}  K {_kstr(best['k'])}  {best['smoothing']}  "
          f"Sharpe {_num(best['metrics']['sharpe'])}  "
          f"nbhd score {_num(best['nbhd_score'])}")

    # ── Baselines, same window, same cost model ─────────────────────────
    svxy_bh = bt.buy_and_hold(df["ret_cc"].tolist())
    spy_bh = bt.buy_and_hold(df["spy_ret"].tolist())
    print(f"\nBASELINES ({start}..{end}, one entry fill at "
          f"{config.CALM_COST_BPS_PER_SIDE:g} bps):")
    chosen_line = summarize(chosen["result"], timestamps, "Calm (chosen)")
    print(f"  {chosen_line}")
    print(f"  {summarize(svxy_bh, timestamps, 'SVXY buy-and-hold (approx -0.5x)')}")
    print(f"  {summarize(spy_bh, timestamps, 'SPY buy-and-hold')}")

    # ── Sensitivity: next-open execution for the chosen config ──────────
    no_res = bt.run_backtest(df, chosen["entry"], chosen["exit"], chosen["k"],
                             smoothing=chosen["smoothing"], mode="next_open")
    no_m = bt.compute_cell_metrics(no_res, timestamps)
    print("\nEXECUTION SENSITIVITY (chosen config, next-open fills vs same-day close):")
    print(f"  next-open: CAGR {_pct(no_m['cagr'])}  Sharpe {_num(no_m['sharpe'])}  "
          f"maxDD {_pct(no_m['max_drawdown'])}")
    print(f"  delta:     CAGR {_pct(no_m['cagr'] - cm['cagr'])}  "
          f"Sharpe {no_m['sharpe'] - cm['sharpe']:+.2f}  "
          f"maxDD {_pct(no_m['max_drawdown'] - cm['max_drawdown'])}")

    # ── Reality check: pure post-flip sub-window, NO scaling ────────────
    post = df[df["timestamp"] >= pd.Timestamp(config.SVXY_LEVERAGE_CHANGE_DATE, tz="UTC")]
    post = post.reset_index(drop=True)
    post_ts = post["timestamp"].tolist()
    post_res = bt.run_backtest(post, chosen["entry"], chosen["exit"], chosen["k"],
                               smoothing=chosen["smoothing"])
    print(f"\nPOST-FLIP REALITY CHECK ({config.SVXY_LEVERAGE_CHANGE_DATE}..{end}, "
          "native -0.5x, NO scaling):")
    print(f"  {summarize(post_res, post_ts, 'Calm (chosen), post-flip only')}")
    print(f"  {summarize(bt.buy_and_hold(post['ret_cc'].tolist()), post_ts, 'SVXY B&H, post-flip only')}")

    # ── Stress episodes vs SVXY B&H ─────────────────────────────────────
    print("\nSTRESS EPISODES (chosen config vs SVXY B&H, leverage-approximated):")
    print(f"  {'episode':<22} {'Calm P&L':>9} {'Calm DD':>8} {'B&H P&L':>9} {'B&H DD':>8}")
    stress_rows = []
    for name, ep_start, ep_end in bt.EPISODES:
        s = bt.episode_stats(chosen["result"], timestamps, ep_start, ep_end)
        b = bt.episode_stats(svxy_bh, timestamps, ep_start, ep_end)
        print(f"  {name:<22} {_pct(s['pnl']):>9} {_pct(s['max_drawdown']):>8} "
              f"{_pct(b['pnl']):>9} {_pct(b['max_drawdown']):>8}")
        stress_rows.append((name, s, b))

    # ── Persist: grid CSV + summary md ──────────────────────────────────
    rows = []
    for c in cells:
        rows.append({
            "smoothing": c["smoothing"], "entry": c["entry"], "exit": c["exit"],
            "k": _kstr(c["k"]), **c["metrics"],
            "nbhd_mean": c["nbhd_mean"], "nbhd_std": c["nbhd_std"],
            "nbhd_min": c["nbhd_min"], "nbhd_score": c["nbhd_score"],
            "chosen": c is chosen,
        })
    grid_csv = os.path.join(RESULTS_DIR, "calm_tuning_grid.csv")
    pd.DataFrame(rows).to_csv(grid_csv, index=False)

    md = [
        "# Calm tuning — chosen config",
        "",
        f"Window {start}..{end} ({len(df)} sessions). {APPROX_LABEL}",
        f"Sealed OOS from {config.OOS_SHORTVOL[0]} untouched.",
        "",
        f"**Chosen**: ENTRY {chosen['entry']:g}, EXIT {chosen['exit']:g}, "
        f"K {_kstr(chosen['k'])}, smoothing {chosen['smoothing']} — selected for "
        f"3x3 neighborhood Sharpe consistency (mean {_num(chosen['nbhd_mean'])}, "
        f"std {_num(chosen['nbhd_std'])}, min {_num(chosen['nbhd_min'])}), not peak.",
        "",
        f"- {chosen_line}",
        f"- {summarize(svxy_bh, timestamps, 'SVXY B&H (approx -0.5x)')}",
        f"- {summarize(spy_bh, timestamps, 'SPY B&H')}",
        f"- next-open delta: CAGR {_pct(no_m['cagr'] - cm['cagr'])}, "
        f"Sharpe {no_m['sharpe'] - cm['sharpe']:+.2f}",
        f"- post-flip (native -0.5x): "
        f"{summarize(post_res, post_ts, 'Calm')}",
        "",
        "| episode | Calm P&L | Calm maxDD | SVXY B&H P&L | B&H maxDD |",
        "|---|---|---|---|---|",
    ]
    for name, s, b in stress_rows:
        md.append(f"| {name} | {_pct(s['pnl'])} | {_pct(s['max_drawdown'])} "
                  f"| {_pct(b['pnl'])} | {_pct(b['max_drawdown'])} |")
    summary_md = os.path.join(RESULTS_DIR, "calm_tuning_summary.md")
    with open(summary_md, "w") as f:
        f.write("\n".join(md) + "\n")

    # ── Equity plot (chosen vs baselines, log scale) ────────────────────
    png = os.path.join(RESULTS_DIR, "calm_tuning_equity.png")
    dates = [t.tz_convert(None) for t in timestamps]
    fig, ax = plt.subplots(figsize=(11.5, 6.0), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    series = [
        (f"Calm (E {chosen['entry']:g} / X {chosen['exit']:g} / "
         f"K {_kstr(chosen['k'])} / {chosen['smoothing']})",
         chosen["result"]["equity"], C_STRAT),
        ("SVXY buy-and-hold (approx -0.5x)", svxy_bh["equity"], C_SVXY),
        ("SPY buy-and-hold", spy_bh["equity"], C_SPY),
    ]
    for label, eq, color in series:
        ax.plot(dates, eq, color=color, linewidth=1.8, label=label,
                solid_capstyle="round", zorder=3)
        ax.annotate(f" {eq[-1]:.1f}x", xy=(dates[-1], eq[-1]),
                    xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK2)
    ax.set_yscale("log")
    ticks = [0.25, 0.5, 1, 2, 4, 8, 16, 32]
    lo = min(min(eq) for _, eq, _ in series)
    hi = max(max(eq) for _, eq, _ in series)
    ticks = [t for t in ticks if lo * 0.8 <= t <= hi * 1.6]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}x" for t in ticks], color=INK2, fontsize=9)
    ax.tick_params(axis="x", colors=INK2, labelsize=9)
    ax.minorticks_off()
    ax.grid(True, which="major", color=GRID_C, linewidth=0.6, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID_C)
    ax.set_title(f"Calm tuning {start}..{end} — equity, log scale (1.0 = start)",
                 color=INK, fontsize=12, loc="left", pad=12)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
    fig.text(0.01, 0.005, APPROX_LABEL, fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(png, facecolor=SURFACE)
    plt.close(fig)

    print(f"\nSaved: {grid_csv}")
    print(f"Saved: {summary_md}")
    print(f"Saved: {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
