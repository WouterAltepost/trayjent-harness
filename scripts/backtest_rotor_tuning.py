"""Rotor (crypto momentum rotation) TUNING run: 16-cell grid, robust
selection, baselines, calendar-year stress, equity plot.

    python scripts/backtest_rotor_tuning.py

Reads ONLY the tuning window (config.ROTOR_TUNING, 2021-01-01..2024-12-31).
The sealed OOS (config.OOS_ROTOR, 2025-01-01 onward) is never loaded —
build_inputs fails closed, and this script never passes allow_sealed.

Survivorship note: the point-in-time universe includes the five delisted
series (config.ROTOR_DELISTED) during their real listing windows — this
backtest is survivorship-safe at the venue level (coins Alpaca never listed
remain invisible, stated in config).

Outputs (results/ is local-only, not tracked):
    results/rotor_tuning_grid.csv
    results/rotor_tuning_summary.md
    results/rotor_tuning_equity.png
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
from rotor import backtest as bt

RESULTS_DIR = os.path.join(_HARNESS_ROOT, "results")

# Validated categorical palette (dataviz reference, light mode), slots 1-4:
# Rotor / BTC B&H / BTC gated / EW top-5. Yellow sits under the light-surface
# contrast bar, so every line also carries a direct end label in ink.
C_ROTOR, C_BTC, C_GATED, C_EW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID_C, SURFACE = "#0b0b0b", "#52514e", "#e3e2de", "#fcfcfb"

COST_LABEL = (f"taker {config.ROTOR_ALPACA_CRYPTO_FEES['taker']:.2%} + "
              f"{config.ROTOR_SLIPPAGE_BPS:g} bps slippage per side, "
              "weekly Monday-00:00Z closes")

YEARS = [2021, 2022, 2023, 2024]


def _pct(x, digits=1):
    return "-" if x is None else f"{100.0 * x:.{digits}f}%"


def _num(x, digits=2):
    return "-" if x is None else f"{x:.{digits}f}"


def _cell_row(c, chosen):
    m = c["metrics"]
    mark = "*" if c is chosen else " "
    return (f"{mark} {c['formation']:>4} {c['n_held']:>2} {c['gate_ma']:>5} "
            f"{'on' if c['ma_filter'] else 'off':<4} {m['start']:<11} "
            f"{_pct(m['cagr']):>7} {_num(m['sharpe']):>6} "
            f"{_pct(m['max_drawdown']):>6} {_pct(m['pos_weeks'], 0):>4} "
            f"{_pct(m['pos_months'], 0):>4} {_pct(m['time_in_market'], 0):>4} "
            f"{m['avg_holdings']:>4.1f} {m['turnover']:>5.1f}x "
            f"{_pct(m['cost_drag']):>6} {_pct(m['worst_week']):>7} "
            f"{_num(c['nbhd_score']):>6}")


def summarize(m, label):
    return (f"{label:<26} CAGR {_pct(m['cagr']):>7}  Sharpe {_num(m['sharpe']):>5}  "
            f"maxDD {_pct(m['max_drawdown']):>6}  TIM {_pct(m['time_in_market'], 0):>4}  "
            f"worst wk {_pct(m['worst_week']):>6}")


def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    start, end = config.ROTOR_TUNING
    inputs = bt.build_inputs(start, end)      # fails closed on the sealed OOS
    print(f"Tuning window {start}..{end}: {len(inputs['calendar'])} UTC days, "
          f"{inputs['close'].shape[1]} coins (survivorship-safe listing map); "
          f"sealed OOS from {config.OOS_ROTOR[0]} untouched.")

    # ── Grid + robust selection ─────────────────────────────────────────
    cells = bt.run_grid(inputs)
    chosen = bt.select_robust(cells)

    print(f"\n{'=' * 108}")
    print(f"TUNING GRID — 16 cells, {COST_LABEL}")
    print("nbhd = plane Sharpe (mean - std). Neighborhood rule, adapted and "
          "stated plainly: the formation x N planes are 2x2, so a")
    print("clipped 3x3 window covers the WHOLE plane for every cell and no "
          "full-3x3 interior exists; the rule therefore selects the most")
    print("consistent (gate, filter) PLANE, then the strongest cell within "
          "it (tie-breaks: plane min — tied — then own Sharpe). * = chosen.")
    print(f"{'=' * 108}")
    hdr = (f"  {'form':>4} {'N':>2} {'gate':>5} {'filt':<4} {'start':<11} "
           f"{'CAGR':>7} {'Sharpe':>6} {'maxDD':>6} {'+wk':>4} {'+mo':>4} "
           f"{'TIM':>4} {'avgH':>4} {'turn':>6} {'drag':>6} {'worstWk':>7} "
           f"{'nbhd':>6}")
    for gate_ma in bt.GATE_GRID:
        for ma_filter in bt.FILTER_GRID:
            print(f"\n-- gate={gate_ma}d  filter={'on' if ma_filter else 'off'} "
                  + "-" * 84)
            print(hdr)
            for c in cells:
                if c["gate_ma"] == gate_ma and c["ma_filter"] == ma_filter:
                    print(_cell_row(c, chosen))

    cm = chosen["metrics"]
    best = max(cells, key=lambda c: c["metrics"]["sharpe"])
    print(f"\nCHOSEN (robustness over peak): formation {chosen['formation']}d  "
          f"N {chosen['n_held']}  gate {chosen['gate_ma']}d  "
          f"filter {'on' if chosen['ma_filter'] else 'off'}")
    print(f"  own Sharpe {_num(cm['sharpe'])}  |  plane: mean "
          f"{_num(chosen['nbhd_mean'])}  std {_num(chosen['nbhd_std'])}  "
          f"min {_num(chosen['nbhd_min'])}  score {_num(chosen['nbhd_score'])}")
    print(f"  equity curve starts {cm['start']} (gate MA {chosen['gate_ma']}d "
          "fully formed)")
    print(f"  best single cell for reference (NOT chosen unless identical): "
          f"formation {best['formation']}d N {best['n_held']} gate "
          f"{best['gate_ma']}d filter {'on' if best['ma_filter'] else 'off'} "
          f"Sharpe {_num(best['metrics']['sharpe'])}")

    # ── Baselines from the chosen cell's start, same cost model ─────────
    start_ts = chosen["result"]["start_ts"]
    bh = bt.btc_buy_hold(inputs, start_ts)
    gated = bt.btc_gated(inputs, start_ts, gate_ma=200)
    ew = bt.run_backtest(inputs, formation=chosen["formation"], n_held=5,
                         gate_ma=None, ma_filter=False, signal="volume")
    # EW baseline shares the chosen start for comparability.
    ew_from = _slice_result(ew, start_ts)
    bh_m = bt.compute_cell_metrics(bh)
    gated_m = bt.compute_cell_metrics(gated)
    ew_m = bt.compute_cell_metrics(ew_from)
    print(f"\nBASELINES (from {bt.rotor_data.bar_day(start_ts)}, same costs):")
    print(f"  {summarize(cm, 'Rotor (chosen)')}")
    print(f"  {summarize(bh_m, 'BTC buy-and-hold')}")
    print(f"  {summarize(gated_m, 'BTC gated (200d MA, weekly)')}")
    print(f"  {summarize(ew_m, 'EW top-5 by volume, weekly')}")

    # ── Calendar-year table + 2022 focus ────────────────────────────────
    print("\nCALENDAR YEARS (chosen vs BTC B&H):")
    print(f"  {'year':<6} {'Rotor ret':>10} {'Rotor DD':>9} {'cash%':>6} "
          f"{'BTC ret':>9} {'BTC DD':>8}")
    year_rows = []
    for y in YEARS:
        s = bt.year_stats(chosen["result"], y)
        b = bt.year_stats(bh, y)
        note = " (from curve start)" if y == 2021 else ""
        print(f"  {y:<6} {_pct(s['ret']):>10} {_pct(s['max_drawdown']):>9} "
              f"{_pct(s['cash_share'], 0):>6} {_pct(b['ret']):>9} "
              f"{_pct(b['max_drawdown']):>8}{note}")
        year_rows.append((y, s, b))
    s22 = next(s for y, s, _ in year_rows if y == 2022)
    b22 = next(b for y, _, b in year_rows if y == 2022)
    print(f"\n2022 BEAR FOCUS: Rotor spent {_pct(s22['cash_share'], 0)} of days "
          f"fully in cash; P&L {_pct(s22['ret'])} vs BTC {_pct(b22['ret'])}; "
          f"worst drawdown {_pct(s22['max_drawdown'])} vs BTC "
          f"{_pct(b22['max_drawdown'])}.")

    # ── Persist: grid csv + summary md ──────────────────────────────────
    rows = []
    for c in cells:
        rows.append({"formation": c["formation"], "n_held": c["n_held"],
                     "gate_ma": c["gate_ma"], "ma_filter": c["ma_filter"],
                     **c["metrics"],
                     "nbhd_mean": c["nbhd_mean"], "nbhd_std": c["nbhd_std"],
                     "nbhd_score": c["nbhd_score"], "chosen": c is chosen})
    grid_csv = os.path.join(RESULTS_DIR, "rotor_tuning_grid.csv")
    pd.DataFrame(rows).to_csv(grid_csv, index=False)

    md = [
        "# Rotor tuning — chosen config",
        "",
        f"Window {start}..{end}; curve starts {cm['start']} (gate warm-up). "
        f"{COST_LABEL}. Survivorship-safe PIT universe incl. delisted series.",
        "",
        f"**Chosen**: formation {chosen['formation']}d, N {chosen['n_held']}, "
        f"gate {chosen['gate_ma']}d, filter "
        f"{'on' if chosen['ma_filter'] else 'off'} — most consistent "
        f"(gate, filter) plane (mean {_num(chosen['nbhd_mean'])}, std "
        f"{_num(chosen['nbhd_std'])}), then strongest in-plane cell.",
        "",
        f"- {summarize(cm, 'Rotor (chosen)')}",
        f"- {summarize(bh_m, 'BTC buy-and-hold')}",
        f"- {summarize(gated_m, 'BTC gated (200d, weekly)')}",
        f"- {summarize(ew_m, 'EW top-5 by volume')}",
        "",
        "| year | Rotor ret | Rotor DD | cash% | BTC ret | BTC DD |",
        "|---|---|---|---|---|---|",
    ]
    for y, s, b in year_rows:
        md.append(f"| {y} | {_pct(s['ret'])} | {_pct(s['max_drawdown'])} | "
                  f"{_pct(s['cash_share'], 0)} | {_pct(b['ret'])} | "
                  f"{_pct(b['max_drawdown'])} |")
    with open(os.path.join(RESULTS_DIR, "rotor_tuning_summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    # ── Equity plot ─────────────────────────────────────────────────────
    png = os.path.join(RESULTS_DIR, "rotor_tuning_equity.png")
    series = [
        (f"Rotor (F{chosen['formation']}/N{chosen['n_held']}/"
         f"G{chosen['gate_ma']}/{'filt' if chosen['ma_filter'] else 'nofilt'})",
         chosen["result"], C_ROTOR),
        ("BTC buy-and-hold", bh, C_BTC),
        ("BTC gated (200d MA)", gated, C_GATED),
        ("EW top-5 by volume", ew_from, C_EW),
    ]
    fig, ax = plt.subplots(figsize=(11.5, 6.0), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for label, res, color in series:
        dates = [t.tz_convert(None) for t in res["timestamps"]]
        ax.plot(dates, res["equity"], color=color, linewidth=1.8,
                label=label, solid_capstyle="round", zorder=3)
        ax.annotate(f" {res['equity'][-1]:.2f}x", xy=(dates[-1], res["equity"][-1]),
                    xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK2)
    ax.set_yscale("log")
    ticks = [0.25, 0.5, 1, 2, 4, 8]
    lo = min(min(r["equity"]) for _, r, _ in series)
    hi = max(max(r["equity"]) for _, r, _ in series)
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
    ax.set_title(f"Rotor tuning {cm['start']}..{end} — equity, log scale (1.0 = start)",
                 color=INK, fontsize=12, loc="left", pad=12)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
    fig.text(0.01, 0.005,
             f"Survivorship-safe PIT universe (delisted series included). {COST_LABEL}.",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(png, facecolor=SURFACE)
    plt.close(fig)

    print(f"\nSaved: {grid_csv}")
    print(f"Saved: {os.path.join(RESULTS_DIR, 'rotor_tuning_summary.md')}")
    print(f"Saved: {png}")
    return 0


def _slice_result(res: dict, start_ts) -> dict:
    """Re-anchor a result at ``start_ts`` (equity rebased to 1.0 there) so a
    gateless baseline compares from the chosen cell's start."""
    i = res["timestamps"].index(start_ts)
    base = res["equity"][i]
    # Rebase to the value at start_ts; the first day's return becomes 0
    # (the baseline's own entry cost was paid at ITS start, before this
    # window — comparable to a curve that begins here).
    return {
        "timestamps": res["timestamps"][i:],
        "equity": [e / base for e in res["equity"][i:]],
        "day_returns": [0.0] + res["day_returns"][i + 1:],
        "invested": res["invested"][i:],
        "rebalances": [r for r in res["rebalances"] if r["ts"] >= start_ts],
        "traded_frac": res["traded_frac"],   # window-approximate, unused in
        "cost_frac": res["cost_frac"],       # baseline reporting emphasis
        "start_ts": start_ts,
    }


if __name__ == "__main__":
    sys.exit(main())
