"""Calm (shortvol sleeve) SEALED OUT-OF-SAMPLE read — one shot.

    python scripts/backtest_shortvol_oos.py

THE ONE DELIBERATE UNSEAL. This script passes ``allow_sealed=True`` to read
config.OOS_SHORTVOL (2022-01-01 onward) exactly once, with the configuration
FROZEN as committed from the tuning phase. No parameter may change here; the
results stand whatever they are. The only variants run are the two execution
modes specified for the read (same-day close primary, next-open sensitivity)
plus the two buy-and-hold baselines.

The sealed window is entirely post-leverage-flip, so SVXY returns are native
-0.5x — the engine's pre-flip x0.5 approximation never triggers (asserted).

Outputs (results/ is local-only, not tracked):
    results/calm_oos_summary.md
    results/calm_oos_equity.png
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
from data_layer import shortvol

RESULTS_DIR = os.path.join(_HARNESS_ROOT, "results")

# ── FROZEN configuration (tuning commit; do not edit) ───────────────────
FROZEN = {"entry": 0.95, "exit_": 1.05, "k": 25, "smoothing": "3d"}

# OOS stress episodes (all inside the sealed window).
OOS_EPISODES = [
    ("2022 bear market", "2022-01-01", "2022-12-31"),
    ("2024-08 VIX spike", "2024-08-01", "2024-08-31"),
    ("2025-04 vol event", "2025-03-01", "2025-05-31"),
]

# Same palette/ink roles as the tuning plot (dataviz reference, light mode).
C_STRAT, C_SVXY, C_SPY = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID_C, SURFACE = "#0b0b0b", "#52514e", "#e3e2de", "#fcfcfb"


def _pct(x, digits=1):
    return "-" if x is None else f"{100.0 * x:.{digits}f}%"


def _num(x, digits=2):
    return "-" if x is None else f"{x:.{digits}f}"


def _metric_row(label, m):
    return (f"{label:<28} {_pct(m['cagr']):>7} {_num(m['sharpe']):>7} "
            f"{_pct(m['max_drawdown']):>7} {_pct(m['daily_win']):>6} "
            f"{_pct(m['pos_weeks'], 0):>5} {_pct(m['pos_months'], 0):>5} "
            f"{_pct(m['time_in_market'], 0):>5} {m['round_trips']:>4} "
            f"{_pct(m['worst_day']):>7}")


def _transitions(df):
    """Replay the frozen config's state walk (same pure next_state the engine
    uses on the same ratio series -> identical sequence) and return
    [(date, 'entry'|'exit')]."""
    ratios = df["ratio_3d"].tolist()
    days = [t.strftime("%Y-%m-%d") for t in df["timestamp"]]
    state, out = bt.FLAT, []
    for day, ratio in zip(days, ratios):
        new_state, _ = bt.next_state(state, ratio, FROZEN["entry"], FROZEN["exit_"])
        if new_state != state:
            out.append((day, "entry" if new_state == bt.LONG else "exit"))
        state = new_state
    return out


def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Last completed session with ALL required series present.
    lasts = {t: shortvol.load_bars(t)["timestamp"].iloc[-1].strftime("%Y-%m-%d")
             for t in ("SVXY", "SPY", "VIX", "VIX3M")}
    start, end = config.OOS_SHORTVOL[0], min(lasts.values())
    print("=" * 78)
    print("CALM SEALED OOS READ — one shot, configuration frozen from tuning")
    print(f"  ENTRY {FROZEN['entry']:g} / EXIT {FROZEN['exit_']:g} / K {FROZEN['k']:g} "
          f"/ smoothing {FROZEN['smoothing']} / same-day-close / "
          f"{config.CALM_COST_BPS_PER_SIDE:g} bps per side / daily resize while long")
    print(f"  Window: {start}..{end} (last completed session across "
          f"{', '.join(f'{t} {d}' for t, d in sorted(lasts.items()))})")
    print("=" * 78)

    df = bt.build_inputs(start, end, allow_sealed=True)   # THE deliberate unseal
    timestamps = df["timestamp"].tolist()
    flip = pd.Timestamp(config.SVXY_LEVERAGE_CHANGE_DATE, tz="UTC")
    assert timestamps[0] >= flip, "sealed window must be entirely post-flip"
    print(f"{len(df)} sessions; entirely post-flip -> native -0.5x SVXY, no scaling.")

    # ── The four runs ───────────────────────────────────────────────────
    frozen = bt.run_backtest(df, FROZEN["entry"], FROZEN["exit_"], FROZEN["k"],
                             smoothing=FROZEN["smoothing"], mode="close")
    nxt = bt.run_backtest(df, FROZEN["entry"], FROZEN["exit_"], FROZEN["k"],
                          smoothing=FROZEN["smoothing"], mode="next_open")
    svxy_bh = bt.buy_and_hold(df["ret_cc"].tolist())
    spy_bh = bt.buy_and_hold(df["spy_ret"].tolist())

    rows = [
        ("Calm frozen (close)", bt.compute_cell_metrics(frozen, timestamps)),
        ("Calm frozen (next-open)", bt.compute_cell_metrics(nxt, timestamps)),
        ("SVXY buy-and-hold", bt.compute_cell_metrics(svxy_bh, timestamps)),
        ("SPY buy-and-hold", bt.compute_cell_metrics(spy_bh, timestamps)),
    ]
    print(f"\nSEALED WINDOW {start}..{end} — native -0.5x SVXY, "
          f"{config.CALM_COST_BPS_PER_SIDE:g} bps/side:")
    print(f"{'':28} {'CAGR':>7} {'Sharpe':>7} {'maxDD':>7} {'dWin':>6} "
          f"{'+wk':>5} {'+mo':>5} {'TIM':>5} {'RT':>4} {'worst':>7}")
    for label, m in rows:
        print(_metric_row(label, m))
    fm, nm = rows[0][1], rows[1][1]
    print(f"\nExecution delta (next-open minus close): "
          f"CAGR {_pct(nm['cagr'] - fm['cagr'])}  "
          f"Sharpe {nm['sharpe'] - fm['sharpe']:+.2f}  "
          f"maxDD {_pct(nm['max_drawdown'] - fm['max_drawdown'])}")

    # ── Stress episodes + state-machine narrative ───────────────────────
    trans = _transitions(df)
    days = [t.strftime("%Y-%m-%d") for t in timestamps]
    print("\nSTRESS EPISODES (frozen config vs SVXY B&H, native -0.5x):")
    ep_rows = []
    for name, ep_s, ep_e in OOS_EPISODES:
        s = bt.episode_stats(frozen, timestamps, ep_s, ep_e)
        b = bt.episode_stats(svxy_bh, timestamps, ep_s, ep_e)
        idx = [i for i, d in enumerate(days) if ep_s <= d <= ep_e]
        cash_days = sum(1 for i in idx if frozen["weights"][i] == 0)
        ep_trans = [(d, kind) for d, kind in trans if ep_s <= d <= ep_e]
        # State walking INTO the episode = last transition before it (FLAT if none).
        prior = [kind for d, kind in trans if d < ep_s]
        state_in = "LONG" if (prior and prior[-1] == "entry") else "FLAT"
        print(f"\n  {name}  ({ep_s}..{ep_e}, {len(idx)} sessions)")
        print(f"    Calm P&L {_pct(s['pnl']):>7}   maxDD {_pct(s['max_drawdown']):>6}   "
              f"in cash {cash_days}/{len(idx)} days")
        print(f"    B&H  P&L {_pct(b['pnl']):>7}   maxDD {_pct(b['max_drawdown']):>6}")
        print(f"    entered episode {state_in}; transitions: "
              + ("; ".join(f"{kind} {d}" for d, kind in ep_trans) if ep_trans else "none"))
        ep_rows.append((name, ep_s, ep_e, len(idx), s, b, cash_days, state_in, ep_trans))

    # ── Tuning vs sealed side-by-side (frozen config) ───────────────────
    # Tuning metrics recomputed live on the tuning window (not sealed) so the
    # comparison is same-code, same-data — not a hand-copied number.
    tdf = bt.build_inputs(*config.CALM_TUNING)
    tuned = bt.run_backtest(tdf, FROZEN["entry"], FROZEN["exit_"], FROZEN["k"],
                            smoothing=FROZEN["smoothing"], mode="close")
    tm = bt.compute_cell_metrics(tuned, tdf["timestamp"].tolist())
    print("\nTUNING vs SEALED (frozen config, same-day close):")
    print(f"  {'':14} {'CAGR':>7} {'Sharpe':>7} {'maxDD':>7} {'+wk':>5}")
    print(f"  {'tuning 11-21':<14} {_pct(tm['cagr']):>7} {_num(tm['sharpe']):>7} "
          f"{_pct(tm['max_drawdown']):>7} {_pct(tm['pos_weeks'], 0):>5}")
    print(f"  {'sealed 22-26':<14} {_pct(fm['cagr']):>7} {_num(fm['sharpe']):>7} "
          f"{_pct(fm['max_drawdown']):>7} {_pct(fm['pos_weeks'], 0):>5}")
    print(f"  {'delta':<14} {_pct(fm['cagr'] - tm['cagr']):>7} "
          f"{fm['sharpe'] - tm['sharpe']:>+7.2f} "
          f"{_pct(fm['max_drawdown'] - tm['max_drawdown']):>7} "
          f"{100 * (fm['pos_weeks'] - tm['pos_weeks']):>+4.0f}%")

    # ── Summary md ──────────────────────────────────────────────────────
    md = [
        "# Calm SEALED OOS read (one shot)",
        "",
        f"Window {start}..{end} ({len(df)} sessions), native -0.5x SVXY (no "
        f"scaling — post-flip only). Frozen: ENTRY {FROZEN['entry']:g}, EXIT "
        f"{FROZEN['exit_']:g}, K {FROZEN['k']:g}, {FROZEN['smoothing']}, "
        f"same-day close, {config.CALM_COST_BPS_PER_SIDE:g} bps/side.",
        "",
        "| run | CAGR | Sharpe | maxDD | dWin | +wk | +mo | TIM | RT | worst |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for label, m in rows:
        md.append(f"| {label} | {_pct(m['cagr'])} | {_num(m['sharpe'])} | "
                  f"{_pct(m['max_drawdown'])} | {_pct(m['daily_win'])} | "
                  f"{_pct(m['pos_weeks'], 0)} | {_pct(m['pos_months'], 0)} | "
                  f"{_pct(m['time_in_market'], 0)} | {m['round_trips']} | "
                  f"{_pct(m['worst_day'])} |")
    md += ["", "| episode | Calm P&L | Calm DD | B&H P&L | B&H DD | cash days | transitions |",
           "|---|---|---|---|---|---|---|"]
    for name, ep_s, ep_e, n, s, b, cash, state_in, ep_trans in ep_rows:
        tr = "; ".join(f"{kind} {d}" for d, kind in ep_trans) or "none"
        md.append(f"| {name} | {_pct(s['pnl'])} | {_pct(s['max_drawdown'])} | "
                  f"{_pct(b['pnl'])} | {_pct(b['max_drawdown'])} | {cash}/{n} | "
                  f"(in {state_in}) {tr} |")
    md += ["",
           "| window | CAGR | Sharpe | maxDD | +wk |",
           "|---|---|---|---|---|",
           f"| tuning 2011-2021 | {_pct(tm['cagr'])} | {_num(tm['sharpe'])} | "
           f"{_pct(tm['max_drawdown'])} | {_pct(tm['pos_weeks'], 0)} |",
           f"| sealed 2022-2026 | {_pct(fm['cagr'])} | {_num(fm['sharpe'])} | "
           f"{_pct(fm['max_drawdown'])} | {_pct(fm['pos_weeks'], 0)} |"]
    summary_md = os.path.join(RESULTS_DIR, "calm_oos_summary.md")
    with open(summary_md, "w") as f:
        f.write("\n".join(md) + "\n")

    # ── Equity plot, sealed window only ─────────────────────────────────
    png = os.path.join(RESULTS_DIR, "calm_oos_equity.png")
    dates = [t.tz_convert(None) for t in timestamps]
    fig, ax = plt.subplots(figsize=(11.5, 6.0), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    series = [
        (f"Calm frozen (E {FROZEN['entry']:g} / X {FROZEN['exit_']:g} / "
         f"K {FROZEN['k']:g} / {FROZEN['smoothing']})", frozen["equity"], C_STRAT),
        ("SVXY buy-and-hold (native -0.5x)", svxy_bh["equity"], C_SVXY),
        ("SPY buy-and-hold", spy_bh["equity"], C_SPY),
    ]
    for label, eq, color in series:
        ax.plot(dates, eq, color=color, linewidth=1.8, label=label,
                solid_capstyle="round", zorder=3)
        ax.annotate(f" {eq[-1]:.2f}x", xy=(dates[-1], eq[-1]),
                    xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK2)
    ax.set_yscale("log")
    ticks = [0.25, 0.5, 0.75, 1, 1.5, 2, 3]
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
    ax.set_title(f"Calm SEALED OOS {start}..{end} — equity, log scale (1.0 = start)",
                 color=INK, fontsize=12, loc="left", pad=12)
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK)
    fig.text(0.01, 0.005,
             "One-shot sealed read. Native -0.5x SVXY (window is post-flip; no "
             f"scaling). Same-day-close fills, {config.CALM_COST_BPS_PER_SIDE:g} bps/side.",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(png, facecolor=SURFACE)
    plt.close(fig)

    print(f"\nSaved: {summary_md}")
    print(f"Saved: {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
