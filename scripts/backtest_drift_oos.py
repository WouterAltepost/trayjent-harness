"""Drift SEALED OOS read — one shot, spends the last intact crypto window.

    python scripts/backtest_drift_oos.py --spend-the-window

Registered spec: ../planning/drift_spec.md (v3, APPROVED 2026-07-29; Decision
4 amended pre-read to weekly same-close — exactly what ``btc_gated()``
implements, no convention shims). This script contains NO strategy code: it
calls ``rotor.backtest.btc_gated(gate_ma=200)`` and ``btc_buy_hold`` as-is
over the registered window and prints the section-4 metrics plus the
registered pass/fail bar. The result stands whatever it says.

Refuses to run without the explicit ``--spend-the-window`` flag, and refuses
to spend the window on truncated data (the store must cover the window's
final bar). Warm-up reads from 2024-06-01 (open data) so the 200d MA is
fully formed entering 2025.

Outputs: results/drift_oos_summary.md (verbatim copy of the printed read).
"""
import os
import sys

_HARNESS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HARNESS_ROOT not in sys.path:
    sys.path.insert(0, _HARNESS_ROOT)

import config
from data_layer.rotor import bar_day
from rotor import backtest as bt
from runner.metrics import _cagr, _max_drawdown, _sharpe, _years_between

WARMUP_START = "2024-06-01"   # spec §4: open-data MA warm-up, seal untouched
GATE_MA = 200
# Spec §4: Sharpe is daily, annualized on 365 (registered constant, not the
# harness's empirical periods-per-year — 24/7 market, no calendar holes).
SHARPE_PPY = 365.0


def count_round_trips(invested: list) -> int:
    """Completed long -> cash cycles from the daily invested-fraction series.
    A position still open at the window end is not counted (Calm precedent)."""
    return sum(1 for prev, cur in zip(invested, invested[1:])
               if prev > 0 and cur == 0)


def verdict(strat_ret: float, strat_dd: float, bh_ret: float, bh_dd: float):
    """The registered pass bar (spec §4), pure so the synthetic test can pin
    it: PASS iff maxDD <= B&H maxDD AND (return >= 0.75 x B&H return when
    B&H > 0, else return > B&H return)."""
    dd_ok = strat_dd <= bh_dd
    ret_ok = (strat_ret >= 0.75 * bh_ret) if bh_ret > 0 else (strat_ret > bh_ret)
    return dd_ok and ret_ok, dd_ok, ret_ok


def _metrics(res: dict) -> dict:
    eq, rets, ts = res["equity"], res["day_returns"], res["timestamps"]
    years = _years_between(ts[0], ts[-1])
    return {
        "total_return": eq[-1] - 1.0,
        "cagr": _cagr(1.0, eq[-1], years),
        "sharpe": _sharpe(rets, SHARPE_PPY),
        "max_drawdown": _max_drawdown([1.0] + eq),
        "time_in_market": sum(res["invested"]) / len(res["invested"]),
        "round_trips": count_round_trips(res["invested"]),
        "cost_drag": res["cost_frac"],
    }


def _row(label: str, m: dict) -> str:
    return (f"{label:<18} ret {100 * m['total_return']:+7.1f}%  "
            f"CAGR {100 * m['cagr']:+6.1f}%  Sharpe {m['sharpe']:.2f}  "
            f"maxDD {100 * m['max_drawdown']:.1f}%  TIM {100 * m['time_in_market']:.0f}%  "
            f"RT {m['round_trips']}  cost {100 * m['cost_drag']:.2f}%")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--spend-the-window" not in argv:
        print("REFUSED: this script spends the LAST intact crypto sealed window "
              f"(OOS_ROTOR, {config.OOS_ROTOR[0]}..{config.OOS_ROTOR[1]}), one "
              "shot, result stands. Run with --spend-the-window only per the "
              "registered protocol in planning/drift_spec.md.")
        return 1

    start, end = config.OOS_ROTOR
    inputs = bt.build_inputs(WARMUP_START, end, allow_sealed=True)  # THE spend
    cal = inputs["calendar"]
    last_day = bar_day(cal[-1])
    if last_day != end:
        raise ValueError(
            f"Store covers only through {last_day}, window ends {end} — "
            "refusing to spend the window on truncated data. Run "
            "scripts/pull_rotor_data.py after 00:00 UTC and retry."
        )
    start_ts = next(t for t in cal if bar_day(t) >= start)

    gated = bt.btc_gated(inputs, start_ts, gate_ma=GATE_MA)
    bh = bt.btc_buy_hold(inputs, start_ts)
    gm, bm = _metrics(gated), _metrics(bh)
    passed, dd_ok, ret_ok = verdict(gm["total_return"], gm["max_drawdown"],
                                    bm["total_return"], bm["max_drawdown"])

    lines = [
        "# Drift sealed OOS read — one shot (result stands)",
        "",
        f"Spec: planning/drift_spec.md v3 (registered 2026-07-29). Window "
        f"{start}..{end} ({bar_day(start_ts)} first tradable bar day), warm-up "
        f"from {WARMUP_START} (open data). Weekly Monday-00:00Z same-close "
        f"execution via rotor.backtest.btc_gated(gate_ma={GATE_MA}), costs "
        f"taker {config.ROTOR_ALPACA_CRYPTO_FEES['taker']:.2%} + "
        f"{config.ROTOR_SLIPPAGE_BPS:g} bps slippage per side. Sharpe daily, "
        f"annualized on {SHARPE_PPY:g}.",
        "",
        _row("Drift (gated)", gm),
        _row("BTC buy-and-hold", bm),
        "",
        f"Registered bar: (1) drawdown, Drift maxDD <= B&H maxDD -> "
        f"{'OK' if dd_ok else 'FAILED'} "
        f"({100 * gm['max_drawdown']:.1f}% vs {100 * bm['max_drawdown']:.1f}%); "
        f"(2) return, "
        + (f"B&H > 0 so Drift >= 0.75 x B&H -> {'OK' if ret_ok else 'FAILED'} "
           f"({100 * gm['total_return']:+.1f}% vs 0.75 x "
           f"{100 * bm['total_return']:+.1f}% = "
           f"{75 * bm['total_return']:+.1f}%)"
           if bm["total_return"] > 0 else
           f"B&H <= 0 so Drift > B&H -> {'OK' if ret_ok else 'FAILED'} "
           f"({100 * gm['total_return']:+.1f}% vs {100 * bm['total_return']:+.1f}%)"),
        "",
        f"VERDICT: {'PASS' if passed else 'FAIL'}",
        "",
        "Power caveat (registered): a PASS means 'no disqualifying behavior "
        "on fresh data', not 'proven edge' — 19 months of a 200d gate is a "
        "handful of flips.",
    ]
    out = "\n".join(lines)
    print(out)
    os.makedirs(os.path.join(_HARNESS_ROOT, "results"), exist_ok=True)
    md_path = os.path.join(_HARNESS_ROOT, "results", "drift_oos_summary.md")
    with open(md_path, "w") as f:
        f.write(out + "\n")
    print(f"\nSaved: {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
