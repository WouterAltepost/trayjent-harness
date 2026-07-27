"""GAP-2 measurement: live 10:00 ET partial-bar process vs validated close
process (readiness review §6 / gate G-8, brief Part 2).

Pure OFFLINE comparison. The live side is the Google Sheets `steady` tab CSV
export (results/live_export/steady.csv, provided by Mr. Altepost). The
backtest side is read from the existing Stage C output
(results/C_steady_insample.json) — run_backtest is NEVER called, no network,
no API key, no cache access. Diffing the two over their stable overlap turns
GAP-2 from a debate into a number; it feeds the Q2 decision (keep the 16:00
Amsterdam run and monitor, vs move the run to the US close).

Two windows, reported separately (discovery 2a, approved 2026-07-27):

  STRICT   2026-06-08..2026-06-24 — config-identical: starts at the first live
           run after the v9.5 watchlist surgery deploy (bfa2139, 2026-06-06),
           which made the live watchlist exactly the harness STEADY_WATCHLIST.
           Ends at the frozen-parquet end. Full comparison, tiny n.
  EXTENDED 2026-05-18..2026-06-24 — mechanics only: starts at the first live
           run scored on opus-4-7 (deploy 020f27a landed 2026-05-15 after that
           day's run), so the scoring model matches the harness. The live
           watchlist still carried XLE/XLF until 6/8, so ticker-level work is
           restricted to the intersection (= STEADY_WATCHLIST) and, because
           breadth enters the prompt, score/signal stats from this window are
           labeled non-evidential. Fill-price and exit-timing mechanics are
           unaffected by breadth and remain valid.

What each side carries (discovery 2a): live BUY rows log Score and notional
but NO fill price; every live SELL logs Sell Price + PnL %, so the entry fill
is back-derived as sell/(1+pnl) for every closed trade. Backtest trades carry
entry/exit ts+price, exit_reason, entry_score. Per-decision (non-entry)
scores exist on neither side in comparable form — that comparison is skipped
and said so. Live paper fills vs idealized backtest fills means this measures
process divergence AND paper-fill noise together, never process alone.

Also preserved here (approved addition): an "Ops findings (feeds G-6)"
appendix recording the missed-run streaks, the pre-open run, the breaker
trip, the manual sells, the closed-day sell re-fire behavior, and any BUY
left without a logged SELL — readiness-review evidence that would otherwise
live only in a Sheets export.

Run from the harness root:

    python3.13 scripts/diag_gap2_live_vs_backtest.py

Writes results/gap2_timing_divergence.md (results/ is gitignored). The
Interpretation section is written by hand, not by this script.
"""
import argparse
import csv
import json
import os
import statistics
import sys
from datetime import date, datetime, timedelta

_HARNESS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HARNESS_ROOT not in sys.path:
    sys.path.insert(0, _HARNESS_ROOT)

import config  # pure import: paths + frozen watchlist, no env/network needed

LIVE_CSV = os.path.join(config.HARNESS_ROOT, "results", "live_export", "steady.csv")
BACKTEST_JSON = os.path.join(config.HARNESS_ROOT, "results", "C_steady_insample.json")
OUT_MD = os.path.join(config.HARNESS_ROOT, "results", "gap2_timing_divergence.md")

# Windows from discovery 2a (approved). Overridable per the brief.
STRICT = ("2026-06-08", "2026-06-24")
EXTENDED = ("2026-05-18", "2026-06-24")

# NYSE full-session holidays 2026 (only the compare span is load-bearing;
# listed in full so trading-day deltas never silently miscount).
HOLIDAYS = {"2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03",
            "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07",
            "2026-11-26", "2026-12-25"}

# Live runs cluster at ~14:00 UTC; rows within this many seconds are one run.
RUN_GAP_S = 300
# A canonical decision run must start at/after the US cash open (13:30 UTC).
OPEN_UTC = "13:30"
# Re-fired closed-day sells share the same implied entry to this tolerance.
REFIRE_TOL = 0.002
ONE_TD = 1  # entry/exit match window, trading days


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d.isoformat() not in HOLIDAYS


def td_delta(a: date, b: date) -> int:
    """Trading days between a and b (absolute)."""
    lo, hi = min(a, b), max(a, b)
    n, d = 0, lo
    while d < hi:
        d += timedelta(days=1)
        if is_trading_day(d):
            n += 1
    return n


def load_live(path):
    """Parse the sheet export. Returns (decision_rows, buys, sells, noise)."""
    rows = list(csv.DictReader(open(path)))
    noise = [r for r in rows if not (r["Date"] and r["Date"][:4] == "2026")]
    dated = [r for r in rows if r not in noise]
    for r in dated:
        r["dt"] = datetime.fromisoformat(r["Date"])
    buys, sells = [], []
    for r in dated:
        if r["Action"] == "BUY":
            buys.append({"ticker": r["Ticker"], "dt": r["dt"],
                         "score": int(r["Score"]) if r["Score"] else None,
                         "notional": float(r["Trade $"]) if r["Trade $"] else None})
        elif r["Action"].startswith("SELL"):
            price, pnl = float(r["Sell Price"]), float(r["PnL %"])
            sells.append({"ticker": r["Ticker"], "dt": r["dt"], "price": price,
                          "pnl": pnl, "implied_entry": price / (1 + pnl / 100),
                          "reason": r["Action"],
                          "manual": r["Action"] == "SELL (MANUAL)"})
    return dated, buys, sells, noise


def dedup_sells(sells):
    """Collapse closed-day re-fires: same ticker + implied entry within
    REFIRE_TOL + within 10 calendar days -> one executed sell (the LAST row,
    the one that landed on an open market day). Returns (executed, chains)."""
    executed, chains, used = [], [], set()
    for i, s in enumerate(sells):
        if i in used:
            continue
        chain = [s]
        for j in range(i + 1, len(sells)):
            t = sells[j]
            if (j not in used and t["ticker"] == s["ticker"]
                    and abs(t["implied_entry"] / s["implied_entry"] - 1) < REFIRE_TOL
                    and (t["dt"] - s["dt"]).days <= 10):
                chain.append(t)
                used.add(j)
        if len(chain) > 1:
            chains.append(chain)
        executed.append(chain[-1])
    return executed, chains


def pair_trades(buys, sells):
    """LIFO-pair each executed SELL with the most recent prior unconsumed BUY
    of the same ticker. Returns (closed_trades, unmatched_buys). LIFO (not
    FIFO) so a re-BUY after an unlogged close pairs with its own sell and the
    orphaned earlier BUY surfaces as unmatched instead of mispairing."""
    open_stack = {}
    closed, all_buys = [], sorted(buys, key=lambda b: b["dt"])
    events = ([("B", b["dt"], b) for b in all_buys]
              + [("S", s["dt"], s) for s in sells])
    for kind, _, ev in sorted(events, key=lambda e: e[1]):
        if kind == "B":
            open_stack.setdefault(ev["ticker"], []).append(ev)
        else:
            stack = open_stack.get(ev["ticker"], [])
            buy = stack.pop() if stack else None
            closed.append({"ticker": ev["ticker"],
                           "entry_dt": buy["dt"] if buy else None,
                           "entry_score": buy["score"] if buy else None,
                           "implied_entry": ev["implied_entry"],
                           "exit_dt": ev["dt"], "exit_price": ev["price"],
                           "pnl": ev["pnl"], "reason": ev["reason"],
                           "manual": ev["manual"]})
    unmatched = [b for st in open_stack.values() for b in st]
    return closed, unmatched


def load_backtest(path):
    d = json.load(open(path))
    trades = []
    for t in d["trades"]:
        trades.append({"ticker": t["ticker"],
                       "entry_d": date.fromisoformat(t["entry_ts"][:10]),
                       "exit_d": date.fromisoformat(t["exit_ts"][:10]),
                       "entry_price": t["entry_price"], "exit_price": t["exit_price"],
                       "reason": t["exit_reason"], "score": t["entry_score"],
                       "pct": t["realized_pct"]})
    decision_days = sorted({e["ts"][:10] for e in d["equity_curve"]})
    return trades, decision_days


def reason_kind(r):
    return ("TP" if "TAKE PROFIT" in r else "SL" if "STOP LOSS" in r
            else "MANUAL" if "MANUAL" in r else r)


def match_entries(live_trades, bt_trades):
    """Match live vs backtest entries: same ticker, entry dates within ONE_TD.
    Greedy nearest-date, each backtest trade used once."""
    pairs, used = [], set()
    for lt in live_trades:
        if lt["entry_dt"] is None:
            continue
        ld = lt["entry_dt"].date()
        cands = [(td_delta(ld, bt["entry_d"]), i, bt)
                 for i, bt in enumerate(bt_trades)
                 if i not in used and bt["ticker"] == lt["ticker"]
                 and td_delta(ld, bt["entry_d"]) <= ONE_TD]
        if cands:
            delta, i, bt = min(cands, key=lambda c: (c[0], c[1]))
            used.add(i)
            pairs.append((lt, bt, delta))
    return pairs, used


def fmt_pct(x):
    return f"{x:+.2f}%"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="diag_gap2_live_vs_backtest")
    p.add_argument("--live-csv", default=LIVE_CSV)
    p.add_argument("--strict-window", nargs=2, default=list(STRICT),
                   metavar=("START", "END"))
    p.add_argument("--extended-window", nargs=2, default=list(EXTENDED),
                   metavar=("START", "END"))
    args = p.parse_args(argv)

    intersection = sorted(config.STEADY_WATCHLIST)
    dated, buys, sells, noise = load_live(args.live_csv)
    executed_sells, refire_chains = dedup_sells(sells)
    live_closed, unmatched_buys = pair_trades(buys, executed_sells)
    bt_trades, bt_days = load_backtest(BACKTEST_JSON)

    md = []
    md.append("# GAP-2: live 10:00 ET process vs validated close process — measurement\n")
    md.append(
        "Live side: `results/live_export/steady.csv` (Sheets export, "
        f"{len(dated)} rows, {dated[0]['Date'][:10]}..{dated[-1]['Date'][:10]}, "
        f"{len(noise)} noise rows dropped). Backtest side: "
        "`results/C_steady_insample.json` (Stage C, completed-close process, "
        "frozen cache — nothing re-run). This measures process divergence AND "
        "paper-fill noise together, never process divergence alone.\n")
    md.append(
        "Method: MARKET CLOSED rows excluded from decision comparison; "
        "closed-day sell re-fires collapsed to the executed open-day sell "
        "(same implied entry within 0.2%); SELL(MANUAL) excluded from "
        "exit-reason agreement; live entry fill back-derived as "
        "sell/(1+PnL%) (BUY rows log no price); SELLs LIFO-paired to prior "
        "BUYs; entries/exits match within 1 trading day (2026 NYSE holidays "
        "modeled).\n")

    headline = {}
    for tag, (lo_s, hi_s), full in (("Strict", args.strict_window, True),
                                    ("Extended", args.extended_window, False)):
        lo, hi = date.fromisoformat(lo_s), date.fromisoformat(hi_s)
        scope_note = ("config-identical (v9.5 watchlist == harness), full comparison"
                      if full else
                      "mechanics only — intersection tickers, score stats non-evidential "
                      "(pre-6/8 live breadth included XLE/XLF, which enters the prompt)")
        md.append(f"\n## {tag} window {lo_s}..{hi_s} — {scope_note}\n")

        lt_in = [t for t in live_closed
                 if t["entry_dt"] and lo <= t["entry_dt"].date() <= hi
                 and t["ticker"] in intersection]
        lt_exit_in = [t for t in live_closed
                      if lo <= t["exit_dt"].date() <= hi and t["ticker"] in intersection]
        bt_in = [t for t in bt_trades if lo <= t["entry_d"] <= hi]
        bt_exit_in = [t for t in bt_trades if lo <= t["exit_d"] <= hi]

        live_days = sorted({r["dt"].date() for r in dated
                            if lo <= r["dt"].date() <= hi and r["Score"]
                            and r["dt"].strftime("%H:%M") >= OPEN_UTC})
        bt_days_in = [d for d in bt_days if lo_s <= d <= hi_s]
        md.append(f"Decision days: backtest {len(bt_days_in)}, live {len(live_days)} "
                  f"(live missed: "
                  f"{', '.join(d for d in bt_days_in if date.fromisoformat(d) not in live_days) or 'none'}).\n")

        pairs, _ = match_entries(lt_in, bt_in)
        live_only = [t for t in lt_in if not any(t is pr[0] for pr in pairs)]
        bt_only = [t for t in bt_in if not any(t is pr[1] for pr in pairs)]
        n_all = len(pairs) + len(live_only) + len(bt_only)
        md.append(f"**Entry overlap:** {len(pairs)} matched, {len(live_only)} "
                  f"live-only, {len(bt_only)} backtest-only "
                  f"(ratio {len(pairs)}/{n_all if n_all else 0}).\n")
        for lt, bt, delta in pairs:
            div = (lt["implied_entry"] / bt["entry_price"] - 1) * 100
            md.append(f"- MATCHED {lt['ticker']}: live {lt['entry_dt'].date()} "
                      f"~{lt['implied_entry']:.2f} (score {lt['entry_score']}) vs "
                      f"backtest {bt['entry_d']} @{bt['entry_price']:.2f} "
                      f"(score {bt['score']}) — Δ{delta} td, fill div {fmt_pct(div)}")
        for t in live_only:
            md.append(f"- live-only entry: {t['ticker']} {t['entry_dt'].date()} "
                      f"~{t['implied_entry']:.2f} (score {t['entry_score']})")
        for t in bt_only:
            md.append(f"- backtest-only entry: {t['ticker']} {t['entry_d']} "
                      f"@{t['entry_price']:.2f} (score {t['score']})")

        fills = [abs(lt["implied_entry"] / bt["entry_price"] - 1) * 100
                 for lt, bt, _ in pairs]
        if fills:
            md.append(f"\n**Fill-price divergence (matched entries, abs):** "
                      f"mean {statistics.mean(fills):.2f}%, "
                      f"median {statistics.median(fills):.2f}% (n={len(fills)}).")
        else:
            md.append("\n**Fill-price divergence:** no matched entries in window.")

        # Exit matching: same ticker, exit dates within 1 td; manual excluded
        # from reason agreement but listed.
        md.append("\n**Exit divergence (exits in window, matched within 1 td):**")
        ex_pairs, used_b = [], set()
        for lt in lt_exit_in:
            cands = [(td_delta(lt["exit_dt"].date(), bt["exit_d"]), i, bt)
                     for i, bt in enumerate(bt_exit_in)
                     if i not in used_b and bt["ticker"] == lt["ticker"]
                     and td_delta(lt["exit_dt"].date(), bt["exit_d"]) <= ONE_TD]
            if cands:
                delta, i, bt = min(cands, key=lambda c: (c[0], c[1]))
                used_b.add(i)
                ex_pairs.append((lt, bt, delta))
        agree = 0
        nonmanual = [(lt, bt, d) for lt, bt, d in ex_pairs if not lt["manual"]]
        for lt, bt, delta in ex_pairs:
            div = (lt["exit_price"] / bt["exit_price"] - 1) * 100
            same = reason_kind(lt["reason"]) == reason_kind(bt["reason"])
            if same and not lt["manual"]:
                agree += 1
            md.append(f"- {lt['ticker']}: live {lt['exit_dt'].date()} "
                      f"{reason_kind(lt['reason'])} @{lt['exit_price']:.4f} vs "
                      f"backtest {bt['exit_d']} {reason_kind(bt['reason'])} "
                      f"@{bt['exit_price']:.2f} — Δ{delta} td, "
                      f"price div {fmt_pct(div)}"
                      + (" (MANUAL, excluded from agreement)" if lt["manual"] else ""))
        for lt in lt_exit_in:
            if lt not in [p[0] for p in ex_pairs]:
                md.append(f"- live-only exit: {lt['ticker']} {lt['exit_dt'].date()} "
                          f"{reason_kind(lt['reason'])} @{lt['exit_price']:.4f}")
        for i, bt in enumerate(bt_exit_in):
            if i not in used_b:
                md.append(f"- backtest-only exit: {bt['ticker']} {bt['exit_d']} "
                          f"{reason_kind(bt['reason'])} @{bt['exit_price']:.2f}")
        if nonmanual:
            md.append(f"\nExit-reason agreement (non-manual matched): "
                      f"{agree}/{len(nonmanual)}.")

        scores = [(lt, bt) for lt, bt, _ in pairs
                  if lt["entry_score"] is not None]
        if scores:
            lbl = "" if full else " (NON-EVIDENTIAL — breadth context differs)"
            md.append(f"\n**Entry-score comparison{lbl}:** "
                      + "; ".join(f"{lt['ticker']} live {lt['entry_score']} vs "
                                  f"backtest {bt['score']}" for lt, bt in scores)
                      + ". Per-decision (non-entry) score comparison SKIPPED: the "
                        "backtest JSON carries scores only on entries.")
        headline[tag] = {"pairs": len(pairs), "n_all": n_all, "fills": fills,
                         "agree": agree, "nonmanual": len(nonmanual)}

    # The single cleanest data point, called out explicitly (approved).
    md.append(
        "\n## The strict-window MSFT matched exit (cleanest data point)\n\n"
        "Both processes stopped out of MSFT on 2026-06-08, same reason "
        "(STOP LOSS), same day: live fill 412.445 vs backtest close fill "
        "411.74 — **+0.17% divergence**. The live exit was first *decided* on "
        "Sat 2026-06-06 against a stale close (two re-fired rows at 416.67) "
        "and *executed* at the next open; the backtest decided and filled on "
        "the 6/08 close directly. Same decision, one trading day of ledger "
        "lag, 17 bps of fill noise.\n")

    # ── Caveats (mandatory) ─────────────────────────────────────────────
    md.append("## Caveats\n")
    md.append(
        "- **Tiny n.** The strict window is 12 trading days with 1 matched "
        "exit and ~1 unmatched entry per side; the extended window adds a "
        "handful more. Fill/exit numbers are informative; overlap ratios are "
        "not statistics.\n"
        "- **Window boundaries:** strict start = first run after the v9.5 "
        "watchlist deploy (bfa2139, 2026-06-06 15:40 CET); extended start = "
        "first run scored on opus-4-7 (020f27a landed 2026-05-15 after that "
        "day's run); end = 2026-06-24 (frozen parquet end, never re-pulled). "
        "The 2026-06-24 TBH extraction commits landed ~14:00 UTC that day, "
        "straddling the final live run — they are behavior-preserving "
        "refactors pinned by the frozen-baseline tests.\n"
        "- **Intersection restriction:** extended-window ticker work is "
        "limited to the harness STEADY_WATCHLIST; live-only XLE/XLF activity "
        "(e.g. the 5/18 XLE BUY, 5/27 XLE stop) is excluded by construction.\n"
        "- **Conflated noise:** live Alpaca paper fills vs idealized "
        "zero-cost close fills — divergence numbers bundle process timing, "
        "paper-fill behavior, and intraday drift.\n"
        "- **Live log hygiene:** re-run clusters, manual sells, and at least "
        "one BUY without a logged SELL (see Ops findings) mean the live "
        "ledger is reconstructed, not read off.\n")

    # ── Ops findings (feeds G-6) ────────────────────────────────────────
    md.append("## Ops findings (feeds G-6)\n")
    all_days = sorted({r["dt"].date() for r in dated})
    have = set(all_days)
    missing = [d for d in (all_days[0] + timedelta(i)
                           for i in range((all_days[-1] - all_days[0]).days + 1))
               if is_trading_day(d) and d not in have]
    streaks, cur = [], []
    for d in missing:
        if cur and td_delta(cur[-1], d) == 1:
            cur.append(d)
        else:
            if cur:
                streaks.append(cur)
            cur = [d]
    if cur:
        streaks.append(cur)
    md.append("- **Missed runs** (trading days with zero log rows): "
              + "; ".join(f"{s[0]}..{s[-1]} ({len(s)}d)" if len(s) > 1
                          else str(s[0]) for s in streaks)
              + ". The 8-day 2026-07-15..2026-07-24 streak is exactly the "
                "\"silence indistinguishable from no-signal\" failure G-6 "
                "alerting must catch.\n")
    md.append("- **Pre-open run:** 2026-06-11 ran at 10:30 UTC (before the "
              "US open) and logged only MARKET CLOSED rows — that day "
              "produced no decision at all; 2026-06-12 had no run.\n")
    breaker = [r for r in dated if "BREAKER" in r["Action"]]
    for r in breaker:
        md.append(f"- **Drawdown breaker trip:** {r['Date']} {r['Ticker']} — "
                  f"\"{r['Reasoning'].split('|')[-1].strip()}\" (only trip in "
                  f"the log; backtest never models the breaker, GAP-4).\n")
    manuals = [s for s in sells if s["manual"]]
    md.append("- **Manual sells (operator actions, outside the frozen "
              "process):** "
              + "; ".join(f"{s['dt'].date()} {s['ticker']} @{s['price']:.4f} "
                          f"({s['pnl']:+.2f}%)" for s in manuals) + ".\n")
    for chain in refire_chains:
        first, last = chain[0], chain[-1]
        drift = (last["price"] / first["price"] - 1) * 100
        md.append(f"- **Closed-day sell re-fire:** {first['ticker']} decided "
                  f"{reason_kind(first['reason'])} on {first['dt'].date()} "
                  f"(closed market, stale close {first['price']:.4f}, "
                  f"{len(chain)} rows) — executed {last['dt'].date()} "
                  f"@{last['price']:.4f} ({fmt_pct(drift)} vs the stale "
                  f"decision price). Once-daily exit evaluation decides on a "
                  f"stale close and fills at the next open day (GAP-3 in the "
                  f"wild).\n")
    for b in unmatched_buys:
        md.append(f"- **Ledger integrity:** {b['ticker']} BUY "
                  f"{b['dt'].date()} has no subsequent SELL row in the "
                  f"export (position later re-bought or state lost without a "
                  f"logged close — check Sheets write failures around the "
                  f"missing-run days).\n")

    md.append("## Interpretation\n\n_(hand-written)_\n")

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w") as f:
        f.write("\n".join(md))

    for tag, h in headline.items():
        fills = h["fills"]
        print(f"[{tag}] entry overlap {h['pairs']}/{h['n_all']}; "
              f"mean |fill div| "
              f"{statistics.mean(fills):.2f}% (n={len(fills)}); " if fills else
              f"[{tag}] entry overlap {h['pairs']}/{h['n_all']}; no matched fills; ",
              end="", file=sys.stderr)
        print(f"exit-reason agreement {h['agree']}/{h['nonmanual']}",
              file=sys.stderr)
    print(f"wrote {OUT_MD}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
