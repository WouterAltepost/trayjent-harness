"""Rotor (crypto momentum rotation) v1 backtest engine — TUNING phase.

Strategy v1:

- **Cadence**: rebalance weekly at the bar close that lands on Monday
  00:00 UTC (i.e., the Sunday UTC bar's close — stored timestamps ARE close
  instants). Decisions use only bars closed by that instant; execution fills
  at that same close.
- **Point-in-time universe** at each rebalance T: coins live on the venue
  per the DERIVED listing windows (a coin has a bar at T), with >= 90
  calendar days since their first bar, ranked top ``UNIVERSE_TOP`` by
  trailing 30-day median dollar volume (volume x vwap; the full 30-day
  window must be gap-free or the coin is out that week). Survivorship-safe:
  delisted coins are in the store and simply stop having bars.
- **Signal**: trailing ``formation``-day return, requiring a gap-free
  window (NO return is ever computed across a venue gap). Hold the top
  ``n_held`` by signal, equal weight (1/N of equity per slot).
- **BTC regime gate**: if BTC's close at T is below its ``gate_ma``-day MA,
  hold 100% cash. The equity curve STARTS at the first weekly rebalance
  where the gate MA is fully formed, so each cell's start depends on its
  gate (report it).
- **Optional per-coin filter**: with ``ma_filter`` on, a selected coin is
  held only if its close is above its own 50-day MA; failing slots stay in
  cash (never backfilled by the next-ranked coin).
- **Forced exits**: a held coin whose bar at day D is its listing-segment
  end (delisting or venue-gap start) is sold at that final close with
  normal costs.
- **Costs**: ``config.ROTOR_ALPACA_CRYPTO_FEES['taker']`` +
  ``config.ROTOR_SLIPPAGE_BPS`` per side on every fill (rebalances and
  forced exits), charged on traded notional.

SEALED WINDOW: everything from ``config.OOS_ROTOR[0]`` (2025-01-01) onward
is out-of-sample. ``build_inputs`` fails closed unless ``allow_sealed=True``
is passed deliberately.

Offline + deterministic: inputs are stored Parquet via ``data_layer.rotor``;
scalar metric helpers reuse ``runner.metrics`` (one source of truth).
"""
import statistics

import numpy as np
import pandas as pd

import config
from data_layer import rotor as rotor_data
from runner.metrics import _cagr, _max_drawdown, _periods_per_year, _sharpe, _years_between

# ── Tuning grid (locked — 2 x 2 x 2 x 2 = 16 cells, do not expand) ──────
FORMATION_GRID = [21, 30]        # trailing-return window, days
N_GRID = [3, 5]                  # coins held, equal weight
GATE_GRID = [100, 200]           # BTC trend-gate MA, days
FILTER_GRID = [False, True]      # per-coin 50d MA filter

UNIVERSE_TOP = 15                # liquidity cut at each rebalance
MIN_HISTORY_DAYS = 90            # calendar days since a coin's first bar
VOLUME_WINDOW = 30               # trailing median dollar-volume window
COIN_MA = 50                     # the per-coin filter's MA length


def _cost_per_side() -> float:
    return config.ROTOR_ALPACA_CRYPTO_FEES["taker"] + config.ROTOR_SLIPPAGE_BPS / 1e4


# ── Inputs ──────────────────────────────────────────────────────────────
def build_inputs(start: str, end: str, allow_sealed: bool = False) -> dict:
    """Assemble the wide daily matrices for [start, end] (ISO days, incl).

    Returns ``{"calendar", "close", "dollar_vol", "ret", "first_bar",
    "segment_ends"}``:

    - ``calendar``: the master daily close-instant index — BTC's timestamps
      (asserted continuous; every UTC day exists).
    - ``close`` / ``dollar_vol``: DataFrames [calendar x coin]; NaN where a
      coin is not listed (pre-listing, delisted, venue gap).
    - ``ret``: daily returns, NaN on each segment's first bar — a return is
      NEVER computed across a venue gap.
    - ``first_bar``: {coin: its first stored close-instant ever} (history
      age for the 90-day rule; predates ``start`` when the coin is older
      than the window).
    - ``segment_ends``: {coin: set of close-instants that end a listing
      segment inside the window} — forced-exit triggers.

    Fails closed if the window touches the sealed OOS (``config.OOS_ROTOR``)
    without ``allow_sealed``.
    """
    if end >= config.OOS_ROTOR[0] and not allow_sealed:
        raise ValueError(
            f"Window end {end} touches the sealed Rotor OOS "
            f"(starts {config.OOS_ROTOR[0]}). Pass allow_sealed=True only "
            "for a deliberate, one-shot validation read."
        )

    day = pd.Timedelta(hours=24)
    windows = rotor_data.listing_windows()
    coins = sorted(windows)

    closes, dvols, rets, first_bar, seg_ends = {}, {}, {}, {}, {}
    for coin in coins:
        df = rotor_data.load_bars(coin)
        ts = df["timestamp"]
        first_bar[coin] = ts.iloc[0]
        s_close = pd.Series(df["close"].values, index=ts)
        closes[coin] = s_close
        dvols[coin] = pd.Series((df["volume"] * df["vwap"]).values, index=ts)
        r = s_close.pct_change()
        r[ts.diff().fillna(pd.Timedelta(0)).values != day] = np.nan  # segment firsts
        rets[coin] = r
        seg_ends[coin] = {seg_last for _, seg_last in windows[coin]}

    close = pd.DataFrame(closes)
    dollar_vol = pd.DataFrame(dvols)
    ret = pd.DataFrame(rets)

    btc_ts = rotor_data.load_bars("BTC")["timestamp"]
    if (btc_ts.diff().dropna() != day).any():
        raise ValueError("BTC calendar has gaps — cannot serve as the master calendar")

    day_str = btc_ts.apply(rotor_data.bar_day)
    mask = (day_str >= start) & (day_str <= end)
    calendar = pd.DatetimeIndex(btc_ts[mask])
    if len(calendar) == 0:
        raise ValueError(f"No bars in [{start}, {end}]")

    return {
        "calendar": calendar,
        "close": close.reindex(calendar),
        "dollar_vol": dollar_vol.reindex(calendar),
        "ret": ret.reindex(calendar),
        "first_bar": first_bar,
        "segment_ends": seg_ends,
    }


# ── Engine ──────────────────────────────────────────────────────────────
def run_backtest(inputs: dict, formation: int, n_held: int, gate_ma,
                 ma_filter: bool, signal: str = "formation") -> dict:
    """Simulate one parameter cell. Deterministic, equity base 1.0.

    ``gate_ma=None`` disables the BTC gate (baseline use); ``signal`` is
    ``"formation"`` (rank by trailing return) or ``"volume"`` (rank by the
    liquidity measure itself — the no-signal equal-weight baseline).

    Returns ``{"timestamps", "equity", "day_returns", "invested",
    "rebalances": [{ts, gated, held}], "traded_frac", "cost_frac",
    "start_ts"}`` — ``invested[i]`` is the invested fraction during day i
    (for time-in-market), ``traded_frac``/``cost_frac`` accumulate traded
    notional and costs as fractions of equity at trade time.
    """
    cal = inputs["calendar"]
    close, dvol, ret = inputs["close"], inputs["dollar_vol"], inputs["ret"]
    first_bar, seg_ends = inputs["first_bar"], inputs["segment_ends"]
    cost = _cost_per_side()

    btc = close["BTC"]
    gate = btc.rolling(gate_ma).mean() if gate_ma else None

    # Signal / eligibility matrices, all computed once, all no-lookahead
    # (rolling windows END at the decision bar). log1p-rolling-sum keeps a
    # NaN anywhere in the window -> a gap inside the formation window
    # disqualifies the coin that week.
    formation_ret = np.expm1(np.log1p(ret).rolling(formation).sum())
    med_dvol = dvol.rolling(VOLUME_WINDOW).median()  # NaN in window -> NaN
    coin_ma = close.rolling(COIN_MA).mean() if ma_filter else None

    # Equity curve starts at the first Monday-00:00Z close where the gate MA
    # is fully formed (gateless runs start at the first Monday).
    mondays = [t for t in cal if t.dayofweek == 0]
    start_ts = None
    for t in mondays:
        if gate is None or not np.isnan(gate.loc[t]):
            start_ts = t
            break
    if start_ts is None:
        raise ValueError(f"gate MA ({gate_ma}d) never forms inside the window")

    positions = {}            # coin -> units
    cash = 1.0
    traded_frac = cost_frac = 0.0
    timestamps, equity_curve, day_returns, invested = [], [], [], []
    rebalances = []
    prev_equity = 1.0

    for t in cal[cal.get_loc(start_ts):]:
        # a) mark to market. Invariant: every held coin has a bar at t —
        #    its segment end was processed at that end bar (below), and
        #    selection never buys a coin at its own final bar.
        equity = cash + sum(u * close.at[t, c] for c, u in positions.items())

        # b) weekly rebalance at the Monday-00:00Z close
        if t.dayofweek == 0:
            gated = bool(gate is not None and btc.loc[t] < gate.loc[t])
            held = []
            if not gated:
                held = _select(t, close, med_dvol, formation_ret, coin_ma,
                               first_bar, seg_ends, n_held, signal)
            # Fee reserve: targets sum to equity x (1 - cost) so a full-
            # turnover buy pass can always pay its own fees from cash —
            # never implicit leverage. The sliver left over sits in cash.
            targets = {c: equity * (1.0 - cost) / n_held for c in held}
            equity, cash, traded, paid = _trade_to(targets, positions, close,
                                                   t, cost, cash)
            if equity > 0:
                traded_frac += traded / (equity + paid)   # vs pre-cost equity
                cost_frac += paid / (equity + paid)
            rebalances.append({"ts": t, "gated": gated, "held": list(held)})

        # c) forced exits AFTER the rebalance: t is a held coin's listing-
        #    segment end (delisting or venue-gap start) -> sold at this
        #    final close, normal costs. Runs last so a coin the rebalance
        #    already sold is not double-processed.
        for c in [c for c in positions if t in seg_ends[c]]:
            notional = positions.pop(c) * close.at[t, c]
            fee = notional * cost
            cash += notional - fee
            traded_frac += notional / equity
            cost_frac += fee / equity
            equity -= fee

        timestamps.append(t)
        equity_curve.append(equity)
        day_returns.append(equity / prev_equity - 1.0)
        invested.append((equity - cash) / equity if equity > 0 else 0.0)
        prev_equity = equity

    return {"timestamps": timestamps, "equity": equity_curve,
            "day_returns": day_returns, "invested": invested,
            "rebalances": rebalances, "traded_frac": traded_frac,
            "cost_frac": cost_frac, "start_ts": start_ts}


def _select(t, close, med_dvol, formation_ret, coin_ma, first_bar,
            seg_ends, n_held, signal) -> list:
    """The point-in-time selection at rebalance instant ``t``."""
    eligible = []
    for c in close.columns:
        px = close.at[t, c]
        if np.isnan(px):                       # not listed at t
            continue
        if t in seg_ends[c]:                   # t is its final bar: venues
            continue                           # announce delistings — never buy one
        if (t - first_bar[c]).days < MIN_HISTORY_DAYS:
            continue
        dv = med_dvol.at[t, c]
        if np.isnan(dv):                       # <30d continuous data
            continue
        eligible.append((c, dv))
    top = sorted(eligible, key=lambda x: -x[1])[:UNIVERSE_TOP]

    if signal == "volume":
        ranked = [c for c, _ in top]
    else:
        scored = [(c, formation_ret.at[t, c]) for c, _ in top
                  if not np.isnan(formation_ret.at[t, c])]
        ranked = [c for c, _ in sorted(scored, key=lambda x: -x[1])]

    held = ranked[:n_held]
    if coin_ma is not None:
        held = [c for c in held
                if not np.isnan(coin_ma.at[t, c]) and close.at[t, c] > coin_ma.at[t, c]]
    return held


def _trade_to(targets: dict, positions: dict, close, t, cost, cash):
    """Trade ``positions`` (units, mutated in place) to ``targets`` (dollar
    values) at ``t``'s closes. Returns ``(equity_after, cash_after,
    traded_notional, fees_paid)``."""
    traded = paid = 0.0

    for c in list(positions):                  # sells: drop or resize down
        px = close.at[t, c]
        have = positions[c] * px
        want = targets.get(c, 0.0)
        if want < have:
            notional = have - want
            fee = notional * cost
            traded += notional
            paid += fee
            cash += notional - fee
            if want == 0.0:
                positions.pop(c)
            else:
                positions[c] = want / px
    for c, want in targets.items():            # buys: add or resize up
        px = close.at[t, c]
        have = positions.get(c, 0.0) * px
        if want > have:
            notional = want - have
            fee = notional * cost
            traded += notional
            paid += fee
            cash -= notional + fee
            positions[c] = want / px

    equity = cash + sum(u * close.at[t, c] for c, u in positions.items())
    return equity, cash, traded, paid


# ── Baselines ───────────────────────────────────────────────────────────
def btc_buy_hold(inputs: dict, start_ts) -> dict:
    """BTC bought at ``start_ts``'s close (one costed fill), held."""
    cal = inputs["calendar"]
    btc = inputs["close"]["BTC"]
    cost = _cost_per_side()
    days = list(cal[cal.get_loc(start_ts):])
    base = btc.loc[start_ts]
    equity = [(1.0 - cost) * btc.loc[t] / base for t in days]
    rets = [equity[0] - 1.0] + [equity[i] / equity[i - 1] - 1.0
                                for i in range(1, len(equity))]
    return {"timestamps": days, "equity": equity, "day_returns": rets,
            "invested": [1.0] * len(days), "rebalances": [],
            "traded_frac": 1.0, "cost_frac": cost, "start_ts": start_ts}


def btc_gated(inputs: dict, start_ts, gate_ma: int = 200) -> dict:
    """The simple-trend competitor: BTC long/cash on its own ``gate_ma`` MA,
    checked at the same weekly Monday closes, same per-side costs."""
    cal = inputs["calendar"]
    btc = inputs["close"]["BTC"]
    gate = btc.rolling(gate_ma).mean()
    cost = _cost_per_side()
    days = list(cal[cal.get_loc(start_ts):])

    units = 0.0
    cash = 1.0
    traded_frac = cost_frac = 0.0
    equity_curve, rets, invested = [], [], []
    prev = 1.0
    for t in days:
        equity = cash + units * btc.loc[t]
        if t.dayofweek == 0:
            long_ok = btc.loc[t] >= gate.loc[t]
            if long_ok and units == 0.0:
                notional = cash / (1.0 + cost)
                fee = notional * cost
                traded_frac += notional / equity
                cost_frac += fee / equity
                units = notional / btc.loc[t]
                cash -= notional + fee
            elif not long_ok and units > 0.0:
                notional = units * btc.loc[t]
                fee = notional * cost
                traded_frac += notional / equity
                cost_frac += fee / equity
                cash += notional - fee
                units = 0.0
            equity = cash + units * btc.loc[t]
        equity_curve.append(equity)
        rets.append(equity / prev - 1.0)
        invested.append(0.0 if units == 0.0 else (units * btc.loc[t]) / equity)
        prev = equity
    rets[0] = equity_curve[0] - 1.0
    return {"timestamps": days, "equity": equity_curve, "day_returns": rets,
            "invested": invested, "rebalances": [], "traded_frac": traded_frac,
            "cost_frac": cost_frac, "start_ts": start_ts}


# ── Metrics ─────────────────────────────────────────────────────────────
def compute_cell_metrics(res: dict) -> dict:
    """The 10 tuning metrics. Conventions (mirroring Calm where they exist):

    - Sharpe: full daily net return stream (cash days at 0), rf=0,
      annualized at the curve's empirical frequency.
    - pos_weeks / worst_week: weekly = rebalance-close to rebalance-close
      (the strategy's own cadence); pos_weeks excludes weeks with zero
      exposure throughout. pos_months: calendar months, same exclusion.
    - time_in_market: mean invested fraction across days (partial exposure
      counts partially).
    - avg_holdings: mean held-slot count over UNGATED rebalances (gate time
      is already visible in time_in_market).
    - turnover: two-way traded notional as a fraction of equity, per year.
    - cost_drag: fees+slippage as a fraction of equity, per year.
    """
    eq, rets = res["equity"], res["day_returns"]
    ts = res["timestamps"]
    years = _years_between(ts[0], ts[-1])
    ppy = _periods_per_year(ts)

    idx = pd.DatetimeIndex(ts)
    eq_s = pd.Series(eq, index=idx)
    inv_s = pd.Series(res["invested"], index=idx)
    # invested[i] is the exposure carried OUT of day i (post-trade), i.e.
    # the exposure DURING day i+1 — shift so period-exposure tests ask
    # "was anything held during this period", not "at its last instant".
    exp_during = inv_s.shift(1).fillna(0.0)

    # Weekly slice on the strategy's own Monday-close instants: return over
    # (prev Monday close, this Monday close], active iff any exposure inside.
    monday_pos = [i for i, t in enumerate(ts) if t.dayofweek == 0]
    week_rets, week_active = [], []
    for a, b in zip(monday_pos, monday_pos[1:]):
        week_rets.append(eq[b] / eq[a] - 1.0)
        week_active.append(any(v > 0 for v in exp_during.iloc[a + 1:b + 1]))
    active_weeks = [r for r, ok in zip(week_rets, week_active) if ok]

    base = pd.Series([1.0], index=[idx[0] - pd.Timedelta(days=int(idx[0].day))])
    months = pd.concat([base, eq_s]).resample("ME").last().dropna().pct_change().dropna()
    month_exp = exp_during.resample("ME").max().reindex(months.index).fillna(0.0)
    active_months = months[month_exp > 0]

    ungated = [r for r in res["rebalances"] if not r["gated"]]

    return {
        "start": rotor_data.bar_day(ts[0]),
        "cagr": _cagr(1.0, eq[-1], years),
        "sharpe": _sharpe(rets, ppy),
        "max_drawdown": _max_drawdown([1.0] + eq),
        "pos_weeks": (sum(1 for r in active_weeks if r > 0) / len(active_weeks)
                      if active_weeks else None),
        "pos_months": (active_months > 0).mean() if len(active_months) else None,
        "time_in_market": float(exp_during.mean()),
        "avg_holdings": (statistics.mean(len(r["held"]) for r in ungated)
                         if ungated else 0.0),
        "turnover": res["traded_frac"] / years,
        "cost_drag": res["cost_frac"] / years,
        "worst_week": min(week_rets) if week_rets else None,
    }


def year_stats(res: dict, year: int) -> dict:
    """Return, max DD (from the year's start), and cash share for one
    calendar year of a result."""
    pairs = [(t, r) for t, r in zip(res["timestamps"], res["day_returns"])
             if rotor_data.bar_day(t).startswith(str(year))]
    eq, curve = 1.0, [1.0]
    for _, r in pairs:
        eq *= 1.0 + r
        curve.append(eq)
    inv = [i for t, i in zip(res["timestamps"], res["invested"])
           if rotor_data.bar_day(t).startswith(str(year))]
    cash_share = (sum(1 for i in inv if i == 0.0) / len(inv)) if inv else None
    return {"ret": eq - 1.0, "max_drawdown": _max_drawdown(curve),
            "days": len(pairs), "cash_share": cash_share}


# ── Grid + robust selection ─────────────────────────────────────────────
def run_grid(inputs: dict) -> list:
    """All 16 cells. Each dict: params + metrics + result."""
    cells = []
    for gate_ma in GATE_GRID:
        for ma_filter in FILTER_GRID:
            for formation in FORMATION_GRID:
                for n_held in N_GRID:
                    res = run_backtest(inputs, formation, n_held, gate_ma, ma_filter)
                    cells.append({
                        "formation": formation, "n_held": n_held,
                        "gate_ma": gate_ma, "ma_filter": ma_filter,
                        "metrics": compute_cell_metrics(res),
                        "result": res,
                    })
    return cells


def select_robust(cells: list) -> dict:
    """Robustness over peak, adapted to this grid's shape — stated plainly:

    The neighborhood plane is formation x N per (gate, filter) combo, but
    those planes are 2x2 — a 3x3 window clipped to a 2x2 plane covers the
    ENTIRE plane for every cell, and no cell has a full 3x3 neighborhood, so
    Calm's interior-only rule would leave nothing eligible. The faithful
    degeneration: every cell's neighborhood IS its whole plane (4 cells), so
    the score (neighborhood mean - std) first picks the most CONSISTENT
    (gate, filter) plane; within that plane the neighborhood stats tie, and
    the tie-breaks (neighborhood min — also tied — then own Sharpe) pick the
    strongest cell of the winning plane. All cells are eligible under this
    definition.
    """
    by_plane = {}
    for c in cells:
        by_plane.setdefault((c["gate_ma"], c["ma_filter"]), []).append(c)
    for plane in by_plane.values():
        sharpes = [c["metrics"]["sharpe"] for c in plane]
        if any(s is None for s in sharpes):
            for c in plane:
                c["nbhd_mean"] = c["nbhd_std"] = c["nbhd_min"] = None
                c["nbhd_score"] = float("-inf")
            continue
        m, sd = statistics.mean(sharpes), statistics.stdev(sharpes)
        for c in plane:
            c["nbhd_mean"], c["nbhd_std"] = m, sd
            c["nbhd_min"] = min(sharpes)
            c["nbhd_score"] = m - sd
    return max(cells, key=lambda c: (
        c["nbhd_score"],
        c["nbhd_min"] if c["nbhd_min"] is not None else float("-inf"),
        c["metrics"]["sharpe"] if c["metrics"]["sharpe"] is not None else float("-inf"),
    ))
