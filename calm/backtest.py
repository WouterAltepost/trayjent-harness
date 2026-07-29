"""Calm (shortvol sleeve) v1 backtest engine — TUNING phase.

Strategy v1, deliberately minimal:

- **Signal**: daily VIX/VIX3M ratio (``data_layer.shortvol`` term structure),
  raw or 3-day mean.
- **State machine with hysteresis**: FLAT -> LONG SVXY when ratio < entry;
  LONG -> FLAT (cash) when ratio > exit; between the two thresholds keep the
  current state, no new entries.
- **Sizing**: target fraction of equity = ``min(1, K / VIX)``, ``K=None`` ->
  always full size. The target is re-evaluated DAILY while long — the literal
  reading of the sizing rule, so rising VIX de-risks an open position — and
  every resize is a fill that pays costs. (Entry-only sizing is a one-line
  change in ``_target_weight`` if ever wanted.)
- **Execution**: ``mode="close"`` (primary — signal and fill at the SAME
  day's close, matching a live cron ~15:45 ET using near-final values) or
  ``mode="next_open"`` (sensitivity — signal at close, fill at the next
  session's open, overnight gap earned at the old weight). Cost:
  ``config.CALM_COST_BPS_PER_SIDE`` per side on every fill, charged on the
  traded weight delta.

LEVERAGE APPROXIMATION (label every output that uses it): SVXY was -1x
before ``config.SVXY_LEVERAGE_CHANGE_DATE`` and -0.5x after. The primary
return series approximates a continuous -0.5x instrument by scaling pre-flip
SVXY daily returns by 0.5. The pure post-flip sub-window needs no scaling.

SEALED WINDOW: everything from ``config.OOS_SHORTVOL[0]`` (2022-01-01)
onward is sealed out-of-sample for this sleeve. ``build_inputs`` fails
closed on any window that touches it unless ``allow_sealed=True`` is passed
deliberately.

Offline + deterministic: inputs are stored Parquet via data_layer.shortvol;
nothing here touches the network. Metric scalars reuse the hand-checkable
helpers in ``runner.metrics`` (one source of truth for CAGR/Sharpe/MDD).
"""
import statistics

import pandas as pd

import config
from data_layer import shortvol
from runner.metrics import _cagr, _max_drawdown, _periods_per_year, _sharpe, _years_between

FLAT, LONG = "FLAT", "LONG"

# ── Tuning grid (locked — 4 x 2 x 4 x 2 = 64 cells, do not expand) ──────
ENTRY_GRID = [0.90, 0.925, 0.95, 0.975]
EXIT_GRID = [1.00, 1.05]
K_GRID = [15, 20, 25, None]        # ascending; None = no vol target (K=inf)
SMOOTHING_GRID = ["raw", "3d"]

# Stress episodes for the chosen-config table (all inside the tuning window).
EPISODES = [
    ("2015-08 selloff", "2015-08-01", "2015-08-31"),
    ("2018-02 Volmageddon", "2018-02-01", "2018-02-28"),
    ("2020 COVID crash", "2020-02-01", "2020-04-30"),
]


# ── Inputs ──────────────────────────────────────────────────────────────
def _leverage_adjust(ts: "pd.Series", ret: "pd.Series") -> "pd.Series":
    """Scale returns of bars dated before the SVXY leverage flip by 0.5 (the
    continuous -0.5x approximation). The flip date itself is already -0.5x."""
    flip = pd.Timestamp(config.SVXY_LEVERAGE_CHANGE_DATE, tz="UTC")
    return ret.where(ts >= flip, ret * 0.5)


def build_inputs(start: str, end: str, allow_sealed: bool = False) -> "pd.DataFrame":
    """Assemble the daily backtest frame for [start, end] (ISO dates, incl).

    Columns: ``timestamp, ret_cc, ret_on, ret_id`` (SVXY close-to-close /
    overnight / intraday returns, leverage-approximated), ``ratio_raw,
    ratio_3d, vix`` (signal inputs; the 3-day mean is seeded from history
    BEFORE the window, so it is valid from the first row), ``spy_ret``.

    Fails closed: raises if the window touches the sealed OOS
    (``config.OOS_SHORTVOL``) without ``allow_sealed``, if any in-window
    signal value is missing, or if a return is missing anywhere but SVXY's
    very first bar (a data hole must not be silently zero-filled).
    """
    if end >= config.OOS_SHORTVOL[0] and not allow_sealed:
        raise ValueError(
            f"Window end {end} touches the sealed shortvol OOS "
            f"(starts {config.OOS_SHORTVOL[0]}). Pass allow_sealed=True only "
            "for a deliberate, one-shot validation read."
        )

    term = shortvol.load_term_structure()
    term = term.rename(columns={"vix_ratio": "ratio_raw"})
    # Smoothing computed on the FULL term history before slicing, so the
    # window's first rows use real prior days, not a truncated seed.
    term["ratio_3d"] = term["ratio_raw"].rolling(3, min_periods=3).mean()

    svxy = shortvol.load_bars("SVXY")
    spy = shortvol.load_bars("SPY")

    df = pd.DataFrame({"timestamp": svxy["timestamp"]})
    df["ret_cc"] = svxy["close"].pct_change()
    df["ret_on"] = svxy["open"] / svxy["close"].shift(1) - 1.0
    df["ret_id"] = svxy["close"] / svxy["open"] - 1.0
    for col in ("ret_cc", "ret_on", "ret_id"):
        df[col] = _leverage_adjust(df["timestamp"], df[col])

    df = df.merge(term[["timestamp", "vix", "ratio_raw", "ratio_3d"]],
                  on="timestamp", how="left")
    df = df.merge(
        pd.DataFrame({"timestamp": spy["timestamp"],
                      "spy_ret": spy["close"].pct_change()}),
        on="timestamp", how="left")

    day = df["timestamp"].dt.strftime("%Y-%m-%d")
    win = df[(day >= start) & (day <= end)].reset_index(drop=True)
    if win.empty:
        raise ValueError(f"No SVXY bars in [{start}, {end}]")

    if win[["vix", "ratio_raw", "ratio_3d"]].isna().any().any():
        bad = win[win[["vix", "ratio_raw", "ratio_3d"]].isna().any(axis=1)]
        raise ValueError(
            f"Missing signal data on {len(bad)} in-window day(s), first "
            f"{bad['timestamp'].iloc[0]} — data foundation hole, fix before tuning"
        )
    first_bar = svxy["timestamp"].iloc[0]
    for col in ("ret_cc", "ret_on", "ret_id", "spy_ret"):
        holes = win[win[col].isna() & (win["timestamp"] != first_bar)]
        if len(holes):
            raise ValueError(
                f"{col}: {len(holes)} missing return(s) beyond the inception "
                f"bar, first {holes['timestamp'].iloc[0]} — data hole"
            )
    return win


# ── State machine + sizing (pure) ───────────────────────────────────────
def next_state(state: str, ratio: float, entry: float, exit_: float):
    """One hysteresis step. Returns ``(new_state, completed_round_trip)``."""
    if state == FLAT and ratio < entry:
        return LONG, False
    if state == LONG and ratio > exit_:
        return FLAT, True
    return state, False


def _target_weight(state: str, k, vix: float) -> float:
    if state != LONG:
        return 0.0
    if k is None:
        return 1.0
    return min(1.0, k / vix)


# ── Engine ──────────────────────────────────────────────────────────────
def run_backtest(df: "pd.DataFrame", entry: float, exit_: float, k,
                 smoothing: str = "raw", mode: str = "close",
                 cost_bps: float = None) -> dict:
    """Simulate one parameter cell over ``df``. Deterministic, equity base 1.0.

    Returns ``{"equity": [float], "day_returns": [float], "weights": [float],
    "round_trips": int, "final_state": str}`` — ``equity``/``day_returns``
    are per close; ``weights[i]`` is the exposure held INTO day i's close
    (what earned day i's return), which is what time-in-market and the daily
    win rate should count.

    ``round_trips`` counts COMPLETED LONG->FLAT cycles; a position still
    open at the window end is not counted.
    """
    if mode not in ("close", "next_open"):
        raise ValueError(f"unknown mode {mode!r}")
    cost = (config.CALM_COST_BPS_PER_SIDE if cost_bps is None else cost_bps) / 1e4

    ratios = (df["ratio_raw"] if smoothing == "raw" else df["ratio_3d"]).tolist()
    vix = df["vix"].tolist()
    # NaN returns only exist on SVXY's inception bar (validated in
    # build_inputs); zero-fill is exact there (nothing is held yet).
    rcc = df["ret_cc"].fillna(0.0).tolist()
    ron = df["ret_on"].fillna(0.0).tolist()
    rid = df["ret_id"].fillna(0.0).tolist()

    state = FLAT
    w = 0.0                 # weight currently held
    pending = None          # next_open only: target decided at prior close
    equity = 1.0
    round_trips = 0
    curve, day_returns, weights = [], [], []

    for i in range(len(df)):
        e0 = equity
        if mode == "close":
            weights.append(w)
            equity *= 1.0 + w * rcc[i]
            state, completed = next_state(state, ratios[i], entry, exit_)
            round_trips += completed
            target = _target_weight(state, k, vix[i])
            equity *= 1.0 - cost * abs(target - w)   # fill at THIS close
            w = target
        else:  # next_open: overnight at old weight, fill at open, intraday at new
            equity *= 1.0 + w * ron[i]
            if pending is not None:
                equity *= 1.0 - cost * abs(pending - w)
                w = pending
                pending = None
            weights.append(w)                        # exposure into day i's close
            equity *= 1.0 + w * rid[i]
            state, completed = next_state(state, ratios[i], entry, exit_)
            round_trips += completed
            pending = _target_weight(state, k, vix[i])

        curve.append(equity)
        day_returns.append(equity / e0 - 1.0)

    return {"equity": curve, "day_returns": day_returns, "weights": weights,
            "round_trips": round_trips, "final_state": state}


def buy_and_hold(returns: list, cost_bps: float = None) -> dict:
    """Baseline: buy at the first close (paying one entry fill), hold to the
    end. Same result shape as :func:`run_backtest`."""
    cost = (config.CALM_COST_BPS_PER_SIDE if cost_bps is None else cost_bps) / 1e4
    equity = 1.0 - cost
    curve, day_returns, weights = [equity], [equity - 1.0], [0.0]
    for r in returns[1:]:
        e0 = equity
        equity *= 1.0 + (0.0 if pd.isna(r) else r)
        curve.append(equity)
        day_returns.append(equity / e0 - 1.0)
        weights.append(1.0)
    return {"equity": curve, "day_returns": day_returns, "weights": weights,
            "round_trips": 0, "final_state": LONG}


# ── Metrics ─────────────────────────────────────────────────────────────
def compute_cell_metrics(res: dict, timestamps: list) -> dict:
    """The 9 tuning metrics for one cell result.

    Definitions (stated once, used everywhere):
    - daily_win: share of days with exposure (weight > 0) whose net day
      return was > 0. Flat days are excluded — they carry no information.
    - pos_weeks / pos_months: share of calendar weeks (W-FRI) / months with
      net return > 0, among periods that had ANY exposure; fully-flat
      periods are excluded from the denominator so time-in-market does not
      mechanically dilute the number (it is reported separately).
    - time_in_market: share of days with exposure.
    - Sharpe: full daily net return stream (flat days included at 0), rf=0,
      annualized at the curve's empirical frequency (runner.metrics).
    - max_drawdown: from equity INCLUDING the 1.0 start, so a first-day
      loss counts.
    """
    eq, rets, wts = res["equity"], res["day_returns"], res["weights"]
    years = _years_between(timestamps[0], timestamps[-1])
    ppy = _periods_per_year(timestamps)

    exposed = [i for i, w in enumerate(wts) if w > 0]
    daily_win = (sum(1 for i in exposed if rets[i] > 0) / len(exposed)
                 if exposed else None)

    idx = pd.DatetimeIndex(timestamps)
    eq_s = pd.Series(eq, index=idx)
    exp_s = pd.Series([1.0 if w > 0 else 0.0 for w in wts], index=idx)

    def _positive_share(freq: str, base_offset: "pd.Timedelta"):
        # Prepend the 1.0 base far enough back to land in the PRIOR resample
        # bucket, so the first (possibly partial) week/month's return is
        # measured from inception rather than dropped as the NaN seed.
        base = pd.Series([1.0], index=[idx[0] - base_offset])
        per = pd.concat([base, eq_s]).resample(freq).last().dropna().pct_change().dropna()
        had_exp = exp_s.resample(freq).max().reindex(per.index).fillna(0.0)
        active = per[had_exp > 0]
        return (active > 0).mean() if len(active) else None

    week_base = pd.Timedelta(days=7)                  # always the prior W-FRI bucket
    month_base = pd.Timedelta(days=int(idx[0].day))   # last day of the prior month

    return {
        "cagr": _cagr(1.0, eq[-1], years),
        "sharpe": _sharpe(rets, ppy),
        "max_drawdown": _max_drawdown([1.0] + eq),
        "daily_win": daily_win,
        "pos_weeks": _positive_share("W-FRI", week_base),
        "pos_months": _positive_share("ME", month_base),
        "time_in_market": sum(1 for w in wts if w > 0) / len(wts),
        "round_trips": res["round_trips"],
        "worst_day": min(rets),
    }


def episode_stats(res: dict, timestamps: list, ep_start: str, ep_end: str) -> dict:
    """P&L and max drawdown of a result inside [ep_start, ep_end] (ISO,
    inclusive), measured from an equity base of 1.0 at the episode start."""
    days = [t.strftime("%Y-%m-%d") for t in timestamps]
    rets = [r for d, r in zip(days, res["day_returns"]) if ep_start <= d <= ep_end]
    eq, curve = 1.0, [1.0]
    for r in rets:
        eq *= 1.0 + r
        curve.append(eq)
    return {"pnl": eq - 1.0, "max_drawdown": _max_drawdown(curve), "days": len(rets)}


# ── Grid + robust selection ─────────────────────────────────────────────
def run_grid(df: "pd.DataFrame") -> list:
    """All 64 cells (mode="close"). Each dict: params + metrics + result."""
    timestamps = df["timestamp"].tolist()
    cells = []
    for smoothing in SMOOTHING_GRID:
        for exit_ in EXIT_GRID:
            for entry in ENTRY_GRID:
                for k in K_GRID:
                    res = run_backtest(df, entry, exit_, k, smoothing=smoothing)
                    cells.append({
                        "entry": entry, "exit": exit_, "k": k,
                        "smoothing": smoothing,
                        "metrics": compute_cell_metrics(res, timestamps),
                        "result": res,
                    })
    return cells


def annotate_neighborhoods(cells: list) -> None:
    """Attach 3x3-neighborhood Sharpe stats to every cell, in place.

    The neighborhood lives in the ENTRY x K plane (the two quasi-continuous
    axes; K ordered 15 < 20 < 25 < None==inf), holding EXIT and smoothing
    fixed — those are 2-value structural switches, not axes a 3x3 window can
    slide over. Score = mean - std of the (edge-clipped, center-inclusive)
    neighborhood Sharpe: high AND flat beats a lone spike (robustness over
    peak).
    """
    by_key = {(c["smoothing"], c["exit"], c["entry"], c["k"]): c for c in cells}
    for c in cells:
        ei = ENTRY_GRID.index(c["entry"])
        ki = K_GRID.index(c["k"])
        vals = []
        for de in (-1, 0, 1):
            for dk in (-1, 0, 1):
                e2, k2 = ei + de, ki + dk
                if 0 <= e2 < len(ENTRY_GRID) and 0 <= k2 < len(K_GRID):
                    n = by_key[(c["smoothing"], c["exit"], ENTRY_GRID[e2], K_GRID[k2])]
                    s = n["metrics"]["sharpe"]
                    if s is not None:
                        vals.append(s)
        c["nbhd_n"] = len(vals)
        if len(vals) >= 2 and c["metrics"]["sharpe"] is not None:
            c["nbhd_mean"] = statistics.mean(vals)
            c["nbhd_std"] = statistics.stdev(vals)
            c["nbhd_min"] = min(vals)
            c["nbhd_score"] = c["nbhd_mean"] - c["nbhd_std"]
        else:  # degenerate cell (no valid Sharpe) can never be chosen
            c["nbhd_mean"] = c["nbhd_std"] = None
            c["nbhd_min"] = None
            c["nbhd_score"] = float("-inf")


def select_robust(cells: list) -> dict:
    """The selection rule: NOT the best cell — the cell whose 3x3
    neighborhood Sharpe is most consistent. Ranked by neighborhood
    (mean - std), tie-broken by neighborhood min, then own Sharpe.

    Only cells with a FULL 3x3 neighborhood (the interior of the ENTRY x K
    plane) are eligible. A clipped corner/edge neighborhood has fewer,
    mutually correlated members and wins on a small-sample artifact exactly
    when the Sharpe surface rises toward the grid boundary — which also makes
    the corner the peak cell, the thing this rule exists to avoid. "Its 3x3
    neighborhood" must actually exist for the robustness claim to mean
    anything; edge cells still carry (clipped) annotations for the display.
    """
    annotate_neighborhoods(cells)
    interior = [c for c in cells if c["nbhd_n"] == 9]
    return max(interior, key=lambda c: (
        c["nbhd_score"],
        c["nbhd_min"] if c["nbhd_min"] is not None else float("-inf"),
        c["metrics"]["sharpe"] if c["metrics"]["sharpe"] is not None else float("-inf"),
    ))
