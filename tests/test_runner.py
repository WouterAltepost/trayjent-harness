"""Orchestrator tests (brief Step 4) — the two L5 ordering invariants and the
previous_score threading (L6), the subtleties that bit live and must be
reproduced exactly. Plus the Steady trailing-exit branch and the sizing_config
plumbing (Steady redesign Step 1, Commit 4).

All synthetic + offline: a tmp DATA_DIR with SPY/^VIX/A..E daily bars and a FAKE
score_batch, so the runner's orchestration is exercised without the scorers,
cache, or network. The fake returns controlled scores so buys/sells are
deterministic and the invariants are observable in portfolio state.

    python tests/test_runner.py
"""
import dataclasses
import os
import sys
import tempfile

import pandas as pd

_HARNESS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HARNESS_ROOT not in sys.path:
    sys.path.insert(0, _HARNESS_ROOT)

import config
from runner.configs import RunConfig
from runner.run import run_backtest


# ── Synthetic fixture ───────────────────────────────────────────────────
# 302 daily bars: warm-up (MIN_ROWS=300) clears at index 299, so decision marks
# are indices 299, 300, 301. Closes are flat 100 (indicator values are unused —
# the fake scorer drives decisions); only A jumps to 110 at index 300 to fire a
# +10% take-profit on the second tick. Trailing-branch paths: TR rises to 112
# at 300 then falls to 104 at 301 (trailing stop fires on the pullback); HS
# drops to 96 at 300 (through the 3% hard floor while its trailing level is
# still sub-entry).
_FIXTURE = {}
_TICKERS = ["A", "B", "C", "D", "E"]
_N = 302


def _write(data_dir, ticker, closes, vol=1000):
    ts = pd.date_range("2022-01-03 21:00", periods=len(closes), freq="B", tz="UTC")
    df = pd.DataFrame({
        "timestamp": ts, "open": closes, "high": closes, "low": closes,
        "close": closes, "volume": [vol] * len(closes),
        "ticker": ticker, "timeframe": "1d",
    })
    path = os.path.join(data_dir, "1d", f"{ticker}.parquet")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)
    return ts


def _setup():
    if _FIXTURE:
        config.DATA_DIR = _FIXTURE["dir"]
        return _FIXTURE
    data_dir = tempfile.mkdtemp(prefix="tbh_runner_")
    config.DATA_DIR = data_dir
    ts = _write(data_dir, "SPY", [100.0] * _N)          # calendar anchor
    _write(data_dir, "^VIX", [15.0] * _N, vol=0)        # get_vix_asof source
    for t in _TICKERS:
        closes = [100.0] * _N
        if t == "A":
            closes[300] = 110.0                          # +10% -> take-profit
        _write(data_dir, t, closes)
    tr = [100.0] * _N
    tr[300], tr[301] = 112.0, 104.0                      # run up, pull back
    _write(data_dir, "TR", tr)
    hs = [100.0] * _N
    hs[300] = 96.0                                       # straight through -3%
    _write(data_dir, "HS", hs)
    _FIXTURE.update({"dir": data_dir, "ts": ts})
    return _FIXTURE


def _strategy(watchlist):
    """A Steady-shaped strategy with a controllable watchlist (no rsi ceiling,
    no max-hold — keeps the cascade to the held/sizing path under test)."""
    return {
        "name": "steady",
        "buy_threshold": 7,
        "take_profit": 0.05,
        "stop_loss": 0.03,
        "cash_safety_pct": 0.80,
        "watchlist": watchlist,
    }


def _trailing_strategy(watchlist):
    """Steady-shaped strategy with the redesign's trailing exit enabled. Small
    dials (ATR period 3, mult 1.0) so the 302-bar fixture can move the stop."""
    return dict(_strategy(watchlist),
                use_trailing_stop=True, trailing_atr_mult=1.0, atr_period=3)


def _run_config(watchlist, end_index, strategy=None):
    f = _setup()
    end = f["ts"][end_index]                              # daily close instant
    return RunConfig(
        name="steady", strategy=strategy or _strategy(watchlist),
        indicator_tf="1d", indicator_bars=300, decision_tf="1d", cadence="daily",
        start=pd.Timestamp("2000-01-01", tz="UTC"), end=end, mode="rules_only",
    )


class FakeScorer:
    """score_batch(signals, strategy, mc) -> decisions, with scores from a
    per-call plan. Records the signals each call receives so previous_score
    threading is observable."""

    def __init__(self, plan):
        self.plan = plan          # list of {ticker: score} per decision tick
        self.i = 0
        self.seen = []            # snapshot of signals per call

    def __call__(self, signals, strategy, market_context):
        self.seen.append([dict(s) for s in signals])
        scores = self.plan[self.i] if self.i < len(self.plan) else {}
        self.i += 1
        out = []
        for s in signals:
            score = scores.get(s["ticker"], 0)
            out.append({
                "ticker": s["ticker"], "score": score, "reasoning": "fake",
                "action": "BUY" if score >= strategy["buy_threshold"] else "SKIP",
            })
        return out


# ── L5 invariant 1: held snapshot before exits blocks same-tick rebuy ───
def test_held_snapshot_blocks_same_tick_rebuy():
    rc = _run_config(["A"], end_index=300)               # ticks 299, 300
    # Tick 0: BUY A @100. Tick 1: A @110 hits TP in the exit pass AND the fake
    # re-issues a BUY for A. Because `held` is snapshotted before exits, A is
    # still "held" for the cascade -> SKIP (ALREADY HELD), no rebuy.
    fake = FakeScorer([{"A": 10}, {"A": 10}])
    res = run_backtest(rc, score_batch=fake)

    assert len(res.closed_trades) == 1, "exactly one round-trip (the TP), no rebuy"
    t = res.closed_trades[0]
    assert t.ticker == "A" and t.exit_reason == "SELL (TAKE PROFIT)"
    assert round(t.realized_pct, 1) == 10.0
    # A is flat after the TP tick; if the snapshot were taken AFTER exits, A
    # would have been re-bought and still open here.
    assert res.equity_curve[1]["n_positions"] == 0, "sold ticker not re-bought same tick"
    assert "A" not in res.portfolio.held_tickers()


# ── L5 invariant 2: PV anchor fixed once, ctx cash decrements per buy ────
def test_pv_anchor_fixed_and_cash_decrements_per_buy():
    saved = config.INITIAL_CAPITAL
    config.INITIAL_CAPITAL = 10_000.0                    # make cash_cap bind late
    try:
        rc = _run_config(_TICKERS, end_index=299)        # single tick (idx 299)
        # All five score 10 -> multiplier 4 -> desired = 4*0.05*PV = 0.20*PV.
        fake = FakeScorer([{t: 10 for t in _TICKERS}])
        res = run_backtest(rc, score_batch=fake)

        pos = res.portfolio._positions
        assert set(pos) == set(_TICKERS), "all five bought in one tick"
        # PV anchor fixed at 10_000 for every buy: desired = 0.20 * 10_000 = 2000.
        # Buys A..D each take 2000 (score_ladder); cash falls 10k->8k->6k->4k->2k.
        for t in ("A", "B", "C", "D"):
            assert pos[t]["notional"] == 2000.0, f"{t} sized off the fixed PV anchor"
            assert pos[t]["meta"]["sizing_rule_bound"] == "score_ladder"
        # By E, live cash is 2000 -> cash_cap 0.8*2000 = 1600 < desired 2000, so
        # E is clamped to 1600. Only possible if ctx["cash"] decremented per buy.
        assert pos["E"]["notional"] == 1600.0, "later buy sees shrunk live cash"
        assert pos["E"]["meta"]["sizing_rule_bound"] == "cash_cap"
        assert round(res.portfolio.cash, 2) == 400.0     # 10000 - (2000*4 + 1600)
    finally:
        config.INITIAL_CAPITAL = saved


# ── L5 invariant 2 (cont.): buys size off PRE-exit cash, not sale proceeds ─
def test_buy_sized_off_pre_exit_cash_not_sale_proceeds():
    saved = config.INITIAL_CAPITAL
    config.INITIAL_CAPITAL = 10_000.0
    try:
        rc = _run_config(_TICKERS, end_index=300)        # ticks 299, 300
        # Tick 0: buy A,C,D,E (2000 each) -> cash 2000, four positions @100.
        # Tick 1: A jumps to 110 and take-profits in the exit pass (+2200
        # proceeds); B is buy-signaled the same tick. The pre-exit cash snapshot
        # is 2000, so B's cash_cap = 0.8*2000 = 1600 < desired (0.2*PV ~ 2040)
        # -> B is clamped to 1600. If A's proceeds were available (post-exit cash
        # 4200), cash_cap would be 3360 and B would size at 2040 (score_ladder).
        fake = FakeScorer([
            {"A": 10, "C": 10, "D": 10, "E": 10},        # B omitted -> SKIP
            {"B": 10},                                   # only B buys this tick
        ])
        res = run_backtest(rc, score_batch=fake)

        pos = res.portfolio._positions
        assert "A" not in pos, "A took profit this tick"
        assert any(t.ticker == "A" and t.exit_reason == "SELL (TAKE PROFIT)"
                   for t in res.closed_trades)
        assert pos["B"]["notional"] == 1600.0, "B sized off pre-exit cash (proceeds excluded)"
        assert pos["B"]["meta"]["sizing_rule_bound"] == "cash_cap"
    finally:
        config.INITIAL_CAPITAL = saved


# ── L6: previous_score threaded from the prior decision, omitted cold ───
def test_previous_score_threaded_and_omitted_on_cold_start():
    rc = _run_config(["A"], end_index=300)               # ticks 299, 300
    fake = FakeScorer([{"A": 5}, {"A": 8}])              # both below threshold -> no trades
    run_backtest(rc, score_batch=fake)

    first_signal_A = next(s for s in fake.seen[0] if s["ticker"] == "A")
    second_signal_A = next(s for s in fake.seen[1] if s["ticker"] == "A")
    assert "previous_score" not in first_signal_A, "cold start omits previous_score"
    assert second_signal_A["previous_score"] == 5, "prior tick's score threaded in"


# ── Steady redesign Step 1: the trailing branch replaces the fixed TP ───
def test_steady_trailing_replaces_take_profit():
    """TR sits at +12% on tick 1 — past the old 5% TP — and does NOT sell;
    the pullback to 104 then crosses the ratcheted trailing level
    (112 - 1.0 * ATR(3) of 20/3 ≈ 105.33) -> SELL (TRAILING STOP). HS falls
    to 96, through the 97.0 hard floor, while its trailing level is still
    sub-entry -> SELL (STOP LOSS) via the fresh-phase handoff."""
    wl = ["TR", "HS"]
    rc = _run_config(wl, end_index=301, strategy=_trailing_strategy(wl))
    fake = FakeScorer([{"TR": 10, "HS": 10}])            # buy both @100 on tick 0
    res = run_backtest(rc, score_batch=fake)

    reasons = {t.ticker: t.exit_reason for t in res.closed_trades}
    assert reasons == {"TR": "SELL (TRAILING STOP)", "HS": "SELL (STOP LOSS)"}
    assert not any(t.exit_reason == "SELL (TAKE PROFIT)" for t in res.closed_trades)

    f = _setup()
    tr = next(t for t in res.closed_trades if t.ticker == "TR")
    # The +12% tick did not exit (old Steady's TP would have): TR left on the
    # NEXT tick at the 104 close — the winner ran past the old 5% cap.
    assert tr.exit_ts == f["ts"][301] and tr.exit_price == 104.0
    assert round(tr.realized_pct, 1) == 4.0
    hs = next(t for t in res.closed_trades if t.ticker == "HS")
    assert hs.exit_ts == f["ts"][300] and hs.exit_price == 96.0


def test_pulse_path_unchanged_tp_and_max_hold():
    """A strategy WITHOUT use_trailing_stop keeps the exact TP/SL/max-hold
    path — the trailing branch must not leak into Pulse. A's +10% still
    take-profits; flat B ages out (bar 299 is a Friday, so the Monday tick is
    72h later, past the 48h max hold)."""
    wl = ["A", "B"]
    pulse = {
        "name": "pulse", "buy_threshold": 6,
        "take_profit": 0.05, "stop_loss": 0.03, "max_hold_hours": 48,
        "cash_safety_pct": 0.80, "watchlist": wl,
    }
    rc = _run_config(wl, end_index=301, strategy=pulse)
    fake = FakeScorer([{"A": 10, "B": 10}])
    res = run_backtest(rc, score_batch=fake)

    reasons = {t.ticker: t.exit_reason for t in res.closed_trades}
    assert reasons == {"A": "SELL (TAKE PROFIT)", "B": "SELL (MAX HOLD)"}
    assert not any(t.exit_reason == "SELL (TRAILING STOP)" for t in res.closed_trades)


# ── sizing_config plumbing: RunConfig override vs POSITION_SIZING default ─
def test_sizing_config_override_and_default_fallback():
    saved = config.INITIAL_CAPITAL
    config.INITIAL_CAPITAL = 10_000.0
    try:
        rc = _run_config(["C"], end_index=299)           # single tick
        base = run_backtest(rc, score_batch=FakeScorer([{"C": 10}]))
        # sizing_config=None falls back to config.POSITION_SIZING:
        # multiplier 4 * 0.05 * 10k = 2000.
        assert base.portfolio._positions["C"]["notional"] == 2000.0

        halved = dict(config.POSITION_SIZING, base_pct_per_score=0.025)
        rc2 = dataclasses.replace(rc, sizing_config=halved)
        override = run_backtest(rc2, score_batch=FakeScorer([{"C": 10}]))
        # Same run, one dial overridden: 4 * 0.025 * 10k = 1000 — the override
        # reached the cascade through the RunConfig.
        assert override.portfolio._positions["C"]["notional"] == 1000.0
    finally:
        config.INITIAL_CAPITAL = saved


if __name__ == "__main__":
    test_held_snapshot_blocks_same_tick_rebuy()
    test_pv_anchor_fixed_and_cash_decrements_per_buy()
    test_buy_sized_off_pre_exit_cash_not_sale_proceeds()
    test_previous_score_threaded_and_omitted_on_cold_start()
    test_steady_trailing_replaces_take_profit()
    test_pulse_path_unchanged_tp_and_max_hold()
    test_sizing_config_override_and_default_fallback()
    print("test_runner OK: held-snapshot rebuy block, PV anchor fixed + cash "
          "decrements per buy, pre-exit cash sizing, previous_score threaded/omitted, "
          "trailing branch (no TP), pulse path unchanged, sizing_config override.")
