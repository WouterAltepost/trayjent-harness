"""Offline tests for the Phase 3 portfolio simulator.

Centre of gravity is the hand-calc P&L oracle: a scripted multi-trade scenario
where every cash balance, unrealized_pct, realized P&L, and ending PV is
asserted against numbers worked out by hand. Plus guards (fail-closed) and the
fidelity invariants (fractional qty, zero-fee identity, ledger/meta passthrough).
Pure arithmetic — no Parquet, no network.

Runnable via pytest or directly:
    python tests/test_portfolio.py
"""
import math
import os
import sys
from datetime import datetime, timezone

# Make the `simulator` package importable when run from anywhere.
_HARNESS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HARNESS_ROOT not in sys.path:
    sys.path.insert(0, _HARNESS_ROOT)

from simulator.portfolio import ClosedTrade, Portfolio

T0 = datetime(2026, 6, 1, 14, 0, tzinfo=timezone.utc)   # entry bar
T1 = datetime(2026, 6, 2, 14, 0, tzinfo=timezone.utc)   # exit bar
T2 = datetime(2026, 6, 3, 14, 0, tzinfo=timezone.utc)   # re-entry bar


def _close(a, b):
    return math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9)


# ── Hand-calc oracle ────────────────────────────────────────────────────
def test_hand_calc_oracle():
    """Two concurrent positions, one TP win + one SL loss, cash decrementing
    across same-bar buys, and a re-entry after close. Every number by hand."""
    p = Portfolio(starting_cash=100_000.0)

    # Same-bar BUYs. A: $20k @ $100 -> 200 sh. B: $30k @ $50 -> 600 sh.
    p.buy("A", notional=20_000.0, price=100.0, ts=T0, meta={"sizing_rule_bound": "score_ladder"})
    assert _close(p.cash, 80_000.0)                     # 100k - 20k
    p.buy("B", notional=30_000.0, price=50.0, ts=T0)
    assert _close(p.cash, 50_000.0)                     # 80k - 30k (decrement across same-bar buys)

    assert p.held_tickers() == {"A", "B"}

    # PV priced at entry == starting capital (no P&L yet).
    assert _close(p.portfolio_value({"A": 100.0, "B": 50.0}), 100_000.0)

    # Mark: A -> $105 (+5%, TP), B -> $48.50 (-3%, SL).
    dicts = {d["ticker"]: d for d in p.position_dicts({"A": 105.0, "B": 48.50})}
    assert _close(dicts["A"]["unrealized_pct"], 5.0)
    assert _close(dicts["B"]["unrealized_pct"], -3.0)
    assert _close(dicts["A"]["qty"], 200.0)
    assert _close(dicts["B"]["qty"], 600.0)
    assert _close(dicts["A"]["unrealized_pl"], 1_000.0)      # (105-100)*200
    assert _close(dicts["B"]["unrealized_pl"], -900.0)       # (48.50-50)*600
    assert _close(dicts["A"]["market_value"], 21_000.0)      # 200*105
    assert _close(dicts["A"]["avg_entry_price"], 100.0)
    # Valuation alone never ratchets high_water (L4): still the entry peak
    # even though A is marked at 105.
    assert _close(dicts["A"]["high_water"], 100.0)
    assert dicts["A"].keys() == {
        "ticker", "qty", "avg_entry_price", "current_price",
        "market_value", "unrealized_pl", "unrealized_pct", "high_water",
    }
    # PV with the marked prices: 50k cash + 21k + 29.1k = 100.1k.
    assert _close(p.portfolio_value({"A": 105.0, "B": 48.50}), 100_100.0)

    # SELL A @105 (TP win): realized +$1,000, +5%.
    ta = p.sell("A", price=105.0, ts=T1, exit_reason="SELL (TAKE PROFIT)")
    assert _close(ta.realized_dollars, 1_000.0)
    assert _close(ta.realized_pct, 5.0)
    assert ta.exit_reason == "SELL (TAKE PROFIT)"
    assert _close(p.cash, 71_000.0)                     # 50k + 200*105

    # SELL B @48.50 (SL loss): realized -$900, -3%.
    tb = p.sell("B", price=48.50, ts=T1, exit_reason="SELL (STOP LOSS)")
    assert _close(tb.realized_dollars, -900.0)
    assert _close(tb.realized_pct, -3.0)
    assert _close(p.cash, 100_100.0)                    # 71k + 600*48.50 = 71k + 29.1k

    # Flat: ending PV == ending cash == starting + net realized.
    assert p.held_tickers() == set()
    assert _close(p.portfolio_value({}), 100_100.0)
    assert _close(p.portfolio_value({}), 100_000.0 + 1_000.0 - 900.0)
    assert len(p.closed_trades) == 2

    # Re-BUY A after its close succeeds (single-lot store freed on close).
    p.buy("A", notional=10_000.0, price=110.0, ts=T2)
    assert "A" in p.held_tickers()
    assert _close(p.cash, 90_100.0)                     # 100.1k - 10k


# ── Guards (fail closed) ────────────────────────────────────────────────
def test_buy_while_held_raises():
    p = Portfolio(100_000.0)
    p.buy("A", 10_000.0, 100.0, T0)
    try:
        p.buy("A", 5_000.0, 100.0, T0)
        assert False, "expected ValueError on BUY of already-held ticker"
    except ValueError as e:
        assert "already-held" in str(e)


def test_sell_while_flat_raises():
    p = Portfolio(100_000.0)
    try:
        p.sell("A", 100.0, T1, "SELL (TAKE PROFIT)")
        assert False, "expected ValueError on SELL of not-held ticker"
    except ValueError as e:
        assert "not-held" in str(e)


def test_missing_price_raises():
    p = Portfolio(100_000.0)
    p.buy("A", 10_000.0, 100.0, T0)
    for call in (lambda: p.portfolio_value({}), lambda: p.position_dicts({})):
        try:
            call()
            assert False, "expected KeyError for held ticker missing from price map"
        except KeyError as e:
            assert "A" in str(e)


# ── Fidelity invariants ─────────────────────────────────────────────────
def test_fractional_qty():
    p = Portfolio(100_000.0)
    p.buy("A", notional=1_000.0, price=3.0, ts=T0)
    [d] = p.position_dicts({"A": 3.0})
    assert _close(d["qty"], 1_000.0 / 3.0)              # 333.333...
    assert _close(d["qty"] * 3.0, 1_000.0)             # qty*price reconstructs notional


def test_zero_fee_identity():
    """After a full round-trip to flat, cash == starting + Σ realized."""
    p = Portfolio(100_000.0)
    p.buy("A", 20_000.0, 100.0, T0)
    p.buy("B", 30_000.0, 50.0, T0)
    p.sell("A", 105.0, T1, "SELL (TAKE PROFIT)")
    p.sell("B", 48.50, T1, "SELL (STOP LOSS)")
    realized_sum = sum(t.realized_dollars for t in p.closed_trades)
    assert _close(p.cash, 100_000.0 + realized_sum)
    assert _close(p.portfolio_value({}), 100_000.0 + realized_sum)


def test_idealized_fill_is_raw_close():
    """v1: _fill_price returns the raw close on both sides (0 fees, 0 slippage)."""
    p = Portfolio(100_000.0, fees_bps=0.0, slippage_bps=0.0)
    assert _close(p._fill_price(100.0, "buy"), 100.0)
    assert _close(p._fill_price(100.0, "sell"), 100.0)


def test_ledger_meta_and_exit_reason():
    p = Portfolio(100_000.0)
    p.buy("A", 10_000.0, 100.0, T0, meta={"sizing_rule_bound": "score_ladder", "score": 9})
    trade = p.sell("A", 110.0, T1, exit_reason="SELL (TAKE PROFIT)")
    assert isinstance(trade, ClosedTrade)
    assert trade.meta == {"sizing_rule_bound": "score_ladder", "score": 9}
    assert trade.exit_reason == "SELL (TAKE PROFIT)"
    assert trade.entry_ts == T0 and trade.exit_ts == T1
    assert _close(trade.notional, 10_000.0)
    assert _close(trade.qty, 100.0)
    assert _close(trade.realized_dollars, 1_000.0)     # (110-100)*100
    assert p.closed_trades[-1] is trade


def test_filled_at_and_equity_point():
    p = Portfolio(100_000.0)
    assert p.filled_at("A") is None
    p.buy("A", 10_000.0, 100.0, T0)
    assert p.filled_at("A") == T0
    pt = p.equity_point(T0, {"A": 100.0})
    assert pt == {"ts": T0, "portfolio_value": 100_000.0, "cash": 90_000.0, "n_positions": 1}


# ── High-water mark (Steady redesign Step 1, Commit 2) ─────────────────
def test_high_water_starts_at_entry_fill():
    p = Portfolio(100_000.0)
    p.buy("A", 10_000.0, 100.0, T0)
    [d] = p.position_dicts({"A": 100.0})
    assert _close(d["high_water"], 100.0)               # zero-cost fill == buy price


def test_high_water_ratchets_up():
    p = Portfolio(100_000.0)
    p.buy("A", 10_000.0, 100.0, T0)
    p.update_high_water({"A": 108.0})
    [d] = p.position_dicts({"A": 105.0})
    assert _close(d["high_water"], 108.0)               # stored peak, not the mark price


def test_high_water_never_drops():
    p = Portfolio(100_000.0)
    p.buy("A", 10_000.0, 100.0, T0)
    p.update_high_water({"A": 108.0})
    p.update_high_water({"A": 95.0})
    [d] = p.position_dicts({"A": 95.0})
    assert _close(d["high_water"], 108.0)


def test_high_water_tracks_running_peak():
    p = Portfolio(100_000.0)
    p.buy("A", 10_000.0, 100.0, T0)
    for px, expected in ((104.0, 104.0), (98.0, 104.0), (110.0, 110.0)):
        p.update_high_water({"A": px})
        [d] = p.position_dicts({"A": px})
        assert _close(d["high_water"], expected)


def test_update_high_water_missing_held_ticker_raises():
    p = Portfolio(100_000.0)
    p.buy("A", 10_000.0, 100.0, T0)
    try:
        p.update_high_water({})
        assert False, "expected KeyError for held ticker missing from price map"
    except KeyError as e:
        assert "A" in str(e)


def test_high_water_is_per_position():
    p = Portfolio(100_000.0)
    p.buy("A", 10_000.0, 100.0, T0)
    p.buy("B", 10_000.0, 50.0, T0)
    p.update_high_water({"A": 120.0, "B": 51.0})        # A peaks here...
    p.update_high_water({"A": 110.0, "B": 55.0})        # ...B peaks here
    dicts = {d["ticker"]: d for d in p.position_dicts({"A": 110.0, "B": 55.0})}
    assert _close(dicts["A"]["high_water"], 120.0)
    assert _close(dicts["B"]["high_water"], 55.0)


if __name__ == "__main__":
    test_hand_calc_oracle()
    test_buy_while_held_raises()
    test_sell_while_flat_raises()
    test_missing_price_raises()
    test_fractional_qty()
    test_zero_fee_identity()
    test_idealized_fill_is_raw_close()
    test_ledger_meta_and_exit_reason()
    test_filled_at_and_equity_point()
    test_high_water_starts_at_entry_fill()
    test_high_water_ratchets_up()
    test_high_water_never_drops()
    test_high_water_tracks_running_peak()
    test_update_high_water_missing_held_ticker_raises()
    test_high_water_is_per_position()
    print("test_portfolio OK: hand-calc oracle, guards, fractional, zero-fee identity, ledger/meta, high-water.")
