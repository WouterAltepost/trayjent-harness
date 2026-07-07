"""In-memory portfolio simulator (Phase 3) — pure, offline, deterministic.

Replaces the live Alpaca account + positions + trader with arithmetic:
configurable starting cash, notional BUYs, full SELLs, realized/unrealized P&L,
position dicts in the live `get_open_positions` shape, and a portfolio-value
series. No network, no live-code import, no mark-to-market state.

Design invariants (Phase 3 locked decisions):
- L2 Sizing is notional dollars in; qty = notional / fill_price, fractional,
  full float precision (mirrors live Alpaca notional orders).
- L3 Single lot per ticker; buy() of an already-held ticker raises (live never
  pyramids — a duplicate BUY is a runner bug, fail closed).
- L4 Valuation is stateless: portfolio_value(prices) / position_dicts(prices)
  always take a {ticker: price} map. No stored "current price".
- L5 Idealized fills: fill at the raw close, zero fees, zero slippage. Costs
  enter only through _fill_price.
- L7 position_dicts emits the live keys verbatim (unrealized_pct in percent)
  plus harness-only high_water, which the trailing exit consumes. Live Alpaca
  does not return a high-water field; live will track it via its own persisted
  state when the trailing stop is wired live (out of scope for the harness).
- L10 Long-only: sell() with no open position raises.
"""
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ClosedTrade:
    """One completed round-trip. The ledger's unit and the single P&L source of
    truth (L8). Mechanical facts are recorded by the portfolio; `exit_reason`
    and `meta` are opaque passthroughs the runner supplies."""
    ticker: str
    entry_ts: datetime
    entry_price: float
    exit_ts: datetime
    exit_price: float
    qty: float
    notional: float          # entry dollars deployed
    realized_dollars: float  # (exit_price - entry_price) * qty
    realized_pct: float      # (exit_price / entry_price - 1) * 100
    exit_reason: str         # e.g. "SELL (TAKE PROFIT)"
    meta: dict               # opaque passthrough from buy()


class Portfolio:
    """Mechanical trading engine. Owns cash (decrements on buy()), open lots,
    and the closed-trade ledger. Knows nothing about decisions, scoring, or the
    cascade — the runner (Phase 5) drives it and supplies decision metadata."""

    def __init__(self, starting_cash: float, fees_bps: float = 0.0, slippage_bps: float = 0.0):
        self._cash = float(starting_cash)
        self.fees_bps = float(fees_bps)
        self.slippage_bps = float(slippage_bps)
        # ticker -> {entry_price, high_water, qty, entry_ts, notional, meta}. One lot each (L3).
        self._positions: dict[str, dict] = {}
        self.closed_trades: list[ClosedTrade] = []

    # ── Cost hook (L5) — the SOLE place costs ever enter ────────────────
    def _fill_price(self, price: float, side: str) -> float:
        """Effective fill price for a side. v1: both bps are 0, so it returns
        the raw close verbatim (idealized fill). Non-zero costs degrade the
        fill — a BUY fills higher (fewer shares for the notional), a SELL fills
        lower (less proceeds). Keeping this the only cost hook makes turning on
        real fees/slippage a one-line change, never scattered fee math."""
        cost_bps = self.slippage_bps + self.fees_bps
        if cost_bps == 0.0:
            return price
        adj = cost_bps / 10_000.0
        return price * (1.0 + adj) if side == "buy" else price * (1.0 - adj)

    # ── Mutation ────────────────────────────────────────────────────────
    def buy(self, ticker: str, notional: float, price: float, ts: datetime, meta: dict = None) -> None:
        """Open a notional-dollar position at the bar close. Raises if the
        ticker is already held (L3) — the runner must SKIP (ALREADY HELD) first."""
        if ticker in self._positions:
            raise ValueError(
                f"buy() for already-held ticker {ticker!r}: the portfolio is single-lot "
                f"(L3) and live never pyramids. A duplicate BUY is a runner bug — fail closed."
            )
        fill = self._fill_price(price, "buy")
        qty = notional / fill
        self._cash -= notional
        self._positions[ticker] = {
            "entry_price": fill,
            "high_water": fill,  # peak since entry; ratcheted by update_high_water()
            "qty": qty,
            "entry_ts": ts,
            "notional": notional,
            "meta": dict(meta) if meta else {},
        }

    def sell(self, ticker: str, price: float, ts: datetime, exit_reason: str) -> ClosedTrade:
        """Fully close a position at the bar close. Raises if not held (L10,
        long-only). Returns the ClosedTrade and appends it to the ledger."""
        if ticker not in self._positions:
            raise ValueError(
                f"sell() for not-held ticker {ticker!r}: long-only, full-close model (L10). "
                f"A SELL with no open position is a programming error — fail closed."
            )
        pos = self._positions.pop(ticker)
        entry = pos["entry_price"]
        qty = pos["qty"]
        fill = self._fill_price(price, "sell")
        proceeds = qty * fill
        realized_dollars = (fill - entry) * qty
        realized_pct = (fill / entry - 1.0) * 100.0
        self._cash += proceeds
        trade = ClosedTrade(
            ticker=ticker,
            entry_ts=pos["entry_ts"],
            entry_price=entry,
            exit_ts=ts,
            exit_price=fill,
            qty=qty,
            notional=pos["notional"],
            realized_dollars=realized_dollars,
            realized_pct=realized_pct,
            exit_reason=exit_reason,
            meta=pos["meta"],
        )
        self.closed_trades.append(trade)
        return trade

    def update_high_water(self, prices: dict) -> None:
        """Ratchet each held position's high_water up to the passed price map:
        max(stored, prices[ticker]). Never lowers it. A held ticker missing
        from the map fails loud via _price_of — the same rule as
        portfolio_value / position_dicts, never silently skip a held lot."""
        for ticker, pos in self._positions.items():
            pos["high_water"] = max(pos["high_water"], self._price_of(ticker, prices))

    # ── Reads (no mutation) ─────────────────────────────────────────────
    def held_tickers(self) -> set:
        return set(self._positions)

    def filled_at(self, ticker: str):
        """Entry timestamp of the open lot, or None if not held. The sim knows
        this for free (it recorded the BUY), so the live lazy-Alpaca-call for
        max-hold disappears here."""
        pos = self._positions.get(ticker)
        return pos["entry_ts"] if pos else None

    def _price_of(self, ticker: str, prices: dict) -> float:
        """Look up a held ticker's price from the passed map, failing loud if
        absent (a runner bug — never silently skip a held position)."""
        try:
            return prices[ticker]
        except KeyError:
            raise KeyError(
                f"price map is missing held ticker {ticker!r}; valuation needs a price "
                f"for every open position (passed map covered {sorted(prices)})."
            ) from None

    def position_dicts(self, prices: dict) -> list:
        """Open positions in the live get_open_positions shape (L7) plus the
        harness-only high_water key, valued at `prices`. unrealized_pct is in
        percent: (price-entry)/entry*100, matching positions.py:38 so the
        runner's exit loop is a verbatim mirror of the live keys."""
        out = []
        for ticker, pos in self._positions.items():
            price = self._price_of(ticker, prices)
            entry = pos["entry_price"]
            qty = pos["qty"]
            out.append({
                "ticker": ticker,
                "qty": qty,
                "avg_entry_price": entry,
                "current_price": price,
                "market_value": qty * price,
                "unrealized_pl": (price - entry) * qty,
                "unrealized_pct": (price - entry) / entry * 100.0,
                "high_water": pos["high_water"],  # harness-only (L7 note)
            })
        return out

    @property
    def cash(self) -> float:
        return self._cash

    def portfolio_value(self, prices: dict) -> float:
        """cash + Σ qty*price over held tickers — the same identity Alpaca
        reports (cash + market value of positions). Stateless (L4): always from
        the passed price map."""
        total = self._cash
        for ticker, pos in self._positions.items():
            total += pos["qty"] * self._price_of(ticker, prices)
        return total

    def equity_point(self, ts: datetime, prices: dict) -> dict:
        """One point on the equity curve for the metrics layer (Phase 5)."""
        return {
            "ts": ts,
            "portfolio_value": self.portfolio_value(prices),
            "cash": self._cash,
            "n_positions": len(self._positions),
        }
