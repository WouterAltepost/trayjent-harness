"""Backtest orchestrator (Phase 5 brief Step 4).

A faithful re-orchestration of live ``workflow.run()`` with live I/O swapped for
harness primitives. It is orchestration + state threading ONLY — every decision
primitive already exists via the reuse shim (``evaluate_price_exit``,
``should_force_close_for_max_hold``, ``decide_buy_action``, ``compute_indicators*``,
``compute_breadth``) and the portfolio simulator. Nothing here re-implements
scoring, sizing, or exit logic.

Per decision ``as_of`` (from ``runner.schedule``):
  1. Price every watchlist ticker at the decision tf (no-lookahead slicer).
  2. **L5 invariant 1** — snapshot ``held = portfolio.held_tickers()`` BEFORE the
     exit pass, and pass that snapshot as ``ctx["held"]``, so a ticker sold this
     tick still blocks a same-tick rebuy.
  3. **L5 invariant 2** — take the PV anchor ``portfolio.portfolio_value(prices)``
     ONCE; ``ctx["portfolio_value"]`` is that fixed snapshot while ``ctx["cash"]``
     is a pre-exit cash snapshot decremented per buy (mirrors live
     ``current_cash_local``) so sequential buys in one tick see shrinking cash.
  4. Exit pass: ``evaluate_price_exit`` / max-hold → ``portfolio.sell``.
  5. Indicators on the indicator tf; thread ``previous_score`` (L6).
  6. Market context (VIX as-of + breadth); score (mode switch, L8).
  7. BUY cascade: ``decide_buy_action`` → ``portfolio.buy``.

Modeled omissions (L7), each slightly more aggressive than live, surfaced in the
output ``notes[]`` (Commit 6): never-tripped breaker, no-earnings stub, idealized
fills, market always open.

Cash for sizing matches live pre-exit cash sizing: live ``run()`` snapshots the
account (PV and cash) BEFORE the exit loop and only ever debits
``current_cash_local`` by buys, never credits it with this tick's sale proceeds.
The runner mirrors that — ``cash_local`` is snapshotted before the exit pass and
decremented per buy — so same-tick sale proceeds never fund same-tick buys, and
there is no double-count against the (also pre-exit) PV anchor. ``portfolio.cash``
stays the authoritative ledger (it does receive proceeds) and drives the equity
curve; only sizing reads the pre-exit snapshot.
"""
import os
from dataclasses import dataclass

import config
from data_layer.slice import get_window, get_price_asof, get_vix_asof
from harness.reuse import (
    compute_indicators,
    compute_indicators_pulse,
    compute_breadth,
    evaluate_price_exit,
    should_force_close_for_max_hold,
    decide_buy_action,
    build_scoring_prompt,
    parse_scoring_response,
    SCORING_MODEL,
)
from scoring.rules_only import score_rules_only
from scoring.cost import CostTracker, CostCeilingExceeded
from simulator.portfolio import Portfolio
from runner.schedule import decision_points

# Neutral cascade inputs for the modeled omissions (L7). A never-tripped breaker
# (is_tripped -> False) and a no-earnings stub make the backtest take buys live
# would have blocked; documented in the output notes[].
_NEVER_TRIPPED_RISK = {"tripped": False, "peak_pv": 0.0, "baseline_source": "backtest"}
_NO_EARNINGS = {"has_earnings_soon": False}


@dataclass
class RunResult:
    """Raw materials for the metrics + JSON layers (Commits 5-6)."""
    run_config: object
    portfolio: Portfolio
    closed_trades: list
    equity_curve: list
    n_decision_points: int
    cost_tracker: object = None
    halted_on_cost: bool = False


def _score_claude(signals, strategy, market_context, scorer):
    """Harness mirror of ``agent.score_signals``: assemble the prompt, call the
    injected scorer, parse — all via the reused ``scoring_core`` halves."""
    prompt = build_scoring_prompt(signals, strategy, market_context)
    content = scorer(prompt)
    return parse_scoring_response(content, strategy)


def _build_score_batch(run_config, cache):
    """Resolve ``score_batch(signals, strategy, mc) -> decisions`` for the run's
    mode (L8). Returns ``(score_batch, cost_tracker)``.

    rules_only -> the deterministic ladder, no tracker. claude -> a per-run
    CostTracker + the cache-backed Anthropic scorer (heavy imports lazily so a
    rules-only run needs no anthropic / cache / API key).
    """
    if run_config.mode == "rules_only":
        return (lambda sg, st, mc: score_rules_only(sg, st, mc)), None

    tracker = CostTracker(config.COST_CEILING_USD, config.PRICES, SCORING_MODEL)
    if cache is None:
        from scoring.cache import ScoringCache
        cache = ScoringCache(os.path.join(config.CACHE_DIR, "scoring.db"))
    from scoring.anthropic_scorer import make_anthropic_scorer
    from scoring.cached_scorer import make_cached_scorer
    scorer = make_cached_scorer(
        cache, make_anthropic_scorer(), tracker,
        model=SCORING_MODEL, prompt_version=config.PROMPT_VERSION,
        strategy=run_config.strategy["name"],
    )
    return (lambda sg, st, mc: _score_claude(sg, st, mc, scorer)), tracker


def run_backtest(run_config, *, score_batch=None, cache=None) -> RunResult:
    """Replay ``run_config`` over historical bars and return a :class:`RunResult`.

    ``score_batch`` is injectable (tests pass a fake; default resolves per L8).
    ``cache`` overrides the claude-mode scoring cache (default opens the run cache).
    """
    strategy = run_config.strategy
    is_pulse = strategy["name"] == "pulse"
    watchlist = strategy["watchlist"]
    indicator_tf = run_config.indicator_tf
    indicator_bars = run_config.indicator_bars
    decision_tf = run_config.decision_tf
    # evaluate_price_exit takes percent units; strategy stores fractions (L2).
    tp_pct = strategy["take_profit"] * 100.0
    sl_pct = strategy["stop_loss"] * 100.0
    max_hold_hours = strategy.get("max_hold_hours")

    cost_tracker = None
    if score_batch is None:
        score_batch, cost_tracker = _build_score_batch(run_config, cache)

    portfolio = Portfolio(config.INITIAL_CAPITAL, config.FEES_BPS, config.SLIPPAGE_BPS)
    last_score = {}                 # ticker -> prior decision's score (L6)
    equity_curve = []
    marks = decision_points(run_config)
    halted_on_cost = False

    for as_of in marks:
        # Decision-tf price for each ticker (no-lookahead). A ticker with no
        # closed decision-tf bar at T is unpriceable -> excluded this tick
        # (mirrors live's per-ticker try/except continue).
        prices = {}
        for ticker in watchlist:
            try:
                prices[ticker] = get_price_asof(ticker, decision_tf, as_of)
            except (ValueError, FileNotFoundError):
                continue

        # L5 invariant 1: held snapshot BEFORE the exit pass.
        held = portfolio.held_tickers()
        # L5 invariant 2: PV anchor + cash snapshot once, BEFORE the exit pass
        # (mirrors live's pre-exit account snapshot). cash_local is decremented
        # per buy and never credited by this tick's sale proceeds, so same-tick
        # sells never fund same-tick buys (no double-count vs the PV anchor).
        pv_anchor = portfolio.portfolio_value(prices)
        cash_local = portfolio.cash

        # Step 1: exit pass over open positions at the decision-tf price.
        # position_dicts() returns a fresh list, so selling mid-loop is safe.
        for pos in portfolio.position_dicts(prices):
            ticker = pos["ticker"]
            reason = evaluate_price_exit(pos["unrealized_pct"], tp_pct, sl_pct)
            if reason is None and max_hold_hours:
                if should_force_close_for_max_hold(
                        portfolio.filled_at(ticker), max_hold_hours, now=as_of):
                    reason = "SELL (MAX HOLD)"
            if reason:
                portfolio.sell(ticker, prices[ticker], as_of, reason)

        # Steps 3-4: indicators on the indicator tf + previous_score threading.
        signals = []
        indicators_by_ticker = {}
        for ticker in watchlist:
            try:
                window = get_window(ticker, indicator_tf, as_of, indicator_bars)
            except (ValueError, FileNotFoundError):
                continue  # warm-up / missing history -> skip this ticker
            indicators = (compute_indicators_pulse(window) if is_pulse
                          else compute_indicators(window))
            prev = last_score.get(ticker)
            if prev is not None:                  # L6: omit on a ticker's cold start
                indicators["previous_score"] = prev
            signals.append(indicators)
            indicators_by_ticker[ticker] = indicators

        if signals:
            vix = get_vix_asof(as_of)
            market_context = {
                "vix_close": vix["vix_close"],
                "vix_20ma": vix["vix_20ma"],
                "vix_regime": vix["vix_regime"],
                "market_breadth_pct": compute_breadth(signals),
            }

            # Step 5: score (mode switch). The claude path is cost-guarded: a
            # cache-miss that would breach the per-run ceiling halts the run.
            try:
                decisions = score_batch(signals, strategy, market_context)
            except CostCeilingExceeded:
                halted_on_cost = True
                break

            # Step 6: BUY cascade against the live portfolio.
            for decision in decisions:
                ticker = decision["ticker"]
                outcome = decide_buy_action(decision, {
                    "market_open": True,                # always open (L7)
                    "held": held,                       # snapshot (invariant 1)
                    "earnings": _NO_EARNINGS,           # no-earnings stub (L7)
                    "risk_state": _NEVER_TRIPPED_RISK,  # breaker omitted (L7)
                    "portfolio_value": pv_anchor,       # fixed anchor (invariant 2)
                    "cash": cash_local,                 # pre-exit snapshot, decrements per buy
                    "strategy": strategy,
                    "indicators": indicators_by_ticker.get(ticker, {}),
                    "sizing_config": config.POSITION_SIZING,
                    "buys_disabled": False,
                    "now": as_of,
                })
                if outcome["action"] == "BUY" and prices.get(ticker) is not None:
                    portfolio.buy(
                        ticker, outcome["notional"], prices[ticker], as_of,
                        meta={"entry_score": decision["score"],
                              "sizing_rule_bound": outcome["sizing_rule_bound"]},
                    )
                    cash_local -= outcome["notional"]   # mirror live current_cash_local

            # Thread each scored ticker's score into the next decision (L6).
            for decision in decisions:
                last_score[decision["ticker"]] = decision["score"]

        # End-of-tick equity point (post-exit, post-buy).
        equity_curve.append(portfolio.equity_point(as_of, prices))

    return RunResult(
        run_config=run_config,
        portfolio=portfolio,
        closed_trades=portfolio.closed_trades,
        equity_curve=equity_curve,
        n_decision_points=len(marks),
        cost_tracker=cost_tracker,
        halted_on_cost=halted_on_cost,
    )
