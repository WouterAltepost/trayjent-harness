"""Versioned JSON output assembly (Phase 5 brief Step 6 / L10, schema_version 1).

Turns a :class:`RunResult` + its metrics block into the locked schema-v1 dict
and a deterministic pretty-printed string. Determinism (L10/L12): the run path
has no wall-clock or RNG, so for a fixed ``generated_at`` the same config + cache
yields byte-identical JSON. Keys are emitted in a stable construction order
(never ``sort_keys``, so the layout reads top-down: meta, trades, metrics, curve,
notes). Timestamps serialize as ISO-8601 UTC strings.
"""
import json

import config
from harness.reuse import SCORING_MODEL

SCHEMA_VERSION = 1
FILLS_DESC = "idealized (0 fees, 0 slippage, fill at decision-bar close)"


def _iso(ts):
    return ts.isoformat() if ts is not None else None


def _trade_row(t) -> dict:
    """One closed round-trip in schema order. entry_score / sizing_rule_bound
    are the decision metadata the runner stashed on the lot at buy time."""
    return {
        "ticker": t.ticker,
        "entry_ts": _iso(t.entry_ts),
        "entry_price": t.entry_price,
        "exit_ts": _iso(t.exit_ts),
        "exit_price": t.exit_price,
        "qty": t.qty,
        "notional": t.notional,
        "realized_dollars": t.realized_dollars,
        "realized_pct": t.realized_pct,
        "exit_reason": t.exit_reason,
        "sizing_rule_bound": t.meta.get("sizing_rule_bound"),
        "entry_score": t.meta.get("entry_score"),
    }


def build_notes(run_config, run_result) -> list:
    """The modeled-omission and deviation disclosures (L7, L14). Prominent so no
    result is mistaken for fully live-equivalent."""
    notes = [
        "breaker not modeled (never tripped)",
        "earnings filter not modeled (no-earnings stub)",
        FILLS_DESC,
        "market always open (no SKIP (MARKET CLOSED) path)",
        "Sharpe/beta risk-free rate = 0; periods/year derived from the equity curve",
        "sizing uses a pre-exit cash snapshot (mirrors live current_cash_local); "
        "same-tick sale proceeds do not fund same-tick buys",
    ]
    if run_config.strategy["name"] == "steady":
        notes.append(
            "live Steady acts intraday on a partial daily bar; backtest decides/"
            "fills on the completed daily close (L14)")
    if run_config.name == "pulse_30min":
        notes.append(
            "Pulse-30min is mechanical fidelity only (indicators 1h, prices 30m); "
            "overlaps the sealed Pulse quarter by nature — not an edge claim, "
            "do not tune any parameter from it (L11)")
    if run_result.halted_on_cost:
        notes.append("run halted early: per-run Claude cost ceiling reached")
    return notes


def build_output(run_result, metrics, *, generated_at, notes=None) -> dict:
    """Assemble the schema-v1 output dict. ``generated_at`` is supplied by the
    caller (the CLI stamps it outside the deterministic run path, L12)."""
    rc = run_result.run_config
    is_claude = rc.mode == "claude"

    meta = {
        "strategy": rc.strategy["name"],
        "run_config": rc.name,
        "mode": rc.mode,
        "period": {"start": _iso(rc.start), "end": _iso(rc.end), "cadence": rc.cadence},
        "prompt_version": config.PROMPT_VERSION,
        "scoring_model": SCORING_MODEL if is_claude else None,
        "initial_capital": run_result.initial_capital,
        "fills": FILLS_DESC,
        "generated_at": generated_at,
    }
    ct = run_result.cost_tracker
    if ct is not None:
        meta["cost"] = {
            "spent_usd": ct.spent_usd,
            "n_calls": ct.n_calls,
            "ceiling_usd": ct.ceiling_usd,
            "halted": run_result.halted_on_cost,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "meta": meta,
        "trades": [_trade_row(t) for t in run_result.closed_trades],
        "metrics": metrics,
        "equity_curve": [
            {"ts": _iso(pt["ts"]), "portfolio_value": pt["portfolio_value"],
             "cash": pt["cash"], "n_positions": pt["n_positions"]}
            for pt in run_result.equity_curve
        ],
        "notes": notes if notes is not None else build_notes(rc, run_result),
    }


def to_json(output: dict) -> str:
    """Deterministic, stable-order pretty JSON (L10)."""
    return json.dumps(output, indent=2, ensure_ascii=False)
