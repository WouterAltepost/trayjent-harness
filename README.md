# Trayjent Backtest Harness (TBH)

Offline tool that replays Trayjent's scoring, sizing, and exit logic against
historical market data and produces structured JSON results. Mac-only for v1.
No connection to Alpaca, Sheets, or Telegram.

See `../planning/harness_plan.md` for the full plan and
`../discovery/harness/phase0_discovery.md` for the reusability analysis this
scaffold is built from.

## Phase 0 status

Scaffold only. This phase establishes repo layout, dependencies, and the
shared-code approach. It does **not** pull data, simulate, or call Claude.

## Locked decisions (Phase 0)

| # | Decision | Value |
|---|---|---|
| D1 | Code sharing with `trading-agent/` | Path-import `tools/indicators.py` only (pure, zero deps). No live refactor. Sizing/gates/scoring sharing deferred to Phase 2. |
| D2 | Git | This `harness/` directory is its own git repo, separate from `trading-agent/`. |
| D3 | Data source | yfinance (daily for Steady, hourly for Pulse). |
| D4 | Runtime | Mac-only. No Railway. |
| D5 | Storage | Parquet for bars (`data/`), SQLite for the Claude cache (`cache/`). |
| D6 | First exit model | Soft-stop, checked at bar close, matching current production. Bracket mode deferred to post-v10. |

## Key reuse contract

The harness data layer (Phase 1) must emit, per decision timestamp:

```python
{"ticker": str, "close_prices": [float, ...], "volumes": [int, ...]}
```

ordered oldest→newest, ending at the decision bar. Honor this and
`compute_indicators` / `compute_indicators_pulse` work unchanged.

Fidelity note: EMA output depends on how far back the close list starts, so
slice a trailing window matching live (`STEADY_BARS`, `PULSE_BARS` in
`config.py`), not the full history.

## Setup

```bash
cd harness
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -q
```

## Layout

```
harness/
  config.py            harness paths, prompt version, cost ceiling, window lengths
  requirements.txt
  harness/
    reuse.py           path-import shim for trading-agent/tools/indicators.py
  tests/
    test_reuse_import.py
  data/                Parquet bars (gitignored)
  cache/               SQLite Claude cache (gitignored)
```
