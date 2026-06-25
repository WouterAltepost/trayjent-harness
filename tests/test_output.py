"""Output + CLI tests (brief Step 6): schema-v1 shape, determinism, omission
notes, and the sealed-OOS guard.

End-to-end offline: a tmp DATA_DIR with SPY/^VIX/A daily bars + a fake scorer ->
run_backtest -> compute_metrics -> build_output -> to_json, all deterministic.

    python tests/test_output.py
"""
import json
import os
import sys
import tempfile

import pandas as pd
import pytest

_HARNESS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HARNESS_ROOT not in sys.path:
    sys.path.insert(0, _HARNESS_ROOT)

import config
from runner.configs import RunConfig, build_run_config
from runner.run import run_backtest
from runner.metrics import compute_metrics
from runner.output import build_output, to_json, SCHEMA_VERSION
from runner import cli


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


def _run():
    """A small deterministic run: buy A @100 (tick 0), take-profit @110 (tick 1)."""
    data_dir = tempfile.mkdtemp(prefix="tbh_output_")
    config.DATA_DIR = data_dir
    ts = _write(data_dir, "SPY", [100.0] * _N)
    _write(data_dir, "^VIX", [15.0] * _N, vol=0)
    closes = [100.0] * _N
    closes[300] = 110.0
    _write(data_dir, "A", closes)

    strategy = {"name": "steady", "buy_threshold": 7, "take_profit": 0.05,
                "stop_loss": 0.03, "cash_safety_pct": 0.80, "watchlist": ["A"]}
    rc = RunConfig(name="steady", strategy=strategy, indicator_tf="1d",
                   indicator_bars=300, decision_tf="1d", cadence="daily",
                   start=pd.Timestamp("2000-01-01", tz="UTC"), end=ts[300], mode="rules_only")

    def fake(signals, strat, mc):
        return [{"ticker": s["ticker"], "score": 10, "reasoning": "fake", "action": "BUY"}
                for s in signals]

    res = run_backtest(rc, score_batch=fake)
    return res, compute_metrics(res)


# ── Schema v1 shape ─────────────────────────────────────────────────────
def test_schema_v1_shape_and_trade_rows():
    res, metrics = _run()
    out = build_output(res, metrics, generated_at="2026-06-25T00:00:00+00:00")

    assert out["schema_version"] == SCHEMA_VERSION == 1
    assert set(out) == {"schema_version", "meta", "trades", "metrics", "equity_curve", "notes"}
    meta = out["meta"]
    assert meta["strategy"] == "steady" and meta["run_config"] == "steady"
    assert meta["mode"] == "rules_only"
    assert meta["scoring_model"] is None                  # no model in rules_only
    assert meta["period"] == {"start": res.run_config.start.isoformat(),
                              "end": res.run_config.end.isoformat(), "cadence": "daily"}
    assert meta["initial_capital"] == res.initial_capital
    assert "cost" not in meta                              # rules_only -> no cost block

    assert len(out["trades"]) == 1
    row = out["trades"][0]
    assert set(row) == {"ticker", "entry_ts", "entry_price", "exit_ts", "exit_price",
                        "qty", "notional", "realized_dollars", "realized_pct",
                        "exit_reason", "sizing_rule_bound", "entry_score"}
    assert row["ticker"] == "A" and row["exit_reason"] == "SELL (TAKE PROFIT)"
    assert row["entry_score"] == 10 and row["sizing_rule_bound"] == "score_ladder"
    assert isinstance(row["entry_ts"], str)               # ISO string, not Timestamp

    ep = out["equity_curve"][0]
    assert set(ep) == {"ts", "portfolio_value", "cash", "n_positions"}
    assert "benchmark_spy" in out["metrics"]


# ── Omission notes present (L7, L14) ────────────────────────────────────
def test_notes_disclose_omissions():
    res, metrics = _run()
    notes = build_output(res, metrics, generated_at="x")["notes"]
    joined = " | ".join(notes)
    assert "breaker not modeled" in joined
    assert "earnings filter not modeled" in joined
    assert "idealized" in joined
    assert "pre-exit cash snapshot" in joined
    assert "L14" in joined                                # Steady partial-bar deviation


# ── Determinism (L10): same run + generated_at -> byte-identical ────────
def test_json_is_deterministic_and_parseable():
    res, metrics = _run()
    a = to_json(build_output(res, metrics, generated_at="2026-06-25T00:00:00+00:00"))
    b = to_json(build_output(res, metrics, generated_at="2026-06-25T00:00:00+00:00"))
    assert a == b, "fixed generated_at -> byte-identical JSON"
    parsed = json.loads(a)                                # valid JSON
    assert parsed["schema_version"] == 1
    # Only generated_at changes the bytes.
    c = to_json(build_output(res, metrics, generated_at="DIFFERENT"))
    assert c != a and json.loads(c)["meta"]["generated_at"] == "DIFFERENT"


# ── Sealed-OOS guard (L11) ──────────────────────────────────────────────
def test_oos_guard_blocks_sealed_window_and_allows_override():
    # Steady sealed slab is 2020-2021.
    rc = build_run_config("steady", "2020-06-01", "2020-12-31", "rules_only")
    with pytest.raises(SystemExit):
        cli._check_oos(rc, allow_sealed=False)
    cli._check_oos(rc, allow_sealed=True)                 # override -> no raise

    # A non-sealed Steady window is fine.
    cli._check_oos(build_run_config("steady", "2023-01-01", "2023-06-30", "rules_only"), False)
    # pulse_30min has no guard (mechanical fidelity, inherently in-window).
    cli._check_oos(build_run_config("pulse_30min", "2026-05-01", "2026-06-20", "rules_only"), False)


def test_cli_main_rejects_sealed_window():
    # The guard fires before any data is read, so no DATA_DIR needed.
    with pytest.raises(SystemExit):
        cli.main(["--config", "pulse_hourly", "--from", "2026-04-15", "--to", "2026-05-15",
                  "--mode", "rules_only"])


if __name__ == "__main__":
    test_schema_v1_shape_and_trade_rows()
    test_notes_disclose_omissions()
    test_json_is_deterministic_and_parseable()
    test_oos_guard_blocks_sealed_window_and_allows_override()
    test_cli_main_rejects_sealed_window()
    print("test_output OK: schema v1 shape, omission notes, determinism, OOS guard + CLI reject.")
