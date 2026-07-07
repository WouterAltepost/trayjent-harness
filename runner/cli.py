"""Backtest CLI (Phase 5 brief Step 6). Run from the harness root:

    python -m runner --config pulse_hourly --mode rules_only \\
        --from 2024-07-01 --to 2024-12-31 --out out.json

``--config`` selects one of the three presets; ``--mode`` is rules_only (the free
inner loop) or claude (cache-backed, per-run cost ceiling). The run takes an
explicit date range, so the sealed OOS windows (L11) are simply never passed —
and the OOS guard refuses them unless ``--allow-sealed`` (the one post-Phase-6
read). ``generated_at`` is stamped here, outside the deterministic run path (L12).
"""
import argparse
import sys
from datetime import datetime, timezone

import config
from runner.configs import build_run_config, PRESET_NAMES, MODES, _to_utc_ts
from runner.run import run_backtest
from runner.metrics import compute_metrics
from runner.output import build_output, to_json

# Sealed OOS windows per preset (L11), a list per preset — Steady carries the
# consumed 2020-2021 slab (stays fenced) plus the fresh 2016-2019 slab for the
# grower redesign's one read. pulse_30min has no entry: its ~60d of 30m data
# lies inside the Pulse quarter by nature and it is mechanical-fidelity only
# (never tuned from), so there is nothing to guard — the discipline is "don't
# tune," not a date fence.
_SEALED = {
    "steady": [config.OOS_STEADY, config.OOS_STEADY_2016],
    "pulse_hourly": [config.OOS_PULSE_HOURLY],
}


def _check_oos(run_config, allow_sealed: bool) -> None:
    """Refuse a run whose window overlaps any of the preset's sealed OOS
    ranges (L11), unless explicitly overridden for a deliberate one-time
    sealed read."""
    for sealed in _SEALED.get(run_config.name, ()):
        s_start = _to_utc_ts(sealed[0])
        s_end = _to_utc_ts(sealed[1], end=True)
        if run_config.start <= s_end and run_config.end >= s_start and not allow_sealed:
            raise SystemExit(
                f"refusing {run_config.name} over {run_config.start.date()}..{run_config.end.date()}: "
                f"overlaps the sealed OOS window {sealed[0]}..{sealed[1]} (L11). Sealed ranges are "
                f"never passed during dev; use --allow-sealed only for a deliberate one-time sealed read."
            )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m runner",
                                description="TBH Phase 5 backtest runner")
    p.add_argument("--config", required=True, choices=PRESET_NAMES)
    p.add_argument("--mode", default="rules_only", choices=MODES)
    p.add_argument("--from", dest="start", required=True, help="run window start (inclusive)")
    p.add_argument("--to", dest="end", required=True, help="run window end (inclusive)")
    p.add_argument("--out", default=None, help="write JSON here (default: stdout)")
    p.add_argument("--allow-sealed", action="store_true",
                   help="permit a sealed OOS window (L11) — the one post-Phase-6 read")
    args = p.parse_args(argv)

    rc = build_run_config(args.config, args.start, args.end, args.mode)
    _check_oos(rc, args.allow_sealed)

    res = run_backtest(rc)
    metrics = compute_metrics(res)
    generated_at = datetime.now(timezone.utc).isoformat()   # outside the run path (L12)
    text = to_json(build_output(res, metrics, generated_at=generated_at))

    summary = (f"[{rc.name}/{rc.mode}] {res.n_decision_points} decisions, "
               f"{len(res.closed_trades)} trades")
    if res.halted_on_cost:
        summary += " (HALTED on cost ceiling)"
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"{summary} -> {args.out}", file=sys.stderr)
    else:
        print(text)
        print(summary, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
