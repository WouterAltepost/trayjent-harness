"""Run configurations (Phase 5 brief Step 3 / L1, L2).

A ``RunConfig`` is the one abstraction that makes the dual indicator/decision
timeframe clean: **indicators** are always computed on ``indicator_tf`` with
``indicator_bars`` of history (Steady 1d/300, both Pulse tracks 1h/440), while
**exit checks and fill prices** use the ``decision_tf`` price at ``as_of``
(Steady 1d, Pulse-hourly 1h, Pulse-30min 30m). Only Pulse-30min splits the two.

Three presets — ``steady``, ``pulse_hourly``, ``pulse_30min`` — fix everything
except the run window and mode; :func:`build_run_config` fills those in from the
CLI (the runner takes an explicit date range so sealed OOS slices, L11, are
simply never passed during dev).
"""
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

import config


@dataclass(frozen=True)
class RunConfig:
    name: str            # "steady" | "pulse_hourly" | "pulse_30min"
    strategy: dict       # frozen STRATEGY_* mirror (config.STRATEGY_*)
    indicator_tf: str    # "1d" | "1h"  — indicators always computed here
    indicator_bars: int  # STEADY_BARS | PULSE_BARS  — trailing window length
    decision_tf: str     # "1d" | "1h" | "30m"  — exit checks + fill prices
    cadence: str         # "daily" | "hourly" | "30min"
    start: datetime      # run window, inclusive (UTC)
    end: datetime        # run window, inclusive (UTC)
    mode: str            # "claude" | "rules_only"
    sizing_config: dict = None  # None -> config.POSITION_SIZING; sweep runs
                                # override via dataclasses.replace (frozen-safe)


# Fixed per-preset fields (everything but start/end/mode). pulse_hourly and
# pulse_30min share the Pulse strategy + 1h indicators; only the decision tf and
# cadence differ (L1, L2).
_PRESETS = {
    "steady": dict(
        strategy=config.STRATEGY_STEADY,
        indicator_tf="1d", indicator_bars=config.STEADY_BARS,
        decision_tf="1d", cadence="daily",
    ),
    "pulse_hourly": dict(
        strategy=config.STRATEGY_PULSE,
        indicator_tf="1h", indicator_bars=config.PULSE_BARS,
        decision_tf="1h", cadence="hourly",
    ),
    "pulse_30min": dict(
        strategy=config.STRATEGY_PULSE,
        indicator_tf="1h", indicator_bars=config.PULSE_BARS,
        decision_tf="30m", cadence="30min",
    ),
}

PRESET_NAMES = tuple(_PRESETS)
MODES = ("claude", "rules_only")


def _to_utc_ts(value, *, end: bool = False) -> "pd.Timestamp":
    """Coerce a date/datetime/str run bound to a UTC tz-aware Timestamp.

    A bare date (midnight, no time) as the ``end`` bound means "through the end
    of that day", so e.g. ``--to 2024-12-31`` includes that day's session-close
    decision (daily as_of is 21:00 UTC). A bare date as ``start`` already
    includes the whole day. Explicit times are honoured as given.
    """
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    if end and ts == ts.normalize():
        ts = ts.normalize() + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return ts


def build_run_config(name: str, start, end, mode: str) -> RunConfig:
    """Build a :class:`RunConfig` from a preset name plus the run window/mode.

    Raises ``ValueError`` on an unknown preset or mode.
    """
    if name not in _PRESETS:
        raise ValueError(f"unknown run config {name!r}; choose from {PRESET_NAMES}")
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
    return RunConfig(
        name=name,
        start=_to_utc_ts(start),
        end=_to_utc_ts(end, end=True),
        mode=mode,
        **_PRESETS[name],
    )
