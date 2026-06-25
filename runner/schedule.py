"""Decision calendar + ``as_of`` generator (Phase 5 brief Step 3 / L4, L14).

The runner must tick on the same schedule the live system decides on, or the
backtest measures a different system. The trading calendar is the stored **SPY
bars on the run's decision timeframe** (L4) — the bars *are* the calendar, so US
session hours, holidays, and DST come straight from the data with no separate
holiday calendar or hand-rolled session clock.

Each decision's ``as_of`` is the SPY decision bar's **close instant**
(``timestamp + TF_DURATION[decision_tf]``), reusing the slicer's duration map so
the close-time rule has one source of truth. For daily that close instant is the
stored 21:00 UTC session-close stamp (TF_DURATION 0), which is exactly Steady's
decide/fill-on-the-completed-daily-close convention (L14).

Warm-up (L4): a decision can only fire once the indicator window exists, so
marks before ``MIN_ROWS`` indicator-tf SPY bars have closed are skipped. The
runner still defensively skips any *per-ticker* insufficiency (the slicer raises
below the floor); this is the coarse calendar-level gate anchored on SPY, which
sits in every universe.

Pure and offline: reads stored Parquet only, never imports ``store`` (yfinance).
"""
import os

import numpy as np
import pandas as pd

import config
from data_layer.slice import TF_DURATION


def _load_bars(ticker: str, timeframe: str) -> "pd.DataFrame":
    """Load stored bars for ``ticker``/``timeframe``, sorted oldest->newest.

    Mirrors the ``data/{timeframe}/{ticker}.parquet`` convention (L3); kept
    local so this module stays import-clean (no yfinance via ``store``)."""
    path = os.path.join(config.DATA_DIR, timeframe, f"{ticker}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No stored SPY bars for {timeframe} at {path}. Run scripts/pull_data.py first."
        )
    df = pd.read_parquet(path, engine="pyarrow")
    return df.sort_values("timestamp").reset_index(drop=True)


def _close_times(df: "pd.DataFrame", timeframe: str) -> "pd.Series":
    """Each bar's CLOSE instant (start + tf duration), as a UTC Series."""
    return df["timestamp"] + TF_DURATION[timeframe]


def decision_points(run_config) -> list:
    """Return the ordered list of UTC ``as_of`` instants for ``run_config``.

    The marks are SPY decision-tf bar close instants within
    ``[run_config.start, run_config.end]`` (inclusive), filtered to those where
    the indicator window is warm (>= ``MIN_ROWS`` indicator-tf SPY bars have
    closed at or before the mark). Ascending, no duplicates.
    """
    dec_close = _close_times(_load_bars("SPY", run_config.decision_tf),
                             run_config.decision_tf)
    in_window = (dec_close >= run_config.start) & (dec_close <= run_config.end)
    marks = dec_close[in_window].reset_index(drop=True)

    # Warm-up gate: count indicator-tf bars closed at or before each mark.
    ind_close = _close_times(_load_bars("SPY", run_config.indicator_tf),
                             run_config.indicator_tf)
    ind_np = np.sort(ind_close.values.astype("datetime64[ns]"))
    counts = np.searchsorted(ind_np, marks.values.astype("datetime64[ns]"), side="right")

    return list(marks[counts >= config.MIN_ROWS])
