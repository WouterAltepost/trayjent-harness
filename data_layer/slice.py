"""No-lookahead slicer (brief Step 4) — the critical Phase 1 deliverable.

Pure and fully offline (L10): reads stored Parquet, emits the live `price_data`
contract as of any decision timestamp T. Deliberately does NOT import
``store`` (which pulls in yfinance) so the Phase 5 runner can import this freely.

- ``get_window(ticker, timeframe, as_of, n_bars=None)`` -> the live
  ``{"ticker", "close_prices", "volumes"}`` dict, oldest->newest, ending at the
  last bar whose **close time** is ``<= as_of``. Honest about time: a bar that
  closes after T does not exist yet and is never included.
- ``get_price_asof(ticker, timeframe, as_of)`` -> the close of the latest bar
  whose close time is ``<= as_of`` (the decision-tf fill/exit price). No
  MIN_ROWS floor — a single-price lookup, not an indicator window.
- ``get_vix_asof(as_of)`` -> ``{"vix_close", "vix_20ma", "vix_regime"}`` from
  the most recent daily VIX bar <= T, classified with the live regime ladder.

No-lookahead, intraday (Phase 5 PF-2): yfinance stamps intraday bars at the bar
START (1h at :30 ET aligned to the 09:30 open; 30m at :30/:00), and consecutive
bars are contiguous, so a bar with ``start <= as_of`` may still be FORMING — its
final close reflects data after T. Eligibility therefore tests the bar's CLOSE
time (``start + tf_duration``), not its start. Daily bars are already stored at
the session-close instant (store stamps them 21:00 UTC), so their close time is
the stored timestamp (duration 0) and daily behaviour is unchanged.
"""
import os

import pandas as pd

import config
from harness.reuse import classify_vix_regime


def _parquet_path(ticker: str, timeframe: str) -> str:
    """``data/{timeframe}/{ticker}.parquet`` (L3). Mirrors
    ``store.parquet_path``; duplicated here to keep this module import-clean
    (no yfinance) for the runner."""
    return os.path.join(config.DATA_DIR, timeframe, f"{ticker}.parquet")


def _bars_per_timeframe(timeframe: str) -> int:
    """Default trailing-window length per timeframe (L7): daily Steady window,
    hourly Pulse window. Window length is not cosmetic — EMA seeds from the
    first `period` values of the list, so the slice length must match live.

    Only the two indicator timeframes have a live default. ``30m`` is a
    decision/exit *price* timeframe (Pulse-30min indicators come from 1h, L2),
    so it has no indicator window default; callers needing a 30m window pass
    ``n_bars`` explicitly."""
    if timeframe == "1d":
        return config.STEADY_BARS
    if timeframe == "1h":
        return config.PULSE_BARS
    raise ValueError(f"no default window for timeframe {timeframe!r}; pass n_bars")


# Nominal bar duration per timeframe, used to derive each bar's CLOSE time for
# the no-lookahead slice (see module docstring). Daily is 0 (already stored at
# the session-close instant); intraday adds the bar length to the START stamp.
# Stub-bar nuance: the last 1h bar of a US session is START-stamped 15:30 ET but
# really a 30-min stub closing 16:00 ET; the uniform +1h treats it as closing
# 16:30, so Pulse's final daily decision misses that one bar in its (440-bar)
# indicator window. Conservative (lookahead-safe) and immaterial — refine only
# if it ever proves material.
_TF_DURATION = {
    "1d": pd.Timedelta(0),
    "1h": pd.Timedelta(hours=1),
    "30m": pd.Timedelta(minutes=30),
}


def _eligible(df: "pd.DataFrame", timeframe: str, cutoff: "pd.Timestamp") -> "pd.DataFrame":
    """Rows whose bar CLOSE time is ``<= cutoff`` — the no-lookahead filter."""
    try:
        duration = _TF_DURATION[timeframe]
    except KeyError:
        raise ValueError(f"unknown timeframe {timeframe!r}")
    return df[df["timestamp"] + duration <= cutoff]


def _as_utc(as_of) -> "pd.Timestamp":
    """Coerce ``as_of`` to a UTC tz-aware Timestamp so the ``<= as_of``
    comparison against stored UTC timestamps is unambiguous."""
    ts = pd.Timestamp(as_of)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _load(ticker: str, timeframe: str) -> "pd.DataFrame":
    path = _parquet_path(ticker, timeframe)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No stored bars for {ticker} {timeframe} at {path}. Run scripts/pull_data.py first."
        )
    df = pd.read_parquet(path, engine="pyarrow")
    return df.sort_values("timestamp").reset_index(drop=True)


def get_window(ticker: str, timeframe: str, as_of, n_bars: int = None) -> dict:
    """Return the live ``price_data`` dict as of ``as_of``.

    Parameters
    ----------
    ticker, timeframe : str
    as_of : str | datetime | pandas.Timestamp
        Decision timestamp T. Only bars with ``timestamp <= T`` are eligible.
    n_bars : int, optional
        Trailing window length. Defaults to the timeframe's live window (L7):
        ``STEADY_BARS`` for daily, ``PULSE_BARS`` for hourly.

    Returns
    -------
    dict
        ``{"ticker": str, "close_prices": [float], "volumes": [int]}``,
        oldest->newest, ending at the decision bar. Byte-identical contract to
        live ``fetch_price_data`` so ``compute_indicators*`` consume it as-is.

    Raises
    ------
    ValueError
        If fewer than ``MIN_ROWS`` bars exist at or before T (fail-closed
        floor, mirrors live ``fetch_intraday_price_data``). The runner skips
        that decision point rather than scoring on thin data.
    """
    if n_bars is None:
        n_bars = _bars_per_timeframe(timeframe)

    df = _load(ticker, timeframe)
    cutoff = _as_utc(as_of)

    eligible = _eligible(df, timeframe, cutoff)
    if len(eligible) < config.MIN_ROWS:
        raise ValueError(
            f"Insufficient history for {ticker} {timeframe} as of {cutoff}: "
            f"{len(eligible)} bars closed <= T, need at least {config.MIN_ROWS}"
        )

    window = eligible.tail(n_bars)
    return {
        "ticker": ticker,
        "close_prices": [float(p) for p in window["close"].tolist()],
        "volumes": [int(v) for v in window["volume"].tolist()],
    }


def get_price_asof(ticker: str, timeframe: str, as_of) -> float:
    """Return the close of the latest bar on ``timeframe`` whose close time is
    ``<= as_of`` — the decision-tf fill/exit price (Phase 5 PF-5).

    Unlike :func:`get_window` there is **no** ``MIN_ROWS`` floor: this is a
    single most-recent-price lookup (Pulse-30min exit checks and fills), not an
    indicator window, so it must return as soon as one bar has closed.

    Raises
    ------
    ValueError
        If no bar on ``timeframe`` has closed at or before ``as_of``.
    """
    df = _load(ticker, timeframe)
    cutoff = _as_utc(as_of)

    eligible = _eligible(df, timeframe, cutoff)
    if len(eligible) == 0:
        raise ValueError(
            f"No closed {timeframe} bar for {ticker} at or before {cutoff}"
        )
    return float(eligible["close"].iloc[-1])


# ── VIX regime, as of T ─────────────────────────────────────────────────
# The regime ladder (<15 / <20 / <30 / 30+) is the live classifier, imported
# via the reuse shim (trading-agent/tools/vix.classify_vix_regime). The private
# copy that used to live here was deleted in Phase 2 — one source of truth, so a
# live threshold change can never silently desync the backtest. Still pinned by
# tests/test_slice.py.
def get_vix_asof(as_of) -> dict:
    """Return the VIX state from the most recent daily VIX bar <= ``as_of``.

    Matches live ``fetch_vix``: ``vix_close`` is the latest close, ``vix_20ma``
    the mean of the trailing 20 closes (``None`` if fewer than 20 exist), both
    rounded to 2dp, regime classified with the same ladder. VIX is daily even
    when Pulse runs hourly (L9), so the runner calls this with the Pulse
    decision T and gets the most recent daily VIX at or before it.

    Raises
    ------
    ValueError
        If no VIX bar exists at or before T (a setup error, not a market
        condition — the harness has full stored history, unlike live which
        degrades to 'unknown' on a network failure).
    """
    df = _load("^VIX", "1d")
    cutoff = _as_utc(as_of)

    eligible = df[df["timestamp"] <= cutoff]
    closes = [float(p) for p in eligible["close"].tolist()]
    if not closes:
        raise ValueError(f"No VIX data at or before {cutoff}")

    vix_close = closes[-1]
    vix_20ma = sum(closes[-20:]) / 20 if len(closes) >= 20 else None
    return {
        "vix_close": round(vix_close, 2),
        "vix_20ma": round(vix_20ma, 2) if vix_20ma is not None else None,
        "vix_regime": classify_vix_regime(vix_close),
    }
