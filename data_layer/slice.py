"""No-lookahead slicer (brief Step 4) — the critical Phase 1 deliverable.

Pure and fully offline (L10): reads stored Parquet, emits the live `price_data`
contract as of any decision timestamp T. Deliberately does NOT import
``store`` (which pulls in yfinance) so the Phase 5 runner can import this freely.

- ``get_window(ticker, timeframe, as_of, n_bars=None)`` -> the live
  ``{"ticker", "close_prices", "volumes"}`` dict, oldest->newest, ending at the
  last bar with ``timestamp <= as_of``. Honest about time: a bar that closes
  after T does not exist yet and is never included.
- ``get_vix_asof(as_of)`` -> ``{"vix_close", "vix_20ma", "vix_regime"}`` from
  the most recent daily VIX bar <= T, classified with the live regime ladder.
"""
import os

import pandas as pd

import config


def _parquet_path(ticker: str, timeframe: str) -> str:
    """``data/{timeframe}/{ticker}.parquet`` (L3). Mirrors
    ``store.parquet_path``; duplicated here to keep this module import-clean
    (no yfinance) for the runner."""
    return os.path.join(config.DATA_DIR, timeframe, f"{ticker}.parquet")


def _bars_per_timeframe(timeframe: str) -> int:
    """Default trailing-window length per timeframe (L7): daily Steady window,
    hourly Pulse window. Window length is not cosmetic — EMA seeds from the
    first `period` values of the list, so the slice length must match live."""
    if timeframe == "1d":
        return config.STEADY_BARS
    if timeframe == "1h":
        return config.PULSE_BARS
    raise ValueError(f"unknown timeframe {timeframe!r}")


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

    eligible = df[df["timestamp"] <= cutoff]
    if len(eligible) < config.MIN_ROWS:
        raise ValueError(
            f"Insufficient history for {ticker} {timeframe} as of {cutoff}: "
            f"{len(eligible)} bars <= T, need at least {config.MIN_ROWS}"
        )

    window = eligible.tail(n_bars)
    return {
        "ticker": ticker,
        "close_prices": [float(p) for p in window["close"].tolist()],
        "volumes": [int(v) for v in window["volume"].tolist()],
    }


# ── VIX regime, as of T ─────────────────────────────────────────────────
# Lifted from the live classifier in trading-agent/tools/market_data.py
# (``fetch_vix``): the <15 / <20 / <30 / 30+ ladder. Pinned by a test
# (tests/test_slice.py) until the live classifier is extracted into shared
# code (deferred past Phase 1, per the brief's out-of-scope note).
def _classify_vix_regime(vix_close: float) -> str:
    if vix_close < 15:
        return "low"
    elif vix_close < 20:
        return "normal"
    elif vix_close < 30:
        return "elevated"
    return "stressed"


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
        "vix_regime": _classify_vix_regime(vix_close),
    }
