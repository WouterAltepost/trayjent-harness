"""Historical bar pull + Parquet storage (brief Step 2). The ONLY module in
the data layer that touches the network (L10).

- ``pull_ticker(ticker, timeframe)`` -> tidy OHLCV DataFrame in UTC.
- ``_upsert_parquet(path, df)`` -> merge-by-timestamp, never drop old rows (L6).
- ``pull_all()`` -> pull the full L4 universe, sanity-check, upsert, summarize.

Adjusted closes (``auto_adjust=True``, L1) so indicators stay continuous across
splits. Timestamps stored UTC tz-aware (L2): daily stamped at the US session
close (21:00 UTC), hourly converted from yfinance's native tz.
"""
import os

import pandas as pd
import yfinance as yf

import config
from data_layer import sanity

# US equity/ETF regular-session close, used to stamp daily bars (L2). yfinance
# daily bars index on the calendar date (midnight); a daily bar does not exist
# until its session closes, so we stamp it at 21:00 UTC to keep the slicer's
# no-lookahead comparison honest.
_DAILY_CLOSE_UTC = pd.Timedelta(hours=21)

_OHLCV = ["open", "high", "low", "close", "volume"]


def parquet_path(ticker: str, timeframe: str) -> str:
    """``data/{timeframe}/{ticker}.parquet`` (L3)."""
    return os.path.join(config.DATA_DIR, timeframe, f"{ticker}.parquet")


def _extract_series(df: "pd.DataFrame", name: str) -> "pd.Series":
    """Pull one OHLCV column out of a yfinance frame, flattening the MultiIndex
    columns yfinance returns for a single ticker (live does the same)."""
    col = df[name]
    if hasattr(col, "columns"):  # MultiIndex -> DataFrame slice
        col = col.iloc[:, 0]
    return col


def pull_ticker(ticker: str, timeframe: str) -> "pd.DataFrame":
    """Download one ticker/timeframe from yfinance and return a tidy UTC frame.

    Columns: ``timestamp, open, high, low, close, volume, ticker, timeframe``
    (L3), ordered oldest->newest.

    Raises ``ValueError`` if yfinance returns nothing or has no usable closes.
    """
    spec = config.TIMEFRAMES[timeframe]
    interval = spec["interval"]
    if interval == "1d":
        period = f"{config.DAILY_HISTORY_YEARS}y"
    else:
        period = config.HOURLY_PERIOD

    df = yf.download(
        ticker,
        interval=interval,
        period=period,
        auto_adjust=config.AUTO_ADJUST,
        progress=False,
    )
    if df is None or df.empty:
        raise ValueError(
            f"No data returned for {ticker} (interval={interval}, period={period})"
        )

    out = pd.DataFrame({c: _extract_series(df, c.capitalize()).values for c in _OHLCV})
    out.insert(0, "timestamp", _normalize_index(df.index, interval))

    # Drop rows with no close (yfinance occasionally emits NaN bars). Volume can
    # legitimately be 0/NaN (^VIX has no volume) so fill it, never drop on it.
    out = out.dropna(subset=["close"]).copy()
    out["volume"] = out["volume"].fillna(0).astype("int64")
    out["ticker"] = ticker
    out["timeframe"] = timeframe

    out = out.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return out.reset_index(drop=True)


def _normalize_index(index, interval: str) -> "pd.DatetimeIndex":
    """Return a UTC tz-aware DatetimeIndex.

    Daily yfinance bars index on a tz-naive date; stamp them at the US session
    close (21:00 UTC) so they only 'exist' after close (L2). Intraday bars are
    tz-aware (yfinance gives UTC); convert defensively in case of tz drift.
    """
    idx = pd.DatetimeIndex(index)
    if interval == "1d":
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        return (idx.normalize() + _DAILY_CLOSE_UTC).tz_localize("UTC")
    # intraday
    if idx.tz is None:
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")


def _upsert_parquet(path: str, df: "pd.DataFrame") -> "pd.DataFrame":
    """Merge ``df`` into the Parquet at ``path`` by timestamp, keeping the union
    of old and new rows (L6). The hourly 730-day window slides forward, so a
    naive overwrite would silently drop bars an earlier pull captured; this
    upsert never reduces the stored set. Returns the merged frame as written.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        existing = pd.read_parquet(path, engine="pyarrow")
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df.copy()

    combined = (
        combined.drop_duplicates("timestamp", keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    combined.to_parquet(path, engine="pyarrow", index=False)
    return combined


def pull_all() -> list:
    """Pull the full L4 universe across timeframes, sanity-check before writing,
    upsert each to Parquet, and return a per-ticker summary list.

    Fails closed per ticker: a hard sanity failure (duplicate / non-monotonic
    timestamps) or a download error is recorded and that ticker is NOT written,
    but the rest of the pull continues. Each summary dict carries
    ``ticker, timeframe, rows, first, last, warnings, error, path``.
    """
    results = []
    for timeframe, spec in config.TIMEFRAMES.items():
        for ticker in spec["universe"]:
            path = parquet_path(ticker, timeframe)
            rec = {
                "ticker": ticker,
                "timeframe": timeframe,
                "rows": 0,
                "first": None,
                "last": None,
                "warnings": [],
                "error": None,
                "path": path,
            }
            try:
                df = pull_ticker(ticker, timeframe)
                # Sanity BEFORE the write so corrupt data is never persisted.
                rec["warnings"] = sanity.check_integrity(df, ticker, timeframe)
                merged = _upsert_parquet(path, df)
                rec["rows"] = len(merged)
                rec["first"] = merged["timestamp"].iloc[0]
                rec["last"] = merged["timestamp"].iloc[-1]
            except Exception as e:  # noqa: BLE001 — record + continue, fail closed
                rec["error"] = str(e)
            results.append(rec)
    return results
