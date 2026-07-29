"""Shortvol sleeve pull + Parquet storage. Network-touching, like ``store``.

Two sources:

- **CBOE published histories** for the volatility indices (VIX, VIX3M) — the
  free daily CSVs on cdn.cboe.com (``config.SHORTVOL_CBOE_URLS``). These are
  the primary source for the indices: full depth (VIX from 1990, VIX3M from
  2009-09-18), straight from the index publisher.
- **yfinance** for the ETP/equity legs (``config.SHORTVOL_EQUITY_UNIVERSE``),
  ``period="max"``, adjusted per ``config.AUTO_ADJUST`` — the same source and
  adjustment convention as the rest of the harness.

Storage follows the ``store`` conventions exactly: tidy frame (``timestamp``
UTC-stamped at the 21:00 UTC session close, ``open/high/low/close/volume/
ticker/timeframe``), ``store._upsert_parquet`` merge-by-timestamp, sanity
check BEFORE the write. Files land in ``data/shortvol/{ticker}.parquet`` — a
directory of the sleeve's own, so the frozen Steady/Pulse windows under
``data/{1d,1h,30m}`` and the Claude cache are never touched.

Partial-bar guard (shortvol-specific): a pull during the US session would
persist today's still-forming daily bar. ``_finalize`` drops any row whose
stamped close instant is in the future, so only completed sessions are ever
written; run the pull after 21:00 UTC (5pm ET) to capture today's close.

Series traps this sleeve carries (documented in config, enforced here):
- VXX is series B only — see ``config.VXX_HISTORY_START``. The sanity
  large-move check doubles as the unadjusted-split detector.
- SVXY flipped -1x -> -0.5x on ``config.SVXY_LEVERAGE_CHANGE_DATE``; the data
  is continuous but the instrument is not. Strategy code must consume that
  constant.
"""
import io
import os
import urllib.request

import pandas as pd
import yfinance as yf

import config
from data_layer import sanity, store

# cdn.cboe.com serves plain requests, but send a browser-ish UA so a future
# bot filter fails loudly here rather than with an opaque 403 mid-pull.
_UA = "Mozilla/5.0 (Macintosh) trayjent-harness"

# Expected CBOE header (verified 2026-07-29). Indices do not trade, so the
# tidy schema's volume is stored as 0.
_CBOE_COLS = ["DATE", "OPEN", "HIGH", "LOW", "CLOSE"]


def parquet_path(ticker: str) -> str:
    """``data/shortvol/{ticker}.parquet`` — the sleeve's own directory."""
    return os.path.join(config.SHORTVOL_DATA_DIR, f"{ticker}.parquet")


def parse_cboe_csv(raw: bytes, name: str) -> "pd.DataFrame":
    """Parse a CBOE ``*_History.csv`` payload into the tidy store schema.

    Pure (no network) so the parser is testable offline. Expects the header
    ``DATE,OPEN,HIGH,LOW,CLOSE`` with MM/DD/YYYY dates and raises
    ``ValueError`` on any shape drift, so a silently altered feed can never
    slip malformed rows into the sleeve.
    """
    df = pd.read_csv(io.BytesIO(raw))
    missing = [c for c in _CBOE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name}: CBOE CSV missing expected column(s) {missing}; got {list(df.columns)}"
        )

    # Strict format so a date-convention change fails loudly, not as swapped
    # months/days. Stamped at the daily close instant like every daily bar.
    idx = pd.DatetimeIndex(pd.to_datetime(df["DATE"], format="%m/%d/%Y"))
    out = pd.DataFrame({
        "timestamp": store._normalize_index(idx, "1d"),
        "open": df["OPEN"].astype(float).values,
        "high": df["HIGH"].astype(float).values,
        "low": df["LOW"].astype(float).values,
        "close": df["CLOSE"].astype(float).values,
    })
    out = out.dropna(subset=["close"]).copy()
    out["volume"] = 0
    out["volume"] = out["volume"].astype("int64")
    out["ticker"] = name
    out["timeframe"] = "1d"
    return _finalize(out)


def pull_cboe_index(name: str) -> "pd.DataFrame":
    """Download one CBOE index history and return the tidy UTC frame."""
    url = config.SHORTVOL_CBOE_URLS[name]
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    raw = urllib.request.urlopen(req, timeout=60).read()
    out = parse_cboe_csv(raw, name)
    if out.empty:
        raise ValueError(f"{name}: CBOE CSV parsed to zero usable rows ({url})")
    return out


def pull_equity(ticker: str) -> "pd.DataFrame":
    """Download one ETP/equity leg from yfinance, full available history.

    Same tidy-frame construction as ``store.pull_ticker`` but ``period="max"``
    (this sleeve wants full depth, not the frozen data's 10y window).
    """
    df = yf.download(
        ticker,
        interval="1d",
        period="max",
        auto_adjust=config.AUTO_ADJUST,
        progress=False,
    )
    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker} (interval=1d, period=max)")

    out = pd.DataFrame(
        {c: store._extract_series(df, c.capitalize()).values for c in store._OHLCV}
    )
    out.insert(0, "timestamp", store._normalize_index(df.index, "1d"))
    out = out.dropna(subset=["close"]).copy()
    out["volume"] = out["volume"].fillna(0).astype("int64")
    out["ticker"] = ticker
    out["timeframe"] = "1d"
    return _finalize(out)


def _finalize(out: "pd.DataFrame") -> "pd.DataFrame":
    """Sort, dedupe, and drop bars whose close instant hasn't happened yet.

    The future-bar filter is the partial-bar guard: yfinance includes today's
    forming daily bar on an intra-session pull, and a persisted partial close
    would silently poison every later backtest read. Fail-safe: a completed
    bar is at worst re-captured by the next pull's upsert.
    """
    now_utc = pd.Timestamp.now(tz="UTC")
    out = out[out["timestamp"] <= now_utc]
    out = out.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return out.reset_index(drop=True)


def pull_all_shortvol() -> list:
    """Pull the full shortvol universe, sanity-check before writing, upsert
    each to ``data/shortvol/``, and return a per-series summary list.

    Same fail-closed-per-ticker shape as ``store.pull_all``: a download error
    or hard sanity failure is recorded and that series is NOT written, but the
    rest of the pull continues. Summary dicts carry the same keys, with
    ``timeframe`` fixed to the sleeve's ``"shortvol"`` sanity profile.
    """
    pulls = [(name, pull_cboe_index) for name in config.SHORTVOL_CBOE_URLS]
    pulls += [(tkr, pull_equity) for tkr in config.SHORTVOL_EQUITY_UNIVERSE]

    results = []
    for ticker, pull in pulls:
        path = parquet_path(ticker)
        rec = {
            "ticker": ticker,
            "timeframe": "shortvol",
            "rows": 0,
            "first": None,
            "last": None,
            "warnings": [],
            "error": None,
            "path": path,
        }
        try:
            df = pull(ticker)
            # Sanity BEFORE the write so corrupt data is never persisted.
            rec["warnings"] = sanity.check_integrity(df, ticker, "shortvol")
            merged = store._upsert_parquet(path, df)
            rec["rows"] = len(merged)
            rec["first"] = merged["timestamp"].iloc[0]
            rec["last"] = merged["timestamp"].iloc[-1]
        except Exception as e:  # noqa: BLE001 — record + continue, fail closed
            rec["error"] = str(e)
        results.append(rec)
    return results
