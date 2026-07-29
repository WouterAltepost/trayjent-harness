"""Integrity / corporate-actions sanity check, run at ingest (brief L-step 3).

`check_integrity` returns human-readable warnings (row-count band, large
single-bar moves, gaps) for the pull summary and the eventual JSON `notes[]`.

It HARD-FAILS (raises ``ValueError``) only on duplicate or non-monotonic
timestamps, because those corrupt the slicer's ``timestamp <= as_of`` logic.
Everything else is a warning: yfinance is known to drop bars and occasionally
mishandle corporate actions, so ingest flags rather than silently trusts.

Pure and offline: no network, no config import beyond constants used here.
"""

# Soft expected row-count bands per timeframe over the configured history
# depth (daily ~10y, hourly ~2y, 30m ~60d ≈ 60 sessions × 13 bars ≈ 780).
# Outside the band -> warn (partial pull or unexpected surplus), never fail.
_ROW_BANDS = {
    "1d": (2000, 3200),
    "1h": (2500, 4200),
    "30m": (600, 900),
}

# A single adjusted-close bar move beyond this fraction suggests an unadjusted
# split slipped through or bad data (the adjusted series should be continuous
# across splits/dividends).
_MAX_BAR_RETURN = 0.35

# Consecutive-bar gap (calendar) beyond this many days -> warn. Daily tolerates
# long weekends/holidays; intraday tolerates weekends but flags missing weeks.
_GAP_DAYS = {
    "1d": 7,
    "1h": 4,
    "30m": 4,
    # Shortvol sleeve (data/shortvol): daily series, same holiday tolerance as
    # 1d. Deliberately NO _ROW_BANDS entry — the sleeve's series lengths range
    # ~1.9k rows (VXX, series B only) to ~9.2k (VIX since 1990), so one band
    # fits none; scripts/pull_shortvol_data.py prints exact per-series
    # coverage instead.
    "shortvol": 7,
    # Rotor sleeve (data/rotor): 24/7 crypto, no exchange calendar — ANY gap
    # beyond the 1-day bar spacing is missing data, never a holiday. No
    # _ROW_BANDS entry (listing dates vary per coin); the pull script reports
    # exact per-coin coverage.
    "rotor": 1,
}


def check_integrity(df, ticker: str, timeframe: str) -> list:
    """Return a list of warning strings for ``df``; raise on corrupting issues.

    Parameters
    ----------
    df : pandas.DataFrame
        Tidy OHLCV frame with at least ``timestamp`` and ``close`` columns,
        ordered oldest->newest.
    ticker, timeframe : str
        For message context.

    Raises
    ------
    ValueError
        On duplicate or non-monotonic timestamps (slicer-corrupting).
    """
    warnings = []
    label = f"{ticker} {timeframe}"

    if df is None or len(df) == 0:
        raise ValueError(f"{label}: empty frame, nothing to check")

    ts = df["timestamp"]

    # ── Hard fails: timestamp integrity ─────────────────────────────────
    dup_count = int(ts.duplicated().sum())
    if dup_count:
        raise ValueError(f"{label}: {dup_count} duplicate timestamp(s) — corrupts the slicer")
    if not ts.is_monotonic_increasing:
        raise ValueError(f"{label}: timestamps not monotonically increasing — corrupts the slicer")

    # ── Warnings: row-count band ────────────────────────────────────────
    n = len(df)
    band = _ROW_BANDS.get(timeframe)
    if band is not None:
        lo, hi = band
        if n < lo:
            warnings.append(f"{label}: only {n} rows (expected >= {lo}) — possible partial pull")
        elif n > hi:
            warnings.append(f"{label}: {n} rows (expected <= {hi}) — more history than expected")

    # ── Warnings: large single-bar adjusted move ────────────────────────
    rets = df["close"].pct_change().abs()
    big = rets[rets > _MAX_BAR_RETURN]
    if len(big):
        worst = float(big.max())
        when = ts.iloc[int(big.idxmax())] if hasattr(big, "idxmax") else "?"
        warnings.append(
            f"{label}: {len(big)} bar(s) with |return| > {_MAX_BAR_RETURN:.0%} "
            f"(worst {worst:.0%} at {when}) — check for an unadjusted split or bad data"
        )

    # ── Warnings: timestamp gaps ────────────────────────────────────────
    gap_days = _GAP_DAYS.get(timeframe)
    if gap_days is not None and n > 1:
        deltas = ts.diff().dropna()
        big_gaps = deltas[deltas > _to_timedelta_days(gap_days)]
        if len(big_gaps):
            warnings.append(
                f"{label}: {len(big_gaps)} gap(s) > {gap_days}d between consecutive bars "
                f"(largest {big_gaps.max()}) — possible dropped bars"
            )

    return warnings


def _to_timedelta_days(days: int):
    import pandas as pd
    return pd.Timedelta(days=days)
