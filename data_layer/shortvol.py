"""Offline loader + cross-checks for the shortvol sleeve (``data/shortvol/``).

Pure like ``slice``: reads stored Parquet only, never imports yfinance or the
store, so future shortvol strategy code (and the runner) can import this
freely. Nothing here reads the frozen Steady/Pulse data or the cache.

- ``load_bars(ticker)`` -> tidy daily frame, oldest->newest.
- ``load_term_structure()`` -> VIX and VIX3M joined by date with the
  ``vix_ratio`` (VIX / VIX3M) precomputed. Ratio >= 1.0 is backwardation —
  the stress signal this sleeve exists to trade around.
- ``cross_check()`` -> the sleeve's data-integrity report, consumed by
  ``scripts/pull_shortvol_data.py``: per-series coverage vs SPY trading days,
  VIX/VIX3M alignment, VIX3M > 0, backwardation-day counts.

Instrument traps (documented in config, surfaced here for consumers):
- ``config.SVXY_LEVERAGE_CHANGE_DATE`` — SVXY is -1x before, -0.5x after.
- ``config.VXX_HISTORY_START`` — VXX is series B only; no pre-2018 history.
"""
import os

import pandas as pd

import config


def _parquet_path(ticker: str) -> str:
    """``data/shortvol/{ticker}.parquet``. Mirrors ``shortvol_store
    .parquet_path``; duplicated to keep this module import-clean (no
    yfinance), same pattern as ``slice._parquet_path``."""
    return os.path.join(config.SHORTVOL_DATA_DIR, f"{ticker}.parquet")


def load_bars(ticker: str) -> "pd.DataFrame":
    """Return the stored tidy daily frame for one shortvol series."""
    path = _parquet_path(ticker)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No stored shortvol bars for {ticker} at {path}. "
            "Run scripts/pull_shortvol_data.py first."
        )
    df = pd.read_parquet(path, engine="pyarrow")
    return df.sort_values("timestamp").reset_index(drop=True)


def load_term_structure() -> "pd.DataFrame":
    """Return the VIX/VIX3M term-structure frame, ratio precomputed.

    Columns: ``timestamp, vix, vix3m, vix_ratio`` (= VIX / VIX3M),
    oldest->newest. Inner join by timestamp so the ratio is never NaN; the
    joint range starts at the VIX3M history start (2009-09-18), not VIX's
    1990 start. ``cross_check`` asserts the join loses no SPY trading day.

    Calendar note: since 2022 CBOE publishes index values on some NYSE
    holidays, so this frame contains a handful of days the equity legs do
    not trade (``cross_check`` reports them as ``cboe_only_days``). Strategy
    code that decisions on equity sessions should join this frame against an
    equity leg's dates rather than assume they match.
    """
    vix = load_bars("VIX")[["timestamp", "close"]].rename(columns={"close": "vix"})
    vix3m = load_bars("VIX3M")[["timestamp", "close"]].rename(columns={"close": "vix3m"})
    ts = vix.merge(vix3m, on="timestamp", how="inner")
    ts["vix_ratio"] = ts["vix"] / ts["vix3m"]
    return ts.sort_values("timestamp").reset_index(drop=True)


def _dates(df: "pd.DataFrame") -> set:
    """Session dates (YYYY-MM-DD strings) present in a tidy frame."""
    return set(df["timestamp"].dt.strftime("%Y-%m-%d"))


def _missing_vs_spy(series_dates: set, spy_sessions: list, first: str, last: str) -> list:
    """SPY trading days within [first, last] that the series lacks — the
    sleeve's gap definition. SPY sessions ARE the NYSE calendar (holidays
    excluded by construction), and CBOE indices publish every NYSE session,
    so the expected count is zero."""
    return [d for d in spy_sessions if first <= d <= last and d not in series_dates]


def cross_check() -> dict:
    """Cross-check the stored sleeve; return a report dict, judging nothing.

    Printing and exit codes belong to ``scripts/pull_shortvol_data.py``; this
    stays pure so tests can assert on the report directly.

    Report keys:

    - ``series``: per-series ``{ticker, first, last, rows, missing_days,
      missing_dates, known_gaps}`` — missing measured against SPY trading
      days within the series' own [first, last] overlap with SPY;
      ``known_gaps`` counts how many of those sit in the documented
      ``config.CBOE_KNOWN_MISSING_DAYS`` allowlist.
    - ``term``: ``{first, last, rows, missing_dates, vix3m_min,
      inversion_days, inversion_pct, inversion_by_year}`` for the joined
      VIX/VIX3M frame.
    - ``failures``: list of human-readable failure strings; empty == pass.
      Failures: VIX or VIX3M or the join has a non-allowlisted gap vs SPY
      trading days, or any VIX3M <= 0. Both index-gap checks are bounded by
      ``min(last dates)`` on purpose — a post-close pull can catch CBOE's
      file before its EOD update, and that operational lag is not a failure.
    - ``warnings``: non-fatal cross-source findings, two kinds: (a) tail
      staleness — CBOE sessions newer than an equity leg's last bar (Yahoo
      sometimes lags whole completed sessions, and a SPY-only calendar can
      never see SPY's own holes); (b) ETP holes — SPY trading days an
      SVXY/VXX bar is missing. Warnings, not failures, because a re-pull
      upsert-fills both once Yahoo materializes the bars.
    - ``cboe_only_days``: sessions the CBOE file publishes inside SPY's
      range that SPY does not trade. Since 2022 the CBOE history includes
      some NYSE holidays (e.g. 2022-07-04), so these are usually real index
      values on non-equity days, NOT missing data — informational. A normal
      weekday showing up here would instead suggest an interior SPY hole on
      Yahoo.
    """
    frames = {t: load_bars(t) for t in ("VIX", "VIX3M", "SVXY", "VXX", "SPY")}
    spy_sessions = sorted(_dates(frames["SPY"]))
    allow = set(config.CBOE_KNOWN_MISSING_DAYS)
    failures = []
    warnings = []

    series = []
    for ticker, df in frames.items():
        dates = _dates(df)
        first = df["timestamp"].iloc[0].strftime("%Y-%m-%d")
        last = df["timestamp"].iloc[-1].strftime("%Y-%m-%d")
        # Overlap-bound the comparison: VIX predates SPY (1990 vs 1993) and a
        # post-close CBOE pull can trail SPY by a session, or vice versa.
        lo = max(first, spy_sessions[0])
        hi = min(last, spy_sessions[-1])
        missing = _missing_vs_spy(dates, spy_sessions, lo, hi)
        known = [d for d in missing if d in allow]
        series.append({
            "ticker": ticker,
            "first": first,
            "last": last,
            "rows": len(df),
            "missing_days": len(missing),
            "missing_dates": missing,
            "known_gaps": len(known),
        })
        if ticker in ("VIX", "VIX3M"):
            unexplained = [d for d in missing if d not in allow]
            if unexplained:
                failures.append(
                    f"{ticker}: {len(unexplained)} SPY trading day(s) with no index close "
                    f"within [{lo}, {hi}] (beyond the known-gap allowlist): "
                    f"{unexplained[:10]}{'...' if len(unexplained) > 10 else ''}"
                )

    # ── Cross-source: Yahoo equity legs vs the CBOE session calendar ────
    cboe_sessions = sorted(_dates(frames["VIX"]))

    # (a) Tail staleness: CBOE sessions newer than a leg's last bar. (A CBOE
    # holiday row on the very newest date can briefly trip this; it clears at
    # the next real session.)
    for ticker in ("SPY", "SVXY", "VXX"):
        leg_last = frames[ticker]["timestamp"].iloc[-1].strftime("%Y-%m-%d")
        trail = [d for d in cboe_sessions if d > leg_last]
        if trail:
            warnings.append(
                f"{ticker}: last Yahoo bar {leg_last} trails the CBOE calendar by "
                f"{len(trail)} session(s): {trail[:5]}{'...' if len(trail) > 5 else ''} "
                "— likely Yahoo lag; re-run the pull later to upsert-fill"
            )

    # (b) ETP holes vs SPY sessions (already counted in the summary table):
    # promoted to warnings so they aren't lost in a wall of zeros.
    for rec in series:
        if rec["ticker"] in ("SVXY", "VXX") and rec["missing_days"]:
            warnings.append(
                f"{rec['ticker']}: {rec['missing_days']} SPY trading day(s) with no "
                f"Yahoo bar: {rec['missing_dates'][:10]}"
                f"{'...' if rec['missing_days'] > 10 else ''} — likely Yahoo holes; "
                "re-run the pull later to upsert-fill"
            )

    # (c) CBOE-only dates inside SPY's range — see docstring; informational.
    spy_set = set(spy_sessions)
    cboe_only = [
        d for d in cboe_sessions
        if spy_sessions[0] <= d <= spy_sessions[-1] and d not in spy_set
    ]

    # ── Joined term structure: gaps, positivity, backwardation ──────────
    term = load_term_structure()
    t_first = term["timestamp"].iloc[0].strftime("%Y-%m-%d")
    t_last = term["timestamp"].iloc[-1].strftime("%Y-%m-%d")
    t_missing = _missing_vs_spy(
        _dates(term), spy_sessions,
        max(t_first, spy_sessions[0]), min(t_last, spy_sessions[-1]),
    )
    t_unexplained = [d for d in t_missing if d not in allow]
    if t_unexplained:
        failures.append(
            f"VIX/VIX3M join: {len(t_unexplained)} SPY trading day(s) lost by the join "
            f"(beyond the known-gap allowlist): "
            f"{t_unexplained[:10]}{'...' if len(t_unexplained) > 10 else ''}"
        )

    vix3m_min = float(term["vix3m"].min())
    if vix3m_min <= 0:
        failures.append(f"VIX3M has non-positive close(s): min={vix3m_min}")

    inverted = term[term["vix_ratio"] >= 1.0]
    by_year = (
        inverted["timestamp"].dt.year.value_counts().sort_index().to_dict()
        if len(inverted) else {}
    )

    return {
        "series": series,
        "term": {
            "first": t_first,
            "last": t_last,
            "rows": len(term),
            "missing_dates": t_missing,
            "vix3m_min": vix3m_min,
            "inversion_days": len(inverted),
            "inversion_pct": 100.0 * len(inverted) / len(term) if len(term) else 0.0,
            "inversion_by_year": {int(y): int(n) for y, n in by_year.items()},
        },
        "failures": failures,
        "warnings": warnings,
        "cboe_only_days": cboe_only,
    }
