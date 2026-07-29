"""Offline loader + cross-checks for the Rotor sleeve (``data/rotor/``).

Pure like ``slice`` and ``shortvol``: reads stored Parquet only, never
imports the network module, so future Rotor strategy code can import this
freely. Nothing here reads the frozen Steady/Pulse data or the cache.

- ``load_bars(coin)`` -> tidy daily frame, oldest->newest. ``timestamp`` is
  the bar's CLOSE instant (00:00Z of the following UTC day — see the config
  "Rotor sleeve" block); ``bar_day()`` recovers the UTC day a bar covers.
- ``load_universe()`` -> sorted coins currently stored.
- ``cross_check()`` -> the sleeve's data-integrity report: per-coin
  coverage, strict no-gap rule (24/7 market: a missing day is a data
  problem, NEVER a holiday), and the BTC/ETH history-depth check that the
  window design step depends on.

SURVIVORSHIP CAVEAT (config "Rotor sleeve" block): this store holds Alpaca's
CURRENT list only — delisted/dead coins are invisible, a known upward bias
for any backtest on this data.
"""
import os

import pandas as pd

import config

# The majors whose history depth gates window design (tuning vs sealed OOS),
# and the depth below which that split gets structurally hard.
_MAJORS = ("BTC", "ETH")
_MIN_MAJOR_YEARS = 4.0


def _parquet_path(coin: str) -> str:
    """Mirrors ``rotor_store.parquet_path``; duplicated to keep this module
    import-clean of the network module (same pattern as the other sleeves)."""
    return os.path.join(config.ROTOR_DATA_DIR, f"{coin}.parquet")


def load_bars(coin: str) -> "pd.DataFrame":
    """Return the stored tidy daily frame for one coin."""
    path = _parquet_path(coin)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No stored Rotor bars for {coin} at {path}. "
            "Run scripts/pull_rotor_data.py first."
        )
    df = pd.read_parquet(path, engine="pyarrow")
    return df.sort_values("timestamp").reset_index(drop=True)


def load_universe() -> list:
    """Sorted base symbols currently stored in data/rotor/."""
    if not os.path.isdir(config.ROTOR_DATA_DIR):
        return []
    return sorted(f[:-len(".parquet")] for f in os.listdir(config.ROTOR_DATA_DIR)
                  if f.endswith(".parquet"))


def bar_day(ts: "pd.Timestamp") -> str:
    """The UTC day a close-stamped bar covers (close 00:00Z next day)."""
    return (ts - pd.Timedelta(hours=24)).strftime("%Y-%m-%d")


def listing_windows() -> dict:
    """Per-coin venue tradability windows, DERIVED from the stored bars.

    Returns ``{coin: [(first_ts, last_ts), ...]}`` — contiguous listed
    segments as close-instant timestamps, oldest first. A coin is tradable
    on the venue at instant T iff T falls inside one of its segments; a
    delisted coin's final segment simply ends at its last real bar. Derived
    (not hand-written in config) so the map can never drift from the data —
    this is what kills survivorship bias in the point-in-time universe: the
    backtest sees ALGO/MATIC/NEAR/TRX/MKR as live until their delisting and
    gone afterward, and never computes a return across a SOL-style
    delist->relist gap.
    """
    day = pd.Timedelta(hours=24)
    out = {}
    for coin in load_universe():
        ts = load_bars(coin)["timestamp"]
        segments = []
        seg_start = ts.iloc[0]
        for i in range(1, len(ts)):
            if ts.iloc[i] - ts.iloc[i - 1] > day:
                segments.append((seg_start, ts.iloc[i - 1]))
                seg_start = ts.iloc[i]
        segments.append((seg_start, ts.iloc[-1]))
        out[coin] = segments
    return out


def cross_check() -> dict:
    """Cross-check the stored sleeve; return a report dict, judging nothing
    (printing and exit codes belong to scripts/pull_rotor_data.py).

    Report keys:

    - ``coins``: per-coin ``{coin, first_day, last_day, rows, missing_days,
      segments, hole_dates}``. A 24/7 market has one bar per UTC day, so
      ``missing_days`` is (calendar days first..last) - rows. Missing days
      are NEVER excused as holidays; they split into two kinds:
      single-day holes (feed-corruption signature -> FAILURE, listed in
      ``hole_dates``) and multi-day runs (venue delist->relist windows, see
      ``venue_gaps``). ``segments`` counts contiguous listed stretches.
    - ``venue_gaps``: ``[{coin, gap_start, gap_end, days}]`` — every missing
      run of >= 2 consecutive days. Real Alpaca tradability history (e.g.
      the June-2023 SEC delisting wave; SOL relisted 2024-08, PAXG
      2026-02): reported prominently, not failed, because the strategy step
      must consume these as tradability windows — never hold a coin across
      one, never treat a cross-gap return as a daily return.
    - ``majors``: ``{coin: {first_day, last_day, years}}`` for BTC and ETH,
      plus ``majors_flag`` — True if either major has under ~4 years of
      usable history, which the next step's tuning-vs-sealed window design
      must know PROMINENTLY.
    - ``failures``: list of human-readable failure strings; empty == pass.
      Failures: any single-day hole, a major missing from the store, or a
      stablecoin that slipped into the store.
    """
    coins = load_universe()
    failures = []
    day = pd.Timedelta(hours=24)

    stray = sorted(set(coins) & (config.ROTOR_STABLECOINS | config.ROTOR_EXCLUDED))
    if stray:
        failures.append(f"excluded coin(s) in the store: {stray}")

    rows_out = []
    venue_gaps = []
    frames = {}
    for coin in coins:
        df = load_bars(coin)
        frames[coin] = df
        ts = df["timestamp"]
        first, last = ts.iloc[0], ts.iloc[-1]
        missing = int((last - first) / day) + 1 - len(df)

        holes = []
        segments = 1
        for i in range(1, len(ts)):
            run = int((ts.iloc[i] - ts.iloc[i - 1]) / day) - 1
            if run == 0:
                continue
            segments += 1
            gap_start = bar_day(ts.iloc[i - 1] + day)
            gap_end = bar_day(ts.iloc[i] - day)
            if run == 1:
                holes.append(gap_start)
            else:
                venue_gaps.append({"coin": coin, "gap_start": gap_start,
                                   "gap_end": gap_end, "days": run})
        if holes:
            failures.append(
                f"{coin}: {len(holes)} single-day hole(s) in a 24/7 market "
                f"(feed problem, not a holiday, not a listing gap): {holes[:10]}"
                f"{'...' if len(holes) > 10 else ''}"
            )
        rows_out.append({
            "coin": coin,
            "first_day": bar_day(first),
            "last_day": bar_day(last),
            "rows": len(df),
            "missing_days": missing,
            "segments": segments,
            "hole_dates": holes,
        })

    majors = {}
    for coin in _MAJORS:
        if coin not in frames:
            failures.append(f"major {coin} missing from the store entirely")
            continue
        df = frames[coin]
        years = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).days / 365.25
        majors[coin] = {
            "first_day": bar_day(df["timestamp"].iloc[0]),
            "last_day": bar_day(df["timestamp"].iloc[-1]),
            "years": years,
        }
    flag = any(m["years"] < _MIN_MAJOR_YEARS for m in majors.values()) or \
        len(majors) < len(_MAJORS)

    return {"coins": rows_out, "venue_gaps": venue_gaps, "majors": majors,
            "majors_flag": flag, "failures": failures}
