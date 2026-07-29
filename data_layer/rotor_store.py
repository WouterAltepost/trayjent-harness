"""Rotor sleeve pull + Parquet storage. Network-touching, like ``store``.

Source: Alpaca crypto market data v1beta3 (``config.ROTOR_BARS_URL``) —
deliberately the venue Rotor would trade, free and keyless for crypto data
(verified 2026-07-29). Two endpoints:

- ``latest/bars`` with the ``config.ROTOR_CANDIDATES`` superset — keyless
  universe discovery (the assets endpoint needs keys; Alpaca silently omits
  symbols it does not serve, so the responders ARE the current tradable
  set). Stablecoins (``config.ROTOR_STABLECOINS``) are then excluded.
- ``bars?timeframe=1D`` per coin, paginated — full available history
  (Alpaca crypto daily data begins 2021-01-01).

BAR CONVENTION (config block "Rotor sleeve"): Alpaca keys each daily bar at
00:00:00Z covering the UTC day [00:00, 24:00). Stored ``timestamp`` is the
bar's CLOSE instant (key + 24h) — the moment the bar becomes knowable —
mirroring the no-lookahead stamping every sleeve uses. The partial-bar guard
drops bars whose close instant is still in the future (Alpaca serves today's
FORMING bar in responses; persisting it would poison later reads).

Schema deviation from the equity sleeves, deliberate and documented: crypto
``volume`` is fractional coin units (kept float, NOT cast to int64), and two
extra columns are stored — ``vwap`` (Alpaca ``vw``) and ``trades`` (``n``) —
because Rotor ranks by liquidity and dollar volume ≈ volume x vwap.

Files land in ``data/rotor/{COIN}.parquet`` (base symbol; all pairs are
/USD). Frozen Steady/Pulse data and cache/ are never touched.
"""
import json
import os
import urllib.parse
import urllib.request

import pandas as pd

import config
from data_layer import sanity, store

_UA = "trayjent-harness"

# Alpaca daily crypto bars cover [key, key + 24h) UTC (config block).
_BAR_DURATION = pd.Timedelta(hours=24)

# Pagination backstop: full history is ~2,050 daily bars (limit=10000 ->
# one page); anything past a handful of pages means a runaway loop.
_MAX_PAGES = 50


def parquet_path(coin: str) -> str:
    """``data/rotor/{COIN}.parquet`` — the sleeve's own directory."""
    return os.path.join(config.ROTOR_DATA_DIR, f"{coin}.parquet")


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


# "Currently tradable" recency bar: latest/bars serves the last KNOWN bar
# even for pairs Alpaca delisted years ago (verified 2026-07-29: ALGO's
# latest bar is from 2023), so responding is NOT the same as trading. A live
# pair's latest daily bar is at most ~2 days old; 3 days adds slack.
_LATEST_MAX_AGE = pd.Timedelta(days=3)


def fetch_universe() -> list:
    """The CURRENTLY TRADABLE /USD coins: candidate superset -> latest/bars
    responders -> minus stale pairs (delisted; see ``_LATEST_MAX_AGE``) ->
    minus stablecoins. Sorted base symbols (e.g. "BTC")."""
    symbols = ",".join(f"{c}/USD" for c in config.ROTOR_CANDIDATES)
    url = (f"{config.ROTOR_BARS_URL}/latest/bars?"
           f"symbols={urllib.parse.quote(symbols)}")
    body = _get_json(url)
    now = pd.Timestamp.now(tz="UTC")
    found = set()
    for pair, bar in body.get("bars", {}).items():
        key = pd.Timestamp(bar["t"])
        key = key.tz_localize("UTC") if key.tz is None else key.tz_convert("UTC")
        if now - key <= _LATEST_MAX_AGE:
            found.add(pair.split("/")[0])
    if "BTC" not in found or "ETH" not in found:
        raise ValueError(
            f"Universe discovery returned {len(found)} live coins but no "
            "BTC/ETH — feed or endpoint problem, not a thin market"
        )
    return sorted(found - config.ROTOR_STABLECOINS - config.ROTOR_EXCLUDED)


def parse_bars(bars: list, coin: str) -> "pd.DataFrame":
    """Turn one symbol's Alpaca bar list into the tidy Rotor frame.

    Pure (no network) so the parser is testable offline. Raises
    ``ValueError`` on a payload-shape change so a silently altered API can
    never slip malformed rows into the sleeve. Close-stamps timestamps
    (key + 24h) and applies the partial-bar guard.
    """
    required = {"t", "o", "h", "l", "c", "v", "vw", "n"}
    rows = []
    for b in bars:
        missing = required - set(b)
        if missing:
            raise ValueError(f"{coin}: bar missing field(s) {sorted(missing)}: {b}")
        rows.append(b)
    if not rows:
        return pd.DataFrame()

    keys = pd.DatetimeIndex([pd.Timestamp(b["t"]) for b in rows])
    if keys.tz is None:
        keys = keys.tz_localize("UTC")
    else:
        keys = keys.tz_convert("UTC")
    if ((keys.hour != 0) | (keys.minute != 0)).any():
        raise ValueError(
            f"{coin}: daily bar not keyed at 00:00Z — Alpaca changed its bar "
            "boundary; re-verify the convention before storing anything"
        )

    out = pd.DataFrame({
        "timestamp": keys + _BAR_DURATION,        # close instant (config)
        "open": [float(b["o"]) for b in rows],
        "high": [float(b["h"]) for b in rows],
        "low": [float(b["l"]) for b in rows],
        "close": [float(b["c"]) for b in rows],
        "volume": [float(b["v"]) for b in rows],  # fractional coins: float
        "vwap": [float(b["vw"]) for b in rows],
        "trades": [int(b["n"]) for b in rows],
    })
    out = out.dropna(subset=["close"]).copy()
    out["ticker"] = coin
    out["timeframe"] = "1d"
    # Partial-bar guard: Alpaca includes the still-forming UTC day.
    now_utc = pd.Timestamp.now(tz="UTC")
    out = out[out["timestamp"] <= now_utc]
    out = out.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return out.reset_index(drop=True)


def pull_coin(coin: str) -> "pd.DataFrame":
    """Full daily history for one coin, paginated."""
    bars, token, pages = [], None, 0
    while True:
        url = (f"{config.ROTOR_BARS_URL}/bars?"
               f"symbols={urllib.parse.quote(f'{coin}/USD')}"
               f"&timeframe=1D&start=2015-01-01&limit=10000")
        if token:
            url += f"&page_token={urllib.parse.quote(token)}"
        body = _get_json(url)
        bars.extend(body.get("bars", {}).get(f"{coin}/USD", []))
        token = body.get("next_page_token")
        pages += 1
        if not token:
            break
        if pages >= _MAX_PAGES:
            raise ValueError(f"{coin}: pagination did not terminate after {pages} pages")

    out = parse_bars(bars, coin)
    if out.empty:
        raise ValueError(f"{coin}: Alpaca returned no completed daily bars")
    return out


def pull_all_rotor() -> list:
    """Discover the live universe, add the deliberately-pulled delisted
    series (``config.ROTOR_DELISTED`` — survivorship-bias kill), pull every
    coin, sanity-check before writing, upsert each to ``data/rotor/``. Same
    fail-closed-per-coin summary shape as the other sleeves' pulls."""
    universe = sorted(set(fetch_universe()) |
                      (set(config.ROTOR_DELISTED) - config.ROTOR_EXCLUDED))
    results = []
    for coin in universe:
        path = parquet_path(coin)
        rec = {
            "ticker": coin,
            "timeframe": "rotor",
            "rows": 0,
            "first": None,
            "last": None,
            "warnings": [],
            "error": None,
            "path": path,
        }
        try:
            df = pull_coin(coin)
            # Sanity BEFORE the write so corrupt data is never persisted.
            rec["warnings"] = sanity.check_integrity(df, coin, "rotor")
            merged = store._upsert_parquet(path, df)
            rec["rows"] = len(merged)
            rec["first"] = merged["timestamp"].iloc[0]
            rec["last"] = merged["timestamp"].iloc[-1]
        except Exception as e:  # noqa: BLE001 — record + continue, fail closed
            rec["error"] = str(e)
        results.append(rec)
    return results
