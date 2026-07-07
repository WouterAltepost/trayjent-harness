"""Pure ATR (Average True Range) — no config, no I/O (Steady redesign Step 1).

Simple mean of the last ``period`` true ranges, deliberately NOT Wilder's
smoothing: window-local and deterministic, so a sweep result never depends on
how much history the smoother warmed up on — the trailing multiplier is the
swept dial, not the smoothing method.

Harness-side on purpose: the exit *decision* primitive (evaluate_trailing_exit)
went into the live repo for parity, but ATR is a deterministic input and the
grower may not graduate the sweep. If it goes live, this function moves to
trading-agent/tools/indicators.py and re-exports via harness.reuse, the same
pattern as the exit primitive.
"""


def compute_atr(highs, lows, closes, period) -> float:
    """Mean of the last ``period`` true ranges, where TR_i = max(high_i -
    low_i, |high_i - close_{i-1}|, |low_i - close_{i-1}|). Needs ``period + 1``
    bars (a prior close for the first TR); raises ValueError on fewer. Flat
    bars legitimately return 0.0 — the exit primitive's atr<=0 guard owns
    that case, not this function."""
    if len(closes) < period + 1:
        raise ValueError(
            f"ATR({period}) needs at least {period + 1} bars "
            f"(a prior close for the first true range); got {len(closes)}"
        )
    true_ranges = []
    for i in range(1, len(closes)):
        prev_close = closes[i - 1]
        true_ranges.append(max(
            highs[i] - lows[i],
            abs(highs[i] - prev_close),
            abs(lows[i] - prev_close),
        ))
    return sum(true_ranges[-period:]) / period
