"""Rules-only scorer (TBH Phase 4, brief Commit 4 / decisions L10, L11, L11a).

A deterministic reimplementation of the prompts' documented point ladders, for
fast parameter sweeps with zero API calls. It is an HONEST APPROXIMATION OF THE
RUBRIC — it reproduces what the prompt literally spells out, NOT Claude's
holistic scoring. Outputs are self-labelled ("rules-only ...") so no one reads a
rules-only result as a Claude-system result.

Not a `scorer(prompt)` seam injection (L10): the seam only carries the assembled
prompt, but rules-only needs the structured signal dicts. So it is a parallel
entry point — `score_rules_only(signals, strategy, market_context)` — emitting
the same decision shape as parse_scoring_response: {ticker, score, reasoning,
action}, so the Phase-5 runner can treat both modes uniformly.

Scope decisions confirmed with Mr. Altepost before build:
- MACD "turning/improving" (+1) and "deeply negative" (-1) rules are OMITTED in
  both strategies — they need a prior-bar histogram the single-snapshot
  indicator dict doesn't carry. Only `macd_histogram > 0 -> +2` is kept.
- The previous_score momentum nudge (Pulse +/-0.5; Steady qualitative) is
  OMITTED — the runner doesn't thread previous_score until Phase 5, and point
  sums stay integer.
- VIX / breadth market-context modulation is ignored by design (Claude-judgment
  overlays, not point rules). `market_context` is accepted for signature
  uniformity and deliberately unused.
- Mapping (L11a): score = max(0, min(10, round(5 + point_sum))).
"""


def _steady_points(sig):
    """Steady point ladder over compute_indicators keys. Returns (points, fired
    labels). None indicator values never fire (insufficient-history safe)."""
    pts = 0
    fired = []

    rsi = sig.get("rsi")
    if rsi is not None:
        if rsi < 30:
            pts += 2; fired.append("RSI<30 +2")
        elif rsi <= 40:
            pts += 1; fired.append("RSI 30-40 +1")
        elif rsi > 70:
            pts -= 1; fired.append("RSI>70 -1")

    hist = sig.get("macd_histogram")
    if hist is not None and hist > 0:
        pts += 2; fired.append("MACD hist>0 +2")

    close = sig.get("latest_close")
    ma_50 = sig.get("ma_50")
    ma_200 = sig.get("ma_200")
    if close is not None and ma_50 is not None and close > ma_50:
        pts += 1; fired.append("close>50MA +1")
    if close is not None and ma_200 is not None and close > ma_200:
        pts += 1; fired.append("close>200MA +1")
    if ma_50 is not None and ma_200 is not None:
        if ma_50 > ma_200:
            pts += 1; fired.append("golden cross +1")
        elif ma_50 < ma_200:
            pts -= 2; fired.append("death cross -2")

    return pts, fired


def _pulse_points(sig):
    """Pulse point ladder over compute_indicators_pulse keys. Returns (points,
    fired labels). None / missing values never fire."""
    pts = 0
    fired = []

    if sig.get("price_above_ema21"):
        pts += 1; fired.append("price>EMA21 +1")
    if sig.get("price_bounced_ema9"):
        pts += 1; fired.append("bounced EMA9 +1")

    close = sig.get("latest_close")
    ema_50 = sig.get("ema_50")
    if close is not None and ema_50 is not None and close > ema_50:
        pts += 1; fired.append("close>EMA50 +1")

    if sig.get("ema9_cross_above_ema21"):
        pts += 2; fired.append("EMA9 x-up EMA21 +2")
    if sig.get("ema9_cross_below_ema21"):
        pts -= 2; fired.append("EMA9 x-down EMA21 -2")

    rsi = sig.get("rsi")
    if rsi is not None:
        if rsi < 30:
            pts += 2; fired.append("RSI<30 +2")
        elif rsi <= 50:
            pts += 1; fired.append("RSI 30-50 +1")
        elif rsi > 70:
            pts -= 1; fired.append("RSI>70 -1")

    hist = sig.get("macd_histogram")
    line = sig.get("macd_line")
    signal = sig.get("macd_signal")
    if (hist is not None and line is not None and signal is not None
            and hist > 0 and line > signal):
        pts += 2; fired.append("MACD hist>0 & line>signal +2")

    if sig.get("obv_trending_up"):
        pts += 1; fired.append("OBV up +1")
    if sig.get("obv_bearish_divergence"):
        pts -= 1; fired.append("OBV bear-div -1")

    return pts, fired


_LADDERS = {"steady": _steady_points, "pulse": _pulse_points}


def score_rules_only(signals, strategy, market_context=None):
    """Score a batch deterministically from the rubric point ladders (no API).

    `market_context` is accepted for parity with the Claude path and ignored by
    design (VIX/breadth are judgment overlays, not point rules). Returns a list
    of decisions in parse_scoring_response shape: {ticker, score, reasoning,
    action}, with action = BUY iff score >= strategy['buy_threshold'].
    """
    name = strategy["name"]
    buy_threshold = strategy["buy_threshold"]
    ladder = _LADDERS.get(name)
    if ladder is None:
        raise ValueError(f"rules-only: unknown strategy {name!r}")

    decisions = []
    for sig in signals:
        points, fired = ladder(sig)
        score = max(0, min(10, round(5 + points)))   # L11a
        action = "BUY" if score >= buy_threshold else "SKIP"
        detail = ", ".join(fired) if fired else "no rules fired"
        decisions.append({
            "ticker": sig["ticker"],
            "score": score,
            "reasoning": f"rules-only ({name}): point_sum={points:+d} [{detail}]",
            "action": action,
        })
    return decisions
