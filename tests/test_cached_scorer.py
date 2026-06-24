"""Offline tests for the Phase-4 cache-backed scorer (brief Commit 3).

Uses a fake live_scorer (canned content + usage) so nothing hits the network.
Covers: miss -> fill -> hit returns identical decisions through
parse_scoring_response; the synthetic cached block parses; hits never accrue
cost; the fail-closed ceiling raises before the breaching live call; and
prompt_version changes invalidate.

Runnable via pytest or directly:
    python tests/test_cached_scorer.py
"""
import os
import shutil
import sys
import tempfile

_HARNESS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HARNESS_ROOT not in sys.path:
    sys.path.insert(0, _HARNESS_ROOT)

from scoring.cache import ScoringCache, cache_key
from scoring.cached_scorer import make_cached_scorer
from scoring.cost import CostTracker, CostCeilingExceeded
from harness.reuse import parse_scoring_response


PRICES = {"m": {"input_per_mtok": 10.0, "output_per_mtok": 30.0}}
STRAT = {"name": "steady", "buy_threshold": 7}
_INPUT = {
    "market_assessment": "constructive",
    "scores": [
        {"ticker": "NVDA", "score": 8, "reasoning": "strong"},
        {"ticker": "AMD", "score": 5, "reasoning": "mid"},
    ],
}


class _Block:
    def __init__(self, type, name=None, input=None):
        self.type = type
        self.name = name
        self.input = input


class _Usage:
    def __init__(self, input_tokens, output_tokens):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


def _fake_live(usage=None, input_dict=None):
    """A fake live_scorer: counts calls, exposes .usage, returns a text block +
    the submit_ticker_scores tool block (mirrors the real content shape)."""
    payload = input_dict or _INPUT

    def live(prompt):
        live.calls += 1
        live.usage = usage
        return [_Block("text"), _Block("tool_use", "submit_ticker_scores", payload)]

    live.calls = 0
    live.usage = usage
    return live


def _cache():
    tmp = tempfile.mkdtemp(prefix="test_cached_")
    return ScoringCache(os.path.join(tmp, "c.db")), tmp


def test_miss_fill_hit_identical_decisions():
    cache, tmp = _cache()
    try:
        live = _fake_live(usage=_Usage(1000, 200))
        tracker = CostTracker(50.0, PRICES, "m")
        scorer = make_cached_scorer(cache, live, tracker, model="m",
                                    prompt_version="v9.5", strategy="steady",
                                    tool_schema_sha="t", call_params_sha="c")
        c1 = scorer("PROMPT")           # miss
        assert live.calls == 1
        c2 = scorer("PROMPT")           # hit
        assert live.calls == 1          # live not called again
        d1 = parse_scoring_response(c1, STRAT)
        d2 = parse_scoring_response(c2, STRAT)
        assert d1 == d2
        assert d1[0] == {"ticker": "NVDA", "score": 8, "reasoning": "strong", "action": "BUY"}
        assert d1[1]["action"] == "SKIP"   # 5 < threshold 7
    finally:
        cache.close()
        shutil.rmtree(tmp, ignore_errors=True)


def test_hit_does_not_accrue_cost():
    cache, tmp = _cache()
    try:
        live = _fake_live(usage=_Usage(1_000_000, 0))   # $10 per live call
        tracker = CostTracker(50.0, PRICES, "m")
        scorer = make_cached_scorer(cache, live, tracker, model="m",
                                    prompt_version="v9.5", strategy="steady",
                                    tool_schema_sha="t", call_params_sha="c")
        scorer("PROMPT")               # miss -> add $10
        assert abs(tracker.spent_usd - 10.0) < 1e-9 and tracker.n_calls == 1
        scorer("PROMPT")               # hit -> no change
        assert abs(tracker.spent_usd - 10.0) < 1e-9 and tracker.n_calls == 1
    finally:
        cache.close()
        shutil.rmtree(tmp, ignore_errors=True)


def test_ceiling_raises_before_live_call():
    cache, tmp = _cache()
    try:
        live = _fake_live(usage=_Usage(1_000_000, 1_000_000))   # $40 per live call
        tracker = CostTracker(50.0, PRICES, "m")
        scorer = make_cached_scorer(cache, live, tracker, model="m",
                                    prompt_version="v9.5", strategy="steady",
                                    tool_schema_sha="t", call_params_sha="c")
        scorer("P1")                    # miss -> $40 spent, rolling avg 40
        assert live.calls == 1
        try:
            scorer("P2")                # 40 + 40 > 50 -> raise before calling live
            assert False, "expected CostCeilingExceeded"
        except CostCeilingExceeded:
            pass
        assert live.calls == 1          # live NOT called
        assert cache.get(cache_key("P2", "m", "v9.5", "t", "c")) is None  # nothing cached
    finally:
        cache.close()
        shutil.rmtree(tmp, ignore_errors=True)


def test_prompt_version_change_invalidates():
    cache, tmp = _cache()
    try:
        live = _fake_live(usage=_Usage(1000, 200))
        s1 = make_cached_scorer(cache, live, None, model="m", prompt_version="v9.5",
                                strategy="steady", tool_schema_sha="t", call_params_sha="c")
        s1("PROMPT")                    # fill under v9.5
        assert live.calls == 1
        s2 = make_cached_scorer(cache, live, None, model="m", prompt_version="v9.6",
                                strategy="steady", tool_schema_sha="t", call_params_sha="c")
        s2("PROMPT")                    # different key -> miss -> live called again
        assert live.calls == 2
    finally:
        cache.close()
        shutil.rmtree(tmp, ignore_errors=True)


def test_synthetic_block_parses_without_tracker():
    cache, tmp = _cache()
    try:
        live = _fake_live(usage=None)   # no usage, no tracker
        scorer = make_cached_scorer(cache, live, None, model="m", prompt_version="v9.5",
                                    strategy="steady", tool_schema_sha="t", call_params_sha="c")
        content = scorer("PROMPT")
        d = parse_scoring_response(content, STRAT)
        assert d[0]["ticker"] == "NVDA" and d[0]["action"] == "BUY"
        assert cache.stats()["rows"] == 1
    finally:
        cache.close()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_miss_fill_hit_identical_decisions()
    test_hit_does_not_accrue_cost()
    test_ceiling_raises_before_live_call()
    test_prompt_version_change_invalidates()
    test_synthetic_block_parses_without_tracker()
    print("test_cached_scorer OK: miss->fill->hit identical, hits free, "
          "fail-closed ceiling, prompt_version invalidation, synthetic parse.")
