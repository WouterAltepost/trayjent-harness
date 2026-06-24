"""Offline tests for the Phase-4 SQLite scoring cache (brief Commit 2).

Covers the L2 cache-key contract (determinism + field-separation + every
component invalidating), the put/get roundtrip, miss -> None, idempotent
re-store, and stats(). Plus a guard pinning the PF-1 opus-4-7 prices wired into
config.PRICES. Pure: a throwaway SQLite file per test, no network.

Runnable via pytest or directly:
    python tests/test_cache.py
"""
import os
import shutil
import sys
import tempfile

_HARNESS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HARNESS_ROOT not in sys.path:
    sys.path.insert(0, _HARNESS_ROOT)

import config
from scoring.cache import ScoringCache, cache_key


# Canonical key inputs reused across the invalidation battery.
_K = dict(
    prompt="PROMPT BODY",
    model="claude-opus-4-7",
    prompt_version="v9.5",
    tool_schema_sha="toolsha",
    call_params_sha="paramsha",
)


def _fresh_cache():
    """A ScoringCache on a throwaway DB file; returns (cache, tmpdir)."""
    tmp = tempfile.mkdtemp(prefix="test_cache_")
    return ScoringCache(os.path.join(tmp, "scoring.db")), tmp


def test_cache_key_deterministic():
    assert cache_key(**_K) == cache_key(**_K)


def test_cache_key_field_separation():
    """The NUL separators must prevent field-boundary collisions: moving a
    character across the tool_schema/call_params boundary changes the key.
    Without separators 't'+'c' and 'tc'+'' would join to the same string."""
    k1 = cache_key(prompt="x", model="m", prompt_version="pv",
                   tool_schema_sha="t", call_params_sha="c")
    k2 = cache_key(prompt="x", model="m", prompt_version="pv",
                   tool_schema_sha="tc", call_params_sha="")
    assert k1 != k2


def test_cache_key_every_component_invalidates():
    base = cache_key(**_K)
    for field, changed in [
        ("prompt", "PROMPT BODY2"),
        ("model", "claude-opus-4-8"),
        ("prompt_version", "v9.6"),
        ("tool_schema_sha", "toolsha2"),
        ("call_params_sha", "paramsha2"),
    ]:
        assert cache_key(**{**_K, field: changed}) != base, field


def test_put_get_roundtrip():
    cache, tmp = _fresh_cache()
    try:
        key = cache_key(**_K)
        response = {"market_assessment": "constructive",
                    "scores": [{"ticker": "NVDA", "score": 8, "reasoning": "strong"}]}
        assert cache.get(key) is None
        cache.put(key, response, {"model": "claude-opus-4-7", "prompt_version": "v9.5",
                                  "strategy": "steady", "prompt_sha": "abc"})
        assert cache.get(key) == response
    finally:
        cache.close()
        shutil.rmtree(tmp, ignore_errors=True)


def test_get_miss_returns_none():
    cache, tmp = _fresh_cache()
    try:
        assert cache.get("no-such-key") is None
    finally:
        cache.close()
        shutil.rmtree(tmp, ignore_errors=True)


def test_put_is_idempotent():
    """Re-storing the same key replaces, not duplicates (one row, latest wins)."""
    cache, tmp = _fresh_cache()
    try:
        key = cache_key(**_K)
        meta = {"model": "claude-opus-4-7", "prompt_version": "v9.5",
                "strategy": "steady", "prompt_sha": "abc"}
        cache.put(key, {"v": 1}, meta)
        cache.put(key, {"v": 2}, meta)
        assert cache.stats()["rows"] == 1
        assert cache.get(key) == {"v": 2}
    finally:
        cache.close()
        shutil.rmtree(tmp, ignore_errors=True)


def test_stats():
    cache, tmp = _fresh_cache()
    try:
        cache.put("k1", {"a": 1}, {"model": "m", "prompt_version": "v", "strategy": "steady",
                                   "prompt_sha": "s1", "est_cost_usd": 0.10})
        cache.put("k2", {"a": 2}, {"model": "m", "prompt_version": "v", "strategy": "steady",
                                   "prompt_sha": "s2", "est_cost_usd": 0.20})
        cache.put("k3", {"a": 3}, {"model": "m", "prompt_version": "v", "strategy": "pulse",
                                   "prompt_sha": "s3", "est_cost_usd": 0.05})
        st = cache.stats()
        assert st["rows"] == 3
        assert abs(st["total_est_cost"] - 0.35) < 1e-9
        assert st["by_strategy"] == {"steady": 2, "pulse": 1}
    finally:
        cache.close()
        shutil.rmtree(tmp, ignore_errors=True)


def test_config_prices_opus_4_7():
    """Pin the PF-1 confirmed prices wired into config.PRICES (L9)."""
    assert config.PRICES["claude-opus-4-7"] == {"input_per_mtok": 5.00, "output_per_mtok": 25.00}


if __name__ == "__main__":
    test_cache_key_deterministic()
    test_cache_key_field_separation()
    test_cache_key_every_component_invalidates()
    test_put_get_roundtrip()
    test_get_miss_returns_none()
    test_put_is_idempotent()
    test_stats()
    test_config_prices_opus_4_7()
    print("test_cache OK: key determinism + field-separation + invalidation, "
          "put/get, miss, idempotent re-store, stats, PF-1 prices.")
