"""Offline guard for the Phase-4 harness Anthropic scorer (brief Commit 3).

The real call is only exercised in live runs / a manual smoke (no networked
unit test). This checks the two import-time invariants that matter offline:
the module imports without dragging in live `config`/`agent` (it would hard-fail
on Alpaca keys), and the API key is read lazily — a missing ANTHROPIC_API_KEY
raises a clear RuntimeError at call time, not at import.

Runnable via pytest or directly:
    python tests/test_anthropic_scorer.py
"""
import os
import sys

_HARNESS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HARNESS_ROOT not in sys.path:
    sys.path.insert(0, _HARNESS_ROOT)

from scoring.anthropic_scorer import make_anthropic_scorer   # import => no live config


def test_missing_api_key_raises_at_call_time():
    saved = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        scorer = make_anthropic_scorer()    # building the scorer needs no key
        try:
            scorer("PROMPT")                # the call needs it -> RuntimeError, no network
            assert False, "expected RuntimeError on missing ANTHROPIC_API_KEY"
        except RuntimeError as e:
            assert "ANTHROPIC_API_KEY" in str(e)
    finally:
        if saved is not None:
            os.environ["ANTHROPIC_API_KEY"] = saved


if __name__ == "__main__":
    test_missing_api_key_raises_at_call_time()
    print("test_anthropic_scorer OK: lazy key, clean import (no live config).")
