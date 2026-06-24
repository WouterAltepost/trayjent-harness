"""Harness miss-path Anthropic scorer (TBH Phase 4, brief Commit 3 / L4, L5).

The cache-miss call. Byte-identical in call params to agent._live_scorer, but
built here so a backtest needs no Alpaca keys: it reads ANTHROPIC_API_KEY
lazily from the environment and pulls model / tool / call params from the reuse
shim. It NEVER imports `agent` or live `config` (both hard-fail on Alpaca keys
at import). Only exercised in live runs (Phase 5) / a manual smoke — the
cache/cost unit tests inject a fake live_scorer instead.
"""
import os

import anthropic

from harness.reuse import (
    SCORING_MODEL,
    SCORING_TOOL,
    SCORING_MAX_TOKENS,
    SCORING_TOOL_CHOICE,
)


def make_anthropic_scorer():
    """Return scorer(prompt) -> message.content (list of content blocks), doing
    the real Anthropic call. The last call's usage is exposed on scorer.usage so
    the cached scorer can record token counts and accrue cost. ANTHROPIC_API_KEY
    is read at call time, not import time."""

    def scorer(prompt):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set; the harness Anthropic scorer needs "
                "it to make a cache-miss call."
            )
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=SCORING_MODEL,
            max_tokens=SCORING_MAX_TOKENS,
            tools=[SCORING_TOOL],
            tool_choice=SCORING_TOOL_CHOICE,
            messages=[{"role": "user", "content": prompt}],
        )
        scorer.usage = message.usage
        return message.content

    scorer.usage = None
    return scorer
