"""Cache-backed scorer (TBH Phase 4, brief Commit 3 / decision L3, L8).

Wraps an injectable `live_scorer` behind the SQLite cache. The seam contract is
unchanged: scorer(prompt) -> a list of content blocks that parse_scoring_response
consumes. The live_scorer is injectable so unit tests pass a fake (canned
content + usage) and never hit the network; the real one is anthropic_scorer.

Hit  -> rebuild a synthetic content list from the stored raw tool_use.input
        (no live call, no cost, cost tracker untouched).
Miss -> fail-closed cost guard -> live_scorer(prompt) -> store raw input -> return
        the SAME synthetic content list, so hit and miss are byte-identical
        downstream.
"""
import hashlib
import json

from scoring.cache import cache_key
from scoring.cost import CostCeilingExceeded


_TOOL_NAME = "submit_ticker_scores"


class _CachedBlock:
    """Synthetic stand-in for an Anthropic tool_use content block. Exposes only
    .type / .name / .input — exactly what parse_scoring_response reads (PF-3)."""

    def __init__(self, input):
        self.type = "tool_use"
        self.name = _TOOL_NAME
        self.input = input


def _sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _default_shas():
    """Compute the tool-schema and call-param shas from the reuse shim. Imported
    lazily so unit tests can inject fixed shas without requiring the shim."""
    from harness.reuse import SCORING_TOOL, SCORING_MAX_TOKENS, SCORING_TOOL_CHOICE
    tool_schema_sha = _sha(json.dumps(SCORING_TOOL, sort_keys=True))
    call_params_sha = _sha(json.dumps(
        {"max_tokens": SCORING_MAX_TOKENS, "tool_choice": SCORING_TOOL_CHOICE},
        sort_keys=True,
    ))
    return tool_schema_sha, call_params_sha


def make_cached_scorer(cache, live_scorer, cost_tracker=None, *, model,
                       prompt_version, strategy=None,
                       tool_schema_sha=None, call_params_sha=None):
    """Build scorer(prompt) -> content blocks, cache-backed.

    cache_key components (L2) come from `model`, `prompt_version`, and the
    tool-schema / call-param shas. The shas default to values derived from the
    reuse shim (live behaviour); tests inject fixed strings to stay shim-free.
    `cost_tracker` is optional — when present, a miss is fail-closed-guarded
    before the spend and the observed usage is accrued after.
    """
    if tool_schema_sha is None or call_params_sha is None:
        d_tool, d_params = _default_shas()
        tool_schema_sha = tool_schema_sha or d_tool
        call_params_sha = call_params_sha or d_params

    def scorer(prompt):
        key = cache_key(prompt, model, prompt_version, tool_schema_sha, call_params_sha)

        stored = cache.get(key)
        if stored is not None:
            # Hit: free, deterministic replay. Never touch the cost tracker.
            return [_CachedBlock(stored)]

        # Miss: fail closed before the spend (Phase 4 primitive; Phase 5 wires
        # the per-run ceiling). rolling_estimate() is 0.0 before any call, so the
        # first call is never pre-empted.
        if cost_tracker is not None:
            cost_tracker.check(cost_tracker.rolling_estimate())

        content = live_scorer(prompt)
        usage = getattr(live_scorer, "usage", None)

        tool_block = next(
            (b for b in content if b.type == "tool_use" and b.name == _TOOL_NAME),
            None,
        )
        if tool_block is None:
            raise RuntimeError(
                f"live_scorer returned no {_TOOL_NAME} tool_use block; content={content!r}"
            )
        response_json = tool_block.input

        est_cost = None
        if cost_tracker is not None and usage is not None:
            est_cost = cost_tracker.estimate(usage)
            cost_tracker.add(usage)

        cache.put(key, response_json, {
            "model": model,
            "prompt_version": prompt_version,
            "strategy": strategy,
            "prompt_sha": _sha(prompt),
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
            "est_cost_usd": est_cost,
        })

        # Return synthetic content so a miss and a later hit are byte-identical
        # downstream (miss -> fill -> hit yields identical decisions).
        return [_CachedBlock(response_json)]

    return scorer
