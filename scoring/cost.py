"""Fail-closed cost tracker (TBH Phase 4, brief Commit 3 / decision L8).

A primitive: accumulate observed Claude spend from `message.usage` and refuse a
call that would breach the ceiling *before* the spend, never after. Phase 4
builds and unit-tests it; the per-run wiring (instantiate with COST_CEILING_USD,
reset per run) is the Phase-5 runner's job.

Cache hits cost nothing and must never be passed through here — the cached
scorer only calls add()/estimate() on a miss.
"""


class CostCeilingExceeded(Exception):
    """Raised before a live call that would push run spend over the ceiling."""


class CostTracker:
    def __init__(self, ceiling_usd, prices, model):
        self.ceiling_usd = ceiling_usd
        self.prices = prices
        self.model = model
        self.spent_usd = 0.0
        self.n_calls = 0

    def estimate(self, usage):
        """USD cost of one usage record (reads .input_tokens / .output_tokens).
        Pure pricing math from the per-model PRICES table; no state change."""
        p = self.prices[self.model]
        return (
            (usage.input_tokens / 1_000_000) * p["input_per_mtok"]
            + (usage.output_tokens / 1_000_000) * p["output_per_mtok"]
        )

    def add(self, usage):
        """Accumulate observed spend after a live call. Returns new total."""
        self.spent_usd += self.estimate(usage)
        self.n_calls += 1
        return self.spent_usd

    def would_exceed(self, est):
        """True if accruing an estimated `est` would push total over the ceiling.
        Strict '>': landing exactly on the ceiling is allowed."""
        return (self.spent_usd + est) > self.ceiling_usd

    def check(self, est):
        """Fail closed: raise CostCeilingExceeded if `est` would breach. Does not
        mutate state — the spend is only recorded by add() after a real call."""
        if self.would_exceed(est):
            raise CostCeilingExceeded(
                f"Estimated next call ${est:.4f} would push spend "
                f"${self.spent_usd:.4f} over the ${self.ceiling_usd:.2f} ceiling."
            )

    def rolling_estimate(self):
        """Forward estimate for the next call: the mean observed per-call cost so
        far, or 0.0 before any call (discovery §6 — a rolling average is fine)."""
        return self.spent_usd / self.n_calls if self.n_calls else 0.0
