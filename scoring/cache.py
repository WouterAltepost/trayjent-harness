"""SQLite scoring cache (TBH Phase 4, brief Commit 2 / decision L2, L7).

Sits behind the Phase-2 scorer seam: one assembled prompt -> one stored scoring
response. The cache is the *reproducibility* mechanism for backtests — Claude
scoring is non-deterministic, so "rerun and get the same numbers" is only true
when the rerun hits a populated, frozen cache. Cost reduction is the side
effect. Treat the key (L2) as the contract: if two genuinely different inputs
ever collide, a cited backtest silently lies.

Pure: sqlite3 + hashlib + json + datetime. No config / anthropic / reuse
import — the DB path and every cache-key field value are passed in by the
caller (the Phase-4 cached_scorer), so this module never drags in the live
stack.
"""
import hashlib
import json
import sqlite3
from datetime import datetime, timezone


# Field separator for the cache key. A byte that cannot appear in any of the
# joined fields, so two different field splits cannot collide (see cache_key).
_NUL = "\x00"


def cache_key(prompt, model, prompt_version, tool_schema_sha, call_params_sha):
    """Compute the per-batch cache key (L2).

    sha256 over the NUL-joined components, in this fixed order:
        [prompt_version, model, tool_schema_sha, call_params_sha, prompt]

    Anything that, if changed, would change Claude's output must be a component
    here. The assembled prompt already encodes the signals batch (incl.
    previous_score), the market-context block, the ticker set/order, and the
    rubric text — a rubric edit changes the prompt and auto-invalidates.
    PROMPT_VERSION is the manual force-invalidate knob; model, tool-schema, and
    call-params guard the rest. The NUL separators prevent field-boundary
    collisions; the prompt goes last because it is the largest, variable field.
    """
    joined = _NUL.join([prompt_version, model, tool_schema_sha, call_params_sha, prompt])
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS scoring_cache (
  cache_key      TEXT PRIMARY KEY,
  model          TEXT NOT NULL,
  prompt_version TEXT NOT NULL,
  strategy       TEXT,
  prompt_sha     TEXT NOT NULL,
  response_json  TEXT NOT NULL,
  input_tokens   INTEGER,
  output_tokens  INTEGER,
  est_cost_usd   REAL,
  created_at     TEXT NOT NULL
);
"""


class ScoringCache:
    """SQLite-backed store, one row per assembled-prompt batch (L7).

    Stores the raw tool_use.input dict (L3) and re-parses it on every read via
    parse_scoring_response upstream — never post-parse decisions — so a
    buy_threshold or parse-logic change is honored without invalidating the
    cache. WAL mode for concurrent-safe reads. The full prompt is NOT stored
    (85k x multi-KB rows bloats the DB); only its sha for debug/dedup.
    """

    def __init__(self, db_path):
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def get(self, key):
        """Return the stored raw tool_use.input dict for `key`, or None on miss."""
        row = self._conn.execute(
            "SELECT response_json FROM scoring_cache WHERE cache_key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def put(self, key, response_json, meta):
        """Store the raw tool_use.input dict (`response_json`) under `key`.

        `meta` carries the audit/cost columns: required `model`,
        `prompt_version`, `prompt_sha`; optional `strategy`, `input_tokens`,
        `output_tokens`, `est_cost_usd`. created_at is stamped here (ISO8601
        UTC). INSERT OR REPLACE so a re-store of the same key is idempotent.
        """
        self._conn.execute(
            """INSERT OR REPLACE INTO scoring_cache
               (cache_key, model, prompt_version, strategy, prompt_sha,
                response_json, input_tokens, output_tokens, est_cost_usd, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                key,
                meta["model"],
                meta["prompt_version"],
                meta.get("strategy"),
                meta["prompt_sha"],
                json.dumps(response_json),
                meta.get("input_tokens"),
                meta.get("output_tokens"),
                meta.get("est_cost_usd"),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()

    def stats(self):
        """Audit summary: row count, total estimated cost, per-strategy counts."""
        rows = self._conn.execute("SELECT COUNT(*) FROM scoring_cache").fetchone()[0]
        total = self._conn.execute(
            "SELECT COALESCE(SUM(est_cost_usd), 0.0) FROM scoring_cache"
        ).fetchone()[0]
        by_strategy = dict(
            self._conn.execute(
                "SELECT strategy, COUNT(*) FROM scoring_cache GROUP BY strategy"
            ).fetchall()
        )
        return {"rows": rows, "total_est_cost": total, "by_strategy": by_strategy}

    def close(self):
        self._conn.close()
