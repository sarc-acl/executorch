# Research: End-to-End Speedup Target and Validation

## Decision 1: Reuse `001`'s exact e2e JSON schema for both baseline and future re-measurement

**Decision**: The target and any future re-measurement read/write the same
`e2e` object shape `001` already produces in
`specs/001-minipc-baseline-benchmarks/results/raw/<model>_<scheme>.json`:
`status`, `prefill_tokens`, `decode_tokens`, `prefill_tokens_per_sec`,
`decode_tokens_per_sec`, `num_runs`, `variance` (with
`prefill_tokens_per_sec_stdev`/`decode_tokens_per_sec_stdev`), and
`run_metadata` (confirmed by inspecting `llama-3.2-1b_4w.json` directly).
No new capture format is introduced.

**Rationale**: `001` already built and validated the entire capture procedure
(export, `llama_main` runner, 5-rep statistical discipline, cold-start
discard) end to end for exactly this fixed 2048-token prefill / 1024-token
decode workload. Reusing its schema means a future re-measurement is a
drop-in replacement, not a new pipeline to build and separately validate.

**Alternatives considered**: designing a new, dedicated schema for this
feature — rejected; it would need its own validation effort for no benefit,
and would risk the "not directly comparable" failure mode (Edge Case 3 /
FR-008) that this feature explicitly guards against.

## Decision 2: Report granularity is per (model, scheme), not per model alone

**Decision**: The target, re-measurement, and outcome are computed and
reported per (model, scheme) — 6 configurations — matching the granularity
every prior feature in this workstream (`001`-`004`) already used.

**Rationale**: The spec's "per model" language (FR-004, FR-007) is about
never averaging *across models* — it does not require collapsing `4w` and
`8da4w` together. The two schemes may see genuinely different coopmat
behavior (different dtype paths, different eligibility details per `003`'s
research), so keeping them separate preserves exactly the information a
reader needs. A model's "outcome" in prose (e.g., "Llama 3.2 1B met its
target") is naturally read as "both its configs met their targets" without
needing a forced single number.

**Alternatives considered**: averaging `4w`/`8da4w` into one number per
model — rejected; would hide a case where one scheme benefits from coopmat
and the other doesn't, which is exactly the kind of information this
workstream's whole methodology exists to surface, not obscure.

## Decision 3: This pass's actual deliverable is the target + validated tooling, not a real report

**Decision**: Since no build with actual optimization work exists yet, this
feature's buildable scope is:
1. `results/speedup-target.json` — the formalized target, computed directly
   from `001`'s existing baseline numbers (real, not synthetic).
2. `scripts/compute_outcome.py` — the comparison/verdict tool.
3. A **self-test**: synthetic "after" JSON files (same schema as Decision 1),
   deliberately constructed to hit each of five outcomes (exactly 2x/"met",
   above 2x/"exceeded", below 2x/"missed", below baseline/"regressed", and a
   deliberately mismatched `run_metadata` to trigger "not_comparable"), run
   through the same tool used for real data, to prove every verdict branch
   fires correctly. All synthetic files and the self-test's own output are
   kept under `results/selftest/`, clearly separated from (and never named
   the same as) the real `outcome-report.md` path.
4. `quickstart.md` documents exactly how to invoke the tool for real once an
   optimization build and a real re-measurement JSON exist — that invocation
   itself is future work, not part of this pass.

**Rationale**: Producing a real `outcome-report.md` right now would require
either fabricating "after" numbers (a direct violation of Constitution
Principle I, correctness before performance, and of this spec's own
Assumptions) or leaving the feature incomplete with no way to tell if the
tooling even works. The self-test resolves this honestly: it proves the tool
correct using clearly-labeled synthetic data, without ever claiming a real
result exists.

**Alternatives considered**: waiting to plan/implement this feature at all
until real optimization work exists — rejected; the spec is explicit
(FR-002) that the target must be defined *before* implementation begins, and
having the tool built and proven now means the moment a real "after" JSON
exists, producing the real report is a single command, not a new development
effort under time pressure.

## Decision 4: Target computation formula

**Decision**: `target_prefill_tokens_per_sec = baseline.e2e.prefill_tokens_per_sec * 2.0`
per (model, scheme), read directly from each of the six existing
`001` baseline JSON files. `target_multiplier` is a named constant (`2.0`)
per the Clarifications session, not derived from anything else.

**Verdict thresholds** (applied to `observed_multiplier = after.e2e.prefill_tokens_per_sec / baseline.e2e.prefill_tokens_per_sec`):
- `observed_multiplier < 1.0` → `regressed`
- `1.0 <= observed_multiplier < 2.0` → `missed`
- `observed_multiplier == 2.0` (within the baseline's own measured stdev-based noise band, i.e. `variance.prefill_tokens_per_sec_stdev`) → `met`
- `observed_multiplier > 2.0` (beyond that same noise band) → `exceeded`

**Rationale**: Using the baseline's own already-measured stdev as the
noise band (rather than an arbitrary new tolerance) keeps the "met vs.
exceeded" boundary consistent with this workstream's established statistical
rigor (Constitution Principle IV) instead of introducing an unrelated new
threshold.
