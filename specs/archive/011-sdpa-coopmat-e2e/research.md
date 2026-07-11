# Research: SDPA Coopmat E2E Validation

## Decision 1: No new export needed -- confirmed empirically, not just assumed

**Finding** (constitution Principle VI -- verified with tools during
planning, not deferred to implementation): ran `009`'s existing
`Buffer`-storage export for `llama-3.2-1b`/`4w`
(`specs/009-e2e-tokrate-report/results/pte/llama-3.2-1b_4w.pte`, unchanged)
with `ET_VK_SDPA_COOPMAT=1` set, against the same `cmake-out-vk-etdump`
build `009` already used (confirmed via `git status` that zero production
runtime files changed between `009`'s build and now -- `010` only added
test code). ETDump shows `sdpa_compute_attn_weights_coopmat_buffer_buffer_half`
and `sdpa_compute_out_coopmat_buffer_buffer_half` each dispatched 16 times,
matching `llama-3.2-1b`'s 16 transformer layers exactly. A quick sanity
timing from that same run (not a rigorous capture) showed 2303.71 tok/s
prefill vs. `009`'s recorded 1867.40 tok/s baseline for this exact
configuration -- a strong, encouraging signal, not the final measurement.

**Decision**: Confirms spec.md's Assumption -- reuse all six of `009`'s
existing exports directly; no new export step exists anywhere in this
feature's task list. No rebuild is needed either (no production code
changed since `009`'s last build).

**Alternatives considered**: None -- this was a factual question (does the
toggle require a new export or rebuild?) with a single verifiable answer,
now confirmed rather than assumed.

## Decision 2: ETDump dispatch confirmation -- reuse `009`'s exact methodology

**Decision**: For each of the six configurations, capture an ETDump trace
with `ET_VK_SDPA_COOPMAT=1` set (`--max_new_tokens 1` to isolate prefill,
matching `002`/`009`'s established recipe), and confirm via the same
`Inspector`-based kernel-name extraction `009` already used that every
measured SDPA dispatch for that configuration's prefill contains
`_coopmat` for both the QK^T and attn·V positions.

**Rationale**: This is `009`'s own tier-2 dispatch-confirmation mechanism
(constitution Principle VI), already proven and reused verbatim -- no new
tooling needed, only a new invocation per configuration with the toggle
set. The `cmake-out-vk-etdump` build tree already exists from `009` and
needs no changes.

**Alternatives considered**: None -- this is the workstream's already
-established tier-2 verification mechanism; there's no reason to invent a
different one for this feature.

## Decision 3: E2E capture methodology -- reuse `009`'s exact procedure

**Decision**: 5 repeated runs per configuration, `ET_VK_SDPA_COOPMAT=1` set,
against the standard (non-ETDump) `cmake-out-vk` build, no concurrent GPU
load, identical fixed workload (2048-token prefill / 1024-token decode,
`--warmup true`) -- byte-identical procedure to `009`'s own capture, just
with the one additional env var set.

**Rationale**: Reuse, not reinvention -- `009` already validated this exact
methodology satisfies constitution Principle IV for this precise kind of
comparison (a new dispatch arm added to an otherwise-identical export).

**Alternatives considered**: None -- carried forward from `001`/`006`/`009`.

## Decision 4: Baseline source -- parse `009`'s own report table

**Decision**: Read the "WMMA" column of `009`'s already-published
`specs/009-e2e-tokrate-report/results/e2e-tokrate-report.md` per-configuration
table directly (regex-parsed, mirroring how `009`'s own script parsed
`006`'s report table) as this feature's baseline -- not `009`'s raw
per-rep logs, and not a re-capture.

**Rationale**: `009`'s report is the citable, already-validated source for
"linear coopmat enabled, SDPA still tiled" e2e numbers; parsing its stable
table format continues this workstream's established reuse chain
(`004`→`006`→`009`→`011`) without duplicating numbers by hand (transcription
risk) or re-deriving them from raw logs unnecessarily.

**Alternatives considered**: Re-parsing `009`'s raw per-rep logs directly.
Rejected -- `009`'s report table is already the validated, citable
aggregate; re-deriving from raw logs would just risk reintroducing a
transcription or aggregation bug `009` already resolved.

## Decision 5: Report verdict logic

**Decision**: Per configuration, state the e2e prefill/decode tok/s with
SDPA coopmat enabled alongside `009`'s baseline pair, the relative
difference, and a consistency verdict against `010`'s microbenchmark-level
finding for that model (66.8% average, direction-only comparison --
`010`'s number is an isolated-shader tier-1 figure, `011`'s is a real
whole-model tier-2 figure diluted by every other op in the model, so
magnitude is not expected to match, only direction: e2e should show *some*
prefill improvement, not necessarily 60-70%). One overall statement per
scheme is not needed here (unlike `009`'s `4w`/`8da4w` split, which existed
because linear coopmat's own effect diverged by scheme) -- SDPA coopmat's
effect is not expected to diverge by scheme (research.md `010`'s own
Decision 5: SDPA shape/dispatch is scheme-independent), so one overall
statement covering all six configurations is appropriate here, with any
per-configuration outlier still named explicitly (spec.md Edge Cases).

**Alternatives considered**: Splitting the overall verdict by scheme, mirroring
`009`. Rejected -- `009`'s split existed because `4w`/`8da4w` genuinely
diverged in direction for linear coopmat; nothing in this feature's own
prior findings (`010`) suggests SDPA coopmat's effect would diverge by
scheme, so a single overall statement is accurate, not a premature
simplification -- and if the actual measurement contradicts this, that
divergence itself gets named explicitly per FR-006/spec.md Edge Cases,
not hidden by the reporting structure.

## Decision 6: Cross-session variance caveat, inherited again

**Finding**: `006`'s documented cross-session prefill variance (inherited by
`009`) applies again here -- this feature's own capture is yet another new
session vs. `009`'s. Decode has no such precedent of drift.

**Decision**: Every prefill row in this feature's report carries the same
inherited caveat `009` already carries forward from `006`, worded
identically for consistency across this workstream's reports.

**Alternatives considered**: None -- this is a known, already-documented
property of this hardware, not a new finding requiring its own
investigation.
