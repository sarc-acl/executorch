# Implementation Plan: 8da4w Tile/Subgroup Sweep Ranked by End-to-End Throughput

**Branch**: `027-e2e-tile-sweep` | **Date**: 2026-07-11 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/027-e2e-tile-sweep/spec.md`

## Summary

Re-rank the `8da4w` (`dq8ca_q4gsw` int8 WMMA) tile/subgroup candidates that `specs/025`
(subgroup=64) and `specs/026` (subgroup=32) already correctness-verified and
microbenchmark-scored, by real end-to-end throughput instead of isolated-kernel GFLOP/s —
since `specs/026`'s own Tier-2 check found its microbenchmark winner is actually slower
end-to-end. Take the top 8 candidates by combined microbenchmark rank (Clarifications) to
a single e2e screening run each, on the model whose per-layer shapes match the candidate's
own microbenchmark shapes (the specific mistake `specs/026` had to catch and fix); escalate
only screening results within 10% of, or ahead of, the shipped baseline to a 3-run
confirmation (Clarifications' adaptive statistical bar). If no confirmed candidate beats
the baseline, optionally extend to a small, budgeted set of new candidates (User Story 2);
otherwise report the confirmed winner. End with exactly one unambiguous e2e-ranked answer.

## Technical Context

**Language/Version**: Python 3 (e2e orchestration/ranking script, following `025`'s/`026`'s
`staged_search.py` pattern but driving `llama_main` runs instead of `test_coopmat_linear_bench`
runs); no shader/dispatch code changes are anticipated (User Story 1 reuses `025`'s and
`026`'s already-built shader variants as-is) — User Story 2, if triggered, reuses the
existing `linear_dq8ca_q4gsw_coopmat_tsweep.{glsl,yaml}` template and
`ET_VK_DQ8CA_COOPMAT_VARIANT=tsweep_...` dispatch token, adding new `shader_variants`
entries only if new candidates are actually selected.

**Primary Dependencies**: `specs/025`'s `round2_results.json`/`round3_results.json`
(subgroup=64 microbenchmark scores) and `specs/026`'s `correctness_matrix.json`/
`round3_results.json` (subgroup=32 microbenchmark scores + per-shape correctness) as the
combined pre-filter input (spec FR-002); the `llama_main` runner already built this
session in the `dbuf-int8-sweep` execution worktree
(`cmake-out-android-vk/examples/models/llama/llama_main`), rebuilt only if a candidate
needs a shader variant that isn't already in that binary; the existing buffer-storage
`8da4w` PTEs for 1B/3B/8B already staged on NFS
(`llama3_{2_1b,2_3b,1_8b}_8da4w_buffer_ctx3072.pte`) — model choice per candidate is
resolved from which microbenchmark shape family (1B/3B/8B) the candidate's existing score
came from (spec FR-003/Assumptions), not assumed uniformly 8B; `adb` access to M5 EVT1;
clock-pinning script.

**Storage**: N/A — file-based. Combined pre-filter ranking, per-candidate e2e screening/
confirmation results, and the final report are JSON/Markdown under this feature's
`specs/027-e2e-tile-sweep/results/`.

**Testing**: no new correctness methodology — a candidate's existing `025`/`026`
correctness-gate result (all representative shapes passing) is a hard prerequisite for
inclusion in the shortlist (spec FR-004); this feature adds no new correctness check, only
an e2e throughput measurement on top of already-correctness-verified candidates.

**Target Platform**: Samsung M5 EVT1 (Exynos 2500/Xclipse 970), Android, pinned clocks
(Principle VII); driver identity re-verified before every measurement round (Principle
VIII); either board may be used, screening/confirmation results record which one.

**Project Type**: Single project — a bounded, internal research/automation addition,
reusing `025`/`026`'s established pattern, elevated to Tier-2 (Principle IV) throughout
rather than as a single post-hoc check.

**Performance Goals**: produce exactly one unambiguous e2e-ranked answer (spec FR-008): a
specific candidate confirmed faster than the shipped baseline with 3-run statistical
backing, or an explicit, evidence-backed statement that the baseline remains the e2e
winner.

**Constraints**: shortlist size fixed at 8 candidates by combined microbenchmark rank
(Clarifications); adaptive 1-run-screen → 3-run-confirm bar, confirmation triggered only
within 10% of or ahead of baseline (Clarifications, spec FR-005); every e2e measurement
must use the shape-matched model for its candidate (spec FR-003) — no cross-model-size
comparisons; User Story 2's search extension (if triggered) stays within a small,
pre-declared additional budget (spec FR-007/FR-009) and must not build/measure once a real
winner is already confirmed (spec FR-006); no correctness-unverified candidate may be
reported as a contender (spec FR-004); driver identity and device availability
re-verified before every measurement round; production default dispatch is unchanged by
this feature regardless of outcome (spec Assumptions).

**Scale/Scope**: 8 initial candidates (screening: 8 runs; confirmation: 0-8 candidates ×
3 runs, depending how many screen within the 10% band) plus, only if User Story 2
triggers, a small additional set (target: single digits, per spec SC-004) — far smaller
than the 1000+-candidate legal tile/subgroup/subgroup-size space. No new shader variants
needed for User Story 1 (all 8 candidates already exist as built binaries from
`025`/`026`); User Story 2 candidates, if any, need new `shader_variants` entries and a
rebuild, following `026`'s established extension pattern.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Applicability | Status |
|---|---|---|
| I. Correctness Before Performance | Every candidate entering the e2e shortlist already passed `025`'s or `026`'s full-shape correctness gate (spec FR-004); this feature adds no candidate without that prerequisite. | PASS |
| II. M5 EVT1 Is the Only Active Target | All e2e measurement happens on M5 EVT1 (primary and/or secondary board) exclusively. | PASS |
| III. Explicit Eligibility Gating, Safe Fallback Always | No production dispatch-gating logic is modified; all measurement is via the existing opt-in `ET_VK_DQ8CA_COOPMAT_VARIANT` token. Any User-Story-2 candidates are additive shader_variants entries, same as `025`/`026`. | N/A (documented) |
| IV. Two-Tier, Statistically Sound Benchmarking | This feature *is* the Tier-2 rollout this principle calls for, done properly (throughout the search, not as a single post-hoc check as `026` did) — directly implements the constitution's "e2e is the deliverable, microbench is for analysis" framing. The adaptive screen→confirm bar (Clarifications) still reports iteration count and never a single-run comparison as a win. | PASS |
| V. Document Every Driver Workaround at the Point of Use | Not applicable — this feature measures existing shaders, introduces no new driver workaround. | N/A |
| VI. Verify With Tools, Never Assume | e2e throughput read directly from the runner's own `PyTorchObserver` timing output (real measured tok/s, not estimated); model coherence spot-checked (short-prompt sanity output) before trusting a PTE/binary combination, following this session's own `026` Tier-2 practice. | PASS |
| VII. Clock Discipline | Clocks pinned and pin-verified before every measurement round. | PASS |
| VIII. Verify the Driver Before Every Coopmat Measurement | Driver hash and device availability re-checked before every round, on whichever board is in use; recorded per result. | PASS |
| IX. Never Disclose Samsung-Internal Specifics Upstream | Entirely internal workstream work on `origin` (`sarc-acl/executorch`); nothing proposed upstream. | N/A |
| X. Consult `.shared-context/instruction-for-ai` Before Acting | Reuses the Android e2e run recipe (`ET_VK_EXECUTE_NODE_THRESHOLD`, `p2048_exact.txt`, `pin_freqs.sh`) already documented in `access-and-run/README.md` and already exercised in this session's `026` Tier-2 check — no new build/run procedure to derive. | PASS |

No violations requiring justification — Complexity Tracking is not needed.

## Project Structure

### Documentation (this feature)

```text
specs/027-e2e-tile-sweep/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md         # Phase 1 output
├── quickstart.md         # Phase 1 output
├── contracts/
│   └── e2e-ranking-schema.md   # Shape of the pre-filter input, screening/confirmation
│                                 # results, and final report
├── tasks.md              # Phase 2 output (/speckit-tasks)
└── results/               # Phase 3+ output: combined pre-filter ranking, screening
                            # results, confirmation results, any User-Story-2 extension
                            # candidates, final report
```

### Source Code (repository root)

This feature does not introduce a new src/tests tree for User Story 1 (no shader/dispatch
changes — it only runs the existing `llama_main` runner against existing shader variants).
User Story 2, if triggered, extends the existing `025`/`026` shader-variant catalog using
their established pattern. Paths below are relative to the execution worktree (see
Structure Decision).

```text
backends/vulkan/runtime/graph/ops/glsl/
└── linear_dq8ca_q4gsw_coopmat_tsweep.glsl/.yaml   # NOT modified for User Story 1; extended
                                                     # with new shader_variants entries only
                                                     # if User Story 2 triggers, following
                                                     # 025's/026's established pattern exactly

backends/vulkan/runtime/graph/ops/impl/
└── QuantizedLinear.cpp   # NOT modified for User Story 1 (existing tsweep_t...s{32,64}
                            # token parsing already covers every candidate); extended only
                            # if User Story 2 selects a genuinely new tile/grid/subgroup
                            # combination not already representable by the existing token
                            # format (unlikely, since the format is general)

examples/models/llama/    # NOT modified -- llama_main already built this session
                            # (cmake-out-android-vk/examples/models/llama/llama_main),
                            # reused as-is; env-var-driven dispatch selection needs no
                            # runner changes

# Analysis/orchestration tooling (lives with the spec-kit docs in THIS repo,
# not the execution worktree, following 025's/026's precedent):
specs/027-e2e-tile-sweep/scripts/
├── build_prefilter_ranking.py   # Phase 1: merge 025's round2/round3 + 026's
│                                  # correctness_matrix/round3 into one ranked,
│                                  # correctness-filtered candidate list; select top 8
├── run_e2e_screen.py            # Phase 2: one e2e screening run per shortlisted
│                                  # candidate, on its shape-matched model, records
│                                  # PyTorchObserver prefill (and decode) tok/s
├── run_e2e_confirm.py           # Phase 2: 3-run confirmation for any candidate whose
│                                  # screening result is within 10% of, or ahead of,
│                                  # the shipped baseline
└── build_report.py              # Phase 3: produces the final e2e-ranked report,
                                   # including the microbench-vs-e2e rank agreement
                                   # finding (spec SC-006)
```

**Structure Decision**: Single project, mirroring `025`/`026`. Spec/plan/tasks documents
and analysis/orchestration scripts live in this repo's `specs/027-e2e-tile-sweep/`.
Execution (running `llama_main` against existing PTEs/shader variants) happens in the
**existing** `dbuf-int8-sweep` worktree (`023-8da4w-int8-dbuf-sweep-impl` branch) — the
same worktree `026` used, and the one that already has the shader-variant catalog and a
freshly-built `llama_main` from this session. This is consistent with `026`'s own Decision
5 (reuse warm, already-relevant infrastructure rather than branching a fresh worktree per
`025`'s original precedent) and is even more clearly justified here, since User Story 1
needs zero new shader/dispatch code — only a runner and PTEs that already exist there.
`dev/executorch` itself is never checked out onto this feature's working branch.

## Post-Design Constitution Re-Check

Re-evaluated after Phase 1 (data-model.md, contracts/, quickstart.md): no new violations
introduced. The adaptive screen→confirm design keeps Principle IV's statistical-rigor bar
intact while bounding device time (screening is cheap, confirmation is reserved for
plausible contenders only); every `E2EMeasurement` record carries driver hash, board, and
clock-pin state (Principle VII/VIII); the shape-matched-model requirement is enforced as a
data field (`model_used`, cross-checked against `candidate.shape_family`), not left
implicit, directly preventing a repeat of this session's own 1B/8B mismatch error.
Constitution Check table above still holds: PASS on all applicable principles, N/A on the
rest (documented), no Complexity Tracking entries needed.
