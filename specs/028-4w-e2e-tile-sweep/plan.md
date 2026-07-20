# Implementation Plan: 4w Tile/Subgroup Sweep Ranked by End-to-End Throughput

**Branch**: `028-4w-e2e-tile-sweep` | **Date**: 2026-07-12 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/028-4w-e2e-tile-sweep/spec.md`

## Summary

Re-rank the `4w` (`linear_q4gsw_coopmat`) tile/subgroup candidates that `specs/022`
already correctness-verified and microbenchmark-scored, by real end-to-end throughput
instead of isolated-kernel GFLOP/s — mirroring `specs/027`'s already-validated 8da4w
methodology, per the user's explicit request. Take the 8 correctness-passing candidates
from `022`'s Round 2 (all already ranked by measured GFLOP/s) to a single e2e screening
run each, on the Llama 3.1 8B `4w` buffer PTE (the shape family `022`'s scores are keyed
to); escalate only screening results within 10% of, or ahead of, the shipped baseline
(`022`'s own dbuf1 default) to a 3-run confirmation. If no confirmed candidate beats the
baseline, optionally extend to a small, budgeted set of new candidates. Once User Story 3
produces one definitive 8B answer, confirm that exact config end-to-end on 1B and 3B as a
non-searching validation pass (Clarifications 2026-07-12). Every e2e measurement uses the
full stack of optimizations already shipped on `dev` (WMMA coopmat linear/SDPA, node-
threshold watchdog workaround), not an isolated `4w`-kernel build.

**Critical Phase 0 finding, not present in `027`'s equivalent plan**: unlike `8da4w`'s
tsweep infra (fully committed on the `dbuf-int8-sweep` worktree), `4w`'s tile/subgroup
variant infrastructure from `specs/022` (`linear_q4gsw_coopmat_tsweep.{glsl,yaml}`, its
`ET_VK_Q4GSW_COOPMAT_VARIANT` dispatch token) was **never committed to any branch** — it
exists only as an uncommitted patch frozen at
`.archived-artifacts/tmp-origcm-2026-07-08/`, based on old commit `1da18955a`. `dev`'s
current `4w` shader (`linear_q4gsw_coop.glsl`, fixed 128×64/K16/2×2/s32 dispatch, no
variant-selection env var at all) diverged substantially since then (`1da18955a` is not
even an ancestor of `dev`; `linear_q4gsw_coopmat.glsl` itself was deleted/renamed).
Reaching User Story 1 therefore requires a **port**, not a `git apply` — re-derive the
tile/subgroup-parameterized shader and env-var dispatch against `dev`'s current shader,
using the archived patch as reference material only. This is the direct `4w` analogue of
the "`dev` never had the `dbuf2` loop-structure port before this — applying the tile
winner required porting the loop structure too" finding `ACTIVE-STATUS.md` already
recorded for `8da4w`/`specs/027`.

## Technical Context

**Language/Version**: Python 3 (e2e orchestration/ranking script, following `027`'s
`build_prefilter_ranking.py`/`run_e2e_screen.py`/`run_e2e_confirm.py`/`build_report.py`
pattern, driving `llama_main` runs); GLSL 450 / C++17 for the one-time port described
above (new `linear_q4gsw_coopmat_tsweep.{glsl,yaml}` derived from `dev`'s current
`linear_q4gsw_coop.glsl`, plus an `ET_VK_Q4GSW_COOPMAT_VARIANT` dispatch token added to
`QuantizedLinear.cpp`, following the exact pattern `ET_VK_DQ8CA_COOPMAT_VARIANT` already
establishes for `8da4w` in the same file).

**Primary Dependencies**: `specs/022`'s `results/round2_results.json` (8 correctness-
passing candidates with measured GFLOP/s) and `results/round3_results.json` (the
confirmed Round-3 winner, `tsweep_t128x64k16g14s32`) as the pre-filter input (spec
FR-002); the archived reference patch at
`.archived-artifacts/tmp-origcm-2026-07-08/untracked-new-files/.../linear_q4gsw_coopmat_tsweep.{glsl,yaml}`
as the starting point for the port (read-only reference, not applied directly — see
Summary); `dev`'s current `linear_q4gsw_coop.glsl`/`QuantizedLinear.cpp` (the port target,
carrying the full WMMA/SDPA-coopmat stack this feature must measure against, unlike
`022`'s own now-vanished base); buffer-storage `4w` PTEs for 1B/3B/8B already staged on
NFS (`llama3_{2_1b,2_3b,1_8b}_4w_buffer_ctx3072.pte`); `adb` access to M5 EVT1; clock-
pinning script.

**Storage**: N/A — file-based. Combined pre-filter ranking, per-candidate e2e screening/
confirmation results, the 1B/3B confirmation pass, and the final report are JSON/Markdown
under this feature's `specs/028-4w-e2e-tile-sweep/results/`.

**Testing**: no new correctness methodology for the *shortlist* — a candidate's existing
`022` correctness-gate result (fp32-reference pass, per `022`'s Constitution Principle I
gate) is a hard prerequisite for shortlist inclusion (spec FR-004). The one genuinely new
correctness step is re-verifying that the **ported** tsweep shader (built against `dev`'s
current base, not `022`'s old base) still passes `022`'s existing correctness harness for
every shortlisted candidate before any e2e measurement — a port that silently changed
semantics must not reach e2e undetected.

**Target Platform**: Samsung M5 EVT1 (Exynos 2500/Xclipse 970), Android, pinned clocks
(Principle VII); driver identity re-verified before every measurement round (Principle
VIII); either board may be used, screening/confirmation results record which one.

**Project Type**: Single project — a bounded, internal research/automation addition,
reusing `022`'s established pre-filter data and `027`'s established e2e-ranking
methodology, plus a one-time infra port (see Summary) that `027` did not need.

**Performance Goals**: produce exactly one unambiguous e2e-ranked 8B answer (spec FR-008):
a specific candidate confirmed faster than the shipped baseline with 3-run statistical
backing, or an explicit statement that the baseline remains the e2e winner — then confirm
that same answer's config end-to-end on 1B and 3B (spec FR-012).

**Constraints**: shortlist = the 8 correctness-passing candidates from `022`'s Round 2
(spec FR-002; already ≤8, so no further trimming is needed — see research.md Decision 1);
adaptive 1-run-screen → 3-run-confirm bar, confirmation triggered only within 10% of or
ahead of baseline (spec FR-005); every 8B-stage e2e measurement must use the Llama 3.1 8B
`4w` buffer PTE (spec FR-003 — `022`'s scores are all keyed to the 8B-shaped
representative shapes, same convention `027` confirmed for `8da4w`); User Story 2's search
extension (if triggered) stays within a small, pre-declared additional budget (spec
FR-007/FR-009) and must not build/measure once a real winner is already confirmed (spec
FR-006); the final 8B answer's exact config (winner or baseline) is confirmed end-to-end
on 1B and 3B as a non-searching pass, not independently re-swept (spec FR-012); no
correctness-unverified candidate may be reported as a contender (spec FR-004); the
full existing `dev` optimization stack (WMMA coopmat linear/SDPA, node-threshold
workaround) stays enabled throughout — no isolated-kernel measurement (spec FR-011);
driver identity and device availability re-verified before every measurement round;
production default dispatch is unchanged by this feature regardless of outcome (spec
Assumptions).

**Scale/Scope**: 8 initial 8B candidates (screening: 8 runs + 1 baseline; confirmation:
0-8 candidates × 3 runs, depending how many screen within the 10% band), plus the 1B/3B
confirmation pass (2 additional models × the winning config, screen+confirm as needed)
and, only if User Story 2 triggers, a small additional 8B-shape set (target: single
digits, per spec SC-004). Far smaller than `022`'s full 642-candidate legal tile space.
Unlike `027`, this feature is NOT zero-new-shader-code for User Story 1 — the one-time
tsweep infra port (Summary) is required before any candidate beyond the current fixed
default can even be dispatched.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Applicability | Status |
|---|---|---|
| I. Correctness Before Performance | Every candidate entering the e2e shortlist already passed `022`'s full correctness gate (spec FR-004); the port itself is re-verified against that same gate before any candidate it enables is measured (Technical Context, Testing). | PASS |
| II. M5 EVT1 Is the Only Active Target | All e2e measurement happens on M5 EVT1 (primary and/or secondary board) exclusively. | PASS |
| III. Explicit Eligibility Gating, Safe Fallback Always | No production dispatch-gating logic is modified; the ported `ET_VK_Q4GSW_COOPMAT_VARIANT` token is opt-in only (unset = today's fixed default dispatch, unchanged), following `ET_VK_DQ8CA_COOPMAT_VARIANT`'s exact existing pattern. Any User-Story-2 candidates are additive shader_variants entries, same as `022`/`025`/`026`. | N/A (documented) |
| IV. Two-Tier, Statistically Sound Benchmarking | This feature *is* the Tier-2 rollout `022` itself deferred as follow-on work (spec Context) — implements the constitution's "e2e is the deliverable, microbench is for analysis" framing throughout, not as a single post-hoc check. The adaptive screen→confirm bar still reports iteration count and never a single-run comparison as a win. | PASS |
| V. Document Every Driver Workaround at the Point of Use | Not applicable to the e2e-ranking logic itself; the tsweep port must carry forward any driver-workaround comments already present in `dev`'s current `linear_q4gsw_coop.glsl` (e.g. any Xclipse PAL-compiler-bug workaround), not silently drop them. | Conditional — verified during port |
| VI. Verify With Tools, Never Assume | e2e throughput read directly from the runner's own `PyTorchObserver` timing output; model coherence spot-checked before trusting a PTE/binary combination; the port's correctness re-verification (Testing) is itself an instance of this principle — a port is not assumed equivalent to its source, it is checked. | PASS |
| VII. Clock Discipline | Clocks pinned and pin-verified before every measurement round. | PASS |
| VIII. Verify the Driver Before Every Coopmat Measurement | Driver hash and device availability re-checked before every round, on whichever board is in use; recorded per result. | PASS |
| IX. Never Disclose Samsung-Internal Specifics Upstream | Entirely internal workstream work on `origin` (`sarc-acl/executorch`); nothing proposed upstream. | N/A |
| X. Consult `.shared-context/instruction-for-ai` Before Acting | Reuses the Android e2e run recipe (`ET_VK_EXECUTE_NODE_THRESHOLD`, `p2048_exact.txt`, `pin_freqs.sh`) already documented in `access-and-run/README.md`; PTE staging paths already documented in `setup/README.md`. | PASS |

No violations requiring justification — Complexity Tracking below documents the one
extra step (the infra port) this feature needs beyond `027`'s template, but it is not a
constitution violation.

## Project Structure

### Documentation (this feature)

```text
specs/028-4w-e2e-tile-sweep/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md         # Phase 1 output
├── quickstart.md         # Phase 1 output
├── contracts/
│   └── e2e-ranking-schema.md   # Shape of the pre-filter input, screening/confirmation
│                                 # results, 1B/3B confirmation pass, and final report
├── tasks.md              # Phase 2 output (/speckit-tasks)
└── results/               # Phase 3+ output: ported-shader correctness re-verification,
                            # pre-filter ranking, screening/confirmation results, 1B/3B
                            # confirmation pass, any User-Story-2 extension, final report
```

### Source Code (repository root)

Unlike `027`, this feature is NOT zero-shader-code for User Story 1 — reaching the
shortlist requires the one-time port described in Summary/Technical Context first. Paths
below are relative to the execution worktree (see Structure Decision).

```text
backends/vulkan/runtime/graph/ops/glsl/
├── linear_q4gsw_coop.glsl/.yaml     # Port SOURCE (dev's current fixed-dispatch 4w
│                                      # shader) — read for reference, not deleted; the
│                                      # ported tsweep variant is a sibling file, following
│                                      # the same coexistence pattern dq8ca_q4gsw_coopmat_
│                                      # {dbuf1-4,tsweep} already use in this same directory
└── linear_q4gsw_coopmat_tsweep.glsl/.yaml   # NEW: ported tile/subgroup-parameterized
                                               # shader, re-derived from the archived
                                               # reference patch against dev's current base;
                                               # extended with new shader_variants entries
                                               # only if User Story 2 triggers

backends/vulkan/runtime/graph/ops/impl/
└── QuantizedLinear.cpp   # Gains an ET_VK_Q4GSW_COOPMAT_VARIANT env-var dispatch token,
                            # following ET_VK_DQ8CA_COOPMAT_VARIANT's exact existing
                            # pattern in this same file (spec Assumptions: no new
                            # dispatch mechanism invented, this one is copied)

examples/models/llama/    # llama_main rebuilt once after the port lands, then reused
                            # as-is for every screening/confirmation run; env-var-driven
                            # dispatch selection needs no further runner changes

# Analysis/orchestration tooling (lives with the spec-kit docs in THIS repo):
specs/028-4w-e2e-tile-sweep/scripts/
├── build_prefilter_ranking.py   # Phase 1: read 022's round2_results.json (8
│                                  # correctness-passing candidates) + round3_results.json
│                                  # (confirmed winner) into one ranked candidate list
├── run_e2e_screen.py            # Phase 2: one e2e screening run per shortlisted
│                                  # candidate on the 8B model, records PyTorchObserver
│                                  # prefill (and decode) tok/s
├── run_e2e_confirm.py           # Phase 2: 3-run confirmation for any candidate whose
│                                  # 8B screening result is within 10% of, or ahead of,
│                                  # the shipped baseline
├── run_1b3b_confirmation.py     # Phase 2 (User Story 4): re-measure the final 8B
│                                  # answer's exact config end-to-end on 1B and 3B
└── build_report.py              # Phase 3: final e2e-ranked report, incl. microbench-
                                   # vs-e2e rank agreement (SC-006) and the 1B/3B
                                   # confirmation finding (SC-007)
```

**Structure Decision**: Single project. Spec/plan/tasks documents and analysis/
orchestration scripts live in this repo's `specs/028-4w-e2e-tile-sweep/`. Execution
(the shader port, rebuild, and all `llama_main` runs) happens on a **new feature
worktree/branch cut from `dev` (`yanwen/dev-1.3`)**, not inside `dev/` itself — per this
workspace's standing rule that new work goes on a feature branch PR'd into `dev`, never
committed directly there, and per the critical-rule prohibition on repointing an existing
worktree folder. `022`'s own execution worktree no longer exists (retired 2026-07-11), so
there is no warm worktree to reuse here the way `027` reused `dbuf-int8-sweep` — a fresh
`git worktree add 4w-e2e-tile-sweep 028-4w-e2e-tile-sweep` (cut from `yanwen/dev-1.3`) is
required, giving this feature `dev`'s full current optimization stack (WMMA coopmat
SDPA, node-threshold workaround) as its base — which is itself required by spec FR-011
(compare against the full-stack baseline, not an isolated kernel).

## Post-Design Constitution Re-Check

Re-evaluated after Phase 1 (data-model.md, contracts/, quickstart.md): no new violations
introduced. The port step is scoped narrowly (one new shader file + one env-var dispatch
token, both following exact existing patterns already in the same files) and is gated by
its own correctness re-verification before any e2e measurement trusts it (Constitution
Principle I re-applied at the port boundary, not just at `022`'s original boundary). The
adaptive screen→confirm design keeps Principle IV's statistical-rigor bar intact while
bounding device time; every `E2EMeasurement` record carries driver hash, board, and
clock-pin state (Principle VII/VIII); the 1B/3B confirmation pass (User Story 4) is
explicitly a non-searching validation step, not a scope-creeping re-sweep, keeping this
feature's device-time budget bounded (spec FR-009) even with the added model coverage.
Constitution Check table above still holds: PASS on all applicable principles, N/A or
Conditional (resolved during the port) on the rest, no Complexity Tracking entries needed.
