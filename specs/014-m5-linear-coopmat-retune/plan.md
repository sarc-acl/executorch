# Implementation Plan: M5 EVT1 `4w` Linear Coopmat Retune (fp16 Accumulate, Loop Flattening, Vectorized Dequant)

**Branch**: `014-m5-linear-coopmat-retune` | **Date**: 2026-07-05 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/014-m5-linear-coopmat-retune/spec.md`

## Summary

Four changes already exist as uncommitted edits in this branch's working
tree, authored before spec-kit tracking was set up on it: three code changes
to the `4w` weight-only int4 coopmat linear shader
(`linear_qw_coopmat.glsl` / `linear_q4gsw_coopmat`) -- an fp16-accumulate
experiment, a dbuf1 loop-shape flattening, and a vectorized INT4 dequant --
plus one documentation-only comment addition to the sibling `8da4w` shader
(`linear_dq8ca_qw_coopmat.glsl`) and its dispatch code
(`QuantizedLinear.cpp`) recording a prior A/B finding. This feature's core
job is not to build anything new: it is to (1) commit the existing work with
accurate per-change attribution and status (US1), then (2) run this
workstream's required correctness/performance validation on the actual M5
EVT1 target for the two same-math changes (US2) and the one precision-risky
change (US3), recording each change's final disposition independently.
**Revised during clarification (2026-07-05 session, FR-008)**: closing US2/
US3's correctness gate requires one small piece of genuinely new work first
-- extending `test_coopmat_linear_bench.cpp`'s correctness shape coverage to
production K, since direct inspection found its existing coverage stops at
K=256.

## Technical Context

**Language/Version**: GLSL (Vulkan compute shaders, `.glsl`/`.yaml` template
pairs already in tree) for the three shader changes; C++17
(`QuantizedLinear.cpp`, existing ExecuTorch Vulkan backend code) for the
documentation-only dispatch-code change; no new language/runtime introduced.

**Primary Dependencies**:
- `backends/vulkan/runtime/graph/ops/glsl/linear_qw_coopmat.glsl` /
  `.yaml` -- the shader carrying all three code changes (already edited,
  uncommitted).
- `backends/vulkan/runtime/graph/ops/glsl/linear_dq8ca_qw_coopmat.glsl` and
  `backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.cpp`
  (`add_linear_dqa_qw_node`) -- carry the documentation-only change (already
  edited, uncommitted).
- `backends/vulkan/test/custom_ops/test_coopmat_linear_bench.cpp` -- the
  existing INT4 coopmat correctness check (per constitution Principle I),
  via its `kCorrectnessShapes`/`kRank3CorrectnessShapes` deterministic
  well-conditioned-data cases. **Revised per spec Clarifications (2026-07-05
  session) / FR-008**: this file is NOT purely reused as-is -- direct
  inspection found its existing shapes top out at K=256, short of FR-003/
  FR-004's production-K (2048/4096+) requirement, so this feature extends
  it with new cases at production K, reusing its already-validated
  `abs=0.5`/`rel=0.05` tolerance strategy. Also reused unmodified for tier-1
  microbenchmark timing, alongside `test_llama_baseline_bench.cpp` (both
  already in tree from specs `007`/`008`).
- `spirv-dis` (or equivalent) for SPIR-V inspection per Principle VI.
- ETDump / the standard ExecuTorch LLaMA runner for the tier-2 e2e leg
  (only if User Story 2/3's tier-1 results motivate a tier-2 check; not
  required by this feature's minimum scope).

**Storage**: Flat files -- this feature's own `results/` directory for
validation logs and the final disposition report; no database/service
component.

**Testing**: No new test framework, but the existing
`test_coopmat_linear_bench.cpp` correctness harness gains new cases (FR-008)
-- extending its `kCorrectnessShapes`/`kRank3CorrectnessShapes` tables to
production K, not authoring a new harness. Performance gating reuses the
existing `BenchmarkResult`-based tier-1 harness unmodified (constitution
Principle IV). This feature's own "test" of User Story 1 is a
diff/attribution check (every uncommitted hunk maps to exactly one described
change) rather than an automated suite.

**Target Platform**: Samsung M5 EVT1 (Exynos 2500 / Xclipse 970) -- this
workstream's sole active validation target per constitution Principle II.
`rocky-ryzen` MiniPC is retired and used only as historical reference for
where the dbuf1 loop variant was originally chosen (specs `007`-`012`).

**Project Type**: Retroactive documentation + hardware validation of
already-written shader/dispatch code, plus one small, in-scope test-harness
extension (FR-008). No new production subsystem; beyond the three files
already sitting in the working tree, this feature additionally edits one
existing test file (`test_coopmat_linear_bench.cpp`) to add production-K
correctness cases -- test code, not shipped production/runtime code.

**Performance Goals**: No committed target -- this feature's User Stories 2
and 3 *measure* whether each change helps, regresses, or is neutral; it does
not assume a specific speedup going in (per spec Clarifications, the
same-math changes may be kept for maintainability even absent a measured
win, but that is a recorded decision, not an assumed default).

**Constraints**:
- Per spec Clarifications: the three shader changes are validated and
  disposed of **independently** -- a correctness failure in the
  fp16-accumulate experiment (US3) must not block committing or validating
  the two same-math changes (US2), and the documentation-only change (item
  4) ships regardless of either outcome.
- Per constitution's Performance & Portability Standards ("Precision"): "Any
  reduced-precision accumulation path must demonstrate it stays within the
  existing per-op correctness test's tolerance before landing" -- this is
  the literal, pre-existing constitutional bar the fp16-accumulate change
  (US3) must clear.
- Per constitution Principle IV: no performance number is reported without
  iteration count + stddev (tier-1) and, for any tier-2 claim, a *separate*
  ETDump dispatch-confirmation run.
- Per constitution Principle VI: any shader change must have its compiled
  SPIR-V inspected to confirm the expected cooperative-matrix instructions
  are present -- applies to all three shader code changes (fp16 accumulate,
  loop flattening, vectorized dequant), since coopmat instruction shape can
  be sensitive to accumulator component type and loop structure.
- If M5 EVT1 access is unavailable, User Story 1 still completes in full;
  User Stories 2/3 are explicitly reported as blocked, not skipped or
  assumed (spec FR-006).
- Per spec FR-008: FR-003/FR-004's correctness gate cannot be satisfied
  against the existing `test_coopmat_linear_bench.cpp` shape list as-is (it
  tops out at K=256); the harness extension to production K is a
  prerequisite for US2 and US3's correctness steps, not optional prep work.

**Scale/Scope**: 3 shader-level changes to 1 shader file + 1
documentation-only change spanning 2 files + 1 correctness-harness
extension (new production-K cases in `test_coopmat_linear_bench.cpp`, per
FR-008) = 4 committed changes plus one in-scope test-harness extension,
each independently disposed of (keep / keep-with-caveat / revert) for the
three shader changes.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Checked against `.specify/memory/constitution.md` (v2.1.0, current committed
`HEAD`):

- **I. Correctness Before Performance (NON-NEGOTIABLE)**: PASS, by design --
  and this principle is exactly what surfaced FR-008's finding. US2 and US3
  both gate their respective changes on the existing INT4 coopmat
  correctness check before any performance number is trusted (spec
  FR-003/FR-004), and per the 2026-07-05 clarification session that check's
  own shape coverage (K<=256) was confirmed insufficient for the
  production-K claim this feature needs to make -- rather than silently
  measuring performance against an unvalidated shape, FR-008 requires
  extending the check first. The fp16-accumulate change (US3) is explicitly
  the higher-risk case this principle exists for -- coopmat's
  mixed-precision accumulation is called out by name in this principle's
  own rationale.
- **II. Samsung M5 EVT1 Is the Only Active Target**: PASS. All validation in
  this feature runs on M5 EVT1, not `rocky-ryzen` (spec Assumptions). The
  loop-flattening change's *origin* (the dbuf1 sweep) was MiniPC-based, but
  this feature re-validates the flattened code-shape itself on the real
  target rather than trusting the MiniPC-era result to carry over unchanged
  -- consistent with this principle's requirement that MiniPC data is
  historical/comparative only, not a substitute for M5 EVT1 validation.
- **III. Explicit Eligibility Gating, Safe Fallback Always**: PASS, N/A to
  modify. None of the four changes touch `can_use_q4gsw_coopmat` or any
  other eligibility gate; the documentation-only change explicitly exists to
  *prevent* an eligibility-adjacent workaround (the `dq8ca_qw` spec-const
  path) from being dropped by a future contributor who assumes the
  now-fixed driver bugs make it removable.
- **IV. Two-Tier, Statistically Sound Benchmarking**: PASS by scope
  restriction. This feature's minimum scope (per spec) is tier-1 (shader
  microbenchmark) only, matching how `007` preceded `009`'s tier-2 work. A
  tier-2 e2e number is not required for this feature to conclude, but if one
  is captured, it MUST follow the separate-ETDump-run rule.
- **V. Document Every Driver Workaround at the Point of Use**: PASS,
  directly implements this principle. The documentation-only change (item 4)
  *is* this principle being applied retroactively to `add_linear_dqa_qw_node`
  -- naming the specific 2026-06-30 A/B finding at the point of use so a
  future contributor does not revert a load-bearing workaround.
- **VI. Verify With Tools, Never Assume**: PASS by design. Spec FR-005
  requires kernel-dispatch/SPIR-V confirmation for every claim in this
  feature; this is the explicit mechanism, not an afterthought.
- **VII. Clock Discipline**: PASS, inherited. Any M5 EVT1 timing run in US2/
  US3 follows the workspace's standard pinned-clock default
  (`.shared-context/instruction-for-ai/README.md` §Conventions); this
  feature does not introduce a new clock-handling need.
- **VIII. Verify the Driver Before Every Coopmat Measurement**: PASS,
  inherited. Standard pre-measurement driver-identity check applies
  unchanged; this feature does not modify driver-verification tooling.
- **IX. Never Disclose Samsung-Internal Specifics Upstream**: PASS, N/A.
  This feature's work stays entirely within this internal workstream branch
  (`quant-perf-optimization`); none of it targets the public
  `pytorch/executorch` PR (`yanwen/quant-dev`, a different branch/worktree
  entirely per workspace `CLAUDE.md`). No upstream-bound artifact is
  produced here.
- **Performance & Portability Standards / Precision**: Directly applicable
  to US3 -- "any reduced-precision accumulation path must demonstrate it
  stays within the existing per-op correctness test's tolerance before
  landing" is the literal acceptance bar already written into spec FR-004
  before this Constitution Check was performed (confirms the spec was
  already constitution-aligned, not retrofitted).

No violations. Complexity Tracking is not needed.

## Project Structure

### Documentation (this feature)

```text
specs/014-m5-linear-coopmat-retune/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── checklists/
│   └── requirements.md  # Spec quality checklist (already created by /speckit-specify)
└── tasks.md             # Phase 2 output (/speckit-tasks, not this command)
```

No `contracts/` directory: this feature has no external interface (API,
CLI, schema) of its own -- it commits existing internal shader/dispatch code
and produces an internal validation report, matching the no-contracts
precedent set by specs `001`/`004`/`006`/`012` (internal
measurement/analysis features in this same workstream).

### Source Code (repository root)

No new production source files. This feature commits three files already
modified in the working tree, unchanged from their current diff (User Story
1), then may apply corrective edits only if User Story 3's correctness check
fails (the FR-004 revert path). It additionally edits one existing **test**
file (FR-008, new in this revision) before US2/US3's correctness steps can
run:

```text
backends/vulkan/runtime/graph/ops/glsl/linear_qw_coopmat.glsl        # fp16 accumulate + loop flattening + vectorized dequant (already written, uncommitted)
backends/vulkan/runtime/graph/ops/glsl/linear_dq8ca_qw_coopmat.glsl  # documentation-only (already written, uncommitted)
backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.cpp           # documentation-only, in add_linear_dqa_qw_node (already written, uncommitted)
backends/vulkan/test/custom_ops/test_coopmat_linear_bench.cpp        # FR-008: extend kCorrectnessShapes/kRank3CorrectnessShapes with production-K (2048/4096+) cases, same data/tolerance strategy -- new work, not yet written

specs/014-m5-linear-coopmat-retune/
└── results/
    ├── us2-loop-vectorized-dequant-validation.md   # correctness + tier-1 perf for the two same-math changes
    ├── us3-fp16-accumulate-validation.md            # correctness + (if passed) tier-1 perf for the fp16-accumulate change
    └── disposition-summary.md                       # per-change keep/keep-with-caveat/revert, cross-referenced
```

**Structure Decision**: Same lightweight, no-new-production-code structure
as specs `001`/`004`/`006`/`012`: the "implementation" is committing
already-written code, and this feature's own deliverable is the validation
report under its own `results/`. The one exception is FR-008's small,
targeted extension of the existing `test_coopmat_linear_bench.cpp`
correctness harness (new cases, same file, same methodology) -- this is not
new tooling, just closing a shape-coverage gap in tooling that already
exists.

## Complexity Tracking

*No violations -- table not needed.*
