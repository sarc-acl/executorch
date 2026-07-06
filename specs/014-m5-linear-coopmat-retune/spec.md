# Feature Specification: M5 EVT1 `4w` Linear Coopmat Retune (fp16 Accumulate, Loop Flattening, Vectorized Dequant)

**Feature Branch**: `014-m5-linear-coopmat-retune`

**Created**: 2026-07-05

**Status**: Draft

**Input**: User description: "Retroactively document already-written, uncommitted code changes on this branch (authored on this PC before spec-kit tracking was set up here) so they get proper spec-kit provenance before being committed. The changes are three related tunings to the M5 EVT1 `linear_q4gsw_coopmat` int4 weight-only coopmat linear shader plus documentation-only clarifications in the sibling int8-activation shader and its C++ dispatch: (1) fp16 accumulator experiment (precision-risky, not yet correctness-tested), (2) dbuf1 loop-shape flattening (same algorithm, not yet re-measured in this shape), (3) vectorized INT4 dequant (same math, fewer scalar ops), (4) a documentation-only note recording a 2026-06-30 A/B finding that the sibling `dq8ca_qw` shader's spec-const workaround must not be dropped. None of this is hardware-validated yet; the spec must capture that honestly rather than as a measured/complete result."

## Clarifications

### Session 2026-07-05

- Q: Items 1-3 are three independent, separable code changes to the same shader, each with a different validation bar (item 1 is precision-risky and needs a correctness pass before any perf claim; items 2-3 are same-math code-shape changes that only need perf/regression confirmation). Should this feature gate all three together, or validate and decide each independently? → A: Validate and decide independently. Bundling them would let a correctness failure in the fp16-accumulate experiment (item 1) block committing/keeping the two low-risk, same-math changes (items 2-3), and would also make a single combined perf number impossible to attribute to a specific change.
- Q: Item 4 is a comment-only change with no runtime effect. Does it need the same hardware-validation gate as items 1-3? → A: No. It records an already-established fact (the 2026-06-30 A/B finding, cited from `add_linear_dqa_qw_node` / spec 013's line of work) next to the code it protects. It ships as soon as this spec's changes are committed, independent of items 1-3's validation outcomes.
- Q: Given Principle II of this workstream's constitution (Samsung M5 EVT1 is the only active target) and Principle IV (two-tier benchmarking required before any performance claim), can this spec report "improved" or "regressed" for items 1-3 without an actual M5 EVT1 run? → A: No. This spec's own scope is bounded to getting the existing implementation correctly documented and committed; the User Story that actually runs the tier-1/tier-2 validation on M5 EVT1 is this feature's own P1 deliverable, not a prerequisite assumed already done.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Preserve and correctly attribute the existing uncommitted work (Priority: P1) 🎯 MVP

As the contributor picking this workstream back up on the Samsung/Xclipse
target machine, I want the three shader tunings and the one documentation-only
clarification -- all currently sitting as uncommitted edits in the working
tree from before spec-kit tracking existed on this branch -- captured in a
spec, planned, and committed with accurate rationale, so this real work
is not silently lost to a `git stash`/`git clean`/branch-switch accident and
so a future reader knows *why* each change exists without re-deriving it from
the diff alone.

**Why this priority**: This is the literal, immediate risk: real, non-trivial
shader engineering (a precision experiment, a loop restructure, an ALU
optimization, and a hard-won driver-bug-workaround rationale) exists only as
uncommitted working-tree state. Everything else in this spec (validating it)
is moot if the code itself is lost first.

**Independent Test**: Diff the working tree against the spec's description of
each of the four changes; confirm every uncommitted hunk is accounted for by
exactly one described change, with no unattributed edits left over. Confirm a
`git commit` of these files succeeds and the resulting commit message/spec
reference explains each change's origin and current validation status.

**Acceptance Scenarios**:

1. **Given** the four currently-uncommitted changes (fp16 accumulator,
   loop-shape flattening, vectorized dequant, dq8ca_qw documentation note),
   **When** this feature's plan and tasks are written, **Then** each change is
   traceable to its own task with its own validation status (done /
   not-yet-validated), not merged into one undifferentiated commit.
2. **Given** the documentation-only change (item 4), **When** the other three
   items' hardware validation is still pending, **Then** the documentation
   change is committed anyway, since it records an already-established fact
   and carries no runtime risk.

---

### User Story 2 - Validate the two low-risk, same-math changes on M5 EVT1 (Priority: P2)

As the contributor, I want the dbuf1 loop-shape flattening and the vectorized
INT4 dequant -- both same-math, code-shape-only changes -- correctness- and
performance-checked on the actual M5 EVT1 target device, so their claimed
"fewer instructions / less branch overhead" benefit is confirmed rather than
assumed, per this workstream's constitution (Principle II: M5 EVT1 is the
only active target; Principle IV: two-tier benchmarking required before any
performance claim).

**Why this priority**: These two changes carry no numerical-precision risk
(the loop restructure preserves the exact dbuf1 algorithm already measured as
the sweep winner in specs `007`-`012`; the vectorized dequant is algebraically
identical to the scalar form it replaces), so they can be validated as soon
as an M5 EVT1 session is available, independent of whether the higher-risk
fp16-accumulate experiment (User Story 3) ever passes.

**Independent Test**: Build the shader with both changes applied, run the
existing coopmat correctness check on real production shapes, then run the
tier-1 microbenchmark and confirm a kernel-name-dispatch-verified coopmat
timing exists to compare against the pre-change baseline.

**Acceptance Scenarios**:

1. **Given** the flattened-loop, vectorized-dequant shader variant, **When**
   it is built and run against the existing INT4 coopmat correctness check,
   **Then** it passes with no numerical difference from the pre-change
   shader (same algorithm, same math).
2. **Given** a passing correctness run, **When** the tier-1 microbenchmark is
   run on M5 EVT1, **Then** a kernel-dispatch-confirmed timing is produced
   and compared against the last known-good coopmat baseline, reporting an
   explicit percentage difference (improvement, regression, or noise-level
   no-change) rather than an assumed win.

---

### User Story 3 - Validate the fp16-accumulate experiment's correctness before any perf claim (Priority: P3)

As the contributor, I want the fp16-accumulate coopmat variant correctness-
tested against real production K-dimensions (K=2048..4096, the actual
model-shape reduction lengths, not just the existing small synthetic-shape
tests) on M5 EVT1, so the precision risk explicitly flagged in-code is either
confirmed safe or the change is reverted -- before any throughput number for
it is trusted or reported.

**Why this priority**: This is the highest-risk of the three changes (an
accumulator precision change over a long reduction) and is explicitly gated
in the existing code comment on a correctness pass that has never been run.
It is lower priority than User Story 2 only because its outcome (keep fp16
accumulate, or revert to fp32) does not block committing or validating the
other two, independent changes.

**Independent Test**: Run the fp16-accumulate shader variant against the
existing correctness check at real production shapes; compare its numerical
output against the fp32-accumulate reference within an explicitly stated
tolerance.

**Acceptance Scenarios**:

1. **Given** the fp16-accumulate shader variant, **When** it is run against
   production K=2048/4096 shapes through the existing correctness check,
   **Then** the result either passes within a stated numerical tolerance
   (and is then eligible for the tier-1/tier-2 perf measurement) or fails
   explicitly, with the failure reported as a reason to revert this specific
   change -- not silently worked around.
2. **Given** a correctness pass, **When** the tier-1 microbenchmark is run,
   **Then** the throughput comparison against the fp32-accumulate baseline is
   reported with kernel-dispatch confirmation, consistent with User Story 2's
   measurement discipline.
3. **Given** a correctness failure, **When** this feature concludes, **Then**
   the fp16-accumulate change is reverted to fp32 accumulate in the committed
   shader, and the failure (shape, magnitude of divergence) is recorded so it
   is not re-attempted without a different approach.

---

### Edge Cases

- What happens if M5 EVT1 hardware access is unavailable when this feature is
  worked? User Story 1 (preserve + commit with accurate status) still
  completes; User Stories 2 and 3 are explicitly reported as blocked/pending
  hardware access, not silently dropped or assumed passing.
- What happens if the fp16-accumulate variant (User Story 3) fails
  correctness at large K but the loop-flattening/vectorized-dequant changes
  (User Story 2) pass? Each change's disposition is independent, per the
  Clarifications above: User Story 2's changes ship even if User Story 3's
  is reverted.
- What happens if a re-measurement under User Story 2 shows the flattened
  loop is *not* faster (e.g. driver-specific scheduling differences from the
  `rocky-ryzen` MiniPC this loop shape was originally swept on)? The finding
  is reported as-is (including a regression or no-change result); the
  same-math code-shape change may still be kept for
  maintainability/simplicity reasons even absent a measured win, but that is
  an explicit decision recorded in the results, not an assumed default.
- What happens to the `linear_dq8ca_qw_coopmat.glsl` / `QuantizedLinear.cpp`
  documentation-only change if its cited 2026-06-30 A/B finding cannot be
  independently reproduced during this feature's work? Per the Clarifications
  above, it ships regardless, since it records history/rationale, not a new
  runtime claim; if the finding is later found stale, that is a follow-up
  correction, not a blocker for this feature.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: This feature MUST commit the fp16-accumulate change, the
  loop-shape-flattening change, the vectorized-dequant change, and the
  `linear_dq8ca_qw_coopmat.glsl`/`QuantizedLinear.cpp` documentation-only
  change as attributable, separable units (whether as separate commits or
  clearly separable diff hunks referenced individually in this feature's
  results), not as one undifferentiated bundle.
- **FR-002**: The documentation-only change (recording the 2026-06-30 A/B
  finding that `add_linear_dqa_qw_node`'s spec-const workaround must not be
  dropped) MUST be committed independent of User Stories 2 and 3's
  validation outcomes, since it has no runtime effect and records an
  already-established fact.
- **FR-003**: The loop-shape-flattening and vectorized-dequant changes MUST
  each pass the existing INT4 coopmat correctness check on M5 EVT1 at real
  production shapes before being reported as validated.
- **FR-004**: The fp16-accumulate change MUST pass a correctness check
  against real production K-dimensions (K=2048..4096) on M5 EVT1, within an
  explicitly stated numerical tolerance, before any throughput number for it
  is reported; a correctness failure MUST result in reverting this specific
  change to fp32 accumulate, not a silent workaround.
- **FR-005**: Every performance claim this feature makes MUST follow this
  workstream's constitution Principle IV (two-tier benchmarking: a
  dispatch-confirming run plus a separate, statistically sound timed run) and
  Principle VI (kernel-dispatch/SPIR-V confirmation before trusting a coopmat
  measurement) -- no number is reported without both.
- **FR-006**: If M5 EVT1 hardware access is unavailable during this feature's
  work, the feature MUST still complete User Story 1 (commit with accurate,
  honestly-labeled not-yet-validated status) and explicitly report User
  Stories 2 and 3 as blocked-on-hardware-access, rather than reporting an
  assumed or extrapolated result.
- **FR-007**: The final disposition of each of the three code changes (keep
  as-is, keep with caveats, or revert) MUST be recorded independently, so a
  reader can determine each change's fate without needing the other two's
  outcome.

### Key Entities

- **Retuned Shader Change**: One of the three code changes to
  `linear_qw_coopmat.glsl` (fp16 accumulate / loop flattening / vectorized
  dequant), each with its own risk level, validation method, and final
  disposition.
- **Documentation Clarification**: The comment-only addition to
  `linear_dq8ca_qw_coopmat.glsl` and `QuantizedLinear.cpp` recording the
  2026-06-30 A/B finding; has no validation gate of its own.
- **Validation Result**: The correctness and/or performance outcome for one
  Retuned Shader Change on M5 EVT1, including kernel-dispatch confirmation
  status and, where applicable, an explicit revert decision.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All four changes currently sitting uncommitted in the working
  tree are committed to the branch, each attributable to its own described
  change -- none remain as unexplained or at-risk-of-loss working-tree state.
- **SC-002**: A reader of this feature's results can determine, for each of
  the three shader changes, whether it is hardware-validated, still pending
  validation, or reverted -- without needing to re-read the raw diff.
- **SC-003**: No performance or correctness claim in this feature's results
  is made without a kernel-dispatch-confirmed, tool-verified measurement on
  M5 EVT1 backing it; claims blocked on hardware access are labeled as such,
  never presented as measured.
- **SC-004**: If the fp16-accumulate change fails correctness, the shipped
  shader reflects the revert (fp32 accumulate) -- the committed code state
  never carries a known-incorrect experiment forward silently.

## Assumptions

- "Already-written, uncommitted code changes on this branch" refers
  specifically to the working-tree diffs against `01fb136d6` (this branch's
  `HEAD` at the time this spec was written, immediately after the
  constitution amendment retroactively committed in the same session) in
  `backends/vulkan/runtime/graph/ops/glsl/linear_qw_coopmat.glsl`,
  `backends/vulkan/runtime/graph/ops/glsl/linear_dq8ca_qw_coopmat.glsl`, and
  `backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.cpp` -- not any
  other uncommitted state elsewhere in the repo (e.g. the unrelated
  `backends/cadence/utils/FACTO` submodule dirty state, which predates and is
  unrelated to this workstream and is explicitly out of scope for this
  feature).
- The existing INT4 coopmat correctness check (used by prior specs
  `007`/`008`/`010` for their own kernel-family correctness gating) is the
  correctness bar for User Stories 2 and 3; authoring a new, dedicated
  correctness test at the exact production K/N shapes is not required unless
  the existing check is found insufficient during this feature's work.
- Validation runs on Samsung M5 EVT1 (Exynos 2500 / Xclipse 970), this
  workstream's sole active target per the constitution's Principle II -- not
  `rocky-ryzen` MiniPC, which is retired to archived/historical reference
  only (per the same constitution amendment committed alongside this spec).
- This feature's scope is limited to the `4w` (weight-only int4) coopmat
  shader (`linear_qw_coopmat.glsl` / `linear_q4gsw_coopmat`); it does not
  extend the fp16-accumulate or loop-flattening experiments to the `8da4w`
  int8-activation shader (`linear_dq8ca_qw_coopmat.glsl`), which only
  receives the documentation-only change in this feature.
- "Loop-shape flattening... verified as the winning loop variant in the
  earlier `007`-`012` MiniPC sweep" refers to the dbuf1 variant's algorithm
  (already the shipped choice); this feature does not re-run that sweep, only
  re-validates this specific flattened code-shape's correctness and
  performance on the new target hardware.
