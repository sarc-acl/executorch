# Research: M5 EVT1 `4w` Linear Coopmat Retune

## Decision 1: The performance baseline for US2/US3 is a fresh pre-change build on M5 EVT1, not the sibling `quant-dev` branch's existing numbers

**Decision**: Before validating any of the three shader changes, build and
measure the *pre-change* `linear_qw_coopmat.glsl` (this branch's committed
`HEAD`, i.e. `01fb136d6`, before the four working-tree edits are applied) on
M5 EVT1, using this workstream's own tier-1 harness. This is the "last
known-good coopmat baseline" User Story 2/3 compares against -- not the
`quant-dev`/`quant-dev-active` worktree's already-published M5 EVT1 dbuf1
numbers (`.shared-context/report-for-human/RESULTS-SUMMARY.md`).

**Grounding**: The workspace-root `CLAUDE.md` documents `quant-dev` and
`quant-perf-optimization` as separate worktrees/branches with independently
evolved shader code (`quant-dev`'s `linear_q4gsw_coopmat` carries its own
dbuf1-4 variant history and UBO/spec-const decisions, per its own commit
`83dbe9b90`+). This workstream's `linear_qw_coopmat.glsl` -- already
restructured across specs `007`/`008`/`013` -- is not byte-for-byte the same
shader `quant-dev`'s numbers were measured against, even though both trace
back to a common ancestor.

**Rationale**: Constitution Principle VI ("Verify With Tools, Never Assume")
and Principle IV (every performance claim needs its own tool-driven
measurement) both argue against treating a different branch's numbers as
this workstream's own baseline without re-verification. Comparing this
feature's post-change numbers against a same-repo, same-commit-lineage
pre-change build isolates exactly the three changes under test; comparing
against `quant-dev`'s numbers would additionally conflate every divergence
between the two branches' shader histories.

**Alternatives considered**: Citing `quant-dev`'s existing dbuf1 M5 EVT1
numbers directly (rejected as the primary baseline -- cross-branch, unverified
for this exact shader; may still be cited informally in the results as
directional context, clearly labeled as such, never as the pass/fail bar).

## Decision 2 (REVISED 2026-07-05 clarification session): Correctness gate is the existing coopmat correctness harness, EXTENDED to production K -- the gap is confirmed, not hypothetical

**Original decision (superseded)**: This decision originally proposed
gating User Stories 2 and 3 on the generic `backends/vulkan/test/op_tests`
`test_*_linear` correctness check, reused unmodified, with a new test only
"if that gap is found during implementation." That framing was wrong on two
counts, both caught during `/speckit-clarify`: (1) the actually-relevant
coopmat-specific correctness harness is
`backends/vulkan/test/custom_ops/test_coopmat_linear_bench.cpp`'s
`kCorrectnessShapes`/`kRank3CorrectnessShapes` (deterministic,
well-conditioned/positive-only data, `abs=0.5`/`rel=0.05` tolerance) -- not
the generic `op_tests` directory, which has no coopmat-specific tolerance
handling; (2) direct inspection of that harness (not deferred to
"implementation time") shows its existing shapes stop at K=256, well short
of FR-003/FR-004's production-K (2048/4096+) requirement. The gap is
confirmed today, not a hypothetical to check later.

**Revised decision**: User Stories 2 and 3 gate on
`test_coopmat_linear_bench.cpp`'s correctness harness, EXTENDED (spec
FR-008) with new `kCorrectnessShapes`/`kRank3CorrectnessShapes` entries at
production K (2048/4096 at minimum), reusing its existing deterministic,
well-conditioned-data generation and `abs=0.5`/`rel=0.05` tolerance
unchanged -- only the shape list grows, not the methodology.

**Grounding**: `test_coopmat_linear_bench.cpp`'s own in-code comment
explains why a *different* correctness strategy (well-conditioned positive
data, not the generic random-data suite) was needed for coopmat shapes at
all: `test_q4gsw_linear.cpp`'s random-data + sqrt(K)-scaled-tolerance
approach was tried and explicitly rejected for coopmat-eligible shapes
because fp16 accumulation drift exceeded any reasonable tolerance there.
Reusing that already-solved strategy at larger K (rather than reinventing a
third approach) avoids repeating that already-documented failure.

**Rationale**: Confirmed cheap during the clarification session: the
correctness cases in this harness are single-shot small-shape dispatches
(64+ already exist at K<=256); the `M=1024` perf sweep elsewhere in the same
file is what actually dominates the harness's runtime, and is untouched by
this extension. Extending shape coverage is a small, bounded addition, not
a new test suite and not a meaningful slowdown.

**Alternatives considered**: (a) Accepting K<=256 as sufficient and dropping
the production-K requirement from FR-003/FR-004 -- rejected, since it would
mean shipping a claim ("validated on M5 EVT1") not actually backed by a
production-shape correctness check, contrary to constitution Principle I.
(b) Reusing `test_q4gsw_linear.cpp`'s random-data/scaled-tolerance approach
at production K instead of extending the well-conditioned harness --
rejected, since that file's own comment already documents this exact
approach failing for coopmat-eligible shapes at any size, not just large K.

## Decision 3: Independent disposition per change, enforced by keeping the three shader changes separately revertible in the working tree until each is validated

**Decision**: Do not squash the three `linear_qw_coopmat.glsl` changes into
a single commit before validation. Commit User Story 1's diff as a single
commit (since all three are already inseparably interleaved in the current
working-tree diff and none has been invalidated yet), but track each
change's disposition independently in `results/disposition-summary.md`; if
User Story 3 (fp16 accumulate) fails correctness, make a **new, separate**
commit that reverts only that hunk (the accumulator type + the two
`coopmat<float16_t, ...>` accumulator-init/store sites), leaving the loop
flattening and vectorized dequant intact.

**Grounding**: The three changes are physically interleaved in the same
functions in the current diff (e.g. the flattened loop body also contains
the vectorized `dequant_block` calls), so a byte-for-byte split into three
separate initial commits would require non-trivial reconstruction with no
actual benefit if all three end up validated. A targeted revert commit if
and only if US3 fails is simpler and just as attributable.

**Rationale**: Matches spec FR-001/FR-007 (attributable, independently
disposed of) without inventing artificial commit surgery for a
still-hypothetical failure case.

**Alternatives considered**: Three separate initial commits, one per change
(rejected -- the changes are interleaved in the same loop body/functions in
a way that would make an artificial split confusing to review, for no
correctness/attribution benefit over documenting the split in
`disposition-summary.md` and only physically reverting if actually needed).

## Decision 4: SPIR-V inspection focuses on accumulator component type and coopmat instruction count/shape

**Decision**: For each of the three shader changes, disassemble the
compiled `.spv` (`spirv-dis`) and confirm: (a) `OpCooperativeMatrix*KHR`
instructions are still present (Principle VI baseline check, all three
changes), and (b) specifically for the fp16-accumulate change, that the
accumulator-type coopmat declarations now reference a 16-bit float
component type rather than 32-bit, confirming the source-level type change
actually took effect in the compiled binary rather than being silently
promoted back to fp32 by the compiler.

**Grounding**: Principle VI's existing precedent (`007`'s research.md
Decision 4-equivalent) already establishes `spirv-dis`-based instruction
presence checking as this workstream's standard tool-verification method
for shader changes.

**Rationale**: A source-level `coopmat<float16_t, ...>` declaration is not
proof the driver actually compiles a distinct fp16-accumulate coopmat
configuration end to end -- exactly the "shader that looks right in GLSL
source is not evidence of what the driver actually compiled" caution
Principle VI itself states.

**Alternatives considered**: Trusting the GLSL source type declaration alone
(rejected -- explicitly the failure mode Principle VI exists to prevent).
