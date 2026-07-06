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

## Decision 2: Correctness gate reuses the existing INT4 coopmat correctness check; no new test is authored

**Decision**: User Stories 2 and 3 both gate on
`backends/vulkan/test/op_tests`'s existing per-op `test_*_linear`
correctness check for the `4w` (`q4gsw`) kernel family, run at real
production K-dimensions (K=2048/4096) in addition to whatever synthetic
shapes it already covers.

**Grounding**: Constitution Principle I explicitly names this test
directory as the correctness bar for "no coopmat shader change is done
until it passes." Specs `007`/`008`/`010` all reused it the same way rather
than authoring new correctness tests for their own shader changes.

**Rationale**: Consistent with this workstream's own precedent; authoring a
new correctness test is unnecessary duplication unless the existing check
is found to not cover the production K-dimensions this feature specifically
cares about (K=2048/4096) -- if that gap is found during implementation, it
becomes a real task (extend the existing test's shape list), not a new
standalone test suite.

**Alternatives considered**: Writing a brand-new correctness test targeting
exactly K=2048/4096 (deferred unless the existing check's shape coverage
turns out to be insufficient -- checked, not assumed, at task time).

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
