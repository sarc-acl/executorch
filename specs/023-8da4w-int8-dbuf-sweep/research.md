# Phase 0 Research: 8da4w Int8 WMMA Double-Buffer Variant Sweep

## Decision 1: Reuse the existing fp16 dbuf1-4 harness pattern, not the specs/008 pattern

**Decision**: Port the dbuf1-4 loop structures onto the int8 shader using the same pattern
already built (uncommitted) in the sibling `.tmp-origcm` worktree for the **fp16** `4w`
shader: four separate, production-registered `.glsl`/`.yaml` shader files
(`linear_q4gsw_coopmat_dbuf{1,2,3,4}`), selected via an opt-in env var
(`ET_VK_Q4GSW_COOPMAT_VARIANT`) read inside `QuantizedLinear.cpp`'s existing kernel-name
selection logic, timed via `test_coopmat_linear_bench.cpp`. This is the pattern to
replicate for `linear_dq8ca_q4gsw_coopmat` (int8).

**Rationale**: constitution Development Workflow states explicitly: "Before building new
loop-structure variants or a tile-geometry sweep harness for this workstream's shaders,
check the workspace's `quant-dev` worktree first: it already has a dbuf1-4 double-buffer
variant harness... Port/reuse that tooling rather than re-deriving it independently on
Samsung." Inspecting that worktree (and its `.tmp-origcm` sibling, where the actual
uncommitted dbuf ports live per the workspace root `CLAUDE.md`) confirms this exact
harness already exists, is already proven on M5 EVT1 (it produced the fp16
`dbuf-sweep-q4gsw-m2048.md` report this spec's Context cites), and benchmarks through a
real, gated production dispatch path rather than a fully test-only op.

**Alternatives considered**: `specs/008-8da4w-parameter-sweep`'s test-only-shader-copy +
test-only-op pattern (`test/custom_ops/glsl/dq8ca_q4gsw_coopmat_sweep.glsl` +
`TestDq8caTileSweep.cpp`). Rejected as the *primary* mechanism because it duplicates
infrastructure the constitution says to check for and reuse first, and because it was built
for a different axis of variation (tile shape / subgroup size via spec constants on one
shader file), not for loop-structure variants that need to be genuinely separate compiled
shaders. Its process-isolation lesson (Decision 2) is still adopted.

## Decision 2: Process isolation per variant, not per shape

**Decision**: Each dbuf variant is measured in its own process invocation (one env var
value per process); all 6 representative shapes for a given variant are timed within that
same process. Four total invocations, not 24.

**Rationale**: `specs/008`'s own precedent (`test_dq8ca_tile_sweep.cpp`) explicitly isolates
at the *config* (variant) level, not the shape level — one `DQ8CA_SWEEP_CONFIG_ID` per
process, all shapes-in-scope run inside that one process. Its stated reason still applies
here: an Xclipse PAL pipeline-creation crash cannot be caught by in-process `try`/`catch`
(the harness's framework only catches `vkapi::ShaderNotSupportedError`), so isolating at
least at the variant boundary guarantees one bad variant can't erase the other three's
results. A driving shell script invokes the bench binary once per variant and records an
explicit `pipeline_crash` (or equivalent) result for any invocation that doesn't exit 0,
matching spec FR-004's "no silent omission" requirement.

**Alternatives considered**: isolating per (variant, shape) pair (24 invocations) —
rejected as unnecessary extra device time; no prior evidence in this codebase that a crash
risk varies *by shape* independently of variant (the documented Xclipse PAL failure mode is
tied to loop *structure*, not to a specific shape's spec-constant values). Running all four
variants in one process — rejected outright, defeats the isolation purpose.

## Decision 3: New env var name, scoped separately from the fp16 sweep's

**Decision**: Introduce `ET_VK_DQ8CA_COOPMAT_VARIANT` for this feature, rather than reusing
`ET_VK_Q4GSW_COOPMAT_VARIANT`.

**Rationale**: code inspection of `QuantizedLinear.cpp` shows `linear_q4gsw_coopmat` (fp16,
`4w`) and `linear_dq8ca_q4gsw_coopmat` (int8, `8da4w`) share one eligibility function
(`can_use_q4gsw_coopmat`) but resolve to different `kernel_name` values. A single shared env
var would be ambiguous about which op family's dispatch it's overriding, and would block
running both sweeps' variants independently (or side by side) in the same session.

**Alternatives considered**: reusing `ET_VK_Q4GSW_COOPMAT_VARIANT` for both op families —
rejected for the ambiguity above.

## Decision 4: Loop-structure adaptation is a genuine re-derivation, not a mechanical port

**Decision**: Each of the four reference loop structures
(`shmem_double_buf{,2,3,4}.comp`) must be independently re-derived against the int8
shader's own existing structure — nested `groups x chunks` loop (not the fp16 shader's flat
`K` loop), K-slab-split + ColumnMajor-B LDS layout with per-column skew, and a **second**
ping-pong pair for per-group weight sums/scales — rather than copy-pasted from the fp16
`linear_q4gsw_coopmat_dbuf{1..4}` ports, which have none of that extra structure.

**Rationale**: the shipped int8 shader's own header already documents hard,
already-hit Xclipse PAL compiler constraints specific to this nested form (loop trip count
must come from a spec constant, not a UBO-derived bound; the loop must stay nested with an
*unconditional* group epilog — "flattening it with a conditional coopmat epilog crashes the
Xclipse PAL compiler at large spec-resolved trip counts"). Only the current dbuf4 structure
is known today to satisfy these constraints in this nested form; dbuf1/2/3 each need their
own pass at satisfying them (or documenting why they can't, per Edge Cases). Any new
workaround discovered in the process must carry an inline comment per constitution
Principle V, exactly as the existing shader's own header does.

**Alternatives considered**: none — this is a description of necessary work, not a
choice between options.

## Decision 5: Correctness check reuse

**Decision**: Reuse the existing correctness test suite that already validates
`linear_dq8ca_q4gsw_coopmat` (the per-op `test_*_linear` / `op_tests` correctness check at
small, tile-aligned shapes) to validate each of the four variants, rather than writing a
new correctness harness.

**Rationale**: matches constitution Principle I, and matches the existing fp16 dbuf1-4
precedent, whose own bench file states outright that "no CPU reference is run (correctness
is covered by the per-op `test_*_linear` [suite])" rather than re-verifying inline in the
perf bench.

## Decision 6: Dispatch and SPIR-V verification

**Decision**: Confirm each variant's dispatch via the bench harness's own kernel-name
logging (the same mechanism the existing fp16 sweep's bench binary already uses), as the
primary Principle VI check for this feature (a standalone microbench binary, not a full
model graph, so no ETDump trace is applicable here — Principle VI's ETDump requirement
applies to Tier-2/model-level studies specifically). Additionally, disassemble each
variant's compiled SPIR-V once (`spirv-dis`/`spirv-cross` or equivalent) to confirm genuine
int8 cooperative-matrix instructions (`OpCooperativeMatrixMulAddKHR` operating on 8-bit
component types) are present in the generated binary, satisfying Principle VI's
shader-change requirement.

**Alternatives considered**: relying on the eligibility gate (`can_use_q4gsw_coopmat`)
passing as sufficient evidence of correct dispatch — explicitly rejected by Principle VI
itself ("An eligibility check... passing in code is not sufficient evidence").

## Decision 7: New worktree and branch

**Decision**: Create a new git worktree via `git worktree add`, branched from the tip of
`quant-perf-optimization` (not `main`), after this feature's `spec.md`/`plan.md`/`tasks.md`
are committed to `quant-perf-optimization` so the new worktree's checkout includes them.
Bootstrap the new worktree per constitution "Environment & Build Bootstrap" (`uv venv .venv
--seed`, `source .venv/bin/activate`, `./install_executorch.sh --minimal`) before any build.

**Rationale**: branching from `quant-perf-optimization`'s tip (rather than `main`) inherits
this workstream's full spec history and constitution; a *worktree* (rather than continuing
in-place) satisfies the user's explicit instruction and gives a working tree with none of
the current worktree's unrelated uncommitted changes (specs 015/018-022 and others per this
session's git status), avoiding any risk of this feature's commits accidentally bundling
unrelated in-flight work. Matches this workspace's existing convention (workspace-root
`CLAUDE.md`) of one worktree per active line of work.

**Alternatives considered**: continuing in the current, already-dirty
`quant-perf-optimization` worktree — rejected per the user's explicit "on a new worktree"
instruction and to avoid entangling this feature's commits with unrelated uncommitted work
already sitting in this tree.

## Decision 8: Shape/model coverage and tier scope (recorded from spec Clarifications)

**Decision**: 6 shapes total (`wq` + `w1_gate` for each of 1B/3B/8B), Tier-1
(microbenchmark) only, no e2e validation required. Already resolved in `spec.md`'s
Clarifications section during `/speckit-clarify`; recorded here only for completeness so
`research.md` alone documents every open question this feature had.

**Rationale**: see spec.md Clarifications and Assumptions.
