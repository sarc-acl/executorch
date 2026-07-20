# Phase 0 Research: 4w Tile/Subgroup Sweep Ranked by End-to-End Throughput

No `NEEDS CLARIFICATION` items remain from the plan's Technical Context. This feature
reuses `022`'s existing correctness-verified microbenchmark data and `027`'s already-
validated e2e-ranking methodology directly; the one genuinely open question (how the
tsweep infrastructure gets from `022`'s retired worktree onto `dev`'s current base) was
resolved by direct investigation of the repo state, documented as Decision 0 below.

## Decision 0: The `4w` tsweep infra must be ported onto `dev`, not reused as-is — it was never committed

**Decision**: Treat the archived patch at
`.archived-artifacts/tmp-origcm-2026-07-08/untracked-new-files/backends/vulkan/runtime/graph/ops/glsl/linear_q4gsw_coopmat_tsweep.{glsl,yaml}`
as **reference material for a manual port**, not as something to `git apply` or copy in
directly. Verified by direct repo inspection:

- `git cat-file -e` for `linear_q4gsw_coopmat_tsweep.glsl` returns nothing on every branch
  reachable from any current worktree (`quant-perf-optimization`, `yanwen/quant-dev`,
  `yanwen/quant-dev-active`, `dev`) — the file was never committed anywhere. `022`'s own
  `results/` directory only has JSON/Markdown output, not the shader source itself.
- The archive's own `README.md` states the patch's base is commit `1da18955a`
  ("`[ET-VK] dq8ca_q4gsw coopmat: ColumnMajor B slabs + group-invariant epilog hoist`").
  `git merge-base --is-ancestor 1da18955a HEAD` (on `dev`) returns false — `dev` did not
  build on top of that commit's history.
- `git diff --stat 1da18955a dev -- .../QuantizedLinear.cpp .../linear_q4gsw_coopmat.glsl`
  shows the old `linear_q4gsw_coopmat.glsl` (317 lines) was deleted/renamed entirely, and
  `QuantizedLinear.cpp` has 245 changed lines between the two points — substantial
  divergence, not a clean fast-forward.
- `dev`'s current fp16 `4w` shader is `linear_q4gsw_coop.glsl` (not `..._coopmat.glsl`),
  dispatched via a **fixed** 128×64/K16/2×2/s32 configuration in `QuantizedLinear.cpp`
  with no env-var variant-selection mechanism at all (`grep -n
  "ET_VK_Q4GSW_COOPMAT_VARIANT"` on `dev` matches nothing outside a comment referencing
  the mechanism by name, not defining it).

**Rationale**: This is `4w`'s direct analogue of the `8da4w` finding
`ACTIVE-STATUS.md` already recorded for `specs/027` ("`dev` never had the `dbuf2`
loop-structure port from `specs/023` before this — applying the tile winner required
porting the loop structure too") — except here the *entire* tile-sweep dispatch
mechanism is missing from `dev`, not just one loop-structure variant. Attempting a
literal patch apply would fail outright (the target file it modifies no longer exists in
that form) and, even if it partially applied, would silently discard `dev`'s
subsequently-landed WMMA/SDPA-coopmat stack changes to the same dispatch path — exactly
the kind of full-stack regression spec FR-011 exists to prevent.

**Port scope** (bounded, mirroring `ET_VK_DQ8CA_COOPMAT_VARIANT`'s existing pattern in
the same file):
1. Add a new `linear_q4gsw_coopmat_tsweep.{glsl,yaml}` shader, parameterizing tile size
   (`wg_tile_m/n/k`), subgroup grid (`sg_grid_x/y`), and subgroup size — using `dev`'s
   current `linear_q4gsw_coop.glsl` as the structural base (so the port inherits any
   WMMA/driver-workaround logic already present there — Constitution Principle V check)
   and the archived reference patch only for the parameterization pattern itself.
2. Add an `ET_VK_Q4GSW_COOPMAT_VARIANT` env-var dispatch token to `QuantizedLinear.cpp`,
   copying `ET_VK_DQ8CA_COOPMAT_VARIANT`'s existing structure verbatim (same file, same
   token-parsing shape) — unset/empty/unrecognized value falls back to today's fixed
   dispatch, unchanged (spec Assumptions: no production dispatch-gating logic is
   modified).
3. Re-run `022`'s existing fp32-reference correctness check
   (`COOPMAT_BENCH_CORRECTNESS_ONLY=1`) against every one of the 8 shortlisted tokens
   through the ported shader, before trusting any of `022`'s old GFLOP/s numbers as
   still representative of the *ported* shader's behavior (plan.md Technical Context,
   Testing) — a port that silently changed semantics must be caught here, not at e2e.

**Alternatives considered**:
- *Reconstruct `1da18955a` in a detached worktree and cherry-pick forward*: rejected —
  the archive's own recovery instructions already describe this path, but the resulting
  shader would carry `022`'s old base (pre-WMMA/SDPA-coopmat-stack), which is exactly the
  isolated-kernel measurement spec FR-011 forbids. The whole point of this feature is a
  full-stack e2e number.
- *Skip the port and just re-measure `022`'s existing winner as reported (no new e2e
  candidates beyond the current default)*: rejected — this would only produce one data
  point (the current fixed dispatch vs. itself), collapsing User Story 1 to nothing to
  compare against; the entire value of this feature is ranking multiple candidates by
  e2e, which requires the variant-dispatch mechanism to exist.

## Decision 1: Use `022`'s Round 2 results as the pre-filter directly — already ≤8 candidates, no further trimming

**Decision**: The shortlist is the 8 candidates in `022`'s `results/round2_results.json`,
all of which have `correctness_status: "pass"` and a measured `mean_gflops` (or
`gflops`) value, ranked by that score. `022`'s `results/round3_results.json` additionally
confirms the top-ranked one (`tsweep_t128x64k16g14s32`, 2518.77 GFLOP/s, 3-run mean) —
that confirmation is preserved as metadata (`microbenchmark_confirmed: true`) but does not
change the shortlist size or membership.

**Rationale**: `027`'s Clarifications fixed its shortlist at "top 8 by combined
microbenchmark rank" as a deliberate device-time bound; `022`'s Round 2 already is
exactly 8 correctness-passing, GFLOP/s-scored candidates — there is no larger pool to
trim from without re-opening `022`'s own earlier-round exploratory data (Round 1, which
includes zero-device-time analytical scores for the full 642-candidate space, not real
measurements). Using Round 2 as-is keeps this feature's pre-filter step a pure read, with
no new ranking judgment calls introduced.

**Alternatives considered**:
- *Include `022`'s Round 1 analytical-score candidates too, to reach a larger pool before
  cutting to 8*: rejected — Round 1 scores are zero-device-time proxies (occupancy/
  register-pressure heuristics), not measured GFLOP/s; mixing measured and unmeasured
  scores into one rank would violate spec FR-002's own definition of "microbenchmark
  score" as something already measured, not estimated.

## Decision 2: 8B is the shape-matched model for the initial search; 1B/3B are a confirmation-only pass on the final answer

**Decision**: User Stories 1-3 (the actual e2e-ranked search) run exclusively on the
Llama 3.1 8B `4w` buffer PTE (`llama3_1_8b_4w_buffer_ctx3072.pte`), since `022`'s
`test_coopmat_linear_bench` GFLOP/s scores are keyed to this workstream's standard 8B-
shaped representative shapes (`K=4096/14336`, the same `wq`+`w1_gate` convention `027`
already confirmed for `8da4w`). Once User Story 3 produces one definitive 8B answer
(spec FR-008), User Story 4 (Clarifications 2026-07-12) measures that exact same config
end-to-end on `llama3_2_1b_4w_buffer_ctx3072.pte` and `llama3_2_3b_4w_buffer_ctx3072.pte`
— both already staged on NFS, no new export needed — as a confirmation pass using the
same screen→confirm statistical bar (Decision 3), not an independent search over 1B/3B
shapes.

**Rationale**: Running the full staged search independently three times (once per model
size) would triple device time on a shared board for a tile configuration that is
primarily driven by the K/N weight-matrix dimensions, which `022`'s scores already
represent at the 8B scale; `027` itself flagged exactly this generalization gap as
non-blocking future work rather than doing it inline. The user's own "(smartly)"-style
instruction from `027`'s precedent, and this feature's explicit Clarifications answer,
both point at "validate the answer holds broadly" rather than "search three times."

**Alternatives considered**:
- *Full independent staged search on 1B and 3B too*: rejected per Clarifications
  (2026-07-12) — user explicitly chose the confirmation-only approach (Option B) over
  running three parallel searches (Option A).
- *Skip 1B/3B entirely, 8B-only like `027`*: rejected — the user's explicit "also try on
  1b and 3b" instruction is a direct requirement, not optional scope; `027`'s own
  unresolved follow-up note is exactly the gap this closes.

## Decision 3: Reuse `027`'s adaptive statistical bar and escalation formula unchanged

**Decision**: `escalate_to_confirm(candidate) = screen_ratio(candidate) >= -0.10`, where
`screen_ratio = (screen_prefill_tok_s - baseline_screen_prefill_tok_s) /
baseline_screen_prefill_tok_s`; confirmed candidates get a 3-run mean compared against
the baseline's own 3-run mean (never a single baseline data point). Applied identically
for the 8B search and the 1B/3B confirmation pass (User Story 4) — the same statistical
bar decides whether a 1B/3B result needs its own 3-run confirmation or is a clear
win/loss on one screening run.

**Rationale**: This bar is already validated on this exact hardware/methodology by
`027`; there is no reason specific to `4w` to redefine it, and Clarifications (2026-07-12)
did not raise any objection to reusing it for the 1B/3B pass.

**Alternatives considered**: none — directly reused per the user's "just like how last
spec was did" instruction from `/speckit-specify`.

## Decision 4: Execution happens on a new worktree cut from `dev`, not a resurrected `022` worktree

**Decision**: Create a new worktree/branch (`4w-e2e-tile-sweep` / `028-4w-e2e-tile-sweep`,
cut from `yanwen/dev-1.3`) for all shader-port, build, and measurement work, rather than
reconstructing `022`'s old retired worktree or building inside `dev/` directly.

**Rationale**: `022`'s own execution worktree was removed 2026-07-11 along with the
broader `quant-perf-optimization`/`quant-dev` cleanup, and reconstructing it would put
this feature on `022`'s old pre-WMMA/SDPA-coopmat base — the opposite of spec FR-011's
requirement to measure against the current full-stack baseline. `dev/` itself is the
active-development branch that feature work PRs into, not a place to commit directly
(workspace `CLAUDE.md` Development Workflow / Critical Rules) — a dedicated feature
worktree is this workstream's standing pattern for exactly this situation (e.g.
`dbuf-int8-sweep/` for `specs/023`).

**Alternatives considered**:
- *Resurrect the old `quant-dev`/`quant-perf-optimization` worktree*: rejected — its
  branch predates `dev`'s WMMA coopmat SDPA and node-threshold-workaround work; any
  measurement there would be the isolated-kernel-on-old-base number spec FR-011 forbids.
- *Build directly in `dev/`*: rejected — violates the standing rule that new work is a
  feature branch PR'd into `dev`, never committed there directly.
