# Research: WMMA-Optimizable Shader Candidates Report

All unknowns resolved via direct source inspection (`QuantizedLinear.cpp`,
`SDPA.cpp`) and `002`'s already-captured raw ETDump event data — see
file:line citations below. There are no remaining `NEEDS CLARIFICATION`
markers.

## Decision 1: Reuse `002`'s aggregated data directly, no new profiling

**Decision**: Read `specs/002-etdump-shader-profiling/results/raw/<model>_<scheme>.json`'s
`phases.{prefill,decode}.aggregated[]` entries (kernel name, shape, total
time, category) as the sole input. No new `.etdump` capture, no device
access.

**Rationale**: `002` already produced exactly the per-shader, per-config
breakdown this feature classifies — re-capturing would violate the "no new
profiling" assumption in spec.md and add risk (device contention, thermal
variance) for zero benefit.

**Alternatives considered**: none — re-profiling was never a serious option
given `002`'s data already has everything needed (kernel name, shape, time,
category, per-config attribution).

## Decision 2: Classification rule set, grounded in the actual dispatch code

**Decision**: Classify each aggregated entry by matching its `category`
(from `002`) and `kernel_name` pattern against the following rules, each
grounded in a specific source citation:

| Condition | Classification | Cited reason |
|---|---|---|
| `category` is "non-shader overhead" (shape=`null`) | (d) not applicable | Not a matrix multiplication |
| `category` ∈ {attention projection, feed-forward, output/vocab projection} and kernel name matches prefill markers (`gemm`/`_tiled`) | (b) exists, blocked | See "Two blockers" below |
| Same categories, kernel name matches decode markers (`gemv`/`_coop`) | (c) no implementation | See "Decode GEMV" below |
| `category` is "attention (sdpa)" (either phase) | (c) no implementation | See "SDPA" below |
| anything not matching the above | uncertain | Recorded per FR-008, not guessed |

**Two blockers for the prefill linear family (attention projection,
feed-forward, output/vocab projection)** — corrected during this research
phase after reading the code in order, not just spot-checking one line:

`can_use_q4gsw_coopmat()` (`QuantizedLinear.cpp:159-216`, shared by both the
`4w` and `8da4w` dispatch functions) checks, in this order: bias absence →
`adapter->supports_cooperative_matrix()` → `subgroup_size() == 64` →
`graph->dim_of(output) > 2` (line 192-194, "batched (rank > 2) outputs
would silently miscompute") → `storage_type_of(output) == kBuffer`
(line 196-197) → dtype/tile-alignment. The real exported model's linear
activations are **rank-3** (`sizes=[1, M, K]`, confirmed directly from raw
ETDump event args in `002`'s captures — the leading `1` is a batch dim, not
squeezed away) *and* use **`TEXTURE_3D`** storage (also confirmed from the
same raw event data), not `Buffer`. Either the rank-3 check or the storage
check alone already returns `false` — they are two **independent**
blockers, not one, and the rank-3 check is hit first in execution order.
Both apply regardless of `001`'s `ET_VK_FORCE_TILED_LINEAR` toggle (which is
itself checked even earlier, at line 175-176, and is this workstream's own
deliberate baseline mechanism, not an inherent model property). Tile
alignment (M/N%64==0, K%32==0) is *not* a blocker for these shapes — all
observed prefill shapes already satisfy it, confirmed against `001`'s
`results/shapes.json`.

**Decode GEMV (M=1) linears**: `pick_linear_qw_shader`/`pick_linear_dqa_qw_shader`
branch on `is_gemv_case` (`is_gemv(graph, fp_input)`, defined in
`QuantizeDequantize.cpp` as `size_at(-2, fp_input) == 1`) *before*
`can_use_q4gsw_coopmat()` is ever called — confirmed in `002`'s own
research.md. There is no coopmat-eligibility check to block here at all;
there is simply no WMMA-capable GEMV (M=1) kernel in this codebase — the
existing coopmat shaders (`linear_qw_coopmat.glsl`/`linear_dq8ca_qw_coopmat.glsl`)
are tiled, multi-row (128×64 / 128×64 workgroup tile) designs, not applicable
at M=1 even in principle. Classified as (c), not (b): there is no existing
implementation sitting blocked by a fixable condition — a GEMV-shaped
WMMA path would be new work, not a configuration fix.

**SDPA (both phases)**: `grep -n "coopmat" backends/vulkan/runtime/graph/ops/impl/SDPA.cpp`
returns zero matches. `SDPA.cpp:192,301` (`add_sdpa_compute_attn_weights_node`,
`add_sdpa_compute_out_node`) only ever append `_tiled` or `_coop` to the
shader name — there is no third, coopmat-aware branch. `add_matmul_coopmat_node`
exists generically (`GemmCoopmat.h`/`.cpp`, wired into `Matmul.cpp`) but is
not called anywhere in `SDPA.cpp`. Classified as (c) for both prefill and
decode.

**Alternatives considered**: classifying purely from `002`'s data (kernel
name/shape patterns) without reading the dispatch source — rejected; this is
exactly how the single-blocker version of this research would have shipped
if not double-checked against the code in execution order. FR-002/FR-003
require a specific, correct reason, which requires reading the actual gate.

## Decision 3: Grouping and ranking for the consolidated report

**Decision**: Retain fine-grained Shader Classification rows at `002`'s own
granularity (one per config × phase × kernel+shape), but roll them up into a
small number of **Optimization Candidate groups** by shared root cause for
the ranked report (US3 / FR-006):

1. "Prefill linear GEMM (attention projection + feed-forward + output
   projection), blocked by rank-3 output + `TEXTURE_3D` storage" — (b)
2. "Decode linear GEMV (attention projection + feed-forward + output
   projection), no WMMA-capable GEMV kernel exists" — (c)
3. "SDPA prefill (tiled), no WMMA implementation exists" — (c)
4. "SDPA decode (coop), no WMMA implementation exists" — (c)

Each group's absolute time is the sum of `total_time_us` across every
member row (all matching configs/shapes/phases); relative percentage is
shown per-config alongside, not summed (percentages of different phase
totals aren't meaningfully additive) — per the Clarifications session.

**Rationale**: matches how `001`→`002` already moved from raw data to
aggregated categories to a rolled-up report; grouping by root cause is what
makes the ranked report actionable ("fix this once, it unlocks all of
these") rather than a flat list of ~100+ near-duplicate rows.

**Alternatives considered**: ranking every fine-grained row individually
without grouping — rejected; FR-006 explicitly asks for a "single
consolidated report," and 100+ ungrouped rows (most sharing an identical
root cause) would not let a reader find the highest-impact opportunity
without doing the grouping mentally themselves.
