# Research: WMMA Coopmat Improvement Microbenchmark

## Decision 1: Reuse the existing `test_llama_baseline_bench.cpp` harness unchanged; toggle WMMA dispatch purely via `ET_VK_FORCE_TILED_LINEAR`

**Decision**: No new harness, shader, or dispatch-code work is needed. The
already-committed `test_llama_baseline_bench.cpp` (from `001`/`004`) is run a
second time with `ET_VK_FORCE_TILED_LINEAR` **unset**, letting
`can_use_q4gsw_coopmat()` dispatch the real coopmat kernel naturally.

**Grounding -- investigated directly, not assumed**:
- `ET_VK_FORCE_TILED_LINEAR` is read via `std::getenv()` at dispatch-decision
  time inside `can_use_q4gsw_coopmat()` (`QuantizedLinear.cpp`); it is not
  baked into the test binary. `test_llama_baseline_bench.cpp`'s `main()`
  never calls `setenv` -- the toggle is purely an external shell-level
  concern, exactly like `001`'s and `004`'s own captures already relied on.
- Coopmat requires Buffer storage, which is already one of the harness's two
  `kStorageTypes` options (`004` added this axis).
- Tile-alignment was checked, not assumed, against every real per-model
  shape (`shapes.json`): `4w`'s tile is `(WG_TILE_M=128, WG_TILE_N=64,
  WG_TILE_K=16)`, `8da4w`'s is `(128, 64, 32)`. Prefill `M=2048` (`/128=16`).
  Every model's `dim`/`hidden_dim` (2048/3072/4096/8192/14336) and every
  `wk`/`wv` output width (512/1024) divides evenly by both 64 and 16/32. No
  candidate shape is misaligned.

**Rationale**: Zero new code to review or maintain; the exact same harness
that produced `001`/`004`'s trusted numbers produces this feature's WMMA
numbers, so any measurement-methodology bug would already have been caught
by two prior features' worth of scrutiny.

**Alternatives considered**: A new dedicated WMMA-vs-tiled harness (like
`test_coopmat_linear_bench.cpp`, which already does something similar but
only for Llama 3.1 8B shapes at a fixed `M=1024`, not the full six-config
catalog at the established prefill workload) -- rejected as unnecessary
duplication of an already-correct, already-committed harness.

## Decision 2: Tiled baseline = `004`'s existing Buffer-storage, prefill-regime rows (per spec Clarification Q2)

**Decision**: This feature's WMMA numbers are diffed against
`specs/004-linear-storage-comparison/results/raw/storage_bench_raw.log`'s
`storage=buffer`, `regime=prefill` rows -- not `001`'s Texture3D-only data.
No re-capture of the tiled baseline.

**Grounding**: `001`'s microbench CSV format has no storage column at all
(fixed Texture3D); `004` extended the format with a `storage` field and
captured both Texture3D and Buffer under the same
`ET_VK_FORCE_TILED_LINEAR=1` forcing. Since coopmat dispatch requires Buffer
storage, `004`'s Buffer rows are the only existing dataset that holds storage
type constant against this feature's new Buffer-storage WMMA capture.

**Rationale**: Avoids re-introducing the exact storage-vs-dispatch-path
confound this workstream spent `004`→`006` carefully isolating.

**Alternatives considered**: `001`'s Texture3D numbers (rejected -- wrong
storage type, confounds the comparison); re-capturing a fresh tiled baseline
for this feature (rejected -- `004`'s data is already trusted and current;
no reason to spend device time re-measuring an unchanged code path).

## Decision 3: Exclude `lm_head` from this feature's measured op set

**Decision**: Of the harness's 8 ops per model (`wq`, `wk`, `wv`, `wo`,
`w1_gate`, `w3_up`, `w2_down`, `lm_head`), this feature measures only the
first 7. `lm_head` is excluded.

**Grounding -- a real discrepancy found during planning, not a guess**:
`test_llama_baseline_bench.cpp`'s `generate_cases()` applies the same
`regime.second` (`M=2048` for "prefill") uniformly to *every* op, including
`lm_head` -- so `001`/`004`'s existing data contains a synthetic `lm_head`
prefill case at `M=2048`, `N=128256`. But `specs/003-wmma-shader-candidates`'s
classification (built from real ETDump captures of the actual exported
model, not this synthetic harness) shows `lm_head`'s vocab projection is
`M=1` in *every* capture, including the ones bucketed under the "prefill"
phase -- because the real model only ever needs the last token's logits,
regardless of prompt length. The harness's uniform `M=2048` treatment of
`lm_head` is a `001`/`004` measurement-convenience artifact with no
production analogue.

**Rationale**: Reporting a "WMMA speedup" for a shape that never occurs in
the deployed model would misrepresent the real-world question this study
exists to answer (spec Clarification Q3's "real wall-clock impact" framing)
-- and `lm_head`'s `N=128256` is large enough that including it would
dominate the time-weighted overall figure with a non-representative number.
This is consistent with, not a reinterpretation of, FR-006 (already
clarified): the exclusion criterion is "never actually runs as a
WMMA-eligible shape in the real model," which describes `lm_head` exactly as
much as it describes decode.

**Scope note**: This decision applies to `007`'s own analysis only; it does
not retroactively revise `001`/`004`'s already-published reports, which
correctly described what their harness actually measured at the time.

**Alternatives considered**:
(a) Include `lm_head`'s synthetic `M=2048` case anyway, since `001`/`004`
already did -- rejected; perpetuates a measurement artifact into a "does
WMMA help in production" answer.
(b) Re-run `lm_head` specifically at `M=1` to match real behavior --
rejected; per FR-006, *no* WMMA-capable GEMV (`M=1`) kernel exists for any
op today (`003`'s classification "c"), so an `M=1` `lm_head` case would not
be a WMMA candidate at all, identical to decode.

## Decision 4: SPIR-V inspection via `spirv-dis` against the already-compiled `.spv` artifacts

**Decision**: Correctness-confidence evidence (spec Clarification Q1, part
b) is produced by running `spirv-dis` (Vulkan SDK) against
`cmake-out-vk/vulkan_compute_shaders/linear_q4gsw_coopmat_buffer_buffer_half.spv`
and
`.../linear_dq8ca_q4gsw_coopmat_buffer_buffer_half.spv` (the Buffer/Buffer
variants, matching this study's storage choice), grepping the disassembly for
`OpCooperativeMatrixLoadKHR` / `OpCooperativeMatrixMulAddKHR` /
`OpCooperativeMatrixStoreKHR`.

**Grounding**: Both `.spv` files already exist as normal build output (no
extra build step). `spirv-dis` is present at
`~/vulkansdk/1.4.341.1/x86_64/bin/spirv-dis`. The shaders use
`GL_KHR_cooperative_matrix` (confirmed via the `#extension` directive in
`linear_qw_coopmat.glsl` and the `VK_KHR_cooperative_matrix`-gated feature
struct in `vk_api/Device.h` -- not the NV extension), so the KHR opcode names
above are the correct ones to search for. Spot-verified during planning:
`linear_q4gsw_coopmat_buffer_buffer_half.spv` disassembles to multiple
`OpCooperativeMatrixLoadKHR`/`OpCooperativeMatrixMulAddKHR` instructions.

**Rationale**: Directly satisfies constitution Principle VI's "any WMMA
shader claim is backed by tool-driven verification, not source reading
alone" using tooling already installed on this machine, with zero extra
build steps.

**Alternatives considered**: RGA (Radeon GPU Analyzer) for ISA-level
occupancy/register analysis -- checked (`which rga`, common install paths)
and confirmed **not installed** on this machine; noted as an explicit gap
rather than assumed away, per Principle VI. `spirv-cross` (also present)
could regenerate readable GLSL/HLSL instead of raw SPIR-V disassembly --
`spirv-dis`'s direct opcode output is preferred since it is unambiguous
about which instructions are actually present in the compiled binary.

## Decision 5: Kernel-dispatch verification reuses `004`'s existing "contaminated" detection pattern, mirrored

**Decision**: FR-004's dispatch check reuses the harness's existing `kernel`
CSV field (the last column of every `RESULT` line) and the same
string-matching approach `004`'s `compare_storage.py` already uses to detect
an unwanted coopmat name during the tiled run (its "contaminated" check) --
this feature performs the mirror-image check, confirming a `coopmat` kernel
name **is** present during the natural (no-env-var) run.

**Rationale**: An already-proven pattern in this exact codebase; no new
detection logic to invent or validate.

**Alternatives considered**: A new, separate dispatch-verification mechanism
-- rejected; would duplicate logic `004` already wrote and validated.

## Decision 6: Time-weighted overall figure reuses `003`'s existing `pct_of_phase` weights (per spec Clarification Q3)

**Decision**: The single overall improvement figure (FR-008) is computed as
`sum(op_speedup * op_pct_of_phase) / sum(op_pct_of_phase)` across the 42
measured (model, scheme, op) triples, using each op's `pct_of_phase` value
already present in `003`'s classification JSON for that configuration.

**Rationale**: Reuses already-computed, already-trusted weighting data
instead of re-deriving op-size weights from scratch; directly answers "how
much real wall-clock time did WMMA save," per the clarified intent.

**Alternatives considered**: geometric mean of unweighted per-op ratios
(rejected -- treats a rarely-hit op the same as `w1_gate`/`w2_down`, which
dominate runtime); `004`-style pass-count framing (rejected -- doesn't
produce the single figure FR-008 requires).

**Addendum (found during implementation, not anticipated during planning)**:
`003`'s classification JSON does not actually carry a `pct_of_phase` per
*named op* -- it's aggregated by `(kernel_name, shape)`, and several op pairs
share both an identical shape and kernel name within one model (`wq`/`wo`,
`wk`/`wv`, `w1_gate`/`w3_up` -- the same same-shape-sibling ambiguity `004`'s
own cross-check section already ran into). Their real ETDump invocations are
merged into one combined entry, so there is no clean way to attribute a
literal per-op split of `003`'s number without inventing an unstated 50/50
assumption -- which Principle VI's "never guess" bar rules out.

**Revised weight source**: instead, each op is weighted by its own share of
this feature's 7 measured ops' total tiled-baseline time (`004`'s
`tiled_mean_us`, summed per model/scheme) -- data that already exists at
exact per-named-op granularity, requires no splitting, and is equally
faithful to "time-weighted by real wall-clock impact" (Clarification Q3's
actual intent). `003`'s classification data still grounds *which* ops are
candidates and *why* (Decision 3's `lm_head` exclusion, FR-006's GEMV
exclusion) -- only the literal weighting figure's data source changed.

## Decision 7: Correctness-confidence composition (per spec Clarification Q1)

**Decision**: An operation's WMMA measurement is reportable once all three
hold: (a) its `kernel` field names a `coopmat` kernel (Decision 5), (b) that
kernel's compiled SPIR-V contains genuine cooperative-matrix instructions
(Decision 4), and (c) `test_coopmat_linear_bench.cpp`'s existing
`kCorrectnessShapes` already exercises that op's kernel family
(`linear_q4gsw` / `linear_dq8ca_q4gsw`) against a CPU/tiled reference at
small, tile-aligned shapes -- confirmed present for both op families (grep
of `kOps` in that file). No new correctness tests at the exact production
K/N shapes are authored for this feature.

**Rationale**: Satisfies constitution Principle I without the materially
larger scope of authoring new production-shape correctness tests; the
existing tests already establish kernel-level correctness, and Principle
III's eligibility gating already ensures only tile-aligned shapes ever
dispatch the kernel being tested.

**Alternatives considered**: authoring new correctness tests at exact
production shapes -- rejected per the spec's explicit clarification (Option
B was not selected).

## Decision 8: Fix the `4w` coopmat dispatch-wiring gap (found during US1, not anticipated during planning)

**Decision**: T005/T006 (US1's own proof step) found `llama-3.2-1b`/`4w`/`wq`
dispatched the tiled kernel even with `ET_VK_FORCE_TILED_LINEAR` unset. Not
shape misalignment (verified: `2048/2048/2048` satisfies `4w`'s tile
requirement) and not noise -- **every** `4w` row in the full capture showed
the same tiled dispatch, **every** `8da4w` row showed genuine coopmat. Traced
via direct code reading (constitution Principle VI), not assumed:

- `et_vk.linear_q4gsw.default` -- the op name both the real `4w` export path
  (`op_registry.py:455`) and every prototyping harness construct -- resolved
  via `VK_GET_OP_FN` to `Q4gswLinear.cpp`'s `q4gsw_linear()`, a separate,
  older implementation (last touched by an unrelated PR, #20055) with zero
  coopmat awareness.
- The coopmat-capable weight-only path (`can_use_q4gsw_coopmat`,
  `kQ4gswCoopmatDims`, `add_linear_qw_node`'s `nbits==4` branch, and the
  `linear_q4gsw_coopmat_*.spv` shaders -- confirmed via `spirv-dis` to contain
  genuine `OpCooperativeMatrixMulAddKHR` instructions) lives in
  `QuantizedLinear.cpp`, but was only reachable via `linear_q8csw`'s
  registration -- and `linear_q8csw`'s own body hardcodes
  `weight_quant_config(8, kPerChannel, {K})`. No registered op anywhere
  called the coopmat-capable path with a 4-bit weight config.
- Confirmed the complete shader family already existed and was already
  compiled for every storage combination -- `linear_q4gsw_tiled_*.spv` and
  `linear_q4gsw_coop_*.spv` (the GEMV fallback), not just the coopmat
  variant -- meaning `add_linear_qw_node` is a fully generalized weight-only
  implementation, not a coopmat-only partial one missing fallback shaders.

**Fix applied** (user decision: "fix the wiring first, then measure both",
2026-07-04): added `linear_q4gsw()` to `QuantizedLinear.cpp` -- identical
6-arg signature to `Q4gswLinear.cpp`'s `q4gsw_linear()`
(`fp_input, weight_data, weight_scales_data, group_size, bias_data, output`)
so it's a drop-in replacement from the caller's side -- calling
`quantized_linear_impl` with `weight_quant_config(4, kPerGroup,
{group_size_val})`, mirroring `linear_dq8ca_q4gsw`'s pattern. Registered it to
`et_vk.linear_q4gsw.default`; removed that same registration from
`Q4gswLinear.cpp` (which still keeps its `et_vk.q4gsw_linear.default` alias,
untouched, in case anything else depends on that separate name).

**Safety verification performed, not assumed**:
- `test_q4gsw_linear`: 72/72 PASSED (0 failed) after rebuild -- includes
  small ACCU (accuracy-checked) shapes at both `linear_q4gsw_tiled_*` (the
  fallback path, confirming the tiled route through the new function is
  still correct) and real production-scale PERF shapes.
- `test_coopmat_linear_bench`: every `linear_q4gsw` case passed; its own
  summary table shows a genuine, consistent **~5x speedup**
  (`linear_q4gsw (4096,4096): 1729.0 -> 9627.8 GFLOP/s, 5.57x`; similar
  across all four tested shapes) with `linear_q4gsw_coopmat_buffer_texture2d_half`
  actually dispatching.
- Re-ran this feature's own capture (T005/T010) against the rebuilt binary;
  `4w` rows now show genuine coopmat dispatch where shape-eligible.

**Separate finding, explicitly NOT fixed (out of scope, flagged not
silently ignored)**: `test_coopmat_linear_bench`'s same run showed 5 pre-existing
correctness failures, all `linear_dq8ca_q4gsw` at **Texture3D** storage,
small shapes (`M,K,N` in `{128,256}`) -- the matching **Buffer**-storage
cases for the same shapes passed. This is unrelated to the `4w` fix above
(zero `dq8ca` code was touched) and was not introduced by it. Given
Texture3D is the real export path's *default* storage type (`006`'s
finding), this could plausibly affect the real, currently-shipping `8da4w`
model at these shapes -- flagged for separate investigation, not
diagnosed or fixed here.

**Alternatives considered**: leaving `4w` unfixed and reporting 0%
improvement -- rejected by explicit user decision, since that would report
an unreachable-code finding as if it were a measurement. Retiring
`Q4gswLinear.cpp` entirely -- rejected as unnecessarily broad; its
`et_vk.q4gsw_linear.default` alias is left untouched in case anything else
depends on it, minimizing blast radius to exactly the one registration that
was actually wrong.
