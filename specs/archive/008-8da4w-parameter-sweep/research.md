# Research: 8da4w Coopmat Tile/Subgroup Parameter Sweep

## Decision 1: New shader variants live in a fully separate, test-owned template -- zero production files touched

**Decision**: Create `backends/vulkan/test/custom_ops/glsl/dq8ca_q4gsw_coopmat_sweep.{glsl,yaml}` as a copy of the production `linear_dq8ca_qw_coopmat.glsl`'s double-buffered int8-coopmat logic, with its own `shader_variants` list for the swept combinations. The production file
(`backends/vulkan/runtime/graph/ops/glsl/linear_dq8ca_qw_coopmat.{glsl,yaml}`)
is not read, edited, or referenced by this feature at all.

**Correction found at build time (T003/T004)**: the production `.glsl`
`#include "common.glslh"` (generic `div_up`/`mul_*`/`align_up_*` macros);
none of the existing test-owned shaders under `test/custom_ops/glsl/`
happened to need this include before, so it isn't in that directory's
codegen include path and `glslc` fails with "Cannot find or open include
file" until it is. Fixed the same way as the `.glsl` itself: a
byte-identical copy of `common.glslh` into `test/custom_ops/glsl/` (`diff`
confirms exit 0), not a CMake include-path change reaching into the
production shader tree.

**Grounding -- investigated directly, not assumed**: `gen_vulkan_spv.py`'s
`SPVGenerator.addSrcAndYamlFiles()` indexes `.glsl` files by filename stem
(`self.src_files[extract_filename(file, keep_ext=False)] = file`) across
every scanned source directory, and `parseTemplateYaml()` raises `KeyError`
if the same top-level `template_name` key is declared in two yaml files
(`"{template_name} params file is defined twice"`). This rules out adding a
second yaml with the *same* template name (`linear_dq8ca_qw_coopmat`) --
it would hard-error at codegen time. A distinct filename stem
(`dq8ca_q4gsw_coopmat_sweep`) sidesteps this entirely and cannot collide.
`backends/vulkan/test/custom_ops/CMakeLists.txt` already builds its own
separate shader library (`prototyping_shaderlib`, from
`PROTOTYPING_SHADERS_PATH = .../test/custom_ops/glsl`), distinct from the
production `vulkan_backend` library -- and that directory already contains
test-owned shaders distinct from production
(`q4gsw_linear_gemv__w_4x8.glsl`) including an explicit reference *port*
(`coopmat_mm_ref.glsl`, "the test/custom_ops reference port of NVIDIA's
shmem_double_buf4.comp", per its own commit message) -- this feature
follows that exact, already-established pattern.

**Rationale**: Satisfies FR-008 structurally (impossible to accidentally
modify production behavior) rather than by discipline alone, at the cost
of one file's worth of duplication -- acceptable given FR-008's explicit,
hard "MUST NOT modify... shader registration" requirement.

**Alternatives considered**: Adding new `shader_variants` entries directly
to the production yaml (rejected -- even though the existing 2 variants'
output would be byte-identical, this edits a file that is part of the
shipped shader registration, which FR-008 explicitly protects; not worth
the ambiguity for a few hundred lines of duplication).

## Decision 2: New test-only C++ harness bypasses `pick_linear_dqa_qw_shader` entirely, reusing exposed production helpers

**Decision**: `test_dq8ca_tile_sweep.cpp` builds each variant's
`DynamicDispatchNode` directly, using a **fixed** kernel name per variant
(no eligibility gating -- the whole point is to force each specific
variant) -- structurally mirroring `add_linear_dqa_qw_node`'s bindings
(same 9-input/1-output layout, same spec-constant list:
`apply_bias`, `K4_per_group`, `coopmat_k_iters`, `output_N`) but replacing
`pick_linear_dqa_qw_shader` with `[[kernel_name]] { return
VK_KERNEL_FROM_STR(kernel_name); }`.

**Grounding**: `add_linear_dqa_qw_node`'s full body was read directly
(`QuantizedLinear.cpp:616-687`) -- it uses `QuantizedLinear.h`'s exposed
`prepack_quantized_linear_weight()` and `QuantizeDequantize.h`'s
`add_quantize_and_pack_4h4w_with_group_sums_node()` (the dynamic
activation quantization/packing step, needed regardless of which GEMM
kernel runs) directly.

**Correction found during implementation (T006), not assumed from the
header alone**: `quantized_linear_local_wg_size()` *is* exposed via
`QuantizedLinear.h`, but reading its body (`QuantizedLinear.cpp:88-131`)
shows it is not actually reusable here -- both it and
`quantized_linear_global_wg_size` (not exposed) call a shared
`coopmat_tile_dims(shader.kernel_name)` helper that identifies tile
dimensions via an exact string-prefix match
(`kernel_name.rfind("linear_dq8ca_q4gsw_coopmat", 0) == 0`). This
feature's kernel names (`dq8ca_q4gsw_coopmat_sweep_cfgN_...`) do not match
that prefix, so calling the exposed function directly would silently fall
through to the *wrong* (fp16 `q4gsw`) tile dimensions default. Both
workgroup-size functions are reimplemented locally instead, keyed by a
compile-time `config_id -> {WG_TILE_M, WG_TILE_N, WG_TILE_K,
subgroup_size}` table matching research.md Decision 4 (not by kernel-name
parsing), together with `resize_linear_qw_node` (also not exposed, but
simple and shape-only -- no coopmat-specific logic to get wrong).

**Rationale**: No changes to any production header (`QuantizedLinear.h`
is read, not edited) -- reimplementing three small, self-contained
functions in test code, keyed by data this feature already knows at
authoring time, is cheaper and safer than adding public declarations to a
production header for a one-off sweep, and avoids the silent-wrong-default
failure mode the exposed function would hit if called naively.

**Alternatives considered**: Exposing `add_linear_dqa_qw_node`,
`quantized_linear_global_wg_size`, and `resize_linear_qw_node` via
`QuantizedLinear.h` -- rejected; adding declarations to a production
header is itself a production-file change, in tension with FR-008's
spirit even though it wouldn't change runtime behavior. Calling the
already-exposed `quantized_linear_local_wg_size()` as originally planned --
rejected after finding it depends on name-prefix matching that silently
breaks for this feature's kernel names.

**Second correction found during implementation (T006/T010)**: reading
`utils.cpp`'s `execute_test_cases()` shows the harness framework only
catches `vkapi::ShaderNotSupportedError` around a test case's execution --
not a general exception from shader-pipeline creation, and certainly not
an actual driver-level crash (a segfault in `vkCreateComputePipelines`,
the exact Xclipse PAL failure mode this shader's own header comment
documents, is not something any C++ `try`/`catch` can recover from in the
same process regardless). A single crashing config would silently lose
every subsequent row in the same binary invocation, violating FR-004/
SC-001 ("never blank"). Fix: `test_dq8ca_tile_sweep`'s `main()` runs
exactly one `config_id` per process invocation, selected by an env var
(`DQ8CA_SWEEP_CONFIG_ID`, matching this codebase's existing env-var-flag
convention -- e.g. `test_coopmat_linear_bench`'s
`COOPMAT_BENCH_CORRECTNESS_ONLY`) -- T010's capture step becomes a shell
loop over `config_id` 1-12 that invokes the binary once per config and
records a `pipeline_crash` outcome (with the process's exit status/signal
as `failure_detail`) for any invocation that doesn't exit 0, rather than
one single in-process run across all configs.

**Rationale**: Process-level isolation is the only sound way to guarantee
one bad kernel can't erase every other config's data -- matching this
entire workstream's Principle VI ("verify with tools, never assume"): the
framework's actual exception-handling scope was read directly, not
assumed adequate from its use elsewhere.

## Decision 3: Sweep-phase shapes are reduced (two per model: square + rectangular); the winning configuration is validated against the full catalog before recommendation

**Decision**: The sweep phase (US2) measures each of the 11 new variants
against 6 representative shapes -- one `wq` (square, `K=N`) shape and one
`w1_gate` (rectangular, `K!=N`) shape per model:

| Model | `wq` (K, N) | `w1_gate` (K, N) |
|---|---|---|
| `llama-3.1-8b` | 4096, 4096 | 4096, 14336 |
| `llama-3.2-3b` | 3072, 3072 | 3072, 8192 |
| `llama-3.2-1b` | 2048, 2048 | 2048, 8192 |

-- not the full 3-model x 7-op `8da4w` catalog. The best-performing
correctness-verified variant(s) are then re-measured against the full
catalog (all 7 ops, all 3 models -- the same 21 cases `007` already
captured for the shipped configuration) before appearing in the final
recommendation (US3).

**Rationale**: Bounds device time and shader-cache growth across 11
variants x full-catalog shapes (which would be 11 x 21 = 231 measurements
before even reaching a recommendation) while still producing a
statistically confident, broadly-validated final answer.

**Revised during `/speckit-analyze` remediation (finding U3)**: the
original plan used `wq` only (square GEMMs). `003`'s classification shows
the highest-runtime-share `8da4w` ops (`w1_gate`/`w3_up`/`w2_down`) are
rectangular, sometimes by a wide margin (e.g. `K=4096`, `N=14336` for the
8B model) -- a tile geometry tuned to square shapes alone could miss the
actual optimum for the ops that dominate total runtime. Adding one
rectangular shape (`w1_gate`) per model directly addresses this at a
modest cost (33 -> 66 sweep-phase rows), without waiting until the
full-catalog validation step (which only ever checks the *already-chosen*
winner, not whether a different winner would have been picked with better
shape coverage).

**Alternatives considered**: Sweeping the full catalog from the start
(rejected -- disproportionate device time for a curated, hypothesis-driven
sweep whose purpose is narrowing down candidates, not an exhaustive
census); sweeping only one shape total (rejected -- one data point per
variant risks a shape-specific artifact skewing the pick before the
full-catalog validation step ever runs); keeping `wq`-only and documenting
the aspect-ratio risk instead of fixing it (rejected -- the fix is cheap
enough that documenting-not-fixing isn't worth the risk of picking a
winner that isn't actually best for the dominant ops).

## Decision 4: The curated sweep set (13 total: 11 new performance candidates + 1 reused shipped baseline + 1 deliberate negative test)

**Decision**: Vary one axis at a time from the shipped configuration
(`WG_TILE_M=128, WG_TILE_N=64, WG_TILE_K=32, SUBGROUP_SIZE=64`), holding
`SG_GRID_X=SG_GRID_Y=2` constant throughout (verified valid for every
candidate below: `MMA_M=MMA_N=16` fixed, and `WG_TILE_M`/`WG_TILE_N`
divided by 2 always lands on a multiple of 16). `WG_TILE_K` is restricted
to `{16, 32}` for the 11 performance candidates -- `group_size=32` (fixed
per every model in `shapes.json`) does not divide evenly by 64.

| # | WG_TILE_M | WG_TILE_N | WG_TILE_K | Subgroup | Note |
|---|---|---|---|---|---|
| 0 (reused) | 128 | 64 | 32 | 64 | Shipped, already measured in `007` |
| 1 | 128 | 64 | 32 | 32 | Subgroup axis only |
| 2 | 64 | 64 | 32 | 64 | Pre-restructure tile (verified via `git show 49a51b1776^`), shipped subgroup |
| 3 | 64 | 64 | 32 | 32 | Pre-restructure tile, native subgroup |
| 4 | 128 | 64 | 16 | 64 | K-step halved |
| 5 | 128 | 64 | 16 | 32 | K-step halved, native subgroup |
| 6 | 64 | 64 | 16 | 64 | Pre-restructure tile + K halved |
| 7 | 64 | 64 | 16 | 32 | Pre-restructure tile + K halved, native subgroup |
| 8 | 256 | 64 | 32 | 64 | Larger M tile |
| 9 | 256 | 64 | 32 | 32 | Larger M tile, native subgroup |
| 10 | 128 | 128 | 32 | 64 | Wider N tile |
| 11 | 128 | 128 | 32 | 32 | Wider N tile, native subgroup |
| 12 (negative test) | 128 | 64 | **64** | 64 | Deliberately invalid: `group_size=32` doesn't divide `WG_TILE_K=64` -- included only to prove the correctness check catches it (`/speckit-analyze` finding G1), not as a performance candidate |

**Rationale**: Every performance-candidate config isolates one variable
relative to a known reference point (shipped or pre-restructure), so a
performance change can be attributed to a specific axis rather than an
unexplained combination effect -- matching this codebase's own prior
tuning style (commit `49a51b1776`'s "M5 EVT1" comparisons named specific
layouts, not grid search results).

**Revised during implementation (T010): a real, reproducible correctness
bug was found and root-caused precisely -- it is a staging thread-count
provisioning bug, not a "subgroup=32" bug per se.** Config 1 (subgroup
axis only, otherwise identical to shipped) was measured first per US1 and
its exact-reference check failed at a real, aligned shape (`M=256, K=2048,
N=8192`) -- not fp16 noise: the mismatch is large (computed `-21.562` vs
reference `-5.766`) and lands exactly at output element 524288, row 64 of
the `[256, 8192]` result, exactly where the second `SG_TILE_M`-subgroup's
tile begins.

Root cause, verified by deriving and checking the shader's own thread-map
formulas (`linear_dq8ca_qw_coopmat.glsl`'s "A staging thread map" /
"B staging thread map" comments): the A-staging pass needs
`(WG_TILE_M/4) * (WG_TILE_K/4)` threads and the B-staging pass needs
`(WG_TILE_K/4) * (WG_TILE_N/4)` threads to cover one chunk in a single
pass, but the actual workgroup only ever has
`WG_SIZE = SG_GRID_X * SG_GRID_Y * SUBGROUP_SIZE` threads (fixed at
`4 * SUBGROUP_SIZE` for this curated set's `SG_GRID=2x2`). The shipped
config (`M=128,K=32,N=64,SUBGROUP=64`) has A-required `=256` and
`WG_SIZE=256` -- an *exact* match, zero slack, by original design. Any
config where the required count *exceeds* `WG_SIZE` silently leaves part
of the LDS staging buffer unwritten (stale garbage) for exactly the
rows/cols past the covered range -- confirmed empirically via config 8's
per-16x16-tile mismatch map: rows 0-127 (the staged region, matching
`A_ACTIVE_THREADS=256` at `WG_SIZE=256` for `WG_TILE_M=256`... wait,
`WG_TILE_M=256` needs `512` threads at `WG_SIZE=256` -- so rows 0-127 are
covered by the 256 available threads and are **100% correct**, while rows
128-255 (never staged) are **~85-98% wrong** -- an exact match to the
formula, not a guess.

Recomputing this formula for all 11 original candidates (A-required vs
`WG_SIZE`, B-required vs `WG_SIZE`):

| Config | M,K,N,SG | A-required | B-required | Verdict |
|---|---|---|---|---|
| 1 | 128,32,64,32 | 256 > 128 | 128 = 128 | broken (confirmed) |
| 3 | 64,32,64,32 | 128 = 128 | 128 = 128 | zero-slack, same margin as shipped -- **tested, passes** |
| 5 | 128,16,64,32 | 128 = 128 | 64 < 128 | zero-slack on A -- **tested, passes** |
| 7 | 64,16,64,32 | 64 < 128 | 64 < 128 | slack on both -- **tested, passes** |
| 8 | 256,32,64,64 | 512 > 256 | 128 < 256 | broken (confirmed) |
| 9 | 256,32,64,32 | 512 > 128 | 128 = 128 | broken (same issue, worse) |
| 11 | 128,32,128,32 | 256 > 128 | 256 > 128 | broken (both sides) |

**User decision (explicit, in two steps)**: first, exclude all 6
subgroup-32 candidates pending investigation; then, after the precise
formula was derived and configs 3/5/7 shown to sit at the same zero-slack
margin as the shipped baseline (never actually disproven), re-test them
rather than exclude by a coarser subgroup-based heuristic. All three
passed correctness. **Final candidate set: 7** (2, 3, 4, 5, 6, 7, 10),
excluding only the 4 configs mathematically guaranteed broken by the
verified formula (1, 8, 9, 11) -- not re-tested, since the root cause is
already confirmed, not assumed. Configs 1, 8, 9, 11 remain built (harmless
already-compiled variants) but are excluded from the sweep-phase table,
ranking, and full-catalog validation -- reported as a called-out,
root-caused finding in the report's Excluded/Out-of-Scope section, with
config 1's and config 8's evidence (including config 8's tile-mismatch
map) as proof. Sweep-phase row count: 7 candidates x 6 shapes (42) +
config 12's 1 row = **43**.

**Sweep-phase result**: config 5 (`WG_TILE_K=16` vs shipped's 32, native
subgroup 32) is the fastest candidate at every one of the 6 representative
shapes, by a clear margin (e.g. `llama-3.1-8b` `wq`: 13355us vs the next
candidates' 14479-23563us) -- validated against the full catalog next
(Decision 6/T019).

**User follow-up: the shipped-vs-tuned improvement (+13-17%) was judged
too modest, and the mixed (not clearly winning) result vs. tiled --
correctly so; a config that only closes the gap against a bad baseline,
without beating the actual competing dispatch path, is a weak finding.
Directed to fix the root cause (not just document and exclude it) and
retest, on the theory that the untested "larger tile" direction (config
8) might have been the actually-promising one, blocked only by this bug.**

**Fix**: generalized the A-staging thread map in the test-owned shader
(`dq8ca_q4gsw_coopmat_sweep.glsl`) to loop multiple slots per thread when
`A_TOTAL_SLOTS > WG_SIZE`, mirroring the exact pattern the B/INT4-weight
staging path already used (`B_SLOTS_PER_THREAD` -- that path was never
broken, since oversubscription there was already handled). For every
previously-passing config (`A_SLOTS_PER_THREAD == 1`), the rewritten loop
is a no-op restatement of the original single-slot logic -- same bounds
check, same indices -- so this could not silently change behavior for
anything already measured. Applied to `temp_A` (now `temp_A[A_SLOTS_PER_THREAD]`)
and all 4 sites that read/write it (prologue prefetch, prologue store,
main-loop prefetch, main-loop store).

**Result after the fix**: all 4 previously-broken configs (1, 8, 9, 11)
now pass correctness at every tested shape -- re-verified, not assumed,
by re-running the exact-reference check. But the "larger tile" hope did
not pan out: config 8 (`M=256`) measures *slower* than config 5
(15693us vs 13355us at `llama-3.1-8b` `wq`), and configs 9/11 (which
combine the larger-tile axis with subgroup 32) are dramatically slower
(10-40x) -- the multi-slot staging loop works correctly but adds real
serialization overhead once a workgroup is oversubscribed, which is
exactly what's happening for 9/11. Config 1 (the original subgroup-32
discovery case, now fixed) turns out to be statistically indistinguishable
from config 5 across all 6 shapes (e.g. `llama-3.1-8b` `wq`: 13309±29 vs
13355±114 -- overlapping `mean ± 2·stdev` bands, matching bands at every
other shape too) -- a second config effectively tied for the win, but not
a *better* one. **Config 5 remains the recommended winner** after
including all 11 original candidates in the ranking; the fix changed
which configs are *correct*, not which one is *fastest*.

**Revised during `/speckit-analyze` remediation (finding G1)**: spec.md's
US2 Acceptance Scenario 2 promises that a mathematically invalid
combination is "recorded as invalid... not attempted" -- but no task
exercised this, and on inspection it isn't a distinct runtime-detected
category at all (see spec.md's Key Entities note): `WG_TILE_K=64`'s
`CHUNKS_PER_GROUP = K4_per_group * 4 / WG_TILE_K` computation
(`linear_dq8ca_qw_coopmat.glsl:195`) is integer division; with
`group_size=32` this truncates to `0`, so the K-loop never executes and
the shader produces zero/garbage output -- caught by the *existing*
correctness check, not a new pre-flight validity check. Config 12 is
included specifically to prove this: it is measured at exactly **one**
representative shape (not all 6 from Decision 3 -- there is nothing
further to learn from repeating a known-broken kernel), and its expected,
verified-in-advance outcome is `correctness_failure`. If it instead came
back `measured` (i.e. the broken shader passed the correctness check),
that would itself be a critical finding about the correctness check's own
reliability.

**Alternatives considered**: Leaving `WG_TILE_K=64` out entirely (the
original plan) -- rejected after finding G1; the whole point of FR-003's
"verify correctness before ranking" guarantee is undermined if it's never
actually been observed to catch a real broken kernel, only assumed to.
Inventing a new `invalid_combination` outcome category distinct from
`correctness_failure` -- rejected; the shader doesn't fail to compile or
crash at pipeline creation for this combination, it silently miscomputes,
which is exactly what `correctness_failure` already means.

## Decision 5: Correctness verification per variant reuses `test_coopmat_linear_bench.cpp`'s exact-reference pattern

**Decision**: Each new variant's correctness check reuses the exact fp32
CPU reference already established for `dq8ca` correctness testing (commit
`10ef1eaa93`: activation values that are multiples of 1/16 so the fp16
dynamic-quantization round-trip is exact, `scale=1/16`, `zp=0`), applied
at small aligned shapes before any variant's performance number is trusted
-- mirroring constitution Principle I and `007`'s FR-007 correctness bar.

**Rationale**: Reuses an already-proven, exact reference computation
instead of deriving a new one; "exact" (not tolerance-based) removes any
ambiguity about whether a small numeric mismatch is a real bug or
acceptable float error.

**Alternatives considered**: A tolerance-based comparison against the
existing `linear_dq8ca_q4gsw_coopmat` production kernel's output --
rejected; comparing a new kernel against another GPU kernel's output does
not establish ground truth the way an exact CPU reference does.

**Risk noted during `/speckit-analyze` remediation (finding U2), reasoned
through but not yet empirically confirmed**: the existing exact-reference
pattern is only validated at small shapes (64-256) in
`test_coopmat_linear_bench.cpp`; this feature's sweep-phase correctness
check runs at real production shapes (2048-4096, per Decision 3). The
`8da4w` scheme's K-dimension reduction accumulates in **int32** (exact
integer arithmetic, per the constitution's Quantization Scheme Matrix --
`coopmat<int8> x coopmat<int8> -> coopmat<int32>`), not fp16/fp32, so
exactness plausibly holds at any K length: unlike floating-point
accumulation, integer addition introduces no rounding error regardless of
how many terms are summed. The only floating-point steps are the
per-element activation quantization and the final int32-to-fp16
dequantization, neither of which accumulates over K. This is a reasoned
expectation, not a verified fact -- T007 MUST confirm exactness actually
holds at the real production shapes before any sweep-phase result is
trusted, rather than assuming the small-shape validation generalizes.
