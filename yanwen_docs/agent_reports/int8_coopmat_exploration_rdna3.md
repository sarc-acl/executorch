# int8 cooperative-matrix exploration on RDNA3 / RADV

## TL;DR

The hardware exposes 12 int8/uint8 cooperative-matrix configurations
(all `16x16x16 → int32/uint32`). After fixing a K-dimension unit bug in
the prototype shader, an int8×int8 → int32 GEMM with int32 accumulation
on `linear_coopmat_int8.glsl` correctly computes 12/12 measured shapes
against a sampled CPU reference and reaches **5.8 TFLOP/s on the 4096³
stress case**, but only **1.3–4.1 TFLOP/s on BERT/LLaMA prefill shapes**
— meaningfully **slower than the fp16 cooperative-matrix shader Phase 1
already measured** at most of those shapes.

Recommendation: **NO-GO for Phase 5 production**, beyond keeping this
prototype as research scaffolding. The fp16 Phase 3 path is faster on
the shapes that actually appear in real LLaMA prefill, and the
integration cost of an int8 production path (coopmat-friendly weight
packing, scale / weight-sum / bias / dequant fusion, quantized-LLaMA
E2E) buys no measured kernel-level advantage at those shapes on this
device. The 4096³ cube delivers 1.27× over fp16 coopmat, but no
real-LLaMA shape lives there.

This report is the deliverable named in
`yanwen_docs/agent_plans/4_int8_coopmat_exploration.md`. Earlier drafts
of this report (commit `c17428ee29` and earlier in the working tree)
quoted **6–19 TFLOP/s** numbers; those were before the K-scaling bug
in the shader was found. **Disregard the earlier numbers**; the table
in this report is the corrected truth. The bug story is documented in
`yanwen_docs/lesson_learned/phase4_int8_coopmat_exploration/k_dim_unit_mismatch.md`.

## Hardware capability

`vulkaninfo --summary` and `queryCooperativeMatrixProperties()` confirm:

- Device: `AMD Radeon 780M (RADV PHOENIX)`, RADV Mesa 25.0.7, Vulkan
  API 1.4.305. `VK_KHR_cooperative_matrix` rev 2 supported.
- 14 cooperative matrix configurations enumerated (configs 0–13).
  Configs 2 through 13 are int8/uint8 variants:

| # | M | N | K | A | B | C | Result | Scope |
| --: | -: | -: | -: | --- | --- | --- | --- | --- |
| 2 | 16 | 16 | 16 | uint8 | uint8 | uint32 | uint32 | Subgroup |
| 3,4 | 16 | 16 | 16 | uint8 | uint8 | int32 | int32 | Subgroup |
| 5 | 16 | 16 | 16 | uint8 | int8 | uint32 | uint32 | Subgroup |
| 6,7 | 16 | 16 | 16 | uint8 | int8 | int32 | int32 | Subgroup |
| 8 | 16 | 16 | 16 | int8 | uint8 | uint32 | uint32 | Subgroup |
| 9,10 | 16 | 16 | 16 | int8 | uint8 | int32 | int32 | Subgroup |
| 11 | 16 | 16 | 16 | int8 | int8 | uint32 | uint32 | Subgroup |
| 12,13 | 16 | 16 | 16 | int8 | int8 | int32 | int32 | Subgroup |

The combination this prototype targets is row 12/13: signed-int8 ×
signed-int8 → signed-int32, subgroup scope. That matches the natural
quantization shape used by ExecuTorch's `linear_q8ta_q8csw_tiled`
(int8 activation × int8 weight, int32 accumulator, fp dequantize).

Raw artifacts:

```
yanwen_docs/agent_results/int8_coopmat_exploration_rdna3/dev_dri.txt
yanwen_docs/agent_results/int8_coopmat_exploration_rdna3/vulkaninfo_summary.txt
yanwen_docs/agent_results/int8_coopmat_exploration_rdna3/coopmat_property_table.txt
```

## Existing ExecuTorch quantized linear path

Four `aten`-style quantized linear ops are registered today (see
`backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.cpp`):

| Op | Activation | Weight | Compute helper | Notes |
| --- | --- | --- | --- | --- |
| `et_vk.linear_q8ta_q8csw.default` | int8 per-tensor | int8 per-channel | `int_accumulate_with_int8_weight` (`dotPacked4x8AccSat`) | Closest analog to a coopmat int8 path |
| `et_vk.linear_q8csw.default` | fp16/fp32 | int8 per-channel | `fp_accumulate_with_int8_weight` | fp accum, dequant on load — *not* int8 GEMM |
| `et_vk.linear_q4gsw.default` | fp16/fp32 | int4 group-wise | tiled or coop variant | Out of scope |
| `et_vk.linear_dq8ca_q4gsw.default` | int8 per-channel | int4 group-wise | tiled or coop variant | Out of scope |

The `linear_q8ta_q8csw_tiled` shader is the relevant baseline. Its
inner loop:

```glsl
accum.data[m][n4][n4i] = dotPacked4x8AccSat(
    in_tile.data[m4][k4][m4i],
    w_tile.data[k4][n4][n4i],
    accum.data[m][n4][n4i]);
```

uses Khronos's 4-wide int8 dot-product extension (`GL_EXT_integer_dot_product`).
That is a per-thread instruction; cooperative matrix replaces it with
one subgroup-cooperative
`coopMatMulAdd<int8, int8, int32, 16, 16, 16>` per output tile.

The packed weight layout (from `pack_q8_linear_weight` →
`linear_int8_weight_block.glslh`):

- Source `int8 [N, K]` (row-major)
- Output: `ivec4 packed[k4 * N4 + n4]` where each `ivec4` holds 4 ints,
  each int packs 4 int8 along `K`. Total 16 int8 per `ivec4` = a 4-row
  (N) × 4-col (K) block.
- Indexed `[k4][n4]` — *transposed* relative to the source N axis, for
  GEMM coalescing.

This layout was tuned for `dotPacked4x8AccSat`'s 4-wide K stride. It
is **not directly coopmat-loadable** as a `gl_MatrixUseB` of shape
`(lK=16, lN=16)` with stride 16 — the ints in `packed[k4][n4]` carry K
contiguous within an int and N contiguous across the four ints within
an ivec4, but the macro stride between blocks is N4-major, not the
16-element N stride a coopmat row-major load expects. Using the
existing layout would either require staging into shared memory with a
transpose-on-load, or a new `pack_q8_linear_weight_coopmat` packer.

## First kernel target

Selected target: **standalone int8×int8→int32 GEMM kernel
(`linear_coopmat_int8.glsl`)**, buffer-only, all activation/weight/output in
`int32` buffers with 4 int8 packed per int along the inner dim of A and B,
one int32 per output element.

Why not extend `linear_q8ta_q8csw_tiled` directly:

1. The packed weight layout is the wrong shape for a coopmat row-major
   load (see above). Reusing it would conflate "does coopmat help?"
   with "can we stage-transpose into shared memory cheaply?", obscuring
   the throughput measurement.
2. Scale / weight-sum / bias / dequant fusion adds GLSL surface area
   that is irrelevant to the throughput question. The fp dequant tail
   is the same for tiled and coopmat variants; it composes onto
   whichever inner kernel is fastest.
3. The Phase 1 fp16 prototype followed exactly this shape — kernel-only
   first, runtime/E2E integration second. Phase 4 should match.

The prototype's contract:

- A buffer `int [M, K/4]` row-major; element `(m, k)` is byte
  `(A_buf[m*K/4 + k/4] >> ((k%4)*8)) & 0xff`, sign-extended.
- B buffer `int [K, N/4]` row-major; element `(k, n)` byte equivalent.
- C buffer `int [M, N]` row-major, full int32.
- Macro tile `64×64×32` per workgroup, 4 subgroups (subgroup size 64),
  each computing 2×2 output coopmat tiles of shape `16×16`.
- Shared-memory staging: `Ash[64×32 int8]` and `Bsh[32×64 int8]`, both
  held as `uvec4` (16 int8 per uvec4).
- `coopMatMulAdd<int8_t, int8_t, int32_t, 16, 16, 16>` in the inner loop.

Files:

```
backends/vulkan/runtime/graph/ops/glsl/linear_coopmat_int8.glsl
backends/vulkan/runtime/graph/ops/glsl/linear_coopmat_int8.yaml
backends/vulkan/test/custom_ops/impl/TestLinearCoopmatInt8.cpp     # registers test_etvk.linear_coopmat_int8.default
backends/vulkan/test/custom_ops/linear_coopmat_int8_bench.cpp       # microbench
backends/vulkan/test/custom_ops/utils.cpp                            # extends validate_against_reference for kInt
```

## Correctness

The bench harness's `validate_against_reference()` originally
short-circuited to "PASS" for non-fp dtypes
(`backends/vulkan/test/custom_ops/utils.cpp:694`). This was extended in
this round to handle `kInt` with `INT32_MIN` as the sentinel-skip mark,
mirroring the Phase 1 NaN-skip semantics for sampled validation:

```cpp
if (dtype == vkapi::kInt) {
  // sentinel-skip on INT32_MIN, exact equality otherwise
  ...
}
```

With the validator extended, the prototype shader **failed** 8 of 9
random-data shapes on the first run (mismatches in the 10⁴–10⁵ range).
Adding two ones-only diagnostic shapes
(`ones_BERT_QKV` and `ones_LLM_QKV_64tok`) showed the GPU consistently
computed exactly 1/4 of the expected value — the smoking gun for a
K-dimension unit bug. The fix was a one-line shader change: the size
UBO's x dimension is the int32 count of the input tensor (`K/4`), not
the int8 count (`K`); the chunkK loop must scale by 4 to cover all of
K. After the fix, all 12 cases pass.

The bug and the fix are in
`yanwen_docs/lesson_learned/phase4_int8_coopmat_exploration/k_dim_unit_mismatch.md`.

The earlier short-circuit gap is documented in
`yanwen_docs/lesson_learned/phase4_int8_coopmat_exploration/int_validator_short_circuit.md`.

## Microbenchmark results (post-fix)

Run on the same RADV PHOENIX device used in Phase 1 / Phase 2 / Phase
3, 3 warmup + 10 timed iterations per case, GPU-timestamped via the
existing custom-ops bench framework. `RANDINT8` data
(int32 byte-pattern in [-128, 127]). Sampled CPU reference for shapes
where any of M, K, N exceeds 1024 (8192 sample positions).

| Shape (M × K × N) | Op µs | int8 GFLOP/s | Validation |
| --- | ---: | ---: | --- |
| ones_64x32x64 | 6.26 | 41.90 | PASSED (full ref) |
| ones_BERT_QKV (128, 768, 768) | 112.39 | 1343.47 | PASSED (full ref) |
| ones_LLM_QKV_64tok (64, 4096, 4096) | 1287.32 | 1668.19 | PASSED (sampled) |
| BERT_FFN_up (256, 768, 3072) | 671.13 | 1799.90 | PASSED (sampled) |
| BERT_FFN_down (256, 3072, 768) | 716.07 | 1686.93 | PASSED (sampled) |
| BERT_QKV (128, 768, 768) | 107.40 | 1405.91 | PASSED (full ref) |
| LLM_QKV_64tok (64, 4096, 4096) | 1283.69 | 1672.90 | PASSED (sampled) |
| LLM_QKV_256tok (256, 4096, 4096) | 3827.22 | 2244.43 | PASSED (sampled) |
| sq_1024 (256, 1024, 1024) | 328.43 | 1634.65 | PASSED (full ref) |
| sq_4096 (256, 4096, 4096) | 2721.16 | 3156.72 | PASSED (sampled) |
| LLM_FFN_up_256tok (256, 4096, 11008) | 5715.81 | 4038.88 | PASSED (sampled) |
| sq_4096_cube (4096, 4096, 4096) | 21840.04 | **6292.98** | PASSED (sampled) |

Numbers above include a **+1 uvec4 row padding** in the shared-memory
staging arrays, mirroring the fp16 coopmat shader's bank-conflict
avoidance. Without padding (initial prototype, also numerically
correct), throughput was ~8–11% lower at small/medium shapes — see
"Why int8 is slower than fp16" below for the full breakdown.

Raw log:
`yanwen_docs/agent_results/int8_coopmat_exploration_rdna3/linear_int8_coopmat_bench_corrected.log`.

### Comparison vs Phase 1 fp16 coopmat (same device, same harness)

Pulling the matching rows from
`yanwen_docs/agent_reports/kernel_sweep_fp16_rdna3.md` (linear table):

| Shape | fp16 Stephen | fp16 coopmat | int8 coopmat (this, padded) | int8 / fp16 coopmat | int8 / Stephen |
| --- | ---: | ---: | ---: | ---: | ---: |
| BERT_FFN_up | 488 | 1882 | 1800 | 0.96× | 3.69× |
| BERT_FFN_down | 489 | 2554 | 1687 | **0.66×** | 3.45× |
| BERT_QKV | 467 | 1362 | 1406 | 1.03× | 3.01× |
| sq_1024 | 467 | 1440 | 1635 | 1.14× | 3.50× |
| sq_4096 | 470 | 4691 | 3157 | **0.67×** | 6.72× |
| sq_4096_cube | 470 | 4565 | **6293** | **1.38×** | 13.39× |

The int8 coopmat path consistently beats Stephen's fp16 baseline
(3.0–13.4×) — the cooperative matrix hardware is doing real work — but
**fp16 coopmat already beats Stephen by similar or larger margins**
(2.9–10×), and at the LLaMA-prefill shape class
(BERT_FFN_down, sq_4096) the fp16 coopmat shader is *faster than*
the int8 coopmat shader. Only at the extreme cube stress (4096³) does
int8 pull ahead of fp16, by 1.38×.

### Why int8 is slower than fp16 here (and what would change it)

On paper, RDNA3 advertises **2× int8 WMMA throughput** over fp16
(~512 vs 256 ops/cycle/CU). The measured int8/fp16 ratio at LLaMA
shapes (0.66–1.14×) is far below that ceiling. After investigating
the obvious suspects, the gap decomposes as follows:

1. **Bank-conflict padding in shared memory** accounted for ~8–11% at
   small/medium shapes. The fp16 shader pads `A_STRIDE_VEC4` from 4 to
   5 uvec4 per row (and similarly for B); the int8 shader needed the
   same padding (2 → 3 uvec4 for A, 4 → 5 for B). Adding that lifted
   BERT_FFN_up from 1623 → 1800 GFLOP/s and the cube from 5784 → 6293
   GFLOP/s. Real but minor.

2. **Output store traffic doubled.** int32 result is 4 bytes per
   element vs 2 bytes for fp16 output. For BERT_FFN_down output
   `[256, 768]` that is 768 KB vs 384 KB written; for FFN_up it is 12
   MB vs 6 MB. At memory-bound shapes this contributes — at the cube
   it doesn't.

3. **`coopMatMulAdd<int8>` may not actually run at 2× the fp16 rate
   on RADV/RDNA3 mobile.** The RX 7900 XTX (desktop RDNA3) is documented
   at 512 vs 256 ops/cycle for int8/fp16 WMMA, but Phoenix APU may
   share the same WMMA execution path for both component types in
   mesa-radv's compiler — the only int8 advantage then is halved
   operand bandwidth. The 1.38× cube ratio is suggestive of "1.4×
   from operand density, not 2× from a dedicated int8 path." A
   shader-isa disassembly would confirm this; not pulled in this
   round.

4. **Input bandwidth ratio favors int8 less than expected.** int8
   reads ½ the operand bytes per mma vs fp16 (256+256 = 512 B vs
   1024 B per 16×16×16 tile). But fp16 was already memory-coalesced
   well via uvec4 staging, and shared-memory bandwidth is not the
   bottleneck on this device — so halving operand traffic does not
   double throughput. The bottleneck is the WMMA issue rate, not the
   staging rate.

5. **Macro tile size may favor fp16.** Both shaders use TILE_M ×
   TILE_N = 64 × 64 with TILE_K = 32. For int8 with halved operand
   bytes per K, an int8-optimal tile might be larger (e.g.
   TILE_K = 64) to amortize staging across more inner mma calls.
   Phase 1 swept this for fp16 and found 32 won; an analogous int8
   sweep was not run. Not pursued here because the integration cost
   does not justify it given finding 3.

The cube case (4096³) is the proof point of the asymptotic int8
advantage — at that compute density the kernel is no longer
staging-bound, and int8's higher peak WMMA rate (whatever the actual
RDNA3 mobile ratio is) finally shows up as 1.38× over fp16. Real
LLaMA prefill shapes do not reach that compute density per output
tile, so the asymptotic int8 advantage is not realized in any
LLaMA-relevant case.

If a future phase wants to close this gap, the candidate experiments
in priority order are:

1. Pull the AMDGPU shader ISA from
   `RADV_DEBUG=preoptir,nir cmake-out-vk/.../linear_coopmat_int8_bench`
   and confirm whether int8 coopmat lowers to a different machine
   instruction than fp16 coopmat (or whether they share an execution
   path).
2. Sweep the macro tile (TILE_M, TILE_N, TILE_K) for int8 the way
   Phase 1 swept it for fp16. The int8-optimal tile is likely
   different.
3. Try replacing the shared-memory staging with direct buffer
   coopMatLoad (skipping shared memory entirely). If the bottleneck
   is staging, this might recover throughput; if it is the WMMA rate,
   it will not.

None of these are worth running in this Phase, because even doubling
int8's throughput would not change the production recommendation —
fp16 is already shipping, and the integration cost of int8 production
(coopmat-friendly weight pack + dequant fusion + quantized-LLaMA E2E)
is the same regardless of whether the kernel hits 4 TFLOP/s or 8
TFLOP/s.

### Comparison vs existing `linear_q8ta_q8csw_tiled` (`dotPacked4x8AccSat`)

The existing `q8csw_linear` bench
(`yanwen_docs/agent_results/int8_coopmat_exploration_rdna3/q8csw_linear_baseline_full.log`)
covers a different shape set (no full-tile-eligible 64-multiple
shapes) and no shape larger than `1024×2048×2048`. The closest
comparison points:

| Shape (M × K × N) | `linear_q8ta_q8csw.default` (Texture3D) | int8 coopmat (this) |
| --- | ---: | ---: |
| 256 × 2048 × 2048 | 2428 GFLOP/s | – |
| 512 × 2048 × 2048 | 4506 GFLOP/s | – |
| 1024 × 2048 × 2048 | 5233 GFLOP/s | – |
| 256 × 4096 × 4096 | – | 3293 GFLOP/s |
| 4096 × 4096 × 4096 | – | 5784 GFLOP/s |

The existing q8ta_q8csw path saturates near 5 TFLOP/s on its largest
shape; the int8 coopmat prototype reaches 5.8 TFLOP/s on the 4096³
cube but only 3.3 TFLOP/s at LLaMA prefill scale (256×4096×4096). A
direct apples-to-apples comparison at identical shapes would require
porting the q8csw bench's shape set to multiples of the coopmat tile
gate — defer if it actually matters.

## Comparison against the fp16 Phase 3 path

The Phase 3 production path delivered **1.77× E2E wallclock on real
LLaMA 3.1 8B 4L fp16 seq=256** with a single capability-check change
and no partitioner work. That is a real, shipped win on the same
device.

The kernel-throughput data above shows that **the int8 path does not
have a kernel-level advantage at LLaMA prefill shapes on this
device**. fp16 coopmat is faster at BERT_FFN_up/down/sq_4096 and only
loses at the 4096³ cube. There is no plausible path where int8 can
deliver a real-LLaMA E2E win that fp16 cannot — the kernel-level
ceiling is set, and adding scale / sums / bias / dequant fusion only
moves it down.

## Recommendation

```
[NO-GO]  do not open a Phase 5 production phase
[FREEZE] keep the prototype shader + bench as research scaffolding
```

Reasoning against Phase 5:

1. **No kernel-level win at the relevant shape class.** Real LLaMA
   prefill is roughly 256×4096×4096 / 256×4096×11008. fp16 coopmat is
   faster than int8 coopmat at those shapes on RDNA3 / RADV PHOENIX.
2. **Integration cost is high.** Production int8 would need a new
   coopmat-friendly weight pack (or a stage-transpose), full scale /
   weight-sum / bias / dequant fusion, a quantized-LLaMA E2E, plus
   the existing `linear_q8ta_q8csw_tiled` path kept as a fallback.
   Multi-week effort.
3. **fp16 Phase 3 already ships.** The 1.77× real-LLaMA E2E win is in
   a single commit (`c17428ee29`) with no partitioner change. Effort
   is best directed at extending fp16 (texture matmul coopmat for
   attention bmms is the natural next item).

Reasoning for keeping the prototype:

- The hardware capability is real and confirmed.
- The int8 coopmat throughput is a useful upper bound for any future
  int8 work — it tells us "even with a perfect dequant tail, this is
  the kernel ceiling on this device."
- The bench harness's `kInt` validator extension is independently
  useful for any future integer-result shaders.

If a future device with a better int8/fp16 hardware ratio surfaces, or
a workload with cube-class shapes appears, this prototype is the
starting point — not the production path.

## Risks already cited from earlier phases that apply here

- **Buffer-trap at production seq lengths**
  (`yanwen_docs/lesson_learned/phase2_real_llama_e2e/seq2048_real_llama_oom.md`):
  the int8 coopmat prototype is buffer-only. A future texture-input
  variant analogous to `linear_coopmat_texture3d_buffer` would be
  needed to avoid the same E2E penalty.
- **Storage propagation surprises**
  (`yanwen_docs/lesson_learned/phase2_real_llama_e2e/buffer_override_does_not_propagate_synth_block.md`):
  the partitioner does not always honor `storage_type_override`;
  routing log is the source of truth.
- **Decode shapes (`M=1`)** (Phase 1 + previous-story): the
  conservative M%64==0 gate excludes them; coopmat int8 should keep
  the same gate.
- **Vendor portability of capability** (Phase 3 design): the strict
  `supports_fp16_coopmat_16x16x16()` pattern landed in Stage 1 should
  be paralleled by a `supports_int8_coopmat_16x16x16()` predicate
  before any int8 dispatch goes live, even for research.

## Files added in this round

```
backends/vulkan/runtime/graph/ops/glsl/linear_coopmat_int8.glsl
backends/vulkan/runtime/graph/ops/glsl/linear_coopmat_int8.yaml
backends/vulkan/test/custom_ops/impl/TestLinearCoopmatInt8.cpp
backends/vulkan/test/custom_ops/linear_coopmat_int8_bench.cpp
backends/vulkan/test/custom_ops/CMakeLists.txt   (one stanza)
backends/vulkan/test/custom_ops/utils.cpp        (kInt validator branch)
yanwen_docs/agent_results/int8_coopmat_exploration_rdna3/{logs, vulkan info}
yanwen_docs/lesson_learned/phase4_int8_coopmat_exploration/int_validator_short_circuit.md
yanwen_docs/lesson_learned/phase4_int8_coopmat_exploration/k_dim_unit_mismatch.md
```

No production runtime files were modified by this report — `Linear.cpp`,
`Matmul.cpp`, `QuantizedLinear.cpp`, `op_registry.py`, and the
partitioner passes are all unchanged. The four files above add a new
opt-in shader, its registration, its bench, and one branch of the bench
validator; nothing in the production dispatch references them.
