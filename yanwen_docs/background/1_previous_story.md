# The KHR Cooperative Matrix iGPU story

## Cast of characters

- **Stephen Jia (ssjia)** authored ExecuTorch's Vulkan tiled fp16/fp32 GEMM
  shaders: `linear_vec.glsl` and `matmul_vec.glsl`. They run on either
  texture3d or buffer storage and are the existing baseline.
- **Yanwen Xu (this work)** added a `VK_KHR_cooperative_matrix` variant —
  `linear_coopmat.glsl` and `matmul_coopmat.glsl` — that issues the WMMA
  intrinsics (`coopMatLoad` / `coopMatMulAdd` / `coopMatStore`). Today this
  variant requires buffer-storage input/output because `coopMatLoad` only
  accepts buffer-reference pointers.
- **Pavan / SARC-ACL** owns the deployment target — Samsung mobile GPU class
  HW, where the FP16 + INT8 WMMA paths matter.

## The thesis the work was started under

Modern GPUs expose hardware matrix-multiply tiles (Tensor Cores on NVIDIA,
WMMA on AMD/Intel) that should give 3–4× throughput on compute-bound GEMM.
ExecuTorch's Vulkan backend was leaving that throughput on the table by
using software tiling everywhere. Add a `VK_KHR_cooperative_matrix` shader
gated on device support, transparently dispatch on capable hardware, fall
back cleanly otherwise — get the kernel speedup, watch it transfer to E2E.

## Round 1 — the discrete-GPU disappointment

We measured on a Radeon RX 7900 XTX (RDNA3, dedicated card) running real
LLaMA 3.1 8B (4 of 32 layers, fp16, seq_len=2048 prefill).

| Layer                           | Result                                              |
| ------------------------------- | --------------------------------------------------- |
| Microbench (kernel only)        | **2.88× – 3.79×** speedup, peak 17.8 TFLOP/s        |
| E2E linear-only ETDump category | 248 ms → 81 ms = **3.07× kernel win**               |
| **E2E wallclock**               | 1847 ms → 1658 ms = **1.114× — barely above noise** |

The kernel speedup was real. It just couldn't transfer because PCIe copy
dominated wallclock at 52% — every forward pass shovelled activations
between system RAM and 24 GB VRAM nine times across the model's nine
Vulkan subgraphs (split by CPU-fallback ops: `embedding`, `eq.Scalar`,
`logical_not`, `any.dim`, `mul.Scalar`, `cat`).

Amdahl: shrink a 12% slice (linear) by 3× and the wallclock barely moves.

The discrete report (`yanwen_plan/final_report_real_llama.md`) flagged this
as: "the real E2E lever is not kernel perf — it's eliminating subgraph
splits."

## Round 2 — the iGPU hypothesis

The premise we wanted to test: on a system with **unified memory**, PCIe
copy collapses to nothing. The 12% linear slice grows toward 90%. Now the
3× kernel speedup IS the wallclock speedup. The right test is real LLaMA
on an integrated GPU.

System: **AMD Radeon 780M (RDNA3+ Phoenix APU)**, 28 GB shared system RAM,
Mesa 25.0.7 / RADV. Confirmed: `VK_KHR_cooperative_matrix` extension rev 2,
14 supported tile configs (16×16×16 fp16/int8/uint8 → fp16/fp32/int32),
subgroupSize=64.

## What we did this session

### 1. Branch + builds

Created `dev-yanwen-coopmat-igpu-bench` from
`dev-yanwen-coopmat-llama-real`. Two cmake builds:

- `cmake-out-vk` — production runtime + microbench binaries
- `cmake-out-vk-etdump` — runtime with `EXECUTORCH_ENABLE_EVENT_TRACER=ON`
  and `EXECUTORCH_BUILD_DEVTOOLS=ON` for ETDump capture

### 2. Microbench (kernel only, GPU-timestamped)

Used the existing `linear_coopmat_bench` and `matmul_coopmat_bench` C++
binaries that compare `linear_vec` vs `linear_coopmat` per shape, and
similarly for matmul. 3 warmup + 10 timed iterations per case, all
correctness PASSED.

| Shape                         | linear_vec (tex) | linear_coopmat (buf) |   Speedup |
| ----------------------------- | ---------------: | -------------------: | --------: |
| 256×768→3072 (BERT FFN up)    |          3305 µs |               805 µs | **4.11×** |
| 256×3072→768 (BERT FFN dn)    |          3155 µs |               852 µs | **3.70×** |
| 32×4096→4096 (LLM Q/K/V b=32) |          4067 µs |              1485 µs | **2.74×** |
| 256×4096→4096 (sq_4096)       |         17429 µs |              5635 µs | **3.09×** |
| 1×4096→4096 (LLM decode)      |          1451 µs |              1473 µs |     0.99× |
| 1×4096→11008 (FFN-up decode)  |          2968 µs |              3761 µs |   0.79× ⚠ |

Peak throughput on the linear path: **1524 GFLOP/s** (fp32).
matmul fp16 path peaked at **4626 GFLOP/s** on sq_4096 (7.63× over
matmul_vec) — the cm_fp16 shader uses the native fp16 accumulator config
the chip exposes.

**Decode (M=1) is at best a wash on this iGPU** — different from the
discrete GPU's 1.74×–2.19× decode wins. Worth a per-shape gate on
production paths.

### 3. E2E synthetic LLaMA — the iGPU promise validated

Re-ran the existing fp32 4-layer synthetic-LLaMA `.pte` files (LLaMA 3.1
shapes, attention simplified to projections-only, no actual softmax /
bmm / mask). 5 warmup + 10 timed runs.

| Variant                                 |   Wallclock |   Linear ETDump |       Speedup |
| --------------------------------------- | ----------: | --------------: | ------------: |
| `linear_vec` (tex3d, baseline)          | **4356 ms** | 3896 ms (95.8%) |             — |
| `linear_coopmat` (buffer)               | **2333 ms** | 1841 ms (91.3%) |    **1.867×** |
| Sanity: buffer + `VK_DISABLE_COOPMAT=1` |     4389 ms |         2744 ms | 1.007× of tex |

The iGPU thesis worked. Linear became 91% of wallclock (instead of 5%
on discrete), the 2.12× linear-category kernel speedup transferred almost
completely to the 1.87× E2E speedup, and the disabled-coopmat sanity check
proved the win was the shader (4389 ≈ 4356 — buffer storage on linear_vec
is essentially free on this synthetic graph).

This matches the prior iGPU observation in
`yanwen_plan/igpu_shader_breakdown.md` (1.99×) within noise.

### 4. E2E real LLaMA — the surprise

Then we ran the same comparison on real Meta LLaMA 3.1 8B weights from
HuggingFace, 4 of 32 layers, fp16, seq=2048, full attention with
softmax/bmm/causal mask.

Dispatch confirmed correct: **28× `linear_coopmat` + 1× `linear_vec`** on
the LM head (M=1, hits the `M >= 64` guard).

| Variant                         |    Wallclock | Linear ETDump |      Speedup |
| ------------------------------- | -----------: | ------------: | -----------: |
| `linear_vec` (tex3d)            |  **5984 ms** | 3192 ms (52%) |            — |
| `linear_coopmat` (buffer)       | **16131 ms** |    **867 ms** | **0.370×** ⚠ |
| Buffer + `VK_DISABLE_COOPMAT=1` |     17715 ms |       2744 ms |       0.338× |

The kernel got faster — 3192 ms → 867 ms is a **3.68× linear-category win**
on real LLaMA, even bigger than synthetic. And **wallclock got 2.7× SLOWER**.

The disabled-coopmat sanity check is the real story: buffer storage on the
*same* linear_vec shader runs **2.96× slower than texture3d storage** for
this whole graph. Coopmat actually beat linear_vec by 1.10× *within* the
buffer path (16131 < 17715). The 2300 ms of kernel savings just couldn't
overcome the buffer-storage tax that the rest of the graph paid.

### 5. Per-shader ETDump breakdown — where the time actually goes

Dumped via `cmake-out-vk-etdump/executor_runner --etdump_path=...`,
analyzed with a leaf-event aggregator. The top lines tell the story:

**TEX3D + linear_vec (5984 ms wallclock, 6123 ms leaf sum):**

| Shader                                 |        Time |   Share | #disp |
| -------------------------------------- | ----------: | ------: | ----: |
| `linear_vec_texture3d_texture2d_half`  | **3179 ms** | **52%** |    28 |
| `ETVK_COPY_OUTPUTS`                    |      941 ms |     15% |    13 |
| `ETVK_COPY_INPUTS`                     |      425 ms |      7% |    13 |
| `matmul_vec_texture3d_half` (attn bmm) |      389 ms |      6% |     8 |
| CPU fallback (cat/eq/etc)              |     ~478 ms |      8% |    41 |
| `softmax_texture3d_half`               |       63 ms |      1% |     4 |
| ...                                    |             |         |       |

**BUFFER + linear_coopmat (16131 ms wallclock, 5053 ms leaf sum):**

| Shader                                        |        Time |   Share | #disp | vs tex              |
| --------------------------------------------- | ----------: | ------: | ----: | ------------------- |
| **`ETVK_COPY_INPUTS`**                        | **1619 ms** | **32%** |    13 | **+3.8×** ⚠         |
| `linear_coopmat_half`                         |      854 ms |     17% |    28 | **−73%**            |
| `ETVK_COPY_OUTPUTS`                           |      709 ms |     14% |    13 | −25%                |
| **`softmax_buffer_float`**                    |  **338 ms** |  **7%** |     4 | **+5.4×** ⚠         |
| `matmul_vec_texture3d_float`                  |      244 ms |      5% |     4 | +similar            |
| CPU fallback                                  |     ~371 ms |      7% |    41 | unchanged           |
| `nchw_to_buffer_float_float`                  |      151 ms |      3% |    12 | new                 |
| `view_buffer_float`                           |       90 ms |      2% |    56 | new                 |
| `where_buffer_float` (causal mask)            |       71 ms |      1% |     4 | new (was 54 ms tex) |
| `binary_mul_buffer_float`                     |       47 ms |      1% |    54 | similar             |
| `view_convert_buffer_*` (storage transitions) |       53 ms |      1% |    94 | new                 |
| `permute_buffer_*`                            |       48 ms |      1% |    28 | new (was 17 ms tex) |
| ...                                           |             |         |       |

The two parasites that show up clearly:

1. **`ETVK_COPY_INPUTS` blew up 3.8×** (425 → 1619 ms). On unified memory
   there's no PCIe; this is host-side cache-coherency / NCHW reformat per
   subgroup boundary against host-visible buffer memory. Buffer storage
   triggers it; texture storage doesn't.
2. **softmax 5.4× slower on buffer** (63 → 338 ms). Same op, same math,
   just the storage flag flipped. Buffer-stored fp16 softmax loses the
   texture cache's row-coalescing on the per-row reduction.

But these two together account for ~1500 ms of regression. The full
wallclock regression is **~10 seconds** more than that. That's the
wallclock-vs-leaf-event-sum gap:

| Variant                   | Wallclock | Leaf sum |         **Gap** |
| ------------------------- | --------: | -------: | --------------: |
| TEX3D                     |      5984 |     6123 |   **−139 ms** ✓ |
| BUFFER coopmat            |     16131 |     5053 | **+11078 ms** ⚠ |
| BUFFER VK_DISABLE_COOPMAT |     17715 |     6564 | **+11151 ms** ⚠ |

The +11s is host-side `vkQueueSubmit` / `vkWaitForFence` / coherency
overhead at every Vulkan command submission, paid by *any* graph that
uses buffer storage on this iGPU. It's not categorized as a leaf event;
it lives between dispatches.

### 6. Approach A — patch the partitioner to keep non-linear ops on texture

Hypothesis: if we keep linear/matmul on buffer but force everything else
to texture3d, the per-op buffer-storage tax (softmax, elementwise) goes
away while coopmat still dispatches on linears.

Implementation: patched site-packages `tag_memory_meta_pass.py` and
`vulkan_preprocess.py` with a new compile option `coopmat_isolated_buffer`.
When set:

- linear/matmul args constrained to `CONTIGUOUS_BUFFER` (early-return in
  `constrain_op_arg_repset`)
- linear/matmul outputs forced `CONTIGUOUS_BUFFER` (early-return in
  `constrain_op_out_repset`)
- consumers downstream skip coopmat-isolated nodes when tracing
  (modified `constrain_repset_with_user`)

**v1 result:** 16465 ms — basically identical to whole-graph buffer
(16131 ms). The reason: `constrain_op_arg_repset` calls
`get_arg_tensor_source_repset` to seed each arg's repset from its
producer. When the producer is a linear with BUFFER output, that BUFFER
preference propagates forward through `try_constrain_with_arg_repset` to
*all* downstream consumers — they all end up buffer too.

**v2 attempt:** also hide linear's BUFFER output from non-coopmat
consumers in `get_arg_tensor_source_repset`. Result: only 5 of 28 linears
dispatched coopmat, output partly garbage (constant 1.87695 in second half
of LM head logits). Reverted. Hiding the producer's source repset means
the consumer's required arg never gets constrained to TEXTURE — so the
partitioner has nothing to insert transitions against, and the resulting
graph fails partition.

**The deeper finding:** Approach A targets per-op storage tax. ETDump
shows buffer's per-op time is actually faster (5053 ms total vs 6123 ms
for tex3d). The 11s wallclock gap is host-side and lives between leaf
events. **No per-op storage choice can fix that gap.**

Saved findings: `yanwen_plan/igpu_results/approach_a_findings.log`.

### 7. Approach B — rewrite the shader for texture3d I/O

If the buffer-storage tax is structural on this iGPU, the right fix is
to never put activations in buffer at all. Approach B: rewrite
`linear_coopmat.glsl` so its input/output bindings are `texture3d`
(weights stay buffer-prepacked, since `coopMatLoad` requires a buffer
pointer for them).

The encouraging part: the shader **already** stages through shared memory.
`Ash` and `Bsh` shared tiles are loaded from buffer first, then
`coopMatLoad` reads from the shared tile. So switching the outer load
loop from `t_mat1[idx]` to `texelFetch(t_mat1, ivec3(k4, m, 0), 0)` is
a binding swap, not a rewrite.

Wrote the new shader (~250 lines, mostly the same as before). Updated
`Linear.cpp:355` dispatch predicate from `kBuffer` to `kTexture3D`.

**The unexpected blocker.** Linear's `inputs_storage` in `op_registry.py`
is `utils.CONTIGUOUS_ANY` — accepts both buffer and texture. The
partitioner picks based on producer. The first op in the real LLaMA
Vulkan partition is `dim_order_ops._to_dim_order_copy.default` (the
fp32→fp16 cast right after the CPU-fallback `embedding`). At the
CPU-fallback boundary, storage defaults to BUFFER. Linear has
`sync_primary_io_repr=True` → its output matches its input → BUFFER.

Result: even with `storage_type_override=TEXTURE_3D` in compile_options,
linear ends up tagged BUFFER. My new predicate (`requires kTexture3D`)
fails, falls through to `linear_vec`. Only 2 of 29 linears dispatched at
all; rest got partitioned out. Reverted shader/predicate to original.

**To make Approach B actually deploy on real LLaMA:**

- **B-narrow:** change `inputs_storage=CONTIGUOUS_ANY` to `ANY_TEXTURE`
  for linear in `op_registry.py`. Forces texture-only; transitions get
  inserted at CPU-fallback boundaries automatically. Risk: breaks any
  workflow that wanted buffer linear.
- **B-dual:** keep both buffer and texture coopmat shaders. Runtime
  picker selects based on output storage. ~2× shader code, no risk to
  other paths. Most production-ready.
- **B-pre-cast:** add an export pass that promotes any buffer-stored
  tensor flowing into a coopmat-eligible linear back to texture3d via an
  explicit clone op. Conceptually similar to Approach A but inserts
  transitions BEFORE linear, not after.

Realistic estimate for any of these: 3–5 days with full real-LLaMA
correctness verification. Out of session budget.

Saved findings: `yanwen_plan/igpu_results/approach_b_findings.log`.

## Key takeaways

1. **The kernel work is good.** Microbench 2.31×–4.51× prefill, peak
   1524 GFLOP/s; matmul fp16 peak 4626 GFLOP/s. ETDump on real LLaMA
   shows linear category goes 3192 → 867 ms = 3.68× whole-category win.
   The shader does its job.

2. **Synthetic-LLaMA was honest about the kernel; not predictive of E2E.**
   The 1.87× synthetic iGPU number reproduces the prior 1.99× claim, but
   the synthetic graph omits softmax / bmm / where / expand_copy — the
   ops that pay the buffer-storage tax on real LLaMA. Always state the
   workload when reporting an E2E speedup.

3. **On AMD 780M, the dispatch model is the bottleneck, not the shader.**
   The whole-graph buffer dispatch costs ~11 s of host-side overhead per
   forward (vkQueueSubmit / vkWaitForFence / cache-coherency on host-
   visible buffer memory) plus per-op buffer-storage taxes (softmax 5.4×,
   ETVK_COPY_INPUTS 3.8×). The kernel saves ~2.3 s. Net: 5984 ms → 16131
   ms, a 0.37× regression on real LLaMA fp16 seq=2048. None of this would
   have shown up in microbench or in the discrete-GPU workflow.

4. **Approach A (per-op storage rules) doesn't fix it.** The bottleneck is
   structural to buffer storage on this device, not per-op compute time.
   Patches were written and explored; v1 didn't help, v2 broke partition.
   Site-packages still carries the (inert) patches; original behavior
   restored when the new compile option isn't set.

5. **Approach B (texture-storage coopmat shader) IS the right answer**
   architecturally, but landing it on real LLaMA requires graph-pass
   surgery beyond a one-shader-rewrite. The shader-side change is small
   (existing shared-memory staging absorbs almost all the work). The
   non-trivial part is breaking buffer-storage propagation from
   CPU-fallback boundary ops into linear.

6. **Decode (M=1) is no longer a free coopmat win on this iGPU class.**
   Microbench: best 1.03× speedup, worst **0.79× regression** (FFN-up).
   The discrete 7900 XTX numbers (1.74×–2.19× at M=1) don't transfer.
   Anyone wiring coopmat into decode on mobile-class APUs should gate
   per-shape.

7. **No INT8/INT4 measurement happened.** Quantized linear paths
   (`linear_q4gsw`, `linear_q8csw`, `q8ta_linear`) are still on
   `linear_vec`. The device's coopmat property table enumerates eight
   int8/uint8 → int32 configs (#2–#13), so the hardware is there; the
   wiring isn't. This is the largest open item for the actual deployment
   target.

## Status / state of the branch

Branch `dev-yanwen-coopmat-igpu-bench`, currently:

- All shader / runtime / Python source files are at clean state (matching
  branch tip `3d773e3e31`). Approach B's experimental shader was
  reverted; backups exist at `linear_coopmat.{glsl,yaml}.bak`.
- `yanwen_plan/run_real_llama_e2e.py` and `yanwen_plan/run_llama_coopmat_e2e.py`
  carry minimal modifications: `--local`, `--executor_runner`, `--cache_dir`,
  `--tmp_dir`, `--etdump_dir`, and `--fp16` flags. Used for the iGPU
  measurements; ready scaffold for any reproduction.
- `yanwen_plan/igpu_results/` has 18+ artifacts: device info, both
  microbench logs, all three real-LLaMA wallclock variants, all three
  ETDump captures + per-shader breakdowns, both Approach findings logs.
- `yanwen_plan/igpu_final_report.md` has the full numerical breakdown.

`.venv/lib/python3.12/site-packages/executorch/backends/vulkan/_passes/tag_memory_meta_pass.py`
and `vulkan_preprocess.py` carry the Approach A patches (inert without
the `coopmat_isolated_buffer` compile option). To fully revert: copy the
`.bak` files alongside back into place.

## Next-action options, in rough priority

1. **Wire INT8/INT4 coopmat into the production quantized linear paths.**
   `MatMulKHRCoopMat.cpp` + `khr_cm_gemm_int8.glsl` already exist (SARC-ACL
   contribution). Attaching them to `linear_q8csw` / `q8ta_linear` is
   probably the highest-value SARC follow-up — that's the deployment dtype.
2. **Pursue Approach B-dual** (separate texture-storage coopmat shader,
   runtime dispatch picker switches by storage). Estimated 3–5 days.
   Validates whether the texture-storage hypothesis actually delivers a
   real-LLaMA E2E win on this iGPU class. The shader itself is mostly
   already written and saved as `.bak`.
3. **Fix the synthetic-fp16 coopmat shader crash** (separate bug surfaced
   in this session — fp16 export on synthetic LLaMA SIGABRTs after ~17
   dispatches). Real LLaMA fp16 doesn't hit it; might be shape-specific.
4. **Add a per-shape decode gate** so coopmat doesn't dispatch on M=1
   FFN-up where it regresses 0.79×.
5. **Document the iGPU buffer-coherency overhead** as a Vulkan backend
   note. Not unique to LLaMA; any graph mixing buffer and texture storage
   will pay it on this APU class.
