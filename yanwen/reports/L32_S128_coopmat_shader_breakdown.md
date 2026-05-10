# L=32 seq=128 GLSL shader breakdown — `linear_coopmat` variant

**Source:** `yanwen/artifacts/L32_coopmat/S128.etdp` (and `S128.events.tsv`)
**Run:** `--num_executions 8 --etdump-analyze`, steady-state = avg of iter 1..7
**Pavan-report branch** (where the `linear_coopmat.glsl` + dispatch logic live)
**Compile flag in partitioner:** `storage_type_override = VkStorageType.BUFFER`
**Total GPU time per forward (steady-state):** **586.2 ms** — matches wallclock 582.6 ms within timer noise
**Total GPU time iter 0 (cold):** 1357.8 ms (down from baseline's 1914.8 ms)
**Total dispatches per forward:** 566 (across 31 unique kernels)

Companion to [`L32_S128_shader_breakdown.md`](L32_S128_shader_breakdown.md) (linear_vec baseline). Read that first if you want the naming-convention key — same convention applies here.

## Top shaders (% of steady GPU time)

| # | Runtime name | Steady % | Steady ms | GLSL source | Variant decoded |
|---:|---|---:|---:|---|---|
| 1 | **`linear_coopmat_half`** | **68.0%** | **398.58** | `linear_coopmat.glsl` + `linear_coopmat.yaml` (in **pavan-report** tree) | KHR cooperative matrix path; activations=**buffer**, weights=**buffer**, dtype=**half**. Used for **224 / 226** linear dispatches. |
| 2 | `binary_mul_buffer_float` | 4.1% | 24.10 | `binary_op_buffer.glsl` | storage=buffer, op=`X*Y`, dtype=float |
| 3 | `view_buffer_half` | 3.9% | 22.62 | `view_buffer.glsl` | storage=buffer, dtype=half |
| 4 | `view_buffer_float` | 3.2% | 18.86 | `view_buffer.glsl` | storage=buffer, dtype=float |
| 5 | **`linear_vec_buffer_buffer_half`** | **2.2%** | **13.17** | `linear_vec.glsl` (main tree, falls back to vec) | **lm_head [1, 128256]** — M=1 < 64 so coopmat doesn't apply. Same kernel as in baseline run, ~same time. |
| 6 | `view_convert_buffer_half_float` | 2.2% | 13.09 | `view_convert_buffer.glsl` | storage=buffer, half→float upcast |
| 7 | `view_convert_buffer_float_half` | 1.9% | 10.85 | `view_convert_buffer.glsl` | storage=buffer, float→half downcast |
| 8 | `buffer_to_nchw_float_float` | 1.7% | 9.90 | `buffer_to_nchw.glsl` | layout unpack |
| 9 | `binary_mul_buffer_half` | 1.6% | 9.23 | `binary_op_buffer.glsl` | storage=buffer, op=`X*Y`, dtype=half |
| 10 | `permute_buffer_half` | 1.5% | 8.99 | `permute_buffer.glsl` | storage=buffer, dtype=half |
| 11 | `nchw_to_buffer_float_float` | 1.5% | 8.97 | `nchw_to_buffer.glsl` | layout pack |
| 12 | `sigmoid_float_buffer` | 1.2% | 6.79 | (unary activation) | storage=buffer, dtype=float (SwiGLU) |
| 13 | `softmax_buffer_float` | 1.1% | 6.65 | `softmax_buffer.glsl` | storage=buffer, dtype=float |
| 14 | `expand_buffer_half` | 1.0% | 6.11 | `expand_buffer.glsl` | storage=buffer, dtype=half |
| 15 | `binary_add_buffer_float` | 0.7% | 4.01 | `binary_op_buffer.glsl` | storage=buffer, op=`X+Y`, dtype=float (residual) |
| 16 | `slice_buffer_float` | 0.6% | 3.79 | (slice op using `slice.glslh`) | storage=buffer, dtype=float |
| 17 | `binary_add_buffer_half` | 0.5% | 2.89 | `binary_op_buffer.glsl` | storage=buffer, dtype=half |
| 18 | `permute_buffer_float` | 0.5% | 2.83 | `permute_buffer.glsl` | storage=buffer, dtype=float |
| 19 | `nchw_to_buffer_half_half` | 0.5% | 2.83 | `nchw_to_buffer.glsl` | layout pack, half→half |
| 20 | `mean_per_row_buffer_float` | 0.5% | 2.73 | (RMSNorm reduction) | storage=buffer, dtype=float |
| 21 | `buffer_to_nchw_half_half` | 0.4% | 2.57 | `buffer_to_nchw.glsl` | layout unpack, half→half |
| 22 | `where_buffer_float` | 0.4% | 2.27 | `where.glsl` | storage=buffer, dtype=float (causal mask) |
| 23 | `binary_sub_buffer_float` | 0.3% | 1.68 | `binary_op_buffer.glsl` | storage=buffer, op=`X-Y`, dtype=float |
| 24 | **`matmul_coopmat_float`** | **0.2%** | **0.89** | `matmul_coopmat.glsl` + `matmul_coopmat.yaml` (in **pavan-report**) | **NEW path for attention BMMs** (`Q@Kᵀ`, `attn@V`). 64 dispatches, replacing baseline's `matmul_vec_*`. |

(31 unique kernels total; the table covers ~98% of steady GPU time.)

## The headline shader: `linear_coopmat.glsl`

```
backends/vulkan/runtime/graph/ops/glsl/linear_coopmat.glsl    ← in pavan-report tree
backends/vulkan/runtime/graph/ops/glsl/linear_coopmat.yaml
```

Runtime selection happens in `backends/vulkan/runtime/graph/ops/impl/Linear.cpp` at `add_linear_coopmat_node()` (line 291). Activation conditions:

```cpp
bool use_coopmat =
    !getenv("VK_DISABLE_COOPMAT") &&            // env override (off by default)
    supports_cooperative_matrix() &&            // KHR extension on adapter
    storage_type_of(out) == kBuffer &&          // forced via partitioner config
    M >= 64;                                    // GEMV-style fallback below this
```

The variant suffix (`_half`) comes from the `DTYPE` parameter in `linear_coopmat.yaml`. Storage is **buffer** for both activations and weights — the cooperative-matrix shader assumes flat `VkBuffer` access patterns, not texture sampling.

## Linear dispatches by output shape (coopmat)

| Output shape | # disp | sum ms | avg ms | Kernel | Component | Per-shape speedup vs baseline |
|---|---:|---:|---:|---|---|---:|
| `[128, 14336]` | 64 | 255.9 | **4.00** | `linear_coopmat_half` | FFN gate + up (`32 × 2`) | **3.08×** (vs 12.3 ms) |
| `[128, 4096]` | 96 | 203.7 | **2.12** | `linear_coopmat_half` | Q + O + FFN down (`32 × 3`) | **3.16×** (vs 6.7 ms) |
| `[128, 1024]` | 64 | 20.7 | **0.32** | `linear_coopmat_half` | K + V (`n_kv_heads=8 × head_dim=128`) | **3.75×** (vs 1.2 ms) |
| `[1, 128256]` | 2 | 13.4 | 6.69 | `linear_vec_buffer_buffer_half` (FALLBACK) | lm_head | 1.0× (M=1, doesn't go through coopmat) |

Three observations:

1. The three GEMM shapes that LLaMA actually uses cluster around 3× speedup, with K/V (the smallest output dim) seeing the largest 3.75×. This is consistent with cooperative-matrix tiling being more cache-friendly when the output tile fits in fewer rows.
2. The lm_head fallback to `linear_vec_buffer_buffer_half` is unchanged from baseline (~6.6 ms avg, 13.4 ms total for 2 dispatches). Total cost of the fallback: **2.7%** of total linear time. Not worth fixing.
3. **Total linear time: 493.7 ms (coopmat) vs 1527.6 ms (baseline) = 3.09× speedup on linears**, very close to the 3.03× whole-forward speedup.

## Bucketed by op family — coopmat vs baseline

| Family | Coopmat steady ms | Coopmat % | Baseline steady ms | Baseline % | Per-family speedup |
|---|---:|---:|---:|---:|---:|
| matmul / linear | 412.65 | **70.4%** | 1515.9 | 89.7% | **3.67×** |
| layout / copy / slice | 124.97 | 21.3% | 114.5 | 6.8% | 0.92× (slightly slower) |
| binary (add/mul/etc) | 41.91 | 7.2% | 41.2 | 2.4% | 0.98× (same) |
| softmax | 6.65 | 1.1% | 6.7 | 0.4% | 1.00× |

Only the matmul/linear family changed. The composition of "what fraction of forward is in linears" went from 89.7% to 70.4% **as a side effect of the linear shrink** — the absolute time on layout/binary/softmax is unchanged.

## What's NOT used (storage-driven changes vs baseline)

The baseline run used **`linear_vec_buffer_texture2d_half`** (weights as `texture2d`). With `storage_type_override=BUFFER`, the partitioner forces **all activations and all weights to buffer storage**. Side effects:

- **`linear_vec_buffer_texture2d_half` not dispatched.** Replaced by `linear_coopmat_half` for 224 of the 226 linears.
- **`matmul_vec_texture3d_float` not dispatched.** Replaced by `matmul_coopmat_float` for 32 of the 32 attention BMMs (15.6× faster in absolute time, but tiny share of forward at S=128).
- **No `nchw_to_image_*` / `image_to_nchw_*` for textures.** Confirmed absent from the kernel list — there's nothing left to convert to a texture image.

What's still around:

- **`linear_vec_buffer_buffer_half`** for the lm_head (unchanged, M=1 fallback).
- All the `view_*`, `permute_buffer_*`, `nchw_to_buffer_*`, `buffer_to_nchw_*` glue shaders. Same 12-segment delegate split, same buffer-storage conversions at boundaries. Their cost is roughly the same (within ~5%) as in baseline.

## Iter 0 vs steady (cold-start cost)

| | Iter 0 | Steady avg | Cold delta |
|---|---:|---:|---:|
| Coopmat total | 1357.8 ms | 586.2 ms | +131% |
| Baseline total | 1914.8 ms | 1689.5 ms | +13% |
| **`linear_coopmat_half`** alone | 1052.3 ms | 398.58 ms | **+164%** |

The coopmat shader has a much larger cold-start penalty as a fraction (164% over steady, vs the baseline `linear_vec`'s ~14%). Likely cause: cooperative-matrix shaders use larger workgroups (512 threads in the GLSL header) and more LDS, so first-dispatch shader compilation + descriptor set creation + first GTT touch is heavier. Doesn't change the steady-state result — that's exactly why the calibration-subtraction methodology subtracts iter 0.

## Where to optimize next (post-coopmat)

Linear is now 70% of forward, down from 90%. The next levers:

| Lever | Steady ms today | Potential | Notes |
|---|---:|---|---|
| Make `aten.cat` (KV concat) Vulkan | 78 ms (CPU) | Recovers ~13% of forward | Removes CPU↔GPU sync per layer; previously listed as #2 priority pre-coopmat, now MORE important relatively |
| Investigate `ETVK_COPY_OUTPUTS` 75 ms | 75 ms | Recovers ~13% of forward | 256 KB at 3.4 GB/s on UMA is anomalous; likely a staging buffer round-trip |
| Consolidate `view_*` / `view_convert_*` (3272 dispatches) | ~57 ms combined | Recovers a few % | Fragmented at delegate boundaries; consolidation may not be free |
| Layout/copy reduction at delegate boundaries | ~125 ms total | Up to 21% of forward | Likely needs partitioner-side work to merge sub-graphs |

Going below ~415 ms (a perfect-linear floor) requires touching the non-linear path. **Coopmat alone got us 3×; further gains need a different attack.**

## How to regenerate this breakdown

```bash
cd /home/doremy/sarc-acl/executorch/main/executorch
source /home/doremy/sarc-acl/executorch/pavan-report/executorch/.venv/bin/activate

# Categories + top 15 ops:
python /home/doremy/sarc-acl/executorch/pavan-report/executorch/yanwen_plan/analyze_etdump.py \
    yanwen/artifacts/L32_coopmat/S128.etdp

# Per-shader steady-state (iter 1..N-1):
python yanwen/scripts/etvk_breakdown.py \
    yanwen/artifacts/L32_coopmat/S128.events.tsv 8

# Linear dispatches by output shape:
python yanwen/scripts/linear_by_shape.py \
    yanwen/artifacts/L32_coopmat/S128.etdp
```
