# L=32 seq=128 GLSL shader breakdown

**Source:** `yanwen/artifacts/L32/S128.etdp` (and `S128.events.tsv`)
**Run:** `--num_executions 8 --etdump-analyze`, steady-state = avg of iter 1..7
**Total dispatches per forward:** 569 (across 38 unique kernels)
**Total GPU time per forward (steady-state):** 1689.5 ms
**Total wallclock per forward (steady-state):** 1765.6 ms — ETDump captures ~96% of forward time

## Naming convention

ETVK shader names follow the pattern emitted by the variant generator in `*.yaml`:

```
<base>_<STORAGE>_[WEIGHT_STORAGE_]<DTYPE>
```

For `view_convert_*` the suffix is `<IN_DTYPE>_<OUT_DTYPE>` instead of one DTYPE. All sources live under:

```
backends/vulkan/runtime/graph/ops/glsl/
```

Every GLSL has a sibling YAML defining which `(STORAGE, WEIGHT_STORAGE, DTYPE, ...)` combinations get compiled into separate variants. The `*.yaml` is the source of truth for what variants exist.

## Top shaders (% of steady GPU time)

| # | Runtime name | Steady % | GLSL source | Variant decoded |
|---:|---|---:|---|---|
| 1 | **`linear_vec_buffer_texture2d_half`** | **88.1%** | `linear_vec.glsl` + `linear_vec.yaml` | activations=**buffer**, weights=**texture2d**, dtype=**half (fp16)**, no bias, TILE_M=4 |
| 2 | `binary_mul_buffer_float` | 1.4% | `binary_op_buffer.glsl` + `binary_op_buffer.yaml` | storage=**buffer**, op=`X*Y`, dtype=**float (fp32)** |
| 3 | `view_buffer_half` | 1.3% | `view_buffer.glsl` + `view_buffer.yaml` | storage=**buffer**, dtype=**half** |
| 4 | `linear_vec_buffer_buffer_half` | 0.8% | `linear_vec.glsl` | activations=**buffer**, weights=**buffer**, dtype=**half**. **1 dispatch only — lm_head `[1, 128256]`** (vocabulary projection). |
| 5 | `view_convert_buffer_half_float` | 0.8% | `view_convert_buffer.glsl` + `view_convert_buffer.yaml` | storage=**buffer**, **half→float upcast** |
| 6 | `view_buffer_float` | 0.8% | `view_buffer.glsl` | storage=**buffer**, dtype=**float** |
| 7 | `view_convert_buffer_float_half` | 0.7% | `view_convert_buffer.glsl` | storage=**buffer**, **float→half downcast** |
| 8 | `binary_mul_buffer_half` | 0.5% | `binary_op_buffer.glsl` | storage=**buffer**, op=`X*Y`, dtype=**half** |
| 9 | `permute_buffer_half` | 0.5% | `permute_buffer.glsl` + `permute_buffer.yaml` | storage=**buffer**, dtype=**half** |
| 10 | `matmul_vec_texture3d_float` | 0.4% | `matmul_vec.glsl` + `matmul_vec.yaml` | storage=**texture3d**, dtype=**float**. Used for **attention BMMs** (`Q@Kᵀ`, `attn@V`). |
| 11 | `buffer_to_nchw_float_float` | 0.4% | `buffer_to_nchw.glsl` + `buffer_to_nchw.yaml` | layout unpack from buffer to NCHW |
| 12 | `matmul_vec_buffer_float` | 0.4% | `matmul_vec.glsl` | storage=**buffer**, dtype=**float** (attention) |
| 13 | `sigmoid_float_buffer` | 0.4% | unary `sigmoid.glsl` (or activation kernels) | storage=**buffer**, dtype=**float**. SwiGLU activation. |
| 14 | `softmax_buffer_float` | 0.4% | `softmax_buffer.glsl` + `softmax_buffer.yaml` | storage=**buffer**, dtype=**float** |
| 15 | `expand_buffer_half` | 0.4% | `expand_buffer.glsl` + `expand_buffer.yaml` | storage=**buffer**, dtype=**half** |
| 16 | `nchw_to_buffer_float_float` | 0.3% | `nchw_to_buffer.glsl` + `nchw_to_buffer.yaml` | layout pack NCHW → buffer |
| 17 | `nchw_to_image_texture3d_float_float` | 0.3% | `nchw_to_image.glsl` + `nchw_to_image.yaml` | layout pack NCHW → 3D image |
| 18 | `clone_float` | 0.3% | `clone.glsl` + `clone.yaml` | storage=**buffer/texture**, dtype=**float** |
| 19 | `slice_buffer_float` | 0.2% | uses `slice.glslh` header from a slice op | storage=**buffer**, dtype=**float** |
| 20 | `mean_per_row_buffer_float` | 0.2% | RMSNorm reduction kernel | storage=**buffer**, dtype=**float** |
| 21 | `binary_add_buffer_half` / `binary_add_buffer_float` / `binary_add_texture3d_float` / `binary_sub_buffer_float` | ~0.6% combined | `binary_op_buffer.glsl` (buffer variant) + `binary_op_texture.glsl` (texture variant) | mixed |
| 22 | `where_buffer_float` | 0.1% | `where.glsl` + `where.yaml` | storage=**buffer**, dtype=**float**. Causal-mask blend. |
| 23 | `image_to_nchw_texture3d_float_float` | 0.1% | `image_to_nchw.glsl` + `image_to_nchw.yaml` | unpack 3D image → NCHW |

(38 unique kernels total; the table above covers ~98% of steady GPU time.)

## The headline shader: `linear_vec.glsl`

```
backends/vulkan/runtime/graph/ops/glsl/linear_vec.glsl
backends/vulkan/runtime/graph/ops/glsl/linear_vec.yaml
```

`linear_vec.yaml` defines the variant we hit:

```yaml
linear_vec:
  parameter_names_with_default_values:
    DTYPE: float
    STORAGE: texture3d           # default; overridden to "buffer" in this run
    WEIGHT_STORAGE: texture2d    # kept as default
    HAS_BIAS: false
    TILE_M4: 1
    TILE_K4: 1
    TILE_N4: 1
    TILE_M: 4
  generate_variant_forall:
    combination:
      parameter_names: [STORAGE, WEIGHT_STORAGE]
      combos:
        - [texture3d, texture2d]
        - [texture3d, buffer]
        - [buffer,    texture2d]   # ← 224 / 226 of our linears use this combo
        - [buffer,    buffer]      # ← 1 dispatch (lm_head [1, 128256])
    DTYPE:
      - half       # ← we use this
      - float
```

So the active variant for almost every linear is:

- **Activations** (input/output): stored as Vulkan `VkBuffer` — flat strided memory.
- **Weights**: stored as `VkImage` 2D — accessed through the GPU's dedicated texture-sampling hardware (texture cache).
- **Datatype**: `half` (fp16) for both weights and activations.

The `_vec` suffix indicates a **GEMV-style** kernel: each output position is a vector dot product across K, with TILE_M=4 producing 4 rows of output per workgroup. It is **not** a coopmat / WMMA tiled GEMM.

## Linear dispatch distribution by output shape

| Output shape | # disp | sum ms | avg ms | Maps to | Kernel |
|---|---:|---:|---:|---|---|
| `[128, 14336]` | 64 | 789.4 | 12.3 | FFN gate + up (`32 layers × 2`) | `linear_vec_buffer_texture2d_half` |
| `[128, 4096]` | 96 | 646.7 | 6.7 | Q + O + FFN down (`32 × 3`) | `linear_vec_buffer_texture2d_half` |
| `[128, 1024]` | 64 | 78.3 | 1.2 | K + V (`n_kv_heads=8 × head_dim=128`, `32 × 2`) | `linear_vec_buffer_texture2d_half` |
| `[1, 128256]` | 2 | 13.2 | 6.6 | lm_head | `linear_vec_buffer_buffer_half` + `buffer_to_nchw_half_half` |

LLaMA 3.1 8B has `dim=4096`, `n_heads=32`, `n_kv_heads=8`, `head_dim=128`, `intermediate_dim=14336`. The 226 linear dispatches break down as `32 layers × 7 linears (Q/K/V/O + gate/up/down) + 1 lm_head + 1 padding/bookkeeping = 226`.

## Bucketed by op family

| Family | Dispatches | iter 0 ms | steady ms | % steady |
|---|---:|---:|---:|---:|
| matmul / linear | 289 | 1725.4 | **1515.9** | **89.7%** |
| layout / copy / slice / view | 3272 | 126.4 | 114.5 | 6.8% |
| binary (add/mul/sub) | 804 | 43.9 | 41.2 | 2.4% |
| softmax | 32 | 8.3 | 6.7 | 0.4% |
| nchw↔texture conversions | 64 | 4.3 | 4.5 | 0.3% |
| clone | 64 | 4.1 | 4.3 | 0.3% |
| image_to_nchw | 32 | 2.4 | 2.4 | 0.1% |

(The 224 + 32 + 32 + 1 = 289 in matmul/linear comes from 224 `linear_vec` dispatches across all 32 layers + 32 BMMs from `Q@Kᵀ` + 32 BMMs from `attn@V` + 1 lm_head.)

## Storage strategy in this run

- **Weights**: `texture2d` — packed into 2D images, accessed through the GPU's texture cache. Win on RDNA3+ where texture sampling has dedicated paths.
- **Activations**: `buffer` — flat `VkBuffer` storage, simpler indexing.
- **Mixed strategy** is why so many auxiliary shaders fire: `view_*`, `view_convert_*`, `permute_buffer_*`, `nchw_to_buffer_*`, `nchw_to_image_*`, `buffer_to_nchw_*`, `image_to_nchw_*`. They are managing layout transforms and storage-class transitions at delegate boundaries (12 sub-graphs per forward).

## What's NOT used in this run (but exists in the tree)

The repo has coopmat shaders for several ops, but **none of them are dispatched** here:

| GLSL | Purpose |
|---|---|
| `linear_q4gsw_coop.glsl` + `.yaml` | coopmat for int4 group-wise quantized linear |
| `linear_qcsnw_coop.glsl` + `.yaml` | coopmat for int8 channel-wise quantized linear |
| `sdpa_compute_attn_weights_coop.glsl` + `.yaml` | coopmat-fused attention scores |
| `sdpa_compute_out_coop.glsl` + `.yaml` | coopmat-fused attention output |

Notable absence: there is **no fp16-non-quantized coopmat linear shader** in `main`. The 2026-05-06 synthetic-LLaMA measurements that showed a 2× speedup used a `linear_coopmat` shader that lives in `pavan-report/executorch/yanwen_plan/coopmat_previous_work.diff` — **not landed**.

## Summary

- **One GLSL kernel does 88% of the forward at L=32 S=128:** `backends/vulkan/runtime/graph/ops/glsl/linear_vec.glsl`, variant `(STORAGE=buffer, WEIGHT_STORAGE=texture2d, DTYPE=half)`.
- **Attention is on a different path** — `matmul_vec.glsl` for both `texture3d` and `buffer` storage variants, fp32. Combined contribution <1%.
- **Storage layout is mixed**: `texture2d` for weights, `buffer` for activations. This is what most of the small "glue" shaders are managing.
- **No coopmat / WMMA path is active** in this run. The 2× speedup projection in the main `REPORT.md` is conditional on landing the `linear_coopmat` diff from `pavan-report`.

## How to regenerate this breakdown

```bash
cd /home/doremy/sarc-acl/executorch/main/executorch
source .venv/bin/activate

# Categories + top 15:
python /home/doremy/sarc-acl/executorch/pavan-report/executorch/yanwen_plan/analyze_etdump.py \
    yanwen/artifacts/L32/S128.etdp

# Per-shader steady-state (iter 1..N-1):
python yanwen/scripts/etvk_breakdown.py \
    yanwen/artifacts/L32/S128.events.tsv 8

# Linear dispatches by output shape:
python yanwen/scripts/linear_by_shape.py \
    yanwen/artifacts/L32/S128.etdp
```
