# SHADER_DISPATCH — how Vulkan picks which shader runs

This doc explains the decision chain that selects which GLSL shader variant runs at runtime. Understanding this is essential for interpreting bench results and for setting up new experiments correctly.

## Kernel naming convention

ETVK shaders are generated from a `<base>.glsl` template + a `<base>.yaml` that enumerates parameter combinations. The runtime-emitted kernel name follows:

```
<base>_<STORAGE>_[<WEIGHT_STORAGE>_]<DTYPE>
```

Examples:
- `linear_vec_buffer_texture2d_half` → base=`linear_vec`, STORAGE=`buffer` (activations), WEIGHT_STORAGE=`texture2d`, DTYPE=`half`
- `linear_coopmat_half` → base=`linear_coopmat`, DTYPE=`half`. Note: coopmat has NO storage suffix because the shader is hard-coded to buffer storage (no other variants exist in the YAML).
- `view_convert_buffer_half_float` → base=`view_convert`, STORAGE=`buffer`, IN_DTYPE=`half`, OUT_DTYPE=`float` (this base has IN/OUT instead of single DTYPE)
- `binary_mul_buffer_float` → base=`binary_op`, OPERATOR=`mul`, STORAGE=`buffer`, DTYPE=`float`

All GLSL sources are under:
```
backends/vulkan/runtime/graph/ops/glsl/
```

## GLSL → SPIR-V → binary

GLSL files are scanned and compiled to SPIR-V at **CMake configure time** into `cmake-out-vk/vulkan_compute_shaders/spv.cpp` (a C++ source file with embedded SPIR-V byte arrays). The compiled `executor_runner` binary contains all variants from this `spv.cpp`.

**Implication**: adding a new GLSL file to the tree means you must **reconfigure** (not just rebuild). `cmake --build` alone won't pick up new variants. This is why building `pavan-report`'s tree is required to get `linear_coopmat` — main's runner literally doesn't have that SPIR-V in its embedded table.

## Decision chain for fp16 linear (the headline case)

```
1. Partitioner — compile_options
   - VulkanPartitioner({}) → defaults
   - VulkanPartitioner({"storage_type_override": VkStorageType.BUFFER}) → force buffer
        ↓
2. Tagged into ExportedProgram / serialized into .pte at export time
        ↓
3. At runtime, output tensor's storage is what was tagged:
   graph.storage_type_of(out) == kBuffer | kTexture2D | kTexture3D
        ↓
4. add_linear_node() decision site (Linear.cpp ~line 350 in pavan-report):
   use_coopmat = !getenv("VK_DISABLE_COOPMAT")
              && adapter.supports_cooperative_matrix()    // 780M ✓
              && storage_type_of(out) == kBuffer
              && M >= 64                                  // S=128 ✓, lm_head M=1 ✗
        ↓
5. prepack_fp_linear_weight(force_buffer=use_coopmat):
   - if force_buffer → weight_storage = kBuffer
   - else → weight_storage = kTexture2D
            with fallback: if output_width/4 > max_texture2d_dim || output_height > max_texture2d_dim
                          then weight_storage = kBuffer (lm_head case, N=128256)
        ↓
6. Final shader name:
   - use_coopmat=true  → "linear_coopmat" + _half = linear_coopmat_half
   - use_coopmat=false → pick_linear_shader() picks linear_vec, then:
                         <base>_<STORAGE_OUT>_<WEIGHT_STORAGE>_<DTYPE>
                         = linear_vec_buffer_texture2d_half (or _buffer_buffer_half for lm_head)
```

## What today's `main` does by default (no compile_options override)

For fp16 LLaMA linears with `VulkanPartitioner({})`:

| Step | Result |
|---|---|
| Op declaration | `op_registry.py:415-422`: `aten.linear.default` has `inputs_storage=utils.CONTIGUOUS_ANY` |
| Partitioner pass | `pick_representations()` in `utils.py` resolves to **buffer** storage for the activation tensors |
| Prepack | `weight_storage = kTexture2D` default; falls back to `kBuffer` only if `output_width/4 > max_texture2d_dim` (16384 on 780M) |
| Result | activations=`buffer`, weights=`texture2d` for 7/8 linear types |
| lm_head exception | output `[1, 128256]` packed width exceeds 16384 → `weight_storage = kBuffer` |
| Net | `linear_vec_buffer_texture2d_half` for Q/K/V/O + FFN gate/up/down × 32 layers = 224 dispatches; `linear_vec_buffer_buffer_half` for lm_head × 2 dispatches |

The `use_coopmat` gate in step 4 fails on the main runner because `linear_coopmat` doesn't exist in main's source tree. Even if you forced output storage to buffer in main, the runtime would just dispatch `linear_vec_buffer_buffer_half` (the buffer-buffer variant of `linear_vec`), which is **slower** than the texture2d-weights default.

## What `storage_type_override=BUFFER` does (coopmat path)

With `VulkanPartitioner({"storage_type_override": VkStorageType.BUFFER})` AND running against `pavan-report`'s runner:

| Step | Result |
|---|---|
| Partitioner override | Forces output storage to `kBuffer` for every linear (and matmul, addmm, mm — see `tag_memory_meta_pass.py`'s `force_buffer_output_ops`) |
| Runtime dispatch | `use_coopmat` gate passes for all linears with `M >= 64` |
| Prepack | `force_buffer=true` → weights also `kBuffer` |
| Net | `linear_coopmat_half` for 224 linears; `linear_vec_buffer_buffer_half` for lm_head (M=1 fails the gate) |

Bonus: the same partitioner override also routes attention BMMs (`aten.bmm.default` via `Matmul.cpp`) through `matmul_coopmat` — 32 dispatches per forward, ~15.6× faster than baseline `matmul_vec_*`.

## Why coopmat REQUIRES buffer for both activations and weights

The `gl_KHR_cooperative_matrix` GLSL extension's `coopMatLoad` intrinsic accepts only **storage buffers** (or shared memory) — not samplers / images / textures. There is no texture-storage variant possible. So:

- `linear_coopmat.yaml` exposes no `STORAGE` / `WEIGHT_STORAGE` parameters at all
- The shader source assumes flat `VkBuffer` access for both inputs
- If you somehow tried to dispatch coopmat with texture-storage tensors, GLSL compilation would fail upstream

This is why the dispatch gate `storage_type_of(out) == kBuffer` is a hard requirement: it ensures we never dispatch coopmat unless we have buffer-storage tensors to feed it.

## Why baseline DEFAULTS to texture2d for weights (a tuned heuristic)

`prepack_fp_linear_weight()` in `Linear.cpp` was authored to prefer texture2d for fp32/fp16 linear weights because:

1. **Separate cache hierarchy**: RDNA3+ texture loads go through the L1 texture cache, parallel to the L0/L1 used by scalar buffer reads. On a memory-bound GEMV, both caches feed in parallel = wider effective bandwidth.
2. **2D access pattern fit**: GEMV walks weights as `weights[k][n]` — maps to texel `(n/4, k)` naturally. Texture hardware handles bounds/clamp/swizzle for free.
3. **Texture cache reuse**: column tiles get reused across multiple activation rows; cache exploits that spatial locality.

The `max_extent` fallback to buffer is purely a device-limit safety net: `maxImageDimension2D` is 16384 on 780M, so very wide weights (like lm_head's 128256-column matrix) can't be stored as texture2d.

## Apples-to-apples comparison check

When comparing baseline vs coopmat at L=32 S=128:

| Tensor | Baseline (main) | Coopmat (pavan-report) | Same? |
|---|---|---|---|
| Activations (output of linear) | buffer (via `CONTIGUOUS_ANY` default) | buffer (via `storage_type_override`) | ✓ same |
| Weights | texture2d (via prepack default) | buffer (via `force_buffer=use_coopmat`) | ✗ different (necessary for coopmat) |
| Algorithm | GEMV-style (`linear_vec`) | KHR WMMA (`linear_coopmat`) | ✗ different (the whole point) |

So the comparison isolates exactly two things: weight storage type AND algorithm. The activation path is the same on both sides. The 3.03× speedup measures the combined effect.

If a future agent worries "was the comparison fair w.r.t. activation storage", the answer is yes: today's main already uses buffer activations for fp16 linears even without `storage_type_override`. We verified this empirically with a 4-layer test (saw `linear_vec_buffer_texture2d_half` — buffer activations).

## Variants in `linear_vec.yaml` (for reference)

The `linear_vec.yaml` (main tree) generates these variants by Cartesian product:

```
STORAGE: {texture3d, buffer}
WEIGHT_STORAGE: {texture2d, buffer}     (combos restricted; see yaml)
DTYPE: {float, half}
HAS_BIAS: {false, true}
TILE_M: {default=4, 2, 1}
```

So `linear_vec_buffer_texture2d_half` is one of ~48 variants. Today's LLaMA only uses two: `linear_vec_buffer_texture2d_half` (224 dispatches) and `linear_vec_buffer_buffer_half` (2 dispatches, for the lm_head which has weights too wide for texture2d).

## Variants in `linear_coopmat.yaml` (pavan-report tree)

```
DTYPE: {float, half}
HAS_BIAS: {false, true}
PRECISION: highp
```

No `STORAGE` / `WEIGHT_STORAGE` params. Buffer storage is hard-coded inside the shader source. Two effective variants we use: `linear_coopmat_half` (no bias) and `linear_coopmat_bias_half` (with bias, not used by LLaMA).

## Where the dispatch logic lives

- Main tree, baseline dispatch: `backends/vulkan/runtime/graph/ops/impl/Linear.cpp`, `pick_linear_shader()` (~lines 129–161) + `add_linear_node()`
- Pavan-report tree, both paths: `backends/vulkan/runtime/graph/ops/impl/Linear.cpp` with added `add_linear_coopmat_node()` (~line 291) and the dispatch site (~line 350) that picks between them
- Storage decision for weights: `prepack_fp_linear_weight()` in same file (`Linear.cpp:23–80`)
- Partitioner override: `backends/vulkan/_passes/tag_memory_meta_pass.py` (pavan-report has extra `force_buffer_output_ops` parameter)

When in doubt, **read the source**. The dispatch logic is short and clear; trace through it for a specific tensor and you'll know exactly which shader will fire.
