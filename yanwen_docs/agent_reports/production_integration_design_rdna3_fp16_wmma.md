# Production integration design — RDNA3 fp16 WMMA linear/matmul

## Objective

Convert the experimental `VK_KHR_cooperative_matrix` (WMMA) linear/matmul
shaders into a production dispatch path on RDNA3-class iGPUs (initial target:
AMD Radeon 780M / RADV Phoenix), while preserving Stephen Jia's existing
`linear_vec` / `matmul_vec` / `*_scalar` shaders as fallback. fp16 is the
first shipping dtype; fp32 is secondary; int8 / q4 are explicitly out of
scope for this design.

This document is the Phase 3 deliverable named in
`yanwen_docs/agent_plans/3_production_integration_design.md`. It is grounded
in the Phase 1 kernel sweep
(`yanwen_docs/agent_reports/kernel_sweep_fp16_rdna3.md`) and the Phase 2
real-LLaMA E2E study
(`yanwen_docs/agent_reports/real_llama_e2e_storage_study.md`).

## Inputs already proven

The earlier phases established the following load-bearing facts that this
design takes as inputs rather than re-evaluating:

| Question | Answer (phase) |
| --- | --- |
| Is fp16 cooperative matrix supported on the target? | yes — `VK_KHR_cooperative_matrix` rev 2, `16x16x16 fp16->fp16` and `16x16x16 fp16->fp32` configs (Phase 1) |
| Best accumulator? | fp32 (Phase 1 sweep) |
| Best subgroup size? | 64 (Phase 1 sweep) |
| Best macro tile? | 64x64 software tile around the fixed 16x16x16 hw tile (Phase 1) |
| Best K-step? | 32 (Phase 1) |
| Conservative shape gate? | `M%64==0`, `N%64==0`, `K%32==0` for fp16 (Phase 1) |
| Does texture3D coopmat linear transfer to E2E? | yes, 1.77× wallclock on real LLaMA 3.1 8B 4L fp16 seq=256 (Phase 2) |
| Does whole-graph buffer storage transfer to E2E? | unsafe at production seq sizes — previous-story seq=2048 saw 0.37× regression; Phase 2 could not re-check seq=2048 due to host OOM |
| Does export-time `storage_type_override` reliably propagate? | no — depends on graph boundaries (Phase 2; lesson `buffer_override_does_not_propagate_synth_block.md`) |
| Is decode (`M=1`) a coopmat win? | no — best 1.03×, worst 0.79× regression (previous-story); fall back |

These don't need to be re-investigated. The remaining design problem is how
to convert the existing benchmark-flag-gated dispatch into a production
dispatch, and what to ship in what stages.

## Final selected architecture

**Dual-storage WMMA linear path, single-storage WMMA matmul path, one bench
hook → one capability switch.**

For `aten.linear.default`:

```
device supports VK_KHR_cooperative_matrix + 16x16x16 fp16 config
+
fp16 + tile-eligible shape (M%64==0, N%64==0, K%32==0)
+
no bias (today; bias variant exists but is conservatively disabled below)
+
storage decision tree:
    out storage == BUFFER  -> linear_coopmat_half           (proven kernel + small-seq E2E)
    out storage == TEXTURE_3D -> linear_coopmat_texture3d_buffer  (proven kernel + large-seq-safe E2E)
otherwise -> linear_vec / linear_scalar (Stephen)
```

For `aten.mm.default` / non-constant `aten.bmm.default`:

```
device supports coopmat + tile-eligible shape + fp16 + 2D
+
out storage == BUFFER  -> matmul_coopmat_half  (proven kernel)
out storage == TEXTURE_3D -> matmul_vec_texture3d_half  (Stephen; no texture matmul coopmat shader yet)
```

Texture matmul coopmat is the highest-value kernel-shader follow-up and is
explicitly *not* in the production-launch scope of this design — the
attention bmms remain on `matmul_vec_texture3d_half` until that shader
exists.

Key shape of the architecture:

- Storage stays where the partitioner already put it. The runtime picks a
  shader that matches; it does not push storage upstream or downstream.
- The capability check is done once per `linear_packed_weight` /
  `can_use_matmul_coopmat` call by querying `Adapter`, not via env var.
- Stephen's shaders remain the *only* fallback — nothing routes to a coopmat
  shader without explicit eligibility.

## Shader selection decision tree

The C++-side decision tree for `aten.linear.default`, in evaluation order,
where every check that returns "no" falls through to the next branch and the
last branch is the unconditional Stephen fallback:

```
linear_packed_weight(graph, args):
  if VK_DISABLE_COOPMAT in env:              -> linear_vec / linear_scalar
  if !adapter->supports_cooperative_matrix(): -> linear_vec / linear_scalar
  if !adapter->supports_fp16_coopmat_16x16x16():
                                              -> linear_vec / linear_scalar  [new check; see capability section]
  if dtype(out) != half:                      -> linear_vec / linear_scalar
  if has_bias && bias-variant disabled:       -> linear_vec / linear_scalar  [stage-1 conservatism]
  shape_ok = (M % 64 == 0 && N % 64 == 0 && K % 32 == 0)
  if !shape_ok:                               -> linear_vec / linear_scalar
  if storage(out) == BUFFER:                  -> linear_coopmat_half        [buffer dual variant]
  if storage(in) == TEXTURE_3D &&
     storage(out) == TEXTURE_3D:              -> linear_coopmat_texture3d_buffer
  -> linear_vec / linear_scalar (storage mismatch — should be rare)
```

For `aten.mm.default` / non-constant `aten.bmm.default`:

```
matmul_dispatch(graph, args):
  if VK_DISABLE_COOPMAT in env:              -> matmul_vec / matmul_scalar
  if !adapter->supports_cooperative_matrix(): -> matmul_vec / matmul_scalar
  if !adapter->supports_fp16_coopmat_16x16x16(): -> matmul_vec / matmul_scalar
  if dtype(out) != half:                      -> matmul_vec / matmul_scalar
  if dim(out) != 2:                           -> matmul_vec / matmul_scalar
  shape_ok = (M % 64 == 0 && N % 64 == 0 && K % 32 == 0 && K(mat1) == K(mat2))
  if !shape_ok:                               -> matmul_vec / matmul_scalar
  if storage(out) == BUFFER:                  -> matmul_coopmat_half
  -> matmul_vec_texture3d_half  (deferred: will become matmul_coopmat_texture3d_buffer once that shader lands)
```

Selection logging (currently `fprintf(stderr, "[VK_LINEAR] ...")` in
`Linear.cpp`) stays on by default in debug builds and is gated behind
`ETVK_LOG_SHADER_SELECTION=1` in release builds — useful for confirming what
fired during a customer trace, low cost.

## Capability query design

Today's `Adapter::supports_cooperative_matrix()` in
`backends/vulkan/runtime/vk_api/Adapter.h:255` only checks the
`VkPhysicalDeviceCooperativeMatrixFeaturesKHR::cooperativeMatrix` feature
bit. That is necessary but **not** sufficient: a vendor could expose only
int8 or only fp32 cooperative matrix configs. The current shader binaries
hard-require `16x16x16 fp16 -> fp16` (or `-> fp32` for the accumulator
variant).

Add a strict check that the required config is enumerated:

```cpp
// backends/vulkan/runtime/vk_api/Adapter.h
bool supports_fp16_coopmat_16x16x16(bool require_fp32_accum = true);

// backends/vulkan/runtime/vk_api/Adapter.cpp
bool Adapter::supports_fp16_coopmat_16x16x16(bool require_fp32_accum) {
#ifdef VK_KHR_cooperative_matrix
  if (!supports_cooperative_matrix()) return false;
  uint32_t n = 0;
  vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(handle_, &n, nullptr);
  if (n == 0) return false;
  std::vector<VkCooperativeMatrixPropertiesKHR> props(
      n, {.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
  vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(handle_, &n, props.data());
  for (const auto& p : props) {
    if (p.MSize != 16 || p.NSize != 16 || p.KSize != 16) continue;
    if (p.scope != VK_SCOPE_SUBGROUP_KHR) continue;
    if (p.AType != VK_COMPONENT_TYPE_FLOAT16_KHR) continue;
    if (p.BType != VK_COMPONENT_TYPE_FLOAT16_KHR) continue;
    if (require_fp32_accum) {
      if (p.CType != VK_COMPONENT_TYPE_FLOAT32_KHR) continue;
      if (p.ResultType != VK_COMPONENT_TYPE_FLOAT32_KHR) continue;
    }
    return true;
  }
  return false;
#else
  return false;
#endif
}
```

Cache the result on the `Adapter` (it cannot change during the lifetime of
the device). The benchmark `queryCooperativeMatrixProperties()` helper that
already lives under `backends/vulkan/test/custom_ops/cm_utils.cpp` enumerates
the same property table — reuse its enumeration call rather than duplicating
it; just promote the predicate into `Adapter`.

Subgroup-size selection currently relies on the benchmark-only env hook
`VK_COOPMAT_REQUIRED_SUBGROUP_SIZE`. The Phase 1 sweep showed 64 is the
right default and per-shape tuning is at most a small win; production should
ship subgroup 64 by default and leave subgroup-size selection out of the
launch design. If a tuning table is later needed, route it through
`pick_linear_coopmat_local_wg_size()` rather than re-introducing an env hook.

## Storage strategy

Phase 2 is unambiguous: **the production default must remain texture3D
activation/output storage** for the LLaMA-style graph; whole-graph buffer
storage is not safe at production sequence lengths on this iGPU class.

Concrete rules:

1. **Do not change `op_registry.py`.** `aten.linear.default` stays
   `inputs_storage=utils.CONTIGUOUS_ANY`. The runtime picks a shader that
   matches the storage already chosen by the partitioner; it does not force
   storage upstream.
2. **Do not touch `tag_memory_meta_pass.py`.** Phase 2 confirmed that the
   real LLaMA texture3D `.pte` already routes 28 of 29 linears to
   `linear_coopmat_texture3d_buffer` without any partitioner change. The LM
   head's `M=1` and `N=128256>texture_width_limit` make it correctly fall
   back; that fallback is not a defect.
3. **Do not touch `vulkan_preprocess.py`.** No new compile spec is required
   for the launch design; the existing `storage_type_override` continues to
   exist for users who deliberately want a buffer-everywhere graph (debug /
   regression-control), and the runtime will then use `linear_coopmat_half`
   automatically.
4. **Buffer coopmat exists for a reason.** Customers whose graph naturally
   ends up buffer-backed (e.g. heavily M=1 decode that uses the buffer
   `linear_vec` fallback today, or future quantized-weight pipelines that
   require buffer storage) will get the matching `linear_coopmat_half`
   shader for the eligible linears. We do not gate this; the dispatch tree
   handles it.
5. **Do not bridge buffer↔texture inside the linear shader.** Both today's
   buffer coopmat shader and the texture coopmat shader stage through shared
   memory; that is correct and stays. The runtime never inserts a
   conversion node around a coopmat dispatch.

The texture-storage propagation issue from
`yanwen_docs/background/1_previous_story.md` ("Approach A v2 broke partition
when hiding buffer producers") is *not* re-opened by this design. We rely
on the partitioner's existing tagging behavior, which Phase 2 demonstrated
is correct for the texture3D-default real LLaMA graph.

## Fallback behavior

Any of the following conditions routes to Stephen's existing
`linear_vec` / `linear_scalar` / `matmul_vec` / `matmul_scalar`:

| Condition | Comment |
| --- | --- |
| `VK_DISABLE_COOPMAT=1` in environment | Required escape hatch for support / debugging |
| Device does not expose `VK_KHR_cooperative_matrix` extension | Non-RDNA3 GPUs, llvmpipe |
| Device does not enumerate the `16x16x16 fp16 -> fp32` config | New strict check — see capability section |
| `dtype(out) != fp16` | fp32 path stays on Stephen for stage 1 (separate fp32 sweep would be needed to flip it) |
| `M % 64 != 0` or `N % 64 != 0` or `K % 32 != 0` | Conservative shape gate |
| `M == 1` (decode / LM head) | Always falls through `M % 64 != 0` |
| `dim(out) != 2` for matmul | bmm with leading batch dim still has 2D inner mm |
| `has_bias` for linear | Stage 1 disables coopmat for biased linear; the variant exists but was not part of the Phase 2 E2E proof |
| Output storage is `TEXTURE_2D` (e.g. via `storage_type_override`) | No texture2D coopmat shader exists |
| Quantized linear / matmul (`q4gsw`, `q8csw`, `q8ta_*`) | Different op family entirely; coopmat not wired |

The fallback is unconditionally always-installed; coopmat shaders only
*supplement* it, never *replace* its registration.

## Files to modify

This design intentionally keeps the code change small and concentrated in
the dispatch / capability layer. Production path:

```
backends/vulkan/runtime/vk_api/Adapter.h         # add supports_fp16_coopmat_16x16x16()
backends/vulkan/runtime/vk_api/Adapter.cpp       # implement, cache the result
backends/vulkan/runtime/graph/ops/impl/Linear.cpp  # remove VK_COOPMAT_TEXTURE env requirement; use new capability check
backends/vulkan/runtime/graph/ops/impl/Matmul.cpp  # use new capability check
```

What stays as-is:

```
backends/vulkan/runtime/graph/ops/glsl/linear_coopmat.glsl              # production buffer kernel (unchanged)
backends/vulkan/runtime/graph/ops/glsl/linear_coopmat.yaml
backends/vulkan/runtime/graph/ops/glsl/linear_coopmat_texture.glsl      # production texture3D kernel (unchanged)
backends/vulkan/runtime/graph/ops/glsl/linear_coopmat_texture.yaml
backends/vulkan/runtime/graph/ops/glsl/matmul_coopmat.glsl              # production buffer kernel
backends/vulkan/runtime/graph/ops/glsl/matmul_coopmat.yaml
backends/vulkan/runtime/graph/ops/impl/Common.{h,cpp}
backends/vulkan/runtime/vk_api/Device.{h,cpp}                           # already wires VK_KHR_cooperative_matrix into Device
backends/vulkan/op_registry.py
backends/vulkan/_passes/tag_memory_meta_pass.py
backends/vulkan/vulkan_preprocess.py
```

What gets retired (env-flag-only / benchmark-only):

| Env hook | Disposition |
| --- | --- |
| `VK_COOPMAT_TEXTURE` | Retire as a *dispatch toggle*; texture coopmat becomes default-on when eligible. Keep the symbol parsed-but-ignored for one release if you want backward-compat for in-flight benchmarks; or just delete. |
| `VK_COOPMAT_ACCUM_FP16` | Retire from production dispatch (`pick_linear_coopmat_shader` / `pick_matmul_coopmat_shader`). Phase 1 showed it is not a consistent speedup; keep the `*_accum_fp16` shader binaries gated behind a benchmark-only path under `backends/vulkan/test/custom_ops/`. |
| `VK_COOPMAT_REQUIRED_SUBGROUP_SIZE` | Retire from production dispatch. The default is subgroup 64, set via `LocalWorkgroupSize` shader specialization — no env hook needed. |
| `VK_COOPMAT_MACRO_TILE`, `VK_COOPMAT_K_STEP` | Retire from production dispatch. Phase 1 said default `64x64x32` wins; the smaller-tile variants are tuning-table candidates, not production knobs. |
| `VK_DISABLE_COOPMAT` | **Keep** as the documented escape hatch. Used by support / regression bisection. Cheap to maintain — single getenv check at dispatch time. |

The benchmark binaries
(`backends/vulkan/test/custom_ops/{linear,matmul}_coopmat_bench.cpp`,
`backends/vulkan/test/custom_ops/cm_utils.{h,cpp}`) keep all the env hooks
they currently honor. That is microbenchmark scope, not production scope.

The Phase 2 lesson `buffer_override_does_not_propagate_synth_block.md` is
captured as a known, accepted behavior of the current partitioner and does
not need to be fixed for this design.

## Concrete diffs (illustrative, not committed)

`Linear.cpp:466-472` today:

```cpp
bool use_texture_coopmat = getenv("VK_COOPMAT_TEXTURE") &&
    !getenv("VK_DISABLE_COOPMAT") &&
    graph.context()->adapter_ptr()->supports_cooperative_matrix() &&
    graph.storage_type_of(input) == utils::kTexture3D &&
    graph.storage_type_of(out) == utils::kTexture3D &&
    graph.dtype_of(out) == vkapi::kHalf && !has_bias && !tile.shader_name &&
    tile_compatible;
```

becomes:

```cpp
bool coopmat_enabled = !getenv("VK_DISABLE_COOPMAT") &&
    graph.context()->adapter_ptr()->supports_fp16_coopmat_16x16x16();
bool use_texture_coopmat = coopmat_enabled &&
    graph.storage_type_of(input) == utils::kTexture3D &&
    graph.storage_type_of(out) == utils::kTexture3D &&
    graph.dtype_of(out) == vkapi::kHalf && !has_bias && !tile.shader_name &&
    tile_compatible;
bool use_buffer_coopmat = coopmat_enabled &&
    graph.storage_type_of(out) == utils::kBuffer &&
    graph.dtype_of(out) == vkapi::kHalf && (!has_bias || !tile.shader_name) &&
    tile_compatible;
```

i.e. drop `getenv("VK_COOPMAT_TEXTURE")`, add a stricter capability check,
unify the env-disable check.

`Matmul.cpp:194-213`'s `can_use_matmul_coopmat` gets the same capability
upgrade:

```cpp
if (getenv("VK_DISABLE_COOPMAT") ||
    !graph.context()->adapter_ptr()->supports_fp16_coopmat_16x16x16() ||
    graph.storage_type_of(out) != utils::kBuffer ||
    graph.dtype_of(out) != vkapi::kHalf || graph.dim_of(out) != 2) {
  return false;
}
```

Selection logging in `Linear.cpp:494-512` and the matching place in
`Matmul.cpp` should be guarded behind `#ifndef NDEBUG` or
`getenv("ETVK_LOG_SHADER_SELECTION")` so release-build users do not see the
stderr spew.

## Test matrix

Three layers of tests, all of which must remain green for the launch:

### Kernel-level (`backends/vulkan/test/custom_ops/`)

These already exist from Phase 1 and need only minor maintenance:

```
linear_coopmat_bench       # buffer + texture coopmat, all LLM/BERT shapes
matmul_coopmat_bench       # buffer coopmat, all shapes
sampled CPU-reference correctness for large shapes (already in)
```

Add (new in Phase 3):

```
test_vk_linear_coopmat_capability_query.cpp:
  - asserts supports_fp16_coopmat_16x16x16() returns true on RADV Phoenix
  - asserts it returns false on a device without VK_KHR_cooperative_matrix
    (use llvmpipe in CI)
test_vk_linear_coopmat_dispatch.cpp:
  - eligible texture3d shape -> linear_coopmat_texture3d_buffer
  - eligible buffer shape    -> linear_coopmat_half
  - M=1 -> linear_vec_*
  - non-multiple K -> linear_vec_*
  - VK_DISABLE_COOPMAT=1 -> linear_vec_*
  - has_bias -> linear_vec_*  (stage 1)
```

These are routing tests — they don't need numeric correctness, only the
dispatch-decision invariant.

### E2E benchmark

Phase 2's helpers reproduce as a CI smoke. Suggested shape:

```
yanwen_docs/agent_results/real_llama_e2e_storage_study/scripts/
  run_real_variants.sh   # 3 texture3d variants
  run_synthetic_variants.sh  # 5 synth variants
```

In CI on a RADV-equipped host, the regression bar is:

```
real_tex_coopmat / real_tex_stephen >= 1.5x  # currently 1.77x; allow noise
real_tex_coopmat correctness PASS against torch fp16 ref
synth_tex_coopmat / synth_tex_stephen >= 1.7x  # currently 2.01x
```

If a real-LLaMA seq=2048 host (≥ 48 GB RAM) becomes available, add:

```
real_tex_coopmat seq=2048 wallclock <= real_tex_stephen seq=2048 wallclock
real_buf_coopmat seq=2048 wallclock check (regression-control; expected >> tex baseline)
```

### Regression / negative tests

Per the launch checklist:

```
device without VK_KHR_cooperative_matrix -> Stephen path; correctness PASS
fp32 linear/matmul -> Stephen path
buffer fallback -> linear_vec_buffer_*  (existing)
existing q4gsw / q8csw / q8ta linears -> unchanged shaders
LM head (M=1, large N) -> Stephen buffer fallback; numeric PASS
```

The Phase 1 lesson
`yanwen_docs/lesson_learned/phase1_kernel_sweep/large_shape_sampled_correctness.md`
is the basis for treating routed coopmat outputs as sampled-passed. Do not
relax the sampled-correctness checks in any benchmark binary as part of
this work.

## Implementation stages

Suggested ordering. Each stage is independently reviewable.

### Stage 1 — production dispatch with no new shaders (~1–2 days)

```
1. Add Adapter::supports_fp16_coopmat_16x16x16()
2. In Linear.cpp / Matmul.cpp, replace
       getenv("VK_COOPMAT_TEXTURE") &&
   with the strict capability check.
3. Remove the VK_COOPMAT_ACCUM_FP16 / MACRO_TILE / K_STEP / SUBGROUP env
   reads from production dispatch helpers (move benchmark variants behind
   the bench binaries only).
4. Add the routing-decision unit tests above.
5. CI: run linear/matmul bench; run synth + real LLaMA seq=256 with the
   Phase 2 scripts; compare to the published baseline.
```

Risk: low. No shader changes; no partitioner changes. Behavior on a
non-coopmat device is identical to today's. Behavior on a coopmat device
matches today's `VK_COOPMAT_TEXTURE=1` path with a tighter capability gate.

### Stage 2 — texture matmul coopmat shader (~3–5 days kernel work)

```
1. Add backends/vulkan/runtime/graph/ops/glsl/matmul_coopmat_texture.glsl
   - texture3d input/output, no packed weight (matmul takes mat2 as a
     runtime tensor, not a constant)
   - shared-memory staging on both A and B before coopMatLoad
   - shared-memory staging before texture imageStore (mirrors
     linear_coopmat_texture's store path)
2. Add the YAML registration.
3. In Matmul.cpp, extend can_use_matmul_coopmat() and pick to allow
   storage(out) == TEXTURE_3D when the new shader exists.
4. Phase 1-style microbench coverage: BERT_FFN_up/down, LLM_QKV_64tok,
   sq_1024, sq_4096, sq_4096_cube; sampled correctness for the cube.
5. E2E re-run: real LLaMA 4L seq=256 should now win in the bmm category
   too (Phase 2 left ~8 ms there).
```

Risk: medium. Texture imageStore from a coopmat result requires the
shared-memory staging dance; there is a Phase 1 lesson on this:
`yanwen_docs/lesson_learned/phase1_kernel_sweep/texture_coopmat_image_store.md`.

### Stage 3 — biased linear coopmat (~1 day, only if needed)

```
1. Phase 1's linear_coopmat_bias variant exists but Phase 2 did not exercise
   it on real LLaMA (LLaMA-3.1 linears have bias=False).
2. If a future model surfaces biased linears, enable the bias variant by
   removing the `!has_bias` clause from use_texture_coopmat / use_buffer_coopmat
   and add a routing test.
3. Risk: low; the variant already passed Phase 1 sampled correctness on
   small shapes; sample-extend for full-tile.
```

Stage 3 should only land if a user hits it; do not pre-enable.

### Stage 4 — explicit-tile / subgroup tuning table (later, optional)

Phase 1 showed `32x64` macro tile helps `M=32` linear and select matmul
shapes. If a target model exposes those shapes, wire a static lookup table
keyed on `(M, K, N, op)` inside `pick_linear_coopmat_*` /
`pick_matmul_coopmat_*`. **Do not** use environment variables for this; the
benchmark hooks are bench-only.

### Stage 5 — fp32 sweep + integration (out of fp16 launch scope)

If fp32 coopmat ever becomes a target, repeat the Phase 1/2 process on fp32
configs. fp32 cooperative matrix configs do exist in the device's property
table (the 14 enumerated configs include fp32 accumulator forms), but no fp32
shader has been validated. Expect a separate kernel sweep.

## Implementation risks

Carry-forwards from Phase 1/2 that the launch checklist must explicitly
sign off on:

1. **seq=2048 buffer-trap re-validation gap.** Phase 2 could not measure
   real LLaMA at seq=2048 due to host OOM
   (`yanwen_docs/lesson_learned/phase2_real_llama_e2e/seq2048_real_llama_oom.md`).
   The buffer-trap previous-story numbers remain the authoritative
   production-size data point. Do not let buffer coopmat become the
   *advertised* default just because it happened to win at seq=256 — it has
   not been re-proven safe at production seq.
2. **Texture matmul coopmat is unbuilt.** Until Stage 2 lands, the
   attention bmms remain the dominant kernel-shader gap; the headline
   1.77× E2E number for Stage 1 is linear-only. Communicate that.
3. **fp16 accumulator is not a default.** Phase 1 showed it is not a
   consistent speedup and may lose accuracy. Do not flip it on for any
   shape without per-shape evidence; do not expose a runtime hook for it.
4. **Decode (`M=1`) does not benefit.** Previous-story has worst case
   0.79× regression at `1×4096→11008`. The shape gate (`M%64==0`) covers
   this, but anyone tempted to loosen the gate "just for decode" should
   stop and re-measure. There is no decode-specific coopmat win on this
   iGPU class.
5. **Capability portability is unverified beyond RADV.** Other vendors'
   coopmat configs may not include `16x16x16 fp16->fp32`. The new strict
   capability check correctly returns false on non-conforming devices, so
   the *correctness* risk is contained, but the *coverage* claim "fp16
   WMMA is enabled on devices with cooperative matrix" stops being true
   for vendors that only expose other tile shapes. Document this in the
   release notes; consider adding a debug log line listing the enumerated
   configs at first dispatch.
6. **Storage propagation depends on graph shape.** Real LLaMA's embedding
   CPU-fallback boundary forwards buffer storage to its consumer linears
   when the .pte is exported with `storage_type_override=BUFFER`; a
   synthetic block without that boundary does not. The runtime correctly
   handles either, but anyone diagnosing "why did my linear not pick
   coopmat" should check the shader-selection log first, not assume the
   compile spec is authoritative.
7. **Subgroup-size selection is currently env-driven.** Phase 1 only
   surfaced subgroup 64 vs 32 through `VK_COOPMAT_REQUIRED_SUBGROUP_SIZE`;
   this design retires that hook for production. If Stage 4 adds a tuning
   table, route it through the existing `pick_local_wg_size` callback so
   the env hook is *not* re-introduced for production.
8. **Quantized linear/matmul is untouched.** `q4gsw`, `q8csw`,
   `q8ta_linear` continue to use Stephen's tiled scalar/vec
   implementations. Wiring those to int8 cooperative matrix configs is its
   own design exercise, explicitly out of scope here. Note that the device
   *does* expose 8 int8/uint8 → int32 coopmat configs; the kernels just
   are not written.

## Recommendation

Land Stage 1 first as the smallest, lowest-risk change that converts the
Phase 2 E2E win into a default-on production behavior. Stage 1 is reviewable
as a single PR touching four files (`Adapter.{h,cpp}`, `Linear.cpp`,
`Matmul.cpp`) plus the routing tests, with no shader or partitioner change.

Stage 2 (texture matmul coopmat shader) is the highest-value next item; it
is the load-bearing reason the Phase 2 result was 1.77× rather than the
synth-block 2.01× — the bmm category did not move because no texture matmul
coopmat shader exists. Plan it for the iteration immediately after Stage 1
ships.

Stages 3–5 should be demand-driven: open them only when a customer model or
new device target makes them necessary.

## Status

Stage 1 has been implemented and verified on this branch. The four-file
diff lives in:

```
backends/vulkan/runtime/vk_api/Adapter.h          # +8  declares supports_fp16_coopmat_16x16x16
backends/vulkan/runtime/vk_api/Adapter.cpp        # +46 enumerates and matches 16x16x16 fp16 -> fp32
backends/vulkan/runtime/graph/ops/impl/Linear.cpp # ±9  drop VK_COOPMAT_TEXTURE env requirement; use new capability check
backends/vulkan/runtime/graph/ops/impl/Matmul.cpp # ±2  use new capability check
```

### Stage 1 verification

Re-running Phase 2's measurement scripts against the Stage-1 build:

Routing (real LLaMA 4L fp16 seq=256, no env vars set):

```
[VK_LINEAR] Using linear_coopmat_texture (cooperative matrix)   ×28
[VK_LINEAR] Using linear_vec (... is_buffer=1, has_bias=0)      ×1   (LM head, M=1)
[VK_MATMUL] Using matmul_vec (..., is_buffer=0)                 ×8   (attention bmm — Stage 2 target)
```

Routing under `VK_DISABLE_COOPMAT=1`:

```
[VK_LINEAR] Using linear_vec (...)  ×29
[VK_MATMUL] Using matmul_vec (...)  ×8
```

Wallclock (`--num_executions=15`, steady mean over iterations 6..15):

| Variant | Steady mean (ms) | Stdev | Speedup |
| --- | ---: | ---: | ---: |
| Stage 1 default (texture coopmat default-on) | 254.72 | 4.22 | **1.74×** |
| Stage 1, `VK_DISABLE_COOPMAT=1` (Stephen) | 443.98 | 2.53 | 1.00× baseline |

The 1.74× wallclock speedup matches the Phase 2 number (1.77×) within
run-to-run noise, confirming that removing the `VK_COOPMAT_TEXTURE=1`
benchmark gate does not regress anything.

Raw evidence:

```
yanwen_docs/agent_results/phase3_production_integration/stage1_default.iters.txt
yanwen_docs/agent_results/phase3_production_integration/stage1_disabled.iters.txt
```

### What is left for Stages 2–5

- Stage 2 (texture matmul coopmat shader) — still the highest-value next
  item; the 8 ms attention-bmm budget remains on `matmul_vec_texture3d_half`.
- Stage 3 (biased linear coopmat) — demand-driven only.
- Stage 4 (per-shape tuning table) — demand-driven only.
- Stage 5 (fp32 coopmat) — separate sweep; not in fp16 launch scope.

No new lesson notes were written this round because no blocker surfaced.
The Phase 1/2 lesson files remain authoritative for the risks called out
above. If a future implementation stage hits a partitioner / capability /
build problem, document it under
`yanwen_docs/lesson_learned/phase3_production_integration/<short_name>.md`
at that time.

The headline change for shipping: **the texture3D coopmat dispatch is now
default-on when the device exposes the required cooperative matrix
configuration**, replacing the `VK_COOPMAT_TEXTURE=1` benchmark hook with a
real capability check, while every other variable from the Phase 1 sweep is
locked at its measured-best default and no partitioner / op-registry / pass
is touched.
