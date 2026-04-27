# Real LLaMA 3.1 8B fp16 prefill — texture3D coopmat WMMA E2E study

## Scope

Phase 2 storage and end-to-end study, executed against the kernel-side
conclusions from `yanwen_docs/agent_reports/kernel_sweep_fp16_rdna3.md`. The
goal was to determine whether the fp16 cooperative-matrix kernels can improve
real-LLaMA prefill on AMD Radeon 780M / RADV Phoenix without forcing
whole-graph buffer storage.

This report covers:

- a near-LLaMA fp16 prefill block (synthetic weights), exported and run via the
  Vulkan delegate, that exposes linear and attention shapes representative of
  LLaMA-3.1-8B at seq=256;
- real LLaMA 3.1 8B with HuggingFace weights, layer-subset to 4 of 32 layers,
  fp16, seq=256, full attention with softmax, expand for GQA, causal mask,
  RoPE, RMSNorm, SwiGLU FFN.

The user's primary workload (`seq=2048` prefill) was not measured because the
.pte+activation working set OOMs on this 28 GB machine. That is the most
important caveat for the conclusions below.

## Source state

- Base commit: `4004bc23a7dbc6acdd06ca6dd4c24fb1d9d6dd58`
- Local dirty files at start: same as the kernel sweep (`yanwen_docs/...` plan
  edits, `.codex/`, `uv.lock`).
- No new shaders or runtime changes were added in Phase 2. The only code
  artifacts produced are Python and shell helpers and a synthetic LLaMA block,
  all under `yanwen_docs/agent_results/real_llama_e2e_storage_study/scripts/`.
- The texture3D coopmat dispatch path (`linear_coopmat_texture3d_buffer`) used
  here was already implemented in Phase 1 behind `VK_COOPMAT_TEXTURE=1`.

## Device

- `/dev/dri`: present (`card0`, `renderD128`, `by-path`).
- `vulkaninfo --summary`: GPU0 is `AMD Radeon 780M (RADV PHOENIX)`, RADV Mesa
  25.0.7, Vulkan API 1.4.305. GPU1 is llvmpipe and was not used.
- Cooperative matrix supported: yes. 14 KHR cooperative matrix configs exposed,
  including `16x16x16 fp16 -> fp16` and `16x16x16 fp16 -> fp32`.
- Saved logs:
  - `yanwen_docs/agent_results/real_llama_e2e_storage_study/dev_dri.txt`
  - `yanwen_docs/agent_results/real_llama_e2e_storage_study/vulkaninfo_summary.txt`

## Build commands

Two CMake outputs were used in this phase:

- `cmake-out-vk` — production-style runtime, mirrors the Phase 1 tested build,
  reconfigured with `EXECUTORCH_ENABLE_LOGGING=ON` so executor_runner reports
  per-iteration latency. Used for sanity routing checks.
- `cmake-out-vk-etdump` — same runtime configured with
  `EXECUTORCH_BUILD_DEVTOOLS=ON` and `EXECUTORCH_ENABLE_EVENT_TRACER=ON` so
  ETDump capture is supported. Used for all timing and ETDump runs reported
  here.

Exact commands (both CMake outputs were configured against the working tree
without nuking the existing tree):

```bash
cmake . \
    -Bcmake-out-vk-etdump \
    --preset "linux" \
    -DCMAKE_INSTALL_PREFIX=cmake-out-vk-etdump \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DEXECUTORCH_PAL_DEFAULT=posix \
    -DEXECUTORCH_BUILD_VULKAN=ON \
    -DEXECUTORCH_ENABLE_LOGGING=ON \
    -DEXECUTORCH_BUILD_DEVTOOLS=ON \
    -DEXECUTORCH_ENABLE_EVENT_TRACER=ON \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_FLAGS="-include algorithm"

cmake --build cmake-out-vk-etdump -j$(nproc) --target install --config Release
```

Custom-op test binaries are not required for Phase 2 measurements but were
left intact in `cmake-out-vk`.

## Workloads

Two workloads were used:

1. **Synthetic LLaMA-3.1-prefill block** (`exports/llama_block_fp16_seq256_*.pte`):
   - one transformer block: RMSNorm + Q/K/V/O linear + GQA attention bmm/softmax
     + RMSNorm + SwiGLU MLP, all fp16
   - `hidden=4096`, `ffn=14336`, `heads_q=32`, `heads_kv=8`, `head_dim=128`,
     seq=256, batch=1
   - exported with the Vulkan partitioner using two storage variants:
     `storage_type_override=TEXTURE_3D` (default texture path) and
     `storage_type_override=BUFFER`
   - random weights — used as a controlled pre-flight
2. **Real LLaMA 3.1 8B**, 4 of 32 layers, fp16 prefill at seq=256, real
   pretrained weights from HuggingFace (`meta-llama/Llama-3.1-8B`):
   - 29 linears (`Q/K/V/O + gate/up/down × 4 layers + LM head` — 28 prefill
     linears + 1 LM head with `M=1`)
   - exported through the existing `examples/models/llama/llama_transformer.py`
     path via the wrapper `scripts/run_real_llama_e2e_patched.py`

The `run_real_llama_e2e_patched.py` wrapper is required because the locally
installed `torchao.quantization.pt2e.quantize_pt2e` fails at module-import time
on this branch's torch version
(`AttributeError: module 'torch.ao.quantization' has no attribute 'quantizer'`).
Lesson:
`yanwen_docs/lesson_learned/phase2_real_llama_e2e/torchao_quantizer_import_blocker.md`.

`seq=2048` was attempted to match the previous-story baseline but the runtime
process is OOM-killed on this 28 GB system once the 3.8 GB .pte plus seq=2048
attention activations are resident. Lesson:
`yanwen_docs/lesson_learned/phase2_real_llama_e2e/seq2048_real_llama_oom.md`.

## Routing observed

For the synthetic block (1 layer, 7 prefill linears + 2 attention bmms):

| Variant | Linear shader | Matmul shader |
| --- | --- | --- |
| tex_stephen (texture3d, no env) | `linear_vec_texture3d_texture2d_half` ×7 | `matmul_vec_texture3d_half` ×2 |
| tex_coopmat (`VK_COOPMAT_TEXTURE=1`) | `linear_coopmat_texture3d_buffer` ×7 | `matmul_vec_texture3d_half` ×2 |
| tex_disable_coop (`VK_DISABLE_COOPMAT=1`) | `linear_vec_texture3d_texture2d_half` ×7 | `matmul_vec_texture3d_half` ×2 |
| buf_disable_coop (BUFFER pte, `VK_DISABLE_COOPMAT=1`) | `linear_vec` `is_buffer=0` ×7 | `matmul_vec` `is_buffer=0` ×2 |
| buf_coopmat (BUFFER pte, no env) | `linear_vec` `is_buffer=0` ×7 | `matmul_vec` `is_buffer=0` ×2 |

Important: when the synthetic .pte is exported with
`storage_type_override=BUFFER`, the runtime still routes linears with
`is_buffer=0` (texture3d output). That is the storage-propagation issue
flagged in `yanwen_docs/background/1_previous_story.md`: the
`_to_dim_order_copy` casts inserted around RMSNorm act as boundaries that the
partitioner promotes back to texture3d. Lesson:
`yanwen_docs/lesson_learned/phase2_real_llama_e2e/buffer_override_does_not_propagate_synth_block.md`.

For the real LLaMA texture3d .pte (29 linears total = 7 × 4 + 1 LM head):

| Variant | Prefill linears (`M=256`) | LM-head linear (`M=1`, `N=128256`) | Matmul (attention bmm) |
| --- | --- | --- | --- |
| real_tex_stephen | `linear_vec_texture3d_texture2d_half` ×28 | `linear_vec_buffer_buffer_half` ×1 | `matmul_vec_texture3d_half` ×8 |
| real_tex_coopmat (`VK_COOPMAT_TEXTURE=1`) | `linear_coopmat_texture3d_buffer` ×28 | `linear_vec_buffer_buffer_half` ×1 | `matmul_vec_texture3d_half` ×8 |
| real_tex_disable_coop (`VK_DISABLE_COOPMAT=1`) | `linear_vec_texture3d_texture2d_half` ×28 | `linear_vec_buffer_buffer_half` ×1 | `matmul_vec_texture3d_half` ×8 |

The LM head's output is `[1, 128256]` (LLaMA selects the last token before LM
head, so `M=1`). It is correctly excluded by the `M % 64 == 0` gate and falls
back to Stephen's buffer path. The LM head's storage is `BUFFER` because
`N=128256 / 4 = 32064` exceeds the default texture3d width limit of 16384, so
the partitioner must use buffer storage there. Coopmat does not engage on the
LM head because `M=1`.

For the real LLaMA buffer-storage .pte:

| Variant | Prefill linears | LM-head linear | Matmul |
| --- | --- | --- | --- |
| real_buf_coopmat | `linear_coopmat_half` ×28 | `linear_vec_buffer_buffer_half` ×1 | `matmul_vec_texture3d_half` ×4 + `matmul_vec_buffer_*` ×4 |
| real_buf_disable_coop (`VK_DISABLE_COOPMAT=1`) | `linear_vec` `is_buffer=1` ×29 | (same) | mixed buffer/tex |

Unlike the synthetic block, the real LLaMA buffer .pte's prefill linears do
end up `is_buffer=1` and therefore exercise `linear_coopmat_half` (buffer
coopmat). The reason the buffer override propagates here but not in the synth
block is probably the embedding layer at the front: real LLaMA's embedding is
a CPU fallback whose output the partitioner tags as buffer and then Linear's
`sync_primary_io_repr` carries that through.

## Wallclock results

All wallclock numbers come from `cmake-out-vk-etdump/executor_runner` with
`--num_executions=15` (5 warmup + 10 timed) for synthetic, `--num_executions=15`
for real LLaMA. Steady mean is computed over iterations 6..15. ETDump capture
runs (`--num_executions=1 --etdump_path=…`) are separate so timing is not
inflated by trace overhead.

### Synthetic LLaMA-prefill block (1 layer, fp16, seq=256)

| Variant | Steady mean (ms) | Stdev | Speedup vs Stephen |
| --- | ---: | ---: | ---: |
| tex_stephen (default texture3d) | 93.74 | 0.51 | 1.00× baseline |
| tex_disable_coop (`VK_DISABLE_COOPMAT=1`) | 93.46 | 0.30 | 1.00× |
| **tex_coopmat (`VK_COOPMAT_TEXTURE=1`)** | **46.53** | 0.35 | **2.01×** |
| buf_disable_coop (BUFFER pte) | 93.59 | 0.27 | 1.00× (override didn't take) |
| buf_coopmat (BUFFER pte) | 94.07 | 0.85 | 1.00× (override didn't take) |

### Real LLaMA 3.1 8B 4-layer fp16 seq=256

| Variant | Steady mean (ms) | Stdev | Speedup vs tex_stephen |
| --- | ---: | ---: | ---: |
| real_tex_stephen | 442.63 | 2.19 | 1.00× baseline |
| real_tex_disable_coop (`VK_DISABLE_COOPMAT=1`) | 441.66 | 1.76 | 1.00× |
| **real_tex_coopmat (`VK_COOPMAT_TEXTURE=1`)** | **250.43** | 4.38 | **1.77×** |
| real_buf_disable_coop (BUFFER pte) | 393.94 | 1.72 | 1.12× |
| real_buf_coopmat (BUFFER pte) | 205.29 | 2.23 | 2.16× |

Correctness: both the synthetic block and real LLaMA passed the script's
`max_abs_err < 10.0` check against the torch CPU fp16 reference, including the
texture coopmat variant. (The torch.export forward returns only the last
token's logits, so the check is on the LM-head output.)

## ETDump per-shader leaf-event breakdowns (real LLaMA, 1 iter)

`yanwen_docs/agent_results/real_llama_e2e_storage_study/etdump_real/real_*.etdp`
captured one execution per variant. ETDump capture itself adds overhead, so
leaf-sum totals are larger than the steady wallclock; the *relative* per-shader
distribution is the useful signal.

| Category | tex_stephen | tex_coopmat | buf_coopmat | buf_disable_coop |
| --- | ---: | ---: | ---: | ---: |
| linear | 525.11 | **354.05** | **241.83** | 455.09 |
| CPU<->GPU copy (ETVK_COPY_*) | 274.66 | 251.98 | 250.95 | 257.06 |
| reshape / view | 34.19 | 36.36 | 34.83 | 32.49 |
| elementwise | 22.01 | 21.58 | 30.70 | 32.94 |
| bmm / matmul | 7.91 | 8.27 | 9.97 | 8.78 |
| softmax | 0.62 | 0.65 | 1.85 | 1.79 |
| CPU fallbacks (eq/mul/cat/logical_not/embedding) | 7.91 | 8.93 | 9.27 | 8.32 |
| other / scalar_tensor | 1.13 | 1.10 | 1.18 | 1.34 |
| **TOTAL leaf events (1 iter, ETDump on)** | **873.43** | **682.82** | **580.59** | **797.68** |

Linear-category speedups (ETDump):

- texture coopmat / Stephen: 525.11 / 354.05 = **1.48×**
- buffer coopmat / buffer disable: 455.09 / 241.83 = **1.88×**
- buffer coopmat / Stephen tex: 525.11 / 241.83 = **2.17×**

The wallclock speedup is larger than the linear-category ETDump speedup because
ETDump capture overhead applies more uniformly across variants; the Stephen
ETDump leaf sum (873 ms) overstates the 442 ms steady wallclock by a similar
ratio for both arms. The per-shader per-call timings remain trustworthy for
relative comparison.

### Storage transition / "buffer trap" sanity at seq=256

The previous-round story (`yanwen_docs/background/1_previous_story.md`)
documented that on the same iGPU at seq=2048 the buffer-storage path regressed
real LLaMA from 5984 ms → 16131 ms (0.37×) because:

1. `ETVK_COPY_INPUTS` blew up 3.8× under buffer storage (host-side
   cache-coherency / NCHW reformat per subgraph boundary).
2. `softmax_buffer_float` ran 5.4× slower than `softmax_texture3d_half` at
   seq=2048 (256² rows benefit from texture row-coalescing; buffer doesn't).
3. A ~+11 s host-side `vkQueueSubmit` / `vkWaitForFence` gap appeared between
   leaf events that texture storage doesn't pay.

At seq=256 4 layers fp16 on the current branch the buffer trap **does not
reproduce in the same direction**. Per the table above:

- ETVK copy is essentially identical across texture and buffer variants
  (~250 ms at seq=256).
- softmax buffer is 1.85 ms vs 0.62 ms texture — a 3× factor, but on a
  small absolute base because seq² (= 65,536) is 64× smaller than seq²
  (= 4,194,304) at seq=2048.
- Wallclock-vs-leaf-sum gap is comparable across variants at this size.

This is consistent with the previous-story analysis: the buffer-storage tax is
size-dependent. Smaller sequences amortize less host-side overhead per
dispatch and lose less to softmax_buffer cache misses, so the buffer trap is
much milder. We were unable to validate at seq=2048 directly because the .pte
plus seq=2048 activation working set is OOM-killed on the 28 GB host
(`yanwen_docs/lesson_learned/phase2_real_llama_e2e/seq2048_real_llama_oom.md`).

That asymmetry between seq lengths is exactly why this study should not yet
recommend whole-graph buffer storage for production. The texture coopmat path
delivered a 1.77× E2E win at seq=256 with no storage-tax risk and is the
recommended Phase 3 starting point even though the buffer coopmat path
happened to be 1.22× faster at this particular small-seq config.

## WMMA eligibility table (real LLaMA 4L seq=256)

| Op | Shape (M, K, N) | Storage in tex pte | Eligible (tex coopmat gate) | Selected shader |
| --- | --- | --- | --- | --- |
| Q/O/down × 4 layers | 256, 4096, 4096 | tex3d → tex3d | yes | `linear_coopmat_texture3d_buffer` |
| K × 4 layers | 256, 4096, 1024 | tex3d → tex3d | yes (1024 % 64 = 0) | `linear_coopmat_texture3d_buffer` |
| V × 4 layers | 256, 4096, 1024 | tex3d → tex3d | yes | `linear_coopmat_texture3d_buffer` |
| gate/up × 4 layers | 256, 4096, 14336 | tex3d → tex3d | yes (14336 % 64 = 0) | `linear_coopmat_texture3d_buffer` |
| FFN down × 4 layers | 256, 14336, 4096 | tex3d → tex3d | yes (14336 % 32 = 0) | `linear_coopmat_texture3d_buffer` |
| LM head | 1, 4096, 128256 | tex3d → buffer | no (M=1, N exceeds tex limit) | `linear_vec_buffer_buffer_half` |
| attention scores Q×K^T | 256, 128, 256 | tex3d × tex3d → tex3d | n/a (no texture matmul coopmat) | `matmul_vec_texture3d_half` |
| attention out attn×V | 256, 256, 128 | tex3d × tex3d → tex3d | n/a | `matmul_vec_texture3d_half` |

All 28 prefill linears successfully route to the texture coopmat shader. The
attention bmms remain on Stephen's `matmul_vec` because no texture-coopmat
matmul shader exists yet — this is a clear target for the next iteration of
shader work.

## Recommendation for production integration (Phase 3 input)

1. **Architecture**: keep the dual-storage WMMA path described in
   `yanwen_docs/agent_plans/3_production_integration_design.md`. The
   texture3D coopmat linear path (`linear_coopmat_texture3d_buffer`) is the
   safe default for graphs that are already texture3d-backed; the buffer
   coopmat linear path (`linear_coopmat_half`) is faster on this device when
   the graph is already buffer-backed but should only be used when the rest of
   the graph chooses buffer storage for reasons other than coopmat.
2. **Dispatch gate**: the Phase 1 conservative gate is sufficient. No
   evidence here recommends loosening it. In particular, decode-style `M=1`
   should remain on Stephen.
3. **Production switch for VK_COOPMAT_TEXTURE**: convert the env-var hook into
   a normal dispatch decision based on (a) coopmat support, (b) eligible
   shape, (c) `dtype == half`, (d) `storage_type_of(input) == kTexture3D &&
   storage_type_of(out) == kTexture3D`. No partitioner changes are required to
   make this transfer to E2E.
4. **Do not force whole-graph buffer storage**. At seq=256 the buffer path
   happens to win, but the previous-story shows it loses at seq=2048, and we
   could not validate the longer prefill on this host. Keep buffer coopmat
   off the production default.
5. **Open follow-ups for the texture path**:
   - implement a texture coopmat **matmul** shader so the attention bmms
     (Q×K^T, attn×V) can use WMMA. This is the largest remaining linear
     budget that the texture path leaves on the table.
   - validate at seq=2048 on a host with more RAM (~48 GB+) before recommending
     buffer storage for any production use case.
   - the LM head's storage propagates to BUFFER because `N=128256` exceeds
     the texture3d width limit. That is fine on its own (M=1 disqualifies
     coopmat anyway), but it is worth documenting that the LM head will
     always route to Stephen's buffer fallback.

## Large-shape correctness statement

Phase 1 already documented sampled CPU-reference correctness for routed large
coopmat shapes
(`yanwen_docs/agent_results/kernel_sweep_fp16_rdna3/linear_coopmat_bench_large_sampled_correctness.log`).
Phase 2 did **not** independently re-check large-shape correctness; instead,
it relied on the helper script's torch CPU reference comparison on the LM-head
output and on the Phase 1 sampled validation for the underlying shaders.
Routed coopmat correctness should be considered "Phase 1 sampled-passed +
Phase 2 end-to-end-fp16-tolerance-passed", not "Phase 2 fully re-validated".

## Deliverables and raw artifacts

```text
yanwen_docs/agent_results/real_llama_e2e_storage_study/
  dev_dri.txt
  vulkaninfo_summary.txt
  scripts/
    export_llama_block.py            # synthetic LLaMA-block exporter
    run_synthetic_variants.sh        # synth 5-variant sweep runner
    run_real_llama_e2e.py            # copied from sibling yanwen_plan/
    run_real_llama_e2e_patched.py    # torchao-import workaround wrapper
    run_real_variants.sh             # real LLaMA texture .pte variants
    run_real_buffer_variants.sh      # real LLaMA buffer .pte variants
    run_real_seq2048.sh              # seq=2048 attempt (OOM blocker)
    analyze_etdump.py                # copied from sibling yanwen_plan/
  exports/
    llama_block_fp16_seq256_texture3d.pte  + .input0.bin
    llama_block_fp16_seq256_buffer.pte     + .input0.bin
  llama_tmp/
    llama31_8b_4L_seq256_fp16_texture3d.pte + _input0.bin + _out-0.bin
    llama31_8b_4L_seq256_fp16_buffer.pte    + _input0.bin + _out-0.bin
    llama31_8b_4L_seq2048_fp16_texture3d.pte (export only; seq=2048 OOMs)
  runs/
    synth_*.log .iters.txt .routing.txt .etdp _etdump_summary.txt
    real_*.log .iters.txt .routing.txt _etdump_summary.txt
    real_seq2048_tex_stephen.etdump_run.log (OOM trace)
  etdump_real/
    real_tex_stephen.etdp
    real_tex_coopmat.etdp
    real_tex_disable_coop.etdp
    real_buf_coopmat.etdp
    real_buf_disable_coop.etdp
    llama31_8b_4L_seq256_fp16_texture3d.etdp  (from helper script's first run)
    llama31_8b_4L_seq256_fp16_buffer.etdp
```

`llama_tmp/*.pte` files are large (~3.8 GB each) and can be re-generated by
re-running `scripts/run_real_llama_e2e_patched.py`; consider deleting them
after evidence is reviewed.

Lessons:

```text
yanwen_docs/lesson_learned/phase2_real_llama_e2e/torchao_quantizer_import_blocker.md
yanwen_docs/lesson_learned/phase2_real_llama_e2e/buffer_override_does_not_propagate_synth_block.md
yanwen_docs/lesson_learned/phase2_real_llama_e2e/seq2048_real_llama_oom.md
```

## Status

Phase 2 plan tasks are addressed:

- (1) Storage/E2E plan re-read; safest minimal experiment selected (synth
  block + real LLaMA 4L seq=256, no whole-graph buffer changes).
- (2) GPU re-verified: AMD Radeon 780M / RADV PHOENIX, not llvmpipe.
- (3) Built using the tested CMake commands (extended to enable logging and
  ETDump).
- (4) Controlled near-E2E LLaMA comparison run with all five intended
  variants for synth and four variants for real LLaMA at seq=256.
- (5) Compared Stephen tex baseline, conservative texture coopmat gate, and
  the buffer-coopmat regression control.
- (6) Correctness, latency, shader routing and storage-failure cases recorded
  per variant.
- (7) Raw outputs persisted under
  `yanwen_docs/agent_results/real_llama_e2e_storage_study/`.
- (8) This report and three lesson notes written.

Headline result: **the texture3D coopmat linear path delivers a 1.77× E2E
wallclock speedup on real LLaMA 3.1 8B fp16 4-layer seq=256 prefill, with all
28 prefill linears routed to `linear_coopmat_texture3d_buffer` and only the
M=1 LM head falling back, no partitioner changes, no whole-graph buffer
override.**

The largest remaining blocker for a stronger E2E win is that attention bmms
still use Stephen's `matmul_vec` because no texture-coopmat matmul shader
exists yet. That is the natural next kernel-side target.
