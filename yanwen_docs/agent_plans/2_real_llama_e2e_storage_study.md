# Agent Plan 2: Real LLaMA 3.1 E2E WMMA Storage Study

## Status

Complete. Phase 2 concluded with:

```text
yanwen_docs/agent_reports/real_llama_e2e_storage_study.md
yanwen_docs/agent_plans/current_findings_after_phase2_e2e.md
yanwen_docs/agent_results/real_llama_e2e_storage_study/analysis/phase2_real_llama_e2e_web_report.html
```

Do not rerun this plan unless the target workload, host RAM, or shader
implementation changes. The measured real workload was LLaMA 3.1 8B, 4 layers,
fp16, seq=256. The requested seq=2048 workload OOM-killed on the 28 GB host.

Current conclusion:

```text
texture3D linear coopmat transfers to real LLaMA E2E at seq=256:
  real_tex_stephen: 442.63 ms
  real_tex_coopmat: 250.43 ms = 1.77x speedup

buffer coopmat is fastest at seq=256:
  real_buf_coopmat: 205.29 ms = 2.16x vs texture Stephen

but do not force whole-graph buffer storage:
  previous seq=2048 evidence showed severe buffer-storage regression
  Phase 2 could not remeasure seq=2048 due OOM
```

## Objective

Determine whether the best fp16 WMMA kernels can improve real LLaMA 3.1
wallclock on AMD Radeon 780M / RADV Phoenix, without reintroducing the
whole-graph buffer-storage regression seen in the previous experiment.

This session should answer:

```text
Can fp16 WMMA replace Stephen's linear/matmul shaders in a real LLaMA 3.1 model
and improve end-to-end runtime on RDNA3 iGPU?
```

## Required Prior Work

Agents may read anything under `yanwen_docs/`, including background notes,
previous results, future diffs, and reports. Read:

```text
yanwen_docs/background/0_shader_baseline.md
yanwen_docs/background/1_previous_story.md
yanwen_docs/agent_plans/0_overview.md
yanwen_docs/agent_plans/1_kernel_sweep_fp16_rdna3.md
yanwen_docs/agent_plans/current_findings_after_kernel_sweep.md
```

Use the tested Vulkan build commands from
`yanwen_docs/agent_plans/0_overview.md` unless the task explicitly requires a
different build, such as an ETDump-enabled build. If an ETDump build is needed,
document the exact additional CMake flags and commands used.

If available, also read:

```text
yanwen_docs/agent_reports/kernel_sweep_fp16_rdna3.md
```

If the kernel sweep report does not exist, use only a small number of known
good candidate kernels from previous work. Do not run a broad kernel parameter
sweep inside full LLaMA.

If the kernel sweep report exists, start from its conservative dispatch gate
instead of rediscovering it:

```text
dtype == fp16
device supports VK_KHR_cooperative_matrix
operation is 2D linear or non-constant 2D matmul
buffer activation/output storage for the fastest kernel-only path
texture3D activation/output storage for the experimental linear E2E candidate
M % 64 == 0
N % 64 == 0
K % 32 == 0
fp32 cooperative accumulator
64x64x32 software macro tile
default subgroup size 64
K-step 32
```

Treat subgroup 32 and smaller macro tiles as later tuning-table work unless the
E2E trace shows a high-value shape that needs them.

## Baselines

Run these baselines:

```text
Stephen fp16 texture3d linear_vec/matmul_vec
Stephen fp16 buffer path if available
WMMA buffer path
WMMA texture-input path if available
WMMA disabled control
```

The previous story found:

```text
texture3d Stephen real LLaMA: ~5984 ms
buffer coopmat real LLaMA: ~16131 ms
buffer with coopmat disabled: ~17715 ms
```

Do not assume those numbers still reproduce. Re-measure on the current branch
and report exact setup.

Use whole-graph buffer WMMA as a regression/control case, not as the target
architecture. Prior real-LLaMA data indicates that whole-graph buffer storage
can make kernels faster while making wallclock much worse because of host-side
submission/coherency gaps and slower buffer implementations of non-linear ops.

## Model Workloads

Primary workload:

```text
real LLaMA 3.1 8B, fp16, seq_len=2048 prefill
```

If full 32-layer runs are too expensive, use a reduced-layer export, but state:

```text
number of layers
sequence length
dtype
attention implementation
quantization status
whether real weights or synthetic weights
```

Do not mix synthetic and real results in the same conclusion. Synthetic runs may
be useful controls but must be labeled clearly.

## Experiments

### Completed Preflight: Large-Shape Correctness

The Phase 1 kernel sweep now includes sampled CPU-reference validation for
large shapes:

```text
yanwen_docs/agent_results/kernel_sweep_fp16_rdna3/linear_coopmat_bench_large_sampled_correctness.log
yanwen_docs/agent_results/kernel_sweep_fp16_rdna3/matmul_coopmat_bench_large_sampled_correctness.log
```

`linear_coopmat_half` and `matmul_coopmat_half` passed sampled validation for
the routed large full-tile cases, including `4096^3`. The remaining sampled
failure is the existing non-coopmat `M=1` buffer fallback
`linear_vec_tile_row_1_buffer_texture2d_half`; treat that as a decode/fallback
caveat, not a routed coopmat blocker.

### Experiment 1: Reproduce Baseline and Buffer Trap

Run:

```text
texture3d Stephen baseline
whole-graph buffer Stephen or buffer fallback
whole-graph buffer WMMA
WMMA disabled buffer control
```

Collect:

- wallclock
- ETDump
- per-shader breakdown
- shader dispatch counts
- leaf event sum
- wallclock minus leaf event gap

Goal: verify whether the +host-side gap and buffer-storage tax still appear.

Stop after this experiment if the buffer trap reproduces and texture-input WMMA
is not available. Do not spend the session broadening buffer-only tuning unless
the wallclock and ETDump gap show that buffer storage is no longer the dominant
problem on the current branch/device.

### Experiment 2: Texture-Input Linear WMMA

This is the main target experiment after the kernel sweep. The Phase 1 kernel
sweep produced an experimental linear texture3D input/output + buffer-weight
coopmat path that beats Stephen's texture linear shader on measured eligible
full-tile shapes. Test whether that transfers to real LLaMA wallclock.

Important target architecture:

```text
texture3d activation input
buffer packed weight
texture3d activation output
shared-memory staging before coopMatLoad
shared-memory result staging before texture3D imageStore
```

Compare against Stephen texture3d baseline.

Collect:

- number of linears dispatched by WMMA
- number of linears falling back to Stephen
- why fallbacks happened
- storage type of each linear input/output
- transition shaders inserted around linears

Also report whether the graph pass actually tags the eligible linears as
texture-backed. Previous work found that CPU-fallback boundaries and
`sync_primary_io_repr=True` can propagate buffer storage into linears even when
texture storage is requested.

### Experiment 3: Selective WMMA Gates

Test narrow gates to avoid bad shapes:

```text
M >= 64
M >= 128
exclude M=1 decode / LM head
only MLP linears
only attention projection linears
```

If per-op gating is hard, implement logging first and simulate decisions from
the trace.

Only test `M >= 32` if a `32x64` or equivalent macro-tile variant is actually
available in the runtime path being evaluated. The default conservative gate is
`M % 64 == 0`.

### Experiment 4: Matmul/Attention Opportunities

Identify whether WMMA should target:

```text
QK attention matmul
attention weights x V matmul
generic aten.mm/bmm/addmm
```

Do not rewrite SDPA in this session unless the linear path is already understood.
This experiment can be a report-only opportunity analysis.

## What To Inspect

Code areas:

```text
backends/vulkan/op_registry.py
backends/vulkan/_passes/tag_memory_meta_pass.py
backends/vulkan/vulkan_preprocess.py
backends/vulkan/runtime/graph/ops/impl/Linear.cpp
backends/vulkan/runtime/graph/ops/impl/Matmul.cpp
backends/vulkan/runtime/graph/ops/impl/SDPA.cpp
backends/vulkan/runtime/graph/ops/glsl/linear_vec.glsl
backends/vulkan/runtime/graph/ops/glsl/matmul_vec.glsl
```

Key question:

```text
How does storage propagate from CPU fallback boundaries into linear/matmul, and
can WMMA use texture3d activation storage without poisoning downstream ops with
buffer storage?
```

## Deliverables

Raw artifacts:

```text
yanwen_docs/agent_results/real_llama_e2e_storage_study/
```

Lessons learned:

```text
yanwen_docs/lesson_learned/phase2_real_llama_e2e/
```

If the agent encounters a storage propagation issue, ETDump mismatch, host-side
gap, partitioning surprise, model-export blocker, or confusing runtime behavior,
write a short lesson file there for Yanwen.

Final report:

```text
yanwen_docs/agent_reports/real_llama_e2e_storage_study.md
```

The report must include:

- exact model/export details
- device info
- build commands
- run commands
- wallclock table
- ETDump per-shader table
- leaf sum vs wallclock gap
- storage transition analysis
- WMMA dispatch count and fallback reasons
- list of WMMA-eligible LLaMA ops and shapes
- recommendation for production integration
- explicit statement of whether large-shape correctness was independently
  checked or still relies on skipped-reference benchmark rows

## Required Tables

Wallclock summary:

```text
variant
wallclock_ms
linear_leaf_ms
matmul_leaf_ms
sdpa_leaf_ms
copy_inputs_ms
copy_outputs_ms
leaf_sum_ms
wallclock_minus_leaf_sum_ms
speedup_vs_stephen
notes
```

Per-shader summary:

```text
shader_name
dispatch_count
total_ms
avg_ms
storage_type
variant
notes
```

WMMA eligibility summary:

```text
op_name
shape
dtype
input_storage
output_storage
eligible
selected_shader
fallback_reason
```

## Success Criteria

A successful WMMA E2E path must:

- beat Stephen texture3d baseline in real LLaMA wallclock
- keep correctness
- avoid whole-graph buffer-storage regression
- have clear fallback behavior for unsupported shapes/storage

If no E2E speedup is achieved, the report should still be considered successful
if it clearly explains why and identifies the next blocker.
