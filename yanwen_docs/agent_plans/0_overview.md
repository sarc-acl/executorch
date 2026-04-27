# WMMA Research Plan Overview

## Goal

Improve ExecuTorch Vulkan matrix-multiplication-related GLSL shaders using
`VK_KHR_cooperative_matrix` on RDNA3 integrated GPUs, with AMD Radeon 780M /
RADV Phoenix as the first target.

The desired end state is:

```text
If a device supports the right KHR cooperative matrix configuration, ExecuTorch
can substitute Stephen Jia's fp16/fp32 tiled matmul/linear shaders with WMMA
cooperative-matrix shaders and get real model speedups.
```

The first implementation target is fp16. fp32 behavior is secondary. int8 is
deferred to Phase 4 exploration after the fp16 production path is under control.
q4 remains future work unless a later plan explicitly scopes it.

## Background Docs

Before starting, agents may read anything under:

```text
yanwen_docs/
```

This directory is expected to contain background context, previous results,
future branch diffs, benchmark artifacts, and agent reports. In particular,
read:

```text
yanwen_docs/background/0_shader_baseline.md
yanwen_docs/background/1_previous_story.md
```

The key baseline is Stephen Jia's March 14 fp32/fp16 tiled matmul/linear shader
work:

```text
7a63aff49f6c9c269a9bb67bddfffd93232e3aca
[ET-VK][matmul] Re-implement fp32/fp16 matmul and linear with tiled compute and blocked weight packing
```

Baseline shaders:

```text
backends/vulkan/runtime/graph/ops/glsl/matmul_scalar.glsl
backends/vulkan/runtime/graph/ops/glsl/matmul_vec.glsl
backends/vulkan/runtime/graph/ops/glsl/linear_scalar.glsl
backends/vulkan/runtime/graph/ops/glsl/linear_vec.glsl
backends/vulkan/runtime/graph/ops/glsl/pack_fp_linear_weight.glsl
```

## Target Device

Initial target:

```text
AMD Radeon 780M (RADV PHOENIX)
Mesa 25.0.7 / RADV
Vulkan API 1.4.305
Integrated GPU
subgroupSize = 64
minSubgroupSize = 32
maxSubgroupSize = 64
subgroupSizeControl = true
VK_KHR_cooperative_matrix revision 2
```

Supported cooperative matrix configs on the target include:

```text
16x16x16 float16 x float16 -> float16
16x16x16 float16 x float16 -> float32
16x16x16 int8/uint8 variants -> int32/uint32
```

This phase should prioritize:

```text
fp16 input, fp16 accumulator/result
fp16 input, fp32 accumulator/result
```

## Tested Build Commands On This Machine

Use this build sequence for the Vulkan runtime and Vulkan custom op tests on
this machine. These commands were tested by Yanwen and should be the default
build path for coder agents unless a task explicitly needs a different build.

```bash
rm -rf cmake-out-vk

cmake . \
    -Bcmake-out-vk \
    --preset "linux" \
    -DCMAKE_INSTALL_PREFIX=cmake-out-vk \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DEXECUTORCH_PAL_DEFAULT=posix \
    -DEXECUTORCH_BUILD_VULKAN=ON \
    -DEXECUTORCH_BUILD_TESTS=ON \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_FLAGS="-include algorithm"

cmake --build cmake-out-vk -j$(nproc) --target install --config Release

cmake backends/vulkan/test/custom_ops/ \
    -Bcmake-out-vk/backends/vulkan/test/custom_ops \
    -DCMAKE_INSTALL_PREFIX=cmake-out-vk \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DEXECUTORCH_ROOT=$(pwd) \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache

cmake --build cmake-out-vk/backends/vulkan/test/custom_ops -j$(nproc)
```

Notes:

- The `-DCMAKE_CXX_FLAGS="-include algorithm"` flag is intentional for this
  machine.
- Do not replace this with generic ExecuTorch setup commands unless the tested
  sequence fails and the failure is documented.
- If a build fails due to missing dependencies or sandbox restrictions, record
  the exact failure in the agent report.

## Research Sessions

Use four separate sessions:

1. `1_kernel_sweep_fp16_rdna3.md`
   - Complete. Phase 1 built the kernel-side truth table.
   - Swept fp16 WMMA software parameters against Stephen's baseline.
   - Produced a kernel report with best configs, dispatch gates, texture-linear
     prototype results, and sampled large-shape correctness.

2. `2_real_llama_e2e_storage_study.md`
   - Complete. Phase 2 showed texture3D linear coopmat transfers to real
     LLaMA 3.1 8B 4-layer fp16 seq=256 E2E.
   - Produced an E2E report with storage propagation, texture vs buffer,
     ETDump, wallclock, and seq=2048 OOM caveats.

3. `3_production_integration_design.md`
   - Convert findings into a production integration design.
   - Define shader selection, fallback, tests, and risk controls.

4. `4_int8_coopmat_exploration.md`
   - Planned after Phase 3.
   - Explore int8/uint8 cooperative matrix capability, quantized Vulkan path
     compatibility, kernel speed, correctness, and E2E value.
   - This is a research phase; production int8 should be a later phase only if
     Phase 4 shows real model wins.

Do not jump directly from fp16 microbenchmarks to production integration. The
previous experiment showed that strong kernel speedups can regress real LLaMA
E2E if storage layout is wrong. Likewise, do not jump directly from int8
hardware capability to int8 production work; Phase 4 must prove kernel and E2E
value first.

Current Phase 1 summary:

```text
HW WMMA tile: 16x16x16 fixed by AMD/RADV hardware
input type: fp16 x fp16
accumulator sweep: fp16, fp32 -> use fp32
subgroup sweep: 32, 64 -> default 64
macro tile sweep: 16x16, 16x32, 32x16, 32x32, 16x64, 64x16, 32x64, 64x32, default 64x64 -> default 64x64
K-step sweep: 16, 32, 64 -> use 32
storage: buffer input/output + buffer weights; texture3D input/output + buffer weights for linear only
ops: linear and matmul buffer coopmat tested
correctness: full CPU reference for small shapes; sampled CPU reference for large shapes
```

Read the current Phase 1 summary before planning new work:

```text
yanwen_docs/agent_plans/current_findings_after_kernel_sweep.md
yanwen_docs/agent_reports/kernel_sweep_fp16_rdna3.md
```

Current Phase 2 summary:

```text
real_tex_stephen: 442.63 ms
real_tex_coopmat: 250.43 ms = 1.77x speedup
real_buf_coopmat: 205.29 ms = 2.16x vs texture Stephen at seq=256

texture3D linear coopmat is the Phase 3 default candidate.
Do not force whole-graph buffer storage based only on seq=256.
seq=2048 OOM-killed on this 28 GB host; previous seq=2048 evidence showed
severe whole-graph buffer-storage regression.
```

Read the current Phase 2 summary before production design:

```text
yanwen_docs/agent_plans/current_findings_after_phase2_e2e.md
yanwen_docs/agent_reports/real_llama_e2e_storage_study.md
```

## Previous Code Intake

Yanwen may provide previous coopmat code as a branch diff. Preferred formats:

```bash
git diff <baseline_commit>..<coopmat_branch> > yanwen_docs/background/coopmat_previous_work.diff
```

or:

```bash
git format-patch <baseline_commit>..<coopmat_branch> -o yanwen_docs/background/coopmat_patches
```

Agents should inspect the diff selectively. Do not blindly apply all previous
changes without reconciling them with the current branch.

Expected previous-work assets may include:

```text
linear_coopmat.glsl
matmul_coopmat.glsl
coopmat YAML variants
KHR cooperative matrix capability query helpers
benchmark binaries
ETDump helper scripts
real LLaMA runner changes
```

## Reporting Convention

Write reports under:

```text
yanwen_docs/agent_reports/
```

Write raw benchmark artifacts under:

```text
yanwen_docs/agent_results/
```

If an agent encounters a problem, confusing behavior, failed approach, tooling
issue, or non-obvious discovery that Yanwen should know, document it under:

```text
yanwen_docs/lesson_learned/<phase_name>/
```

Use a short Markdown file with a descriptive name, for example:

```text
yanwen_docs/lesson_learned/phase2_real_llama_e2e/buffer_storage_host_gap.md
yanwen_docs/lesson_learned/phase1_kernel_sweep/subgroup_size_control_blocker.md
```

See `yanwen_docs/lesson_learned/README.md` for the current phase grouping.

Each lesson should include:

- what was attempted
- what happened
- why it matters
- exact commands/log snippets if relevant
- recommended next action or workaround

Each report should include:

- exact git commit and local changes used
- device info
- build commands
- run commands
- result tables
- correctness status
- failures or inconclusive findings
- recommended next action
