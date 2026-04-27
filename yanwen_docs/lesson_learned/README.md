# Lessons Learned

Lessons are grouped by research phase.

## Phase 1: Kernel Sweep

Directory:

```text
yanwen_docs/lesson_learned/phase1_kernel_sweep/
```

Scope: fp16 RDNA3 WMMA kernel microbenchmarks, shader variants, benchmark
harness behavior, CMake shader registry issues, macro-tile/K-step/subgroup
sweeps, texture linear coopmat prototype, and sampled large-shape correctness.

## Phase 2: Real LLaMA E2E Storage Study

Directory:

```text
yanwen_docs/lesson_learned/phase2_real_llama_e2e/
```

Scope: real and synthetic LLaMA E2E runs, storage propagation, seq=2048 OOM,
and toolchain/import blockers encountered while running the real LLaMA helper.

## Phase 3: Production Integration

Directory:

```text
yanwen_docs/lesson_learned/phase3_production_integration/
```

Scope: fp16 production dispatch, capability gating, fallback behavior, test
coverage, storage integration, and build-system issues found while landing the
texture-backed linear coopmat path.

## Phase 4: int8 Cooperative-Matrix Exploration

Directory:

```text
yanwen_docs/lesson_learned/phase4_int8_coopmat_exploration/
```

Scope: int8/uint8 cooperative-matrix capability, quantized Vulkan path
compatibility, packing/dequantization cost, correctness, and E2E value relative
to the fp16 coopmat path.

## Future Phases

Create a new subdirectory per phase instead of placing new lesson files at this
directory root. Keep lesson filenames descriptive and link them from the
corresponding agent report.
