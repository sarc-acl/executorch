# Matmul coopmat build typo

## What was attempted

Rebuilt the Vulkan runtime after adding the first `matmul_coopmat` dispatch path.

## What happened

This exact command failed:

```bash
cmake --build cmake-out-vk -j$(nproc) --target install --config Release
```

The compile error was:

```text
backends/vulkan/runtime/graph/ops/impl/Matmul.cpp:159:21: error: base operand of '->' has non-pointer type 'vkcompute::ComputeGraph'
```

## Why it matters

`can_use_matmul_coopmat` receives `ComputeGraph& graph`, not `ComputeGraph*`.
Using `graph->size_at` broke the runtime build before any benchmark could run.

## Fix

Use `graph.size_at<...>` in that helper. The rebuild succeeded after this
one-line correction.
