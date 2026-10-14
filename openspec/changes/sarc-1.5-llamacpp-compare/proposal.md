# sarc-1.5-llamacpp-compare

## Why

Every published SARC number so far compares ExecuTorch with ExecuTorch (stock `release/1.5` against the SARC
kernels). That says how much the kernels gained, not where the result stands against the runtime most people
use for the same models on the same GPUs. This change measures the same Llama prefill workload under llama.cpp
and, on NVIDIA, under ExecuTorch's CUDA backend, so the SARC Vulkan result can be placed against both. It is
wanted this week, for an external presentation.

## What Changes

- A cross-runtime prefill measurement on the five GPUs of `sarc-1.5-e2e-benchmark` (Radeon 780M, Arc B580,
  Arc Pro B70, RTX 4070 Ti SUPER, Jetson Orin Nano): Llama 3.2 1B, 3.2 3B and 3.1 8B, one fixed 2048-token
  prompt, prefill tokens per second.
- ExecuTorch Vulkan arms: stock `release/1.5`, the SARC release measured in `sarc-1.5-e2e-benchmark`, and the
  latest gate-accepted tuned profile of each device's `topic/<gpu>-prefill-refine` branch.
- llama.cpp arms, one pinned commit on every device: Vulkan everywhere, CUDA on the two NVIDIA devices, SYCL on
  the two Intel devices; quantizations Q4_0 (uniform 4-bit, closest to `4w`) and Q4_K_M (the common default);
  three settings tiers (out of the box, aligned to one 2048-token batch, best documented settings).
- ExecuTorch CUDA arm on the RTX 4070 Ti SUPER: upstream plus the minimal export fix that lets 3B and 8B take
  a 2048-token prompt, `4w` only.
- A measurement kit (dev zone, under this change) and aggregated results; raw logs stay outside the repository.
- Order: Arc B580 first as the pilot that fixes the protocol, then the RTX 4070 Ti SUPER, then the other three
  as their tuning campaigns free the device.
- Out of scope here: decode (generation) speed, any kernel change, any promotion, any change to the release
  zone or to another change's results.

## Capabilities

### New Capabilities

- `cross-runtime-benchmark`: how an ExecuTorch result is compared with another runtime on the same device:
  which arms exist, what "the same workload" means, how runs are taken and judged valid, what must be
  disclosed with every number, and how cells that cannot be run are reported.

### Modified Capabilities

None. The repository has no main specs yet.

## Impact

- New files only, all in the dev zone: `openspec/changes/sarc-1.5-llamacpp-compare/` (proposal, design, spec,
  tasks, `kit/`, `results/`). No shader, no selection table, no hook, no golden changes.
- External inputs, none vendored: llama.cpp at one pinned commit, a oneAPI container image for the SYCL builds,
  the CUDA toolkits already on the NVIDIA hosts, ExecuTorch upstream `main` for the CUDA backend.
- Device time: each device must be exclusive while it is measured, so the work on a device waits for that
  device's tuning campaign, except where the owner decides otherwise.
- Results are private to the owner until the owner releases them.
