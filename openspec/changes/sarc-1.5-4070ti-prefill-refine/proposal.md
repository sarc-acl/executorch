# sarc-1.5-4070ti-prefill-refine

Dev zone only. Branch `topic/4070ti-prefill-refine`, forked from `topic/780m-prefill-refine` at `6a7cc8cc6`
(the parent of every comparison: that commit with no profile, i.e. the shipped NVIDIA rows).

## Why

Push the 2048-token Llama prefill on the RTX 4070 Ti SUPER (Vulkan backend) as far as it goes, measured
against the parent in the same session: SDPA prefill kernels for this device first, then 8da4w and 4w linear.

## What

Nothing is measured yet: the campaign is blocked on host write permissions, see `STATUS.md`. In the tree so far:

- `tools/`: the measurement, gate and build tools for gpu-dev-4004, and two generators.
- SDPA prefill kernels as dev-zone families of this device (`glsl/sarc_dev/sarc_sdpa_{qk,av}_coopmat_4070ti*`),
  13 subgroup-32 variants, selected with `ET_VK_SARC_DEV_PROFILE=4070ti-*`. The shaders are the 780M campaign's
  dev twins under new names. They compile; they have not run.
- Phase-timing twins of the shipped 4w and 8da4w kernels (`glsl/sarc_dev/sarc_dev_prof_4070ti_*`), measurement
  only.
- `tools/local-hook-nvidia-sdpa.patch`: the smallest release-zone hook SDPA needs on this device (two
  `kUnverified` rows in `impl/sarc/table_nvidia.cpp`). It is documentation and a build input for candidate
  trees; it is not applied to the branch.
