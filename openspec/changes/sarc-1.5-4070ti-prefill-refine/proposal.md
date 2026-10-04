# sarc-1.5-4070ti-prefill-refine

Dev zone only. Branch `topic/4070ti-prefill-refine`, forked from `topic/780m-prefill-refine` at `6a7cc8cc6`
(the parent of every comparison: that commit with no profile, i.e. the shipped NVIDIA rows).

## Why

Push the 2048-token Llama prefill on the RTX 4070 Ti SUPER (Vulkan backend) as far as it goes, measured
against the parent in the same session: SDPA prefill kernels for this device first, then 8da4w and 4w linear.

## What

So far only `tools/`: the 780M campaign's tools adapted to gpu-dev-4004. No kernel, no profile, no measurement.
The campaign is blocked on host write permissions; see `STATUS.md`.
