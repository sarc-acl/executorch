# Release of the comparison results

**2026-10-06, owner decision: all numbers of this comparison are public.** No arm, device, model or backend is
withheld. This covers every result in this directory: ExecuTorch Vulkan (stock, SARC, tuned), llama.cpp Vulkan,
SYCL and CUDA on the five GPUs, and the ExecuTorch CUDA backend on the RTX 4070 Ti SUPER, including the cells
where llama.cpp is faster than the tuned ExecuTorch kernels.

What goes with the numbers wherever they are shown, as the spec requires: the arms are not identical
configurations (quantization format and bits per weight, batch handling, export path); the tuned ExecuTorch
arms are unmerged development branches, some accepted by the reference-error rule; llama.cpp is quoted at its
best measured setting and by its warm timer; the Radeon 780M cells marked in `780m/README.md` ran at a
workload-dependent clock; the first RTX 4070 Ti SUPER session was discarded and re-measured.

`cells.csv` in this directory collects every cell of every device with its source file. The per-device
`README.md` files give the arms, the settings screens and the result tables.

Releasing the numbers is not the same as publishing the branch: `topic/llamacpp-compare` is a local branch and
has not been pushed.
