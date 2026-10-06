# RTX 4070 Ti SUPER: arms and results (llama.cpp Vulkan and CUDA)

Session `s2`, 2026-10-06, after the tuning campaign had finished and pushed. 16 arms; exact lines in `arms.tsv`.
Validity: the campaign's lowest calibrated clock floor (2502 MHz median in the prefill window), no other GPU
workload, at least two clock samples in the window (nvidia-smi answers every 20 ms; a 1B prefill lasts 60 to
70 ms).

**Session `s1` was discarded.** The first version of this device's sampler left its `nvidia-smi` poller
running after each run; 160 of them had accumulated when it was noticed, run-to-run spread had reached 40 to
66 % and the reference arms read low. The session was stopped, the pollers removed, the sampler fixed (it now
ends its children), and the whole session taken again. `s2` ran with one poller at a time and its ExecuTorch
arms reproduce the campaign's final numbers (tuned 4w: 29681 / 12881 / 5988 here, 29681 / 12800 / 5988 there).

- ExecuTorch: `stock` (the `release/1.5` binary built for the B580 session), `sarc` = `6a7cc8cc6`, `tuned` =
  `6397f868f` with `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=4070ti-refine1` (the committed branch that
  the campaign's last gate accepted; unmerged development branch).
- llama.cpp `b11430`: Vulkan built for this host's CPU (`-march=znver4`, in an Ubuntu 24.04 image), CUDA built
  on the host (`-DGGML_CUDA=ON -DGGML_NATIVE=ON`, CUDA 13.4). llama.cpp's Vulkan backend reports
  `matrix cores: NV_coopmat2` on this card.
- Settings screen (1B, Q4_0, llama-bench tok/s, `screen.csv`): Vulkan default 30038, `-ub 1024` flash attention
  on 32117 / off 17854, `-ub 2048` on 31032 / off 14241; CUDA default 40889, `-ub 1024` on 39741, `-ub 2048` on
  36181 / off 15949. Three settings per backend were then carried through all models; the table takes the
  fastest per model.

## Result (tok/s, median of 5; llama.cpp by llama-bench, Q4_0, its fastest of the three settings)

| model | stock 4w | SARC 4w | SARC 8da4w | tuned 4w | tuned 8da4w | llama.cpp Vulkan | llama.cpp CUDA | tuned 4w / Vulkan | tuned 4w / CUDA | tuned 8da4w / CUDA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1B | 6804 | 19692 | 21113 | 29681 | 33032 | 31893 | 40861 | 0.93 | 0.73 | 0.81 |
| 3B | 2554 | 8715 | 9660 | 12881 | 14949 | 12471 | 15837 | 1.03 | 0.81 | 0.94 |
| 8B | 1110 | 4472 | 5007 | 5988* | 6954* | 5539 | 7418 | 1.08 | 0.81 | 0.94 |

\* four valid runs; one more of each read the same speed with a clock sample pattern below the floor, and one
tuned 4w run exited without statistics (the runner's known abort after a slow model load).

On this card, unlike the Intel and AMD ones, llama.cpp's Vulkan backend is level with the tuned ExecuTorch
kernels (7 % ahead on 1B, 3 to 8 % behind on 3B and 8B), and its Q4_K_M files run faster than Q4_0 on Vulkan
(34239 / 12765 / 6235), which puts Q4_K_M ahead of tuned 4w on 8B as well. llama.cpp CUDA is 23 to 38 % faster
than tuned 4w and 6 to 24 % faster than tuned 8da4w. llama.cpp's fresh-process timer agrees with its warm one
within 10 % on CUDA here. Text check: all arms continue the real-text prompt fluently.
