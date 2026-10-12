# Radeon RX 7900 XTX, llama.cpp HIP (ROCm): arms of the timed session (written before it, 2026-10-12)

Request 10.2 of the company-side comparison, on the card of `results/7900xtx/`. The tuning campaign and the 2026-10-10 session
(`../ARMS.md`) are unchanged; this session adds the HIP backend of llama.cpp beside the ExecuTorch arms and the Vulkan reference
arms, in one interleaved session, so that the ratios share the conditions.

## Drivers, per arm

| arms | driver / runtime |
|---|---|
| ExecuTorch `stock`, `sarc`, `tuned` (`4w`, `8da4w`) | Vulkan, AMDVLK 2025.Q2.1 (LLPC), `VK_ICD_FILENAMES` of the campaign; same binaries, `.pte` files and environment as `../ARMS.md` |
| llama.cpp Vulkan reference (`lb-q4_0-best`, `lb-q4_k_m-best`) | Vulkan, AMDVLK 2025.Q2.1, `-dev Vulkan0`; the `b11430` Vulkan build of `../ARMS.md` |
| llama.cpp HIP (`hip-*`) | ROCm 7.2.1 HIP runtime, kernel driver `amdgpu` 6.16.13 (DKMS of the host, unchanged), `-dev ROCm0`; the card reports `gfx1100`, wave size 32 |

## How the HIP build was made (host untouched)

The host has the ROCm 7.2.1 runtime packages only (`/opt/rocm`: no `hipcc`, no HIP headers, no hipBLAS or rocBLAS), and the
distribution (Ubuntu 25.04) is not the one ROCm 7.2.1 is built for (24.04). Nothing was installed on the host. The build ran in a
throw-away container of the public image `rocm/dev-ubuntu-24.04:7.2.1`
(digest `sha256:fe29799baf7b8db986a707110c15579dc0fb6e39d48973fbb82dfc33a0c7e179`, pulled with the user's docker access; the host's other images were not
touched). Inside the container only (root of the container, `apt` of the image's own ROCm 7.2.1 repository): `cmake` 3.28.3,
`ninja-build`, `hipblas-dev` 3.2.0.70201, `rocblas-dev` 5.2.0.70201 (the image has `hipcc` 1.1.1.70201 and the ROCm LLVM, AMD clang 22.0.0 `roc-7.2.1`,
but neither cmake nor the BLAS libraries). The container reached the package repository through the host's CA bundle, mounted read-only
(the company web gateway's CA is not in the image; certificate verification stayed on).

```
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" cmake -S <llama.cpp b11430> -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF
cmake --build build --target llama-completion llama-bench
```

(`-DGGML_NATIVE=OFF` plus AVX2 as for the Vulkan build of this card, whose CPU has no AVX-512.) Source: the pinned worktree of
`b11430` (`8345f333951c661d166b00e6f9362e553768f292`) copied without `.git`; the binaries report `build: 8345f33 (11430)`. The
runtime pieces the binaries load (`libamdhip64`, `libhipblas`, `librocblas` with its `gfx1100` kernel library, `libhipblaslt`,
`librocsolver`, `libhsa-runtime64`, `libamd_comgr`, ...: all ROCm 7.2.1, the same version as the host's runtime) were copied from the
container into a directory next to the binaries, and the binaries carry an `RPATH` to it (`patchelf --force-rpath`), so that the
Vulkan and HIP arms can run in one session with one `LD_LIBRARY_PATH`. No container runs during the timed session: the HIP
binaries run on the host, in the session's own process tree, with the kit's guard, sampler and rules unchanged. sha256 of
`llama-bench` `564ab5c4…`, `llama-completion` `08af573d…`, `libggml-hip.so` `a1ad1881…`.

GPU use is checked, not assumed (`gpu-check.txt`): the verbose log of a HIP `llama-completion` run lists `offloaded 17/17 layers to GPU` on
`ROCm0`, the card's `gpu_busy_percent` reaches 100 % and its VRAM use grows by 1.4 GiB during the run, and every `llama-bench` log of
the HIP arms says `backends: ROCm`. A CPU fallback would be a stop condition.

## Arms of the session (`arms.tsv`)

16 arms on 1B, 3B and 8B: the six ExecuTorch arms (`stock`, `sarc`, `tuned`, each `4w` and `8da4w`, as `../ARMS.md`, commits there),
two Vulkan reference arms (`lb-q4_0-best`, `lb-q4_k_m-best`: the better timer of the 2026-10-10 session, here for the in-session
baseline), and eight HIP arms: `hip-{lc,lb}-{q4_0,q4_k_m}-{default,best}` (requested names `hip-q4_0`, `hip-q4km`; the timer and the
setting are part of the arm name, as `lc-q4_0-best` etc. for Vulkan, and `sy-` for the SYCL rows). `default` = `-ngl 99 -dev ROCm0`;
`best` = `-ngl 99 -dev ROCm0 -b 2048 -ub 1024 -fa on`. `lc` = `llama-completion` in a fresh process, `lb` = `llama-bench`, both
timers for every HIP arm. The aggregated `kind` column keeps the timer (`lc` / `lb`; `aggregate.py` is unchanged): the HIP rows are the ones whose arm name
starts with `hip-`, in `results/cells.csv` too.

Validity: as `../ARMS.md` (the campaign's rule through `kit/hosts/7900xtx/host.sh`, clock floor 2670 MHz, thermal bit 36 masked, card
idle before each run, no foreign GPU user, no host build). The rule is the same for the HIP arms. Known from the screen and from the Vulkan session:
`lc` has no full-prompt warm-up in its process, so its window includes the clock ramp and every `lc` run was `clock_low` in the screen;
how such runs are counted will be the same as in `../README.md` and is stated in the README of this session.

## Settings screen (fixes `best`)

`screen.csv`, `screen-runs.csv`, session `hipscreen1`, 2026-10-12 18:41 to 18:51 UTC, 1B and 3B, both quantizations: `llama-bench` at
the default setting (`-ub 512`, flash attention auto) and at `-b 2048` with `-ub {512, 1024, 2048}` x flash attention {on, off}
(the kit's list, the same as for Vulkan); `llama-completion` default; the tuned `4w` arm as an anchor. `lb` tok/s, median of five
repetitions in one process (the first repetition of a 1B process is up to 20 % slower than the rest: the card's clock
ramp, which the median absorbs):

| model, quant | default | ub512 fa on / off | ub1024 fa on / off | ub2048 fa on / off |
|---|---:|---:|---:|---:|
| 1B Q4_0 | 23180 | 22894 / 15507 | **23989** / 14181 | 22149 / 13611 |
| 3B Q4_0 | 8420 | 8428 / 6992 | 9005 / 6893 | **9018** / 6312 |
| 1B Q4_K_M | 21868 | 21824 / 14974 | **22187** / 13708 | 19356 / 12490 |
| 3B Q4_K_M | 8107 | 8140 / 6794 | **8593** / 6614 | 8324 / 6037 |

Flash attention off is 16 to 41 % slower than on at the same `-ub`. `-ub 1024` with flash attention on is the top setting in three of
four rows; in the fourth (3B Q4_0) `-ub 2048` is 0.15 % higher, inside the spread of the repetitions, so one `best` is used for
both quantizations and all models: `-b 2048 -ub 1024 -fa on` (the same string as the Vulkan `best` of this card), 1.5 to 6.9 %
above the default of the same row (1B Q4_0 3.5 %, 3B Q4_0 6.9 %, 1B Q4_K_M 1.5 %, 3B Q4_K_M 6.0 %). The kit's list was tried in full;
no other llama.cpp option was screened. The `lc` default runs of the screen (five per row: three planned plus two replacements, after one discarded repetition 0; the first version of this file said three, corrected 2026-10-12) are all `clock_low` and are listed in `screen-runs.csv`.

## Baseline before the HIP work

Vulkan cells measured at 18:38 UTC (`hipbase-pre`, 1B, five repetitions, same arms and rules as the session): tuned `4w`
23273 / 23011 / 22022 / 22756 / 24094 tok/s (median 23011; the first version of this file said 22756, which was misread: 22756 is the fourth value, not the median of the five; 2026-10-10 session 22261), `lb` Q4_0 `best` 16294 (2026-10-10 session
16409). The same two cells are measured again after the timed session (`hipbase-post`); the request asks for agreement within 2 %.
