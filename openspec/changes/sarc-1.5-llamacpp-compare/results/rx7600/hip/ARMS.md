# Radeon RX 7600, llama.cpp HIP (ROCm): arms of the timed session (written before it, 2026-10-12)

Request 10.2 of the company-side comparison, on the card of `results/rx7600/`, with the owner decision of 2026-10-12 (21:44 UTC):
**one interleaved session with all arms of the device**, ExecuTorch (stock, SARC, tuned), llama.cpp Vulkan and llama.cpp HIP, so that
every ratio of the HIP result comes from one session and no number is compared across days. The tuning campaign, the Vulkan arms and
the 2026-10-10 session (`../ARMS.md`, session `rx7600-r2`) are unchanged; the Vulkan and ExecuTorch arms below are the same binaries,
`.pte` and GGUF files, environments and settings as there.

## Drivers, per arm

| arms | driver / runtime |
|---|---|
| ExecuTorch `stock`, `sarc`, `tuned` (`4w`, `8da4w`) | Vulkan, user-space RADV Mesa 26.2.3 (`31e9a6b2e9`), `VK_ICD_FILENAMES`; arms, commits and environments as `../ARMS.md` |
| llama.cpp Vulkan (`lc-*`, `lb-*`, `default` and `best`, both quantizations) | Vulkan, the same RADV, `-dev Vulkan0`; the `b11430` Vulkan build and the `best` of `../ARMS.md` (`-b 2048 -ub 512 -fa on`) |
| llama.cpp HIP (`hip-*`) | ROCm 7.2.4 user-space HIP runtime (installed on the workstation, not by this task), the card's kernel driver unchanged, `-dev ROCm0`; the card reports `gfx1102`, wave size 32 |

## How the HIP build was made

Built on this workstation with the installed ROCm 7.2.4, no container, nothing installed, no package or group changed; the build ran as the user,
in a shell whose `id` lists group `render`, and `rocminfo` lists `gfx1102` "AMD Radeon RX 7600" (checked at the start of the round).

```
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" cmake -S <llama.cpp b11430> -B <scratch>/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1102 -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF
cmake --build <scratch>/build --target llama-completion llama-bench
```

`hipconfig` 7.2.53211, AMD clang 22.0.0 (`roc-7.2.4`). The binaries carry an `RPATH` to the ROCm libraries (`patchelf --force-rpath`) so that the Vulkan and
HIP arms run in one session. **The native `gfx1102` target builds and runs: `HSA_OVERRIDE_GFX_VERSION` and the `gfx1100` fallback are not used.** The
binaries report `build: 8345f33 (11430)`. sha256 `llama-bench` `9fbc80c0…`, `llama-completion` `ff549476…`, `libggml-hip.so` `1e50cc5f…`.
This is the build of the earlier (stopped) round, unchanged since.

GPU use is checked, not assumed (`gpu-check.txt`, 1B Q4_0): `backends: ROCm`, device `ROCm0` (`gfx1102`); the verbose `llama-completion` log lists
`offloaded 17/17 layers to GPU`; the card's `gpu_busy_percent` reached 100 % and its VRAM use grew from 33 to 5393 MiB; 16 generated tokens are fluent English.
A CPU fallback would be a stop condition. The session's `check` runs repeat the text check for every `lc` arm.

## Arms of the session (`arms.tsv`)

22 arms on 1B, 3B and 8B:

- six ExecuTorch arms (`stock`, `sarc`, `tuned`, each `4w` and `8da4w`);
- eight Vulkan arms `{lc,lb}-{q4_0,q4_k_m}-{default,best}` (`default` = `-ngl 99 -dev Vulkan0`; `best` = `-ngl 99 -dev Vulkan0 -b 2048 -ub 512 -fa on`);
- eight HIP arms `hip-{lc,lb}-{q4_0,q4_k_m}-{default,best}` (the requested `hip-q4_0` and `hip-q4km`, with the timer and the setting in the name;
  the aggregated `kind` column keeps the timer, `lc` / `lb`, and the rows are the ones whose arm name starts with `hip-`; `aggregate.py` unchanged).
  `default` = `-ngl 99 -dev ROCm0`; `best` = `-ngl 99 -dev ROCm0 -b 2048 -ub 512 -fa on` for Q4_0 and `-b 2048 -ub 1024 -fa on` for Q4_K_M (screen below).

`lc` = `llama-completion` in a fresh process, `lb` = `llama-bench`; both timers for every llama.cpp arm. Validity: as `../ARMS.md` (the campaign's rule through
`kit/hosts/rx7600/host.sh`, unchanged: clock floor 2420 MHz, at least five clock samples, exactly 2048 prompt tokens, no foreign GPU user; no foreign-busy ceiling on this
card). Builds on the workstation during the session are recorded per run and in a one-second watcher, as before.

## Settings screen (fixes `best`)

`screen.csv`, `screen-runs.csv`, session `hipscreen2`, 2026-10-12 22:20 to 22:40 UTC, 1B and 3B, both quantizations: `llama-bench` at the default setting
(`-ub 512`, flash attention auto) and at `-b 2048` with `-ub {512, 1024, 2048}` x flash attention {on, off} (the kit's list, the same as for Vulkan);
`llama-completion` default (three runs after the discarded repetition 0); the tuned `4w` arm as an anchor (three runs). All 51 timed rows are valid. `lb` tok/s, median
of the five repetitions of one process (1B / 3B), and the geometric mean of the two:

| quantization, setting | 1B | 3B | geometric mean |
|---|---:|---:|---:|
| Q4_0 default | 6342 | 2540 | 4014 |
| Q4_0 `-ub 512` fa on / off | 6340 / 4186 | 2540 / 2003 | 4013 / 2896 |
| Q4_0 `-ub 1024` fa on / off | 6175 / 3962 | 2578 / 1925 | 3990 / 2761 |
| Q4_0 `-ub 2048` fa on / off | 5757 / 3507 | 2524 / 1758 | 3812 / 2483 |
| Q4_K_M default | 5937 | 2338 | 3726 |
| Q4_K_M `-ub 512` fa on / off | 5932 / 4004 | 2340 / 1876 | 3725 / 2741 |
| Q4_K_M `-ub 1024` fa on / off | 5924 / 3858 | 2419 / 1829 | **3786** / 2656 |
| Q4_K_M `-ub 2048` fa on / off | 5527 / 3409 | 2371 / 1681 | 3620 / 2393 |

Reading, by the rule of the screen (highest geometric mean of the two models' `lb` medians, per quantization): Q4_0, the default and `-ub 512` on tie within 0.02 %
(the default is the same as `-ub 512` with flash attention auto), so `best` is the highest-ranked setting other than the default, `-b 2048 -ub 512 -fa on` (as for
Vulkan on this card); Q4_K_M, `-ub 1024` on is the top setting, 1.6 % above the default, so `best` = `-b 2048 -ub 1024 -fa on`. Flash attention off is 20 to 39 % slower than on
at the same `-ub`; `-ub 2048` is the slowest flash-attention-on setting at 1B. Unlike Vulkan on this card, HIP shows no slow path at 3B (flash off is slower, not 5 to 20 times).
`best` and `default` are both measured, so the table shows both. The kit's list was tried in full; no other llama.cpp option was screened.

**Observation recorded here, before the session.** The tuned `4w` anchor of this screen reads 11070 tok/s at 1B and 4267 at 3B, which are the values of the
2026-10-10 session (`rx7600-r2`: 11070 and 4267). The Vulkan baseline cell measured at 21:28 UTC the same day (`hip/baseline-before-runs.csv`) read 12118 for the same arm, 9.5 % higher.
Between the two measurements this task did not install, change or restart anything; the workstation, its card and the arm's binary are the same. The cause is UNVERIFIED and not examined; per the owner decision no number is
compared across days, and the timed session measures every arm itself.
