# sarc-1.5-bootstrap

## Why

- **Decision.** On 2026-09-27 the owner moved all SARC ExecuTorch work to release 1.5, on one development
  branch (`dev/1.5`) with a release zone and an additive dev zone. Releases (`sarc/1.5-rN`) are generated
  from it.
- **Problems it fixes.** The 1.4 per-GPU branches (`yanwen/release14-quant-shaders*`):
  - needed every fix cherry-picked five times;
  - drifted apart;
  - let a change for one GPU alter another GPU's shaders;
  - split this openspec content into three diverging copies.

## What

- **Scaffold.**
  - Selection layer: `impl/sarc/Select.*` and the vendor tables.
  - 4w op hook: `impl/sarc/Q4gswCoopmat.cpp`, plus `sarc/HOOKS`.
  - Shaders: shared body plus shipped and sweep wrappers.
  - Dev overrides: `impl/sarc_dev/Overrides.cpp`.
  - Tests: `test/sarc_dev/`.
  - Tools: `sarc/tools/{build,verify,check,make-release}.sh` and `spirv_golden.py`.
- **Proof of concept.** Radeon 780M 4w coopmat (`t128x128k32g42s32f32c`), ported from
  `yanwen/release14-quant-shaders-780m` e4786474b9.
  - The shader is byte-identical to the 1.4 one, split into a body and a wrapper.
  - A `buffer_buffer` variant is added (missing on 1.4).
  - The device match is case-insensitive.
  - Launch dims come from the table instead of kernel-name parsing.
- **Upstream backport.** 03f41d2031 (`<algorithm>` includes); release 1.5 does not build with GCC 15+ without it.

## Done when

- **Mechanical checks:**
  - `check.sh` passes.
  - The release export differs from `release/1.5` only in `glsl/sarc/`, `impl/sarc/` and the `sarc/HOOKS` lines.
- **Microbenchmark on the 780M** (`verify.sh`):
  - every 1B/3B/8B 4w prefill shape dispatches `sarc_linear_q4gsw_coopmat_t128x128k32g42s32f32c_*`;
  - kernel medians within ±3 % of the 1.4 confirmation `.artifacts/roofline-et-study/confirm/780m-final`.
- **Correctness:** correctness and production-diff pass.
- **End to end (1B/3B/8B 4w):**
  - prefill tok/s for stock 1.5, tiled and SARC;
  - next token == tiled on the real-text and the unaligned prompt;
  - decode works.
- **Other devices unaffected:** a device without rows (B580) dispatches exactly the stock 1.5 kernels.

## Results (2026-09-27)

**Setup.**
- Device: Radeon 780M on rocky-ryzen, Mesa 25.2.7 RADV, under the gpu-lab lock, host idle, default DVFS
  (clocks not pinned).
- Builds: `sarc/tools/build.sh` (`localhost/et-vk-build:rocky10`, shaderc v2023.8).
- PTEs: the existing `llama3_*_vulkan_4w.pte` in `/mnt/linux-share/models` (exported with the 1.4 flow).
- Raw logs are in `results/780m/{sarc,stock,release}` and `results/b580-norow`.

**Kernels.**
- All 24 4w prefill shapes (1B/3B/8B × 4 ops × buffer/texture3d) dispatch
  `sarc_linear_q4gsw_coopmat_t128x128k32g42s32f32c_{texture3d,buffer}_texture2d_half`.
- Median kernel time vs the 1.4 confirmation (`confirm/780m-final`) is 0.988×, range 0.984–0.998×
  (single run).

**Correctness.**
- All 4w numeric checks pass, including the rank-3 cases.
- The binary exits with rc = 1 only because the 8da4w rank-3 cases fall back to tiled; 8da4w is not
  ported yet.
- Production-diff passes for 1B/3B/8B × buffer/texture3d (6/6).

**End to end, 2048-token prefill, tok/s:**

| Build / mode | 1B | 3B | 8B |
|---|---:|---:|---:|
| tiled (dev build, `ET_VK_FORCE_TILED_LINEAR`) | 714 | 246 | 104 |
| stock release 1.5 (upstream `q4gsw_linear_gemm__tin__w_4x8`) | 1205 | 421 | 194 |
| SARC (dev build, `ET_VK_SARC_UNVERIFIED=1`) | 1916 | 683 | 352 |
| SARC release export (row verified, no env) | 1930 | 690 | 352 |
| SARC vs stock 1.5 | 1.60× | 1.64× | 1.81× |

**Checks.**
- Next token: the release export and SARC match tiled on the 1973-token real-text prompt and on the
  1304-token unaligned prompt (fallback path).
- Decode, 32 tokens: SARC 70.8–73.5 tok/s vs stock 69.5.
- Other devices unaffected: on the Arc B580 (no rows), the dev build dispatches exactly the stock 1.5
  kernels (48/48 cases).
- `check.sh` passes: zone rule, twins, selection test, host and Android release-export builds, SPIR-V golden.
- The SPIR-V verified on the device (dev build) is byte-identical to the release export.

**Not verified.**
- The `buffer_buffer` variant (buffer weights) builds and is in the golden, but no microbench or PTE
  case uses buffer-stored 4-bit weights, so it has not run on the device.
- Clocks were not pinned.

**Finding: 1.5 vs 1.4 end to end.**
- The 1.4 780M branch reached 2702 tok/s (1B), versus 1916 here, although the linear kernels are ~1 %
  *faster*.
- Both tiled and SARC gain the same ~311 ms per prefill: tiled 2557 → 2868 ms, SARC 758 → 1069 ms. So
  the difference lies outside the 4w linears.
- The most likely cause is the SDPA coopmat work (`sdpa_*_coopmat`) carried by the 1.4 dev branches and
  not yet ported to 1.5. **Not yet confirmed**; the next step is `test_llama_microbench --sdpa` / ETDump
  on both.
