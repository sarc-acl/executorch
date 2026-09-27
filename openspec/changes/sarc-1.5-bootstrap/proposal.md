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

## Results

(filled in by the verification run; raw logs in `results/`)
