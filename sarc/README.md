# SARC development zone (dev/1.5)

`dev/1.5` is SARC's single maintained ExecuTorch branch, forked from upstream `release/1.5`. It holds two
zones:

| Zone | Paths | Rule |
|---|---|---|
| Release | `backends/vulkan/runtime/graph/ops/{glsl,impl}/sarc/`, plus the hook lines listed in `sarc/HOOKS` | Validated content only. A release is exactly this on top of `release/1.5`. |
| Dev | `backends/vulkan/runtime/graph/ops/{glsl,impl}/sarc_dev/`, `backends/vulkan/test/sarc_dev/`, `sarc/`, `openspec/` | New files only; never edits release-zone or upstream files. |

Branches:
- work on `topic/<gpu>-<what>` and open a fork PR into `dev/1.5`;
- never rebase or force-push `dev/1.5`;
- releases are generated: branch `sarc/1.5` and tags `sarc/1.5-rN`, by `make-release.sh`.

## Layout

- **Kernel selection**: `impl/sarc/Select.{h,cpp}` with per-vendor tables in `impl/sarc/table_<vendor>.cpp`.
  - A row is `{device substring, device predicate, op, kernel base name, tile dims, texture IO allowed, status}`.
  - `kUnverified` rows are inert in a release. In a dev build, `ET_VK_SARC_UNVERIFIED=1` enables them.
  - Launch dims come from the row. Nothing parses kernel names.
- **Op hooks**: `impl/sarc/Q4gswCoopmat.cpp` builds 4w linear on the SARC path when the device has an active row.
  - For shapes no row covers, it falls back to release 1.5's own `linear_q4gsw_{tiled,coop}` kernels: decode, unaligned M, bias, fp32.
  - It is called from `Q4gswLinear.cpp:q4gsw_linear` (listed in `sarc/HOOKS`).
- **Shaders**: one untemplated body `glsl/sarc/<family>_body.glslh`, with two thin template wrappers.
  - `glsl/sarc/<family>.glsl` plus a yaml of shipped variants.
  - `glsl/sarc_dev/<family>_sweep.glsl` plus a yaml of sweep candidates.
  - The wrappers must be identical from `#version` on. A template can have only one yaml, which is why there are two wrappers.
- **Dev overrides**: `impl/sarc_dev/Overrides.cpp` holds the env vars:
  - `ET_VK_SARC_UNVERIFIED`;
  - `ET_VK_FORCE_TILED_LINEAR` (tiled baseline);
  - `ET_VK_SARC_Q4GSW_VARIANT=<tile>` (sweep; also builds 4w on the SARC path on devices without rows);
  - the sweep candidate rows.
- **Benchmarks and tests**: `test/sarc_dev/`, a standalone CMake project with `test_llama_microbench` and `test_sarc_select`. The latter is GPU-free.

## Tools (`sarc/tools/`)

| Tool | What it does |
|---|---|
| `build.sh [--android] [--llama] [--traced] <tree> <out>` | Container build (`localhost/et-vk-build:rocky10`, pinned glslc). It regenerates shaders every run. |
| `verify.sh --dir <stage> --lock <uuid> [--models 1b,3b,8b] [--schemes 4w,8da4w] [--pdiff]` | Runs on the GPU host and produces the promotion evidence. |
| `check.sh [--no-build] [--android]` | GPU-free: zone rule, twins, selection test, release-export build, SPIR-V golden. |
| `make-release.sh --export <dir>/executorch` | Writes the release tree, for building. |
| `make-release.sh --commit rN` | Runs `check.sh`, appends to `sarc/1.5`, tags `sarc/1.5-rN`. |
| `spirv_golden.py` | Compares or updates `sarc/golden/spirv.json`. |

## Promotion checklist (one PR into dev/1.5)

1. Move the variant block from the `sarc_dev` sweep yaml to the release yaml, renamed `sarc_<family>_<tile>_<io>_<weight>_<dtype>`.
2. Add a row to `table_<vendor>.cpp`, or flip its status to `kVerified`, and extend `test_sarc_select` with the device fixture.
3. Build with `build.sh`, then run `verify.sh` on the device. Required evidence:
   - the dispatched kernel names;
   - microbench correctness;
   - production-diff for 1B/3B/8B × buffer/texture3d, with nonzero zp for 8da4w;
   - e2e prefill tok/s (tiled vs default);
   - next token == tiled on the real-text and the unaligned prompt;
   - decode.
4. Update the golden with `spirv_golden.py <build>/backend/vulkan_compute_shaders sarc/golden/spirv.json --update --owner <device> --prefix <kernel_base>`.
5. Run `check.sh`. Put the evidence summary in the PR, and the raw logs under the study's `openspec/changes/<change>/`.

A change that alters another device's shipped SPIR-V fails the golden check. It needs that device's owner to
re-verify first.

## Devices (release 1.5 status)

| Device | 4w | 8da4w | SDPA prefill | Evidence |
|---|---|---|---|---|
| Radeon 780M | verified | verified | verified | `openspec/changes/sarc-1.5-{bootstrap,8da4w-port,sdpa-port}` |
| Arc B580 / Arc Pro B70 | verified | verified | stock | `sarc-1.5-4w-port`, `sarc-1.5-8da4w-port` |
| RTX 4070 Ti SUPER | verified | verified | stock | same |
| Jetson Orin | verified (texture3d projections) | verified (texture3d projections) | stock | same |
| Samsung Xclipse (M51) | unverified | unverified | unverified | 1.4 dev-branch defaults; owned by the M51 agent |
| Adreno 840 (S26) | unverified | – (1.4 int8 kernel broken) | stock | owned by the phone agent |
| Mali-G1 | – (1.4 routed to upstream) | – | stock | nothing to port |
| 7900 XTX | – | – | – | no promoted 1.4 default; owned by its agent |

Unverified rows are inert in a release. Their SPIR-V is still pinned in `sarc/golden/spirv.json`
(owner `UNVERIFIED:<device>`), so accidental changes are caught.

Notes for the agents that own a device:
- Run `verify.sh` with `ET_VK_SARC_UNVERIFIED=1`. For phones, build with `build.sh --android` and adapt
  `verify.sh` to `adb`.
- Then flip the row and update the golden entry, in one PR.
- The legacy 1.4 branches are in the remote as `yanwen/release14-quant-shaders*`, and locally as
  `refs/legacy/*` in the dev clone.
