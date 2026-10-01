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
  - A row is `{device substring, device predicate, op, kernel base name, tile dims, storage-combination mask, shape predicate, status, rowmajor_a}`; rows are tried in order.
  - `kUnverified` rows are inert in a release. In a dev build, `ET_VK_SARC_UNVERIFIED=1` enables them.
  - Launch dims come from the row. Nothing parses kernel names.
- **Op hooks**: `impl/sarc/Q4gswCoopmat.cpp` builds 4w linear on the SARC path when the device has an active row.
  - For shapes no row covers, it falls back to release 1.5's own `linear_q4gsw_{tiled,coop}` kernels: decode, unaligned M, bias, fp32.
  - It is called from `Q4gswLinear.cpp:q4gsw_linear` (listed in `sarc/HOOKS`).
- `impl/sarc/Dq8caCoopmat.{h,cpp}`: the 8da4w linear op path (zpg / zpgtr, see [SWEEP-PARAMETERS.md](SWEEP-PARAMETERS.md)).
- `impl/sarc/SdpaCoopmat.{h,cpp}`: the SDPA prefill path. Its shaders are `glsl/sarc/sarc_sdpa_{qk,av}_coopmat` and `sarc_sdpa_attn_weights_softmax`.
- `impl/sarc/GraphInfo.{h,cpp}`: glue between `ComputeGraph` and the Vulkan-free selection layer (device info, predicted weight storage).
- `glsl/sarc/sarc_quantize_and_pack_4w_with_group_sums`: the activation quantize-and-pack shader.
- The upstream files that call into these paths are listed in `sarc/HOOKS`, which is the authoritative hook list.
- **Shaders**: one untemplated body `glsl/sarc/<family>_body.glslh`, with two thin template wrappers.
  - `glsl/sarc/<family>.glsl` plus a yaml of shipped variants.
  - `glsl/sarc_dev/<family>_sweep.glsl` plus a yaml of sweep candidates.
  - The wrappers must be identical from `#version` on. A template can have only one yaml, which is why there are two wrappers.
- **Dev overrides**: `impl/sarc_dev/Overrides.cpp` holds the env vars:
  - `ET_VK_SARC_UNVERIFIED`;
  - `ET_VK_FORCE_TILED_LINEAR` (tiled baseline);
  - `ET_VK_SARC_Q4GSW_VARIANT=<tile>` (sweep; also builds 4w on the SARC path on devices without rows);
  - `ET_VK_SARC_DQ8CA_VARIANT=<tile>` (8da4w sweep, only on devices with active dq8ca rows);
  - `ET_VK_DISABLE_COOPMAT` (SARC SDPA ops use the upstream kernels);
  - the sweep candidate rows.
  - Usage of the sweep variables: [SWEEP-PARAMETERS.md](SWEEP-PARAMETERS.md#running-a-sweep-candidate).
- **Benchmarks and tests**: `test/sarc_dev/`, a standalone CMake project with `test_llama_microbench` and `test_sarc_select`. The latter is GPU-free.

End-to-end benchmark and evidence reports, plus how to add your GPU to them: `openspec/changes/sarc-1.5-e2e-benchmark/` (`CONTRIBUTING-A-GPU.md`).

Every yaml parameter of the 4w/8da4w coopmat shaders, and how to sweep them: [SWEEP-PARAMETERS.md](SWEEP-PARAMETERS.md).

## Tools (`sarc/tools/`)

| Tool | What it does |
|---|---|
| `build.sh [--android] [--llama] [--traced] [--no-tests] <tree> <out>` | Host build with the pinned glslc (shaderc v2026.2, `SARC_GLSLC`; it refuses any other version). It regenerates shaders every run. See the script header for all options. |
| `verify.sh --dir <stage> --lock <uuid> [--models 1b,3b,8b] [--schemes 4w,8da4w] [--pdiff]` | Runs on the GPU host and produces the promotion evidence. See the script header for all options (`--out`, `--no-tiled`, `--device-index`, `--model-root`, `--flat-models`). |
| `check.sh [--no-build] [--android] [--work <dir>]` | GPU-free: zone rule, twins, selection test, release-export build, SPIR-V golden. |
| `make-release.sh --export <dir>/executorch` | Writes the release tree, for building. |
| `make-release.sh --commit rN` | Runs `check.sh`, appends to `sarc/1.5`, tags `sarc/1.5-rN`. |
| `spirv_golden.py` | Compares or updates `sarc/golden/spirv.json` (also `--glslc`). |
| `compare_kernels.py <new.json> <reference>` | Compares `test_llama_microbench --json-out` results: dispatched kernel and median time per case. |
| `zones.sh` | Sourced by the other tools; defines the release/dev zones and the twin wrappers. |

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
| Samsung Xclipse (M51) | unverified, `t128x128k16g22s32f32xp` (fp32 accumulate + one-pass texture3d drain; production diff 1B/3B/8B × buffer/texture3d passes, not yet through the promotion checklist; golden from the shaderc v2023.8 container, needs re-verify with pinned-glslc binaries) | unverified, e2e output correct (`contrib/m51`) | unverified | 1.4 dev-branch defaults; owned by the M51 agent |
| Adreno 840 (S26) | unverified; the shipped fp16-accumulate row fails production-diff at K ≥ 3072. Sweep candidate `t64x64k32g21s64m64x32x16gr` (`ACC_GROUP_FP32_REG`) passes it for 1B/3B/8B; not yet timed | – (no Adreno 8-bit row) | stock | e2e `contrib/s26` (1B/3B; 8B `DEVICE_LOST`): no releasable gain yet. Driver statistics count fp16 coopmat as ALU work (`contrib/s26/isa`); int8 unresolved. Owned by the phone agent |
| Mali-G1 | unverified, `t64x128k32g44s16m16x32x32gahb` (MMA 16×32×32, per-group fp32 total; large linears only, N·K ≥ 2²⁴) | – (no fp32 accumulator for its 4×16×16 int8 shape) | stock | e2e `contrib/mali` partial (1B/3B 4w, 1B 8da4w; GPU latched into its degraded state); golden from the shaderc v2023.8 container, needs re-verify with pinned-glslc binaries |
| Radeon RX 7900 XTX | unverified (t256x128k32g24s32f32cbt) | unverified (780M zpg) | unverified (780M SDPA) | `openspec/changes/sarc-1.5-7900xtx-4w`, e2e `contrib/7900xtx` (2.79× vs stock, pre-release); golden added; needs re-verify with pinned-glslc binaries |
| Radeon RX 7600 | unverified (7900 XTX tile `t256x128k32g24s32f32cbt`) | unverified (780M zpg) | unverified (780M SDPA) | rows in `table_amd.cpp`; need re-verification with pinned-glslc binaries. e2e `contrib/rx7600` (2.48× vs stock, pre-release). Coopmat needs a newer RADV: Mesa 26.2.3 used; the system 23.2.1 has none |

The pinned glslc is shaderc tag v2026.2 built with its own `utils/git-sync-deps` revisions
(glslang 5ed4003a, spirv-tools c1cb30bb); the LunarG Vulkan SDK 1.4.350.1 glslc is the same build and
produces byte-identical SPIR-V. No container is needed. Golden entries recorded with the earlier
container glslc (shaderc v2023.8) that differ under v2026.2 stay as they are until their owner
re-verifies on the device; `check.sh` lists them as `DIFF`.

Unverified rows are inert in a release. Their SPIR-V is still pinned in `sarc/golden/spirv.json`
(owner `UNVERIFIED:<device>`), so accidental changes are caught.

Notes for the agents that own a device:
- Run `verify.sh` with `ET_VK_SARC_UNVERIFIED=1`. For phones, build with `build.sh --android` and adapt
  `verify.sh` to `adb`.
- Then flip the row and update the golden entry, in one PR.
- The legacy 1.4 branches are archived as tags `archive/yanwen/release14-quant-shaders*` and
  `archive/release/1.4-{mali,qualcomm}` (2026-09-27).
