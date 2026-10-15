# sarc-1.5-wmma-families

Proposal (2026-10-14). Branch `topic/wmma-families`, forked from `dev/1.5` at `6178971845` (the merged
campaigns), moved onto `topic/pre-promotion` on 2026-10-15. Upstream branch, later: `pr/wmma-families` off upstream `main`. Decisions behind this
change: the owner's answers of 2026-10-14, recorded as N13 in
`../sarc-1.5-campaign-playbook/OWNER-DECISIONS.md`.

## Why

The campaigns tuned each device separately. `dev/1.5` now carries about 43 distinct cooperative-matrix kernels
for eight devices, selected by device-name substring (`impl/sarc/Select.cpp`). Neither form can go upstream:
upstream ExecuTorch needs kernels that help every device of a kind, not a table of eight cards.

Upstream `main` uses cooperative matrix almost nowhere today. Its fp linear path needs subgroup size 64 and a
discrete GPU; its q4gsw and dq8ca paths additionally need a device name containing "Radeon" or "Xclipse". On
Xe2 (subgroup 16), on NVIDIA and on the 780M (subgroup 32) no cooperative-matrix linear kernel runs.

The evidence so far says kernels transfer within an architecture family and not across: the B580 fused kernel
on the B70 +8 %, the 780M kernel on the RX 7600, the 4070 Ti port on the Orin; but the 780M kernel on the
7900 XTX -13.8 %.

Goal: bring the 4w and 8da4w linear speedups upstream in one PR, as one kernel set per architecture family,
with every device that matches no family left exactly as upstream runs it today.

## Scope

In:

- 4w linear (`q4gsw`) and 8da4w linear (`dq8ca_q4gsw`), prefill shapes.
- Families: **Xe2** (B580, B70), **RDNA3 on RADV** (780M, RX 7600, RX 7900 XTX), **NVIDIA Ampere/Ada** as one
  family (Orin, 4070 Ti), **Xclipse** (M51).
- The full-subgroups pipeline flag (F1, `99989d9a3a`), which the subgroup-16 kernels need.

Out:

- Fused attention and its graph hook: a later, separate upstream PR. The D4.3 review of the fused-node hook is
  therefore not a prerequisite here.
- Decode, fp linear (`coopmat_mm`), and any re-measurement of the baseline on upstream `main` (the owner may do
  that later; it is not part of this PR).
- Adreno, Mali, ImgTec, AMDVLK, and every other device without a family: they keep upstream's current path.
  The PR's claim is "faster on these families, unchanged elsewhere".
- The pinned glslc (`topic/sarc-pinned-glslc`): internal tooling only, not a gate for this change.

## Design

**Family key.** A device maps to a family by vendor ID, driver ID (`VkPhysicalDeviceDriverProperties`) and a
property signature: cooperative-matrix shapes and types, subgroup-size range, maximum compute shared memory.
No device-ID or device-name tables. Driver ID is part of the key because the same RDNA3 silicon behaves
differently under RADV and AMDVLK.

**No family, no change.** A device whose key matches no family gets upstream's current kernel. There is no new
generic fallback kernel.

**Margin.** A family's kernel set is accepted when, on every member device, its geomean over the six cells is
within **3 %** of that device's tuned best (its final campaign profile, linear kernels only) and no cell is more
than **8 %** worse.

**Per-device overrides stay internal.** In the fork, a per-device row may override the family default where it
beats the family kernel by more than the margin. Overrides never go upstream.

**Upstream form.** The family kernels become the cooperative-matrix path of the existing upstream op families
(`gemm/linear/qw/q4gsw/`, `gemm/linear/qw/dq8ca_q4gsw/`), adapted to Stephen Jia's refactor of the
cooperative-matrix path (`impl/gemm/GemmCommon.h`, `gemm/linear/LinearCoopmat.cpp`, the split of
`QuantizedLinear.cpp`). Upstream's current kernel stays only where it still wins: discrete AMD at subgroup 64,
to be measured on the 7900 XTX. Names are neutral: no `sarc` in shader names, files, comments or env vars; no
`ET_VK_SARC_*` hooks, profiles or `Overrides.cpp` upstream.

**Xclipse** defaults go into the upstream table without published numbers. M51 numbers stay local, as the
device owner's rule says. RX 7600 and RX 7900 XTX numbers may be published.

## Work order

1. **Inventory.** From the eight final profiles, list every 4w and 8da4w linear kernel with its shape rule,
   cooperative-matrix shape, subgroup size and shared memory. Mark which kernels are legal on which other member
   of the same family.
2. **7900 XTX under RADV.** Company side (`COMPANY-SIDE.md`): baseline and the device's final profile under
   RADV. This replaces the AMDVLK numbers for the family and removes the open question about kernels that
   declare 40 to 60 KB of shared memory over AMDVLK's 32 KB limit.
3. **Cross-device matrix within each family.** Each member's linear kernels on every other member where legal,
   six cells, the campaigns' measurement protocol (R6). Company-side devices run their share through
   `COMPANY-SIDE.md`.
4. **Choose each family's set** from the existing kernels: the fewest kernels (with shape rules) that meet the
   margin on all members. NVIDIA stays one family unless one set costs a member more than the margin; then
   split it into Ampere and Ada.
5. **New bodies only where step 4 fails**: one body per family, parameters derived from the queried
   properties, gated like any candidate.
6. **Family selection in the release zone.** Add the family key to `impl/sarc/Select.{h,cpp}` and turn the
   chosen sets into family rows; the old per-device rows become internal overrides where step 4 found them
   worth more than the margin.
7. **Promote on `dev/1.5`, one family per PR** (exception to the "one device per PR" rule of the branch model,
   N13): the promotion checklist of `sarc/README.md` with every member's `verify.sh` evidence attached. Open
   promotion prerequisites that apply: M1 (near-tie stage of the gate), M2d (M51 caveats), M5 (independent
   review).
8. **Port to `pr/wmma-families`** off upstream `main`: F1, the family key, the promoted kernels as the
   cooperative-matrix path of `q4gsw` and `dq8ca_q4gsw`, op tests under `test/op_tests/` and `etvk_bench`
   correctness sets for every new kernel. Known port work: every hook path moved in the 2026-10-12 `gemm/`
   restructure; bare-name `linear_*.glslh` includes must become `gemm/tile_utils/...`; the
   `SharedObject.cpp` / `Squeeze.cpp` backport is already on `main` and is dropped. Check whether llvmpipe
   exposes the cooperative-matrix shapes the tests need, so that upstream CI runs them.
9. **Short device check on the port** (all eight devices, port against `dev/1.5` with the promoted families:
   same kernels dispatched, next token identical, cells within the device check's band), then one upstream PR.
10. **After the PR lands**: start a new maintained fork branch from upstream `main` carrying only the per-device
    overrides and the SARC tooling; freeze `dev/1.5`. Separate owner instruction.

Executed by a humanize campaign under the playbook (`../sarc-1.5-campaign-playbook/`). Coordination with upstream
happens offline through the owner; no RFC issue.

## Done when

- Every family's set meets the margin on all its members, from recomputed `runs.csv` medians.
- Each family is promoted on `dev/1.5` by its own PR, `check.sh` PASS, golden updated in the same PR.
- `pr/wmma-families` builds on upstream `main`, passes its op tests and `etvk_bench` correctness sets on every
  member device and on llvmpipe where the shapes allow, contains no `sarc` name, and the device check of step 9
  holds on all eight devices.
- Devices outside the families dispatch exactly upstream's kernels (checked on at least one such device or
  driver, e.g. llvmpipe, and by a unit test that feeds an unmatched key through the selection).

## State 2026-10-15

The first hmz campaign under this change (wave 0 inventory and the five device runs on our side, run split in
the campaign coordinator's `RUN-SPLIT.md`) ended at its $435 ceiling without a complete device matrix:

- Wave 0 inventory: done (`f393c3ff`, independently reviewed).
- 4070 Ti: eleven candidate gates captured on the memory-model build; the native timed matrix never started.
- Orin: memory-model build and golden pass; the parent snapshot failed on a tool bug (`KeyError:
  'sampler_children'` in `device_snapshot.py`); no timed cell.
- 780M: build, selector and CPU checks pass, 19 candidate profiles committed; no GPU gate or timed cell.
- B580, B70: parked on an input-legality question about the 8B model (image extent limits of the gate's
  texture3d fixtures; a two-byte RoPE layout repair prepared, partial). No GPU gate or timed cell.

Carried onto this branch: the Vulkan memory-model fix (`sarc/HOOKS`, kind fix). The device branches
(`topic/wmma-<device>`) stay on their hosts until their matrices finish.
