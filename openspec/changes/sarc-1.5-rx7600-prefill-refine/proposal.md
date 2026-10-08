# sarc-1.5-rx7600-prefill-refine

Second-layer prefill tuning of the Radeon RX 7600 (Navi33, gfx1102, RADV Mesa 26.2.3 user-space build), the
780M's position (`CAMPAIGN.md`, owner decision N1): port the 780M's second-layer results, gate each, stop when
two consecutive gated candidates gain under 2 %. Dev zone only. Branch `topic/rx7600-prefill-refine`, forked
from `origin/topic/780m-prefill-refine` at `f5f1bf10c` (the parent).

## Parent

`f5f1bf10c510da347a556f716c2c9025a85d6228` with `ET_VK_SARC_UNVERIFIED=1` and
`VK_ICD_FILENAMES=<mesa-install>/share/vulkan/icd.d/radeon_icd.x86_64.json`, no profile. The rows
for "rx 7600" are already in `impl/sarc/table_amd.cpp` (kUnverified): 4w `t256x128k32g24s32f32cbt`, 8da4w zpg
`t128x64k32g42s32`, SDPA `qk_coopmat_t128x64k32g22s64` / `av_coopmat_t64x64k32g22s64`, SARC truncated softmax.
The two release-zone hooks of owner decision D4 (softmax variant name `b969e8f1c`, fused attention node
`1c8861aa7`) are already on the starting branch.

## Thresholds (fixed 2026-10-06 before the first measurement; `tools/thresholds.txt`)

| threshold | value | source |
|---|---|---|
| noise band | a difference inside +-2 % is noise, not a gain | R6 |
| baseline tolerance | each of the six parent cells within 3 % of the published value (`sarc-1.5-e2e-benchmark/contrib/rx7600/NOTES.md`: 4w 7787 / 3287 / 1517, 8da4w 7340 / 3080 / 1403 tok/s), else explained before anything is optimised | R4.4 |
| clock floor | 97 % of the lowest per-run median clock (sclk, prefill window) of the valid runs of the A/A session, one device-wide value; written into `thresholds.txt` once, before the first candidate | R6, L16 |
| clock samples | at least 5 samples (20 ms period) inside the prefill window | R6 |
| thermal throttle | a run whose window shows a temperature bit of `gpu_metrics` `indep_throttle_status` (bits 32 to 47) that is in `thermal_mask` is invalid; power and current bits are recorded, not rejected. **Revised 2026-10-06 07:45 UTC, before the A/A and before any candidate:** the smoke test of the parent (two untimed 1.25 s runs, `stage/smoke`) showed bit 36 (`TEMP_HOTSPOT` in the kernel's numbering; UNVERIFIED how the SMU 13.0.7 firmware sets it) in every sample whose junction temperature was 65 to 68 C, with the same gfxclk as samples without it, far below the card's limits (junction / edge critical 110 C, memory 105 C). The A/A session decides each temperature bit it shows: per cell, if the median clock of the A/A runs whose window carries the bit is within 1 % of that of the runs without it (or no clock difference can be shown because every run carries it, and the cell's clocks are within 1 % of the other cells' A/A clocks), the bit is masked (recorded, not rejected); otherwise it rejects. The resulting `thermal_mask` is written into `thresholds.txt` once, with the clock floor | R6, L16 |
| foreign GPU use | any other holder of `/dev/dri/{renderD128,card1}` or GPU runner process before, during (0.5 s poll) or after a run makes it invalid (no per-process engine accounting exists for other users' processes here) | R6, L22 |
| host builds | runs wait until no compiler / linker / ninja process of anyone runs; one that appears during a run makes it invalid | R5 |
| repeats | 5 valid runs per arm and cell; 7 if any cell of the A/A session shows a repeat spread ((max - min) / median) above 2 % in either arm | R6, L22 |
| kernel screen | a kernel replaces the incumbent for a shape only if at least 3 % faster in every round; a tie keeps the incumbent | R8 |
| stop rule | two consecutive gated candidates each under 2 % geometric mean over their parent | R11 |
| cooling | to idle + 5 C, or until the temperature has not fallen for 30 s, at most 300 s, before every run | R6, L20 |
| next-token items | D1 / D3 as written (near-tie evidence; reference-error rule for arithmetic changes) | owner |

## Adoption rules for the last candidates (written 2026-10-08 03:24 UTC, the time of commit `52e39f623`, before the M2a session; the candidate-4 rule only restates the table above)

- **Candidate 4** (whole-texel 8da4w staging everywhere, `rx7600-refine3`, against candidate 3): adopted only if its
  geomean gain over candidate 3 is at least 2 % and its gate passes; otherwise candidate 3's kernel stays and the number is
  reported as a negative result.
- **M2a** (`fused3sb`: `subgroupBarrier()` after every `memoryBarrierShared()` in the fused kernel; its own gated candidate by
  the owner decision of 2026-10-07 23:15 UTC, reported either way): it is a correctness-hygiene change (an execution
  barrier where the 780M kernel relies on lockstep), not a speed-up. It enters the final stack if (a) the gate passes (SDPA tiers
  all / extended / full, 12 passes each, 0 mismatches, `pairing=ok`; `verify.sh` as the snapshot; next token SAME) and its
  output is byte-identical to `fused3` in the SDPA evidence (else D3, error against the fp32 reference not larger) and (b)
  its geomean change over candidate 3 is not worse than -2 % (inside the noise band or better). Otherwise it stays out and
  the stack keeps `fused3` with the formal-race finding recorded.
- The stop rule counts candidates 4 and M2a only as gated candidates that were timed against their parent.

## Host and environment (host-ws1)

- Ubuntu 22.04, kernel 6.8.0-107-generic, 48 cores, 45 GiB RAM (the six models, 14 GB, fit the page cache), RX 7600
  8 GiB at `0000:04:00.0` = `card1`, hwmon1. Governor `auto`, sclk levels 255 / 2356 MHz (as found, not changed).
- Vulkan: only the user-space RADV ICD is loaded with `VK_ICD_FILENAMES` set, so the RX 7600 is device 0.
- Builds: native (`tools/build-native.sh`), because podman fails on this host (owner fact, 2026-10-05): host gcc
  11.4, `<toolchain-share>` Python 3.12.9, Vulkan SDK 1.4.350.1 glslc. The shipped-SPIR-V golden check is therefore
  **pending** for every build of this campaign (glslc differs from the container's shaderc v2023.8).
- Models: the six `*_embq_ctx3072.pte` copied to this host on 2026-09-28
  (`<workspace>/.artifacts/2026-09-28/e2eb/rx7600/models-src/`), read-only, sha256 equal to the
  shared manifest; linked under `.artifacts/models` in `verify.sh`'s flat layout. The copy at
  `<campaign-root>/models` is incomplete (8B 4w truncated at 3.26 of 4.17 GB, 8B 8da4w missing)
  and is not used.
- Another campaign (an Android device over adb) builds natively on this host's CPU. Timed runs wait for its
  builds to end and are invalid if one starts during them (thresholds above).

## Tools (copied from `sarc-1.5-780m-prefill-refine/tools/`; originals untouched)

| tool | change against the 780M copy |
|---|---|
| `env.sh` (new) | host constants: artifact directory, lock `rx7600-host`, hwmon1 / card1 sensors (checked to be the same device), `VK_ICD_FILENAMES`, `ETVK_DEVICE_INDEX=0`, `TMPDIR`; stops on a `GPU_GONE` / `ABORTED` marker |
| `export_commit.sh` (new) | exports one commit and, recursively, each pinned submodule from this clone's object stores; writes `COMMIT` and `MANIFEST` |
| `build-native.sh` (new) | native mirror of `sarc/tools/build.sh --llama [--traced]` (the container cannot run here) |
| `build-both.sh` | export + native build, instead of a container build of the working tree |
| `sampler.py` (new) | one-process sampler (no child processes, L39): sclk, busy, power, max of the three temperatures, `gpu_metrics` v1.3 throttle status, every 20 ms |
| `others.sh` (new) | foreign GPU users by `fuser` on `/dev/dri` and by program name (comm), host builds by compiler / linker processes |
| `e2e5.sh` | sensors and lock of this host; page cache filled per cell (D5) and model load time recorded; throttle, foreign-GPU and host-build rules above; samples every 20 ms and a 5-sample minimum; cooling also ends when the temperature stops falling; next token also on `prompt_real_2048.txt`; flat model layout |
| `gl.sh` | lock `rx7600-host`, `others.sh` guard, waits out builds; the 780M's tracing refusal kept for unattended jobs |
| `hold.sh` | artifact directory of this campaign |
| `stage.sh` | paths of this host; `ET_VK_SARC_UNVERIFIED=1` belongs to both arms' env |
| `summarize.py` | repeat count as an argument; three next-token prompts |
| `thresholds.txt` (new) | the thresholds above |

The other copied tools (search, screen and plot scripts of the 780M) are unchanged and used only where named.

## Plan (`CAMPAIGN.md`, rx7600 section 6)

1. Snapshot `s0-parent-verify`: unmodified `verify.sh` on the parent build, stored once.
2. Baseline and A/A: parent against the same binary, six cells; calibrate the clock floor and repeats.
3. Locate: per-operator ETDump breakdown for the six cells, phase timing of the linear kernels.
4. Port list (each a gated candidate): fused attention kernel; fp32 softmax without the zero tail (D4.1);
   linear kernel per layer shape; whole-texel 8da4w weight staging.
5. Further candidates only where step 3 points. No tile sweep first, no sampled search (N1).

## Results (2026-10-08; all evidence under `results/rx7600/`, raw runs under the artifact directory)

Stop rule met (R11): candidates 4 (-0.15 %) and M2a (+0.00 %) are two consecutive gated candidates under 2 %.

| # | candidate | parent | geomean | state | evidence |
|---|---|---|---:|---|---|
| 1 | softmax `r3` (the 780M's; `ET_VK_SARC_780M_PROFILE=c7`) | pristine parent | +1.48 % | gated, bit-identical to the release softmax in 21 of 21 SDPA cases | `sessions/c1-softmax` |
| 2 | fused attention kernel `fused3` (D4.3 hook, owner review before promotion) | 1 | +18.22 % | **ACCEPTED (reference-error rule, owner decision 2026-10-04)**; next token SAME everywhere | `sessions/c2-fused`, `sdpa-error/`, `probe/`, `acceptance-c2.txt` |
| 3 | linear kernel per layer shape (`rx7600-refine2`, real build `c3`) | 2 | +5.49 % | gated; outputs byte-identical in 24 of 24 linear shapes (no arithmetic change) | `screens/`, `sessions/c3-linear` |
| 4 | whole-texel 8da4w staging everywhere (`rx7600-refine3`, build `c4`) | 3 | -0.15 % | gated, **not adopted** (8B 8da4w -1.51 %) | `sessions/c4-texel` |
| M2a | `fused3sb`: `subgroupBarrier()` after every `memoryBarrierShared()` | 3 | +0.00 % | gated, output byte-identical to `fused3`, adopted under the rule above | `sessions/m2a-sgbarrier` |

Failed first attempt of candidate 3 (rc 127, no binaries; superseded): `sessions/c3-first-attempt-failed`.

**Final stack against the pristine parent** (build `final`, commit `18cc0d53a`, no local patch; `sessions/final`):

| cell | parent | final | gain | published 2026-09-28 |
|---|---:|---:|---:|---:|
| 1B 4w | 7846.74 | 10502.60 | +33.85 % | 7787 |
| 1B 8da4w | 7340.50 | 10343.40 | +40.91 % | 7340 |
| 3B 4w | 3297.91 | 3984.44 | +20.82 % | 3287 |
| 3B 8da4w | 3079.70 | 3953.67 | +28.38 % | 3080 |
| 8B 4w | 1517.04 | 1747.44 | +15.19 % | 1517 |
| 8B 8da4w | 1402.74 | 1738.54 | +23.94 % | 1403 |

Geomean **+26.90 %** (expected range of N1: +20 to +30 %). Recommended configuration, all of it committed code, selected by environment:
`ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_780M_PROFILE=c7 ET_VK_SARC_780M_SDPA_FUSED=fused3sb_d64_t32x32g11s32rko,fused3sb_d128_t16x64g11s32rko
ET_VK_SARC_RX7600_PROFILE=rx7600-refine2` and `VK_ICD_FILENAMES` of the user-space RADV (Mesa 26.2.3). Nothing outside the dev zone changed
since the parent (`sessions/final/files-outside-dev-zone.txt`); `sarc/tools/check.sh --no-build` PASS (`sessions/final/check-no-build.txt`).

Where each gain came from (warm ETDump, 8B 8da4w, ms per 2048-token prefill, final session: `sessions/final/trace-families.csv`, `trace-totals.csv`):
total dispatch time 1447.0 (pristine parent) -> 1162.1 (final). Attention: QK^T 72.4 + softmax 64.6 + attn x V 55.1 = 192.2 ms -> the one-pass fused
kernel with its K / V copy pass 46.7 ms (candidate 2; the softmax `r3` alone, candidate 1, took the softmax from 64.2 to 53.1 ms in the `c1-softmax` trace).
Linear GEMM family: 1110.9 -> 972.1 ms (candidate 3: the 256 x 64 tile, K step 64, on the twelve shapes), now 84 % of the final stack's 1162.1 ms.
(The 1304.2 -> 1163.6 ms in `sessions/c3-linear/trace.out` are candidate 3's total dispatch times against candidate 2, not the linear family.)

Negative results: whole-texel staging on every shape (-0.15 %); the 780M's fused-variant alternatives (no variant >= 3 % faster in every round,
`results/rx7600/fused/`); 4w kernels: only 6 of 12 shapes pass the 3 % rule and give +0.5 to +1.5 % per cell.

What limits further progress: the linear GEMMs are about 77 % of the 8B prefill (1031.8 of 1335.3 ms 4w, 1111.2 of 1446.9 ms 8da4w, parent) and on 6 of 12 shapes
no screened 4w kernel is 3 % faster than the one the release table already ships in every round; 8da4w staging and barriers cost as much as the
MMA (phase timing). Percent of the roofs: not re-measured (no igpu-roofline on this host; owner question in `STATUS.md`); the cited
2026-09-28 roofs are 43.42 TFLOP/s (fp16) and 43.90 TOP/s (int8).

## Round 2 (owner decision 2026-10-08 23:38 UTC): the linear kernels, structure first

Round 1 is closed (DONE, +26.90 % geomean, head `fd2690e85`). Round 2's parent is round 1's final stack: build `final` (commit
`18cc0d53a`), environment `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_780M_PROFILE=c7
ET_VK_SARC_780M_SDPA_FUSED=fused3sb_d64_t32x32g11s32rko,fused3sb_d128_t16x64g11s32rko ET_VK_SARC_RX7600_PROFILE=rx7600-refine2` and
`VK_ICD_FILENAMES` of the user-space RADV. The pristine parent `f5f1bf10c` stays the second reference of the final session. Thresholds
(`tools/thresholds.txt`: noise band 2 %, clock floor 2420 MHz, 5 repeats, 3 % kernel screen) are unchanged and not re-calibrated.

Candidates (owner order), each in new dev-zone files under the device-tag prefix `rx7600`, default off, golden unchanged:
1. 8da4w: a variant of the shipped 8da4w kernel with less synchronisation / LDS cost per K step, proven by phase timing before it is
   timed end to end.
2. 4w: the same treatment for the 4w kernel on the shapes where the shipped kernel won the screen.
3. Only if the trace shows attention above 10 % of a cell: the 780M's best per-head-dimension QK^T / attention x V kernels where the
   fused node does not serve the call.

Rules fixed on 2026-10-08 23:56 UTC, before any round-2 measurement (the commit carrying this text is the time stamp):
- **Kernel screen:** a variant replaces the incumbent of a shape only if at least 3 % faster in every round of a 3-round kernel screen
  (`test_llama_microbench --linear --regime=prefill --storage=texture3d`, the twelve real shapes), as in round 1.
- **Candidate adoption:** a candidate is adopted into the round-2 stack if its gate passes (R7) and its geomean gain over its parent in
  one timed session is at least 2 % (outside the noise band). Otherwise it is recorded as a negative result with its numbers and counts
  as a candidate under 2 % for the stop rule. A candidate that changes arithmetic is judged by the D3 reference-error rule; one that
  claims not to must show bit-identical outputs (`tools/linear_bitwise.sh`).
- **Phase timing:** the share of barrier + LDS-store time of the candidate's measurement twin must fall against the incumbent twin
  (same tile, same shapes) before the candidate is timed end to end; if it does not fall, the candidate is dropped without a session
  and recorded.
- **Measurement-only kernels** (phase-timing twins, ablations) write wrong results by design, are never selected by a profile and are
  never part of a gated binary's selection.
- **Stop rule (round 2):** two consecutive gated candidates each under 2 % geomean over their parent, or all three candidates done.
  Then: final verification of the round-2 stack on the committed head (build from an exported commit, no local patch), a final session
  against round 1's final stack and against the pristine parent, write-up. Never pushed from here.
- **Shared host:** a timed run waits until no compiler, linker or build of anyone runs and is invalid if one appears (R5); timed tools
  keep the names `e2e5`, `gl`, `verify` under `<campaign-root>/.../tools/`.
