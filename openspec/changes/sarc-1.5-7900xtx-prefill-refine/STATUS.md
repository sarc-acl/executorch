# STATUS: sarc-1.5-7900xtx-prefill-refine

Updated 2026-10-09 15:01 UTC (`date -u`). Branch `topic/7900xtx-prefill-refine`, parent `90fe4d013`. **Nothing pushed.** Nothing is running on either machine; the GPU host is idle (no `HOLD`, no queue). Round finished after the reviewer's findings of 2026-10-09: the second set (items 2 and 3) was done before the final verification, which was redone on the build of the committed head after candidate 7.

## Result
Final stack against the pristine parent, one timed session on the build of the committed head (`c10` = commit `5950764fa`; `git diff --name-only 5950764fa HEAD -- . ':!openspec'` is empty; no local patch), 7 valid runs per arm and cell, 84 of 84 timed runs valid (the 24 untimed next-token rows hold 23 `clock_low` invalid runs, kept in the file):

| cell | pristine parent | final stack | gain | published 2026-09-28 |
|---|---:|---:|---:|---:|
| 1B 4w | 20480.00 | 23011.20 | +12.36 % | 20078 |
| 1B 8da4w | 22021.50 | 24381.00 | +10.71 % | 22261 |
| 3B 4w | 10138.60 | 10666.70 | +5.21 % | 10089 |
| 3B 8da4w | 10502.60 | 11770.10 | +12.07 % | 10396 |
| 8B 4w | 4762.79 | 4899.52 | +2.87 % | 4774 |
| 8B 8da4w | 5019.61 | 5375.33 | +7.09 % | 4971 |

Geometric mean **+8.33 %** (earlier sessions of the same stack: +8.24 % on `c9`, +7.91 % on `final`). **Noise caveat:** per-cell gains of a few percent, e.g. 8B 4w +2.87 %, are of the size seen as noise between identical kernels in this setup: in candidate 7's session the 1B 4w and 3B 8da4w cells ran the same kernels in both arms and still read +2.35 % and +2.92 %. The thresholds are unchanged (+-2 %); the large gains (1B, 3B 8da4w, 8B 8da4w) are not in doubt, the small ones are indicative only. N1 expected +20 to +30 %. Recommended configuration: `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_7900XTX_PROFILE=7900xtx-refine5`, AMDVLK ICD.
Evidence: `results/7900xtx/sessions/final3/` (and `inert3/`); full account: `proposal.md`.

| # | candidate | parent | geomean | state |
|---|---|---|---:|---|
| 1 | softmax `r3` | pristine | +1.56 % | gated, bit-identical (21 of 21) |
| 2 | fused attention `fused3sb` | 1 | -13.79 % | rejected on performance |
| 3 | linear kernel per layer shape (`refine2`) | 1 | +2.64 % | gated, outputs byte-identical (24 of 24) |
| 4 | whole-texel 8da4w staging (`refine3`) | 3 | -0.42 % | gated, not adopted |
| 5 | unfused attention kernels (`refine4`) | 3 | +4.09 % | gated, bit-identical (21 of 21) |
| 6 | attn*V t32x32 for head dim 64 (`refine5`) | 5 | +0.67 % | gated, adopted by the rule written beforehand |
| 7 | the sweep 128 x 64 tile on 8B w2 (`refine6`; the one pick of the first second-set screen, 1.034 worst round; 1.026 and 1.024 in two other screens) | 6 | +1.16 % | gated (24 of 24 linear outputs byte-identical, `verify.sh` as the snapshot but the kernel names); only the 8B 8da4w cell differs in kernel: +0.79 %; **not adopted** (single-scheme rule: its one changed cell, 8B 8da4w, +0.79 % < 2 %) |
| -- | second set, items 2 and 3 (8da4w / 4w staging pitch, branch-free loop, interleaved stores) | 6 | none | no variant passes the screen rule (0 of 12 shapes for every pitch / synchronisation variant); the screens' one pick is candidate 7 |

Second set (`results/7900xtx/set2/`): the RX 7600's 24-byte A pitch and 72 / 88-byte rows, a branch-free chunk loop and interleaved stores were built as default-off `7900xtx` variants of copies of the release 8da4w / 4w bodies. Phase timing first: the pitch raises the barrier + LDS share of the main 8da4w tile
(48.9 -> 63.0 %), it does not lower it; the 4w table kernel has the same pattern (barrier 52 %). Screens (3 rounds, against the incumbents of the final profile): 0 shapes qualify. Ablation twins give the ceilings: barrier worth 0.8 % of the 8da4w kernel (4w: 6.9 %), all 8da4w staging 7 to 10 %.

Final verification (on `c10`, the head's code): `verify.sh` 30 of 32 lines identical to `s0-parent-verify` (the 2 differing lines are the dispatched-kernel names), and 32 of 32 with nothing selected; SDPA tiers 39 runs 0 failed 0 mismatches `pairing=ok`; SDPA output of the final stack byte-identical to the parent configuration (no profile) run on the same c10 binary (21 of 21); real-text probe KL 0 against that configuration (32 prompts x 6 cells; the probe binary is c10's for both arms). The parent configuration (`ET_VK_SARC_UNVERIFIED=1`, no profile) on the same c10 binary stands for the pristine parent because (i) `inert3` (`verify.sh` on c10 with nothing selected) is identical to `s0-parent-verify` in 32 of 32 lines and (ii) all 53 shipped SPIR-V variants are byte-identical between build c10 and build `parent` (the reviewer checked it with `cmp`). These two comparisons were not run on the pristine parent's own binaries;
next token SAME everywhere; golden DIFF set equal to the parent build's (14 lines; native glslc, the container golden is pending); `check.sh --no-build` PASS on the head; `git diff --name-status 90fe4d013 HEAD` outside the change directory = dev-zone files only (`glsl/sarc_dev/`, `impl/sarc_dev/`), nothing under `sarc/tools` or `sarc/golden`.
Roofs (igpu-roofline quick + fast, this driver): matrix fp16 / fp32-acc 140.83 TFLOP/s, int8 141.60 TOP/s; the GEMMs run at 50 to 69 % of them. The final stack dispatches the same 139 (model, scheme, kernel, count) rows on `c10` as on `c9`.

## Decision needed from the owner
1. **Workgroup memory above the device limit (known risk, nothing changed).** AMDVLK reports `maxComputeSharedMemorySize` = 32,768 bytes; the SPIR-V of the final stack declares more: QK^T `pk_t128x128k32g42s32nf` 53,248 bytes (every QK^T), the 780M's `afmb1` 8da4w kernel 60,160 (1B w2), the 4w `sweep_t128x128k32g42s32f32cbt` 40,960 (1B), and the parent's own release 4w row
   `t256x128k32g24s32f32cbt` 61,440. It is pre-existing (the parent's table row is the largest) and every gate passes, but it breaks a Vulkan valid-usage rule, and `Select.cpp` checks shared memory only for texture3d linear rows, not for attention rows. Accept it on AMDVLK, or require in-limit kernels before any promotion?
   (An earlier version of this file said the table 4w kernel uses about 30 KB; that was my wrong estimate.)

## Decisions recorded (answered by the owner)
- **Thermal mask: ratified** (owner 2026-10-08 19:30 UTC, again 2026-10-09 00:56 UTC): bit 36 masked (recorded per run, not rejecting); every other temperature bit rejects; clock floor 2670 MHz; raw status words kept.
- **Single-scheme candidates** (owner 2026-10-09 12:57 UTC): a candidate that changes only one scheme is judged on the cells it changes (adopted if each gains >= 2 %); under 2 % geomean it counts as one candidate under 2 % for the stop rule. Applied to candidate 7 (only 8da4w, one changed cell, 8B 8da4w): +0.79 % < 2 %, not adopted. The rule is in the Thresholds table of `proposal.md` (own commit `b786a5719`, before use).
- **Passive monitor: the rule stays** (owner 2026-10-09 00:56 UTC): no run is taken while another process holds the card; the queue waits and logs the wait; a monitor that appears again is noted here and does not invalidate finished runs.
- **Cool-start check on the core temperatures only** (owner 2026-10-08 22:13 UTC, repeated 2026-10-09 00:56 UTC): `gtemp_core` (edge and junction, not memory) in `tools/env.sh`, used by `gate.sh` and the screen, tier and trace waits (own commit `9a61374ed`, synced before the redone final gate, which started at once at 47 C core / 60 C memory). It changes when a session starts, not which runs are valid.
- **The stop rule does not end the campaign after the port list; items 2 and 3 of the second set come before the final verification** (owner 2026-10-08 23:38 UTC, coordinator 2026-10-09 04:26 UTC): done in that order from this round on; the first final verification (`sessions/final/`) was run too early and is superseded by `sessions/final2/`.
- Builds on this workstation may run during GPU measurements (owner 2026-10-09 02:33 UTC).
- **Scratch disk** (coordinator 2026-10-09 08:26 UTC): `/` was at 99 %. Moved (not deleted) to `<scratch>/`, symlinks left at the old paths: `src/7900xtx/{c2,c3,c5,c6b,c7,c8,c9,final,parent}` and `build/7900xtx/{c2,c3,c5,c6b,c7,c8,c9,final,parent}` with their `-traced` twins (list: `MOVED.txt` there; sha256 of the parent and c9 binaries verified equal to the staged copies). `build-both.sh` and `export_commit.sh` now write new exports and builds there (`BIG` in `env.local`); build `c10` was made that way.

## For the reviewer: open or not checked by me
- **Stop rule.** Met: candidate 6 (+0.67 %, adopted by the pre-written rule, not claimed as a gain) and candidate 7 (+1.16 %, gated, not adopted) are two consecutive gated candidates under 2 %. Earlier (after candidate 5) it was not, and an earlier version of this file said the screens' pick did not exist ("0 of 12 shapes in every screen"): wrong, the 8B w2 pick is candidate 7.
- **Further gains.** The only lever the evidence leaves is a different 8da4w / 4w MMA-loop structure (a new kernel family), which the ablations bound at 7 to 10 % of the 8da4w kernel for all staging and which the owner decisions did not order.
- Whether AMDVLK runs the 32-lane fused kernels as wave32 (UNVERIFIED; they were correct, 18 tier passes, but 2.6x slower than the unfused path; rejected on its timed session).
- The very large 4w tile `t128x256 ... cbt` takes about 14 minutes to run its 12-shape microbench job (probably driver compile time; UNVERIFIED); it is not in the final stack. The workgroup-memory overrun of the shipped kernels is decision item 1 above.
- New shaders: `fused3sb` (read for shared writes, not in the final stack); candidate 6's `sarc_sdpa_av_coopmat_sweep_t32x32k32g22s32` is a new yaml variant of an existing template (the reviewer read it: race-free); the second set adds copies of the release linear bodies with options (copies reproducible from the release bodies by the generators; the pitch / drain-in-A-buffer code read by me: address changes only, barriers unchanged,
  none of the new variants is in the final profile).
- The golden against `sarc/golden/spirv.json` is pending (no container image); the DIFF set equals the parent build's in every build measured (`c2`, `c3`, `c5`, `c6b`, `final`, `c7`, `c8`, `c9`, `c10`).
- Submodules of the working copy were fetched from their public GitHub URLs (read-only download) so that R5 exports can pin them.
- The RX 7600's checkout was read (two commits, one generator pair, four generated shader files copied with the names changed to the `7900xtx` prefix; reproducibility of the two bodies from this tree's release bodies verified); nothing was written there.
