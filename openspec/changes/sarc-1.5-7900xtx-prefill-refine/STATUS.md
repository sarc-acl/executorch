# STATUS: sarc-1.5-7900xtx-prefill-refine

Updated 2026-10-09 11:42 UTC (`date -u`). Branch `topic/7900xtx-prefill-refine`, parent `90fe4d013`. **Nothing pushed.** Nothing is running on either machine; the GPU host is idle (no `HOLD`, no queue). Round finished after the reviewer's findings of 2026-10-09: the second set (items 2 and 3) was done before the
final verification, which was then redone on the head's build.

## Result
Final stack against the pristine parent, one timed session on the head's build `c9` (commit `a554a5791`, no local patch; later commits are evidence and documents only), 7 valid runs per arm and cell, 84 of 84 valid:

| cell | pristine parent | final stack | gain | published 2026-09-28 |
|---|---:|---:|---:|---:|
| 1B 4w | 19883.50 | 22260.90 | +11.96 % | 20078 |
| 1B 8da4w | 22260.90 | 24674.70 | +10.84 % | 22261 |
| 3B 4w | 10189.10 | 10556.70 | +3.61 % | 10089 |
| 3B 8da4w | 10502.60 | 11838.20 | +12.72 % | 10396 |
| 8B 4w | 4762.79 | 4911.27 | +3.12 % | 4774 |
| 8B 8da4w | 5007.33 | 5389.47 | +7.63 % | 4971 |

Geometric mean **+8.24 %** (the first verification, before the second set, on the build of `a8dd09570`: +7.91 %). N1 expected +20 to +30 %. Recommended configuration: `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_7900XTX_PROFILE=7900xtx-refine5`, AMDVLK ICD.
Evidence: `results/7900xtx/sessions/final2/` (and `inert/`); full account: `proposal.md`.

| # | candidate | parent | geomean | state |
|---|---|---|---:|---|
| 1 | softmax `r3` | pristine | +1.56 % | gated, bit-identical (21 of 21) |
| 2 | fused attention `fused3sb` | 1 | -13.79 % | rejected on performance |
| 3 | linear kernel per layer shape (`refine2`) | 1 | +2.64 % | gated, outputs byte-identical (24 of 24) |
| 4 | whole-texel 8da4w staging (`refine3`) | 3 | -0.42 % | gated, not adopted |
| 5 | unfused attention kernels (`refine4`) | 3 | +4.09 % | gated, bit-identical (21 of 21) |
| 6 | attn*V t32x32 for head dim 64 (`refine5`) | 5 | +0.67 % | gated, adopted by the rule written beforehand |
| 7+ | second set, items 2 and 3 (8da4w / 4w staging pitch, branch-free loop, interleaved stores) | 6 | none | **no variant passes the screen rule (0 of 12 shapes in every screen)**; nothing to gate |

Second set (`results/7900xtx/set2/`): the RX 7600's 24-byte A pitch and 72 / 88-byte rows, a branch-free chunk loop and interleaved stores were built as default-off `7900xtx` variants of copies of the release 8da4w / 4w bodies. Phase timing first: the pitch raises the barrier + LDS share of the main 8da4w tile
(48.9 -> 63.0 %), it does not lower it; the 4w table kernel has the same pattern (barrier 52 %). Screens (3 rounds, against the incumbents of the final profile): 0 shapes qualify. Ablation twins give the ceilings: barrier worth 0.8 % of the 8da4w kernel (4w: 6.9 %), all 8da4w staging 7 to 10 %.

Final verification (on `c9`): `verify.sh` 30 of 32 lines identical to `s0-parent-verify` (the 2 differing lines are the dispatched-kernel names), and 32 of 32 with nothing selected; SDPA tiers 39 runs 0 failed 0 mismatches `pairing=ok`; SDPA output byte-identical to the pristine parent (21 of 21); real-text probe KL 0 (32 prompts x 6 cells);
next token SAME everywhere; golden DIFF set equal to the parent build's (14 lines; native glslc, the container golden is pending); `check.sh --no-build` PASS; `git diff --name-status 90fe4d013 HEAD` outside the change directory = dev-zone files only (`glsl/sarc_dev/`, `impl/sarc_dev/`), nothing under `sarc/tools` or `sarc/golden`.
Roofs (igpu-roofline quick + fast, this driver): matrix fp16 / fp32-acc 140.83 TFLOP/s, int8 141.60 TOP/s; the GEMMs run at 50 to 69 % of them.

## Decisions recorded (all answered by the owner; nothing is open)
- **Thermal mask: ratified** (owner 2026-10-08 19:30 UTC, again 2026-10-09 00:56 UTC): bit 36 masked (recorded per run, not rejecting); every other temperature bit rejects; clock floor 2670 MHz; raw status words kept.
- **Passive monitor: the rule stays** (owner 2026-10-09 00:56 UTC): no run is taken while another process holds the card; the queue waits and logs the wait; a monitor that appears again is noted here and does not invalidate finished runs.
- **Cool-start check on the core temperatures only** (owner 2026-10-08 22:13 UTC, repeated 2026-10-09 00:56 UTC): `gtemp_core` (edge and junction, not memory) in `tools/env.sh`, used by `gate.sh` and the screen, tier and trace waits (own commit `9a61374ed`, synced before the redone final gate, which started at once at 47 C core / 60 C memory). It changes when a session starts, not which runs are valid.
- **The stop rule does not end the campaign after the port list; items 2 and 3 of the second set come before the final verification** (owner 2026-10-08 23:38 UTC, coordinator 2026-10-09 04:26 UTC): done in that order from this round on; the first final verification (`sessions/final/`) was run too early and is superseded by `sessions/final2/`.
- Builds on this workstation may run during GPU measurements (owner 2026-10-09 02:33 UTC).

## For the reviewer: open or not checked by me
- **Stop rule.** The last gated candidates are 4 (-0.42 %), 5 (+4.09 %), 6 (+0.67 %); the second set has no candidate, so "two consecutive gated candidates under 2 %" is not literally met after candidate 5. The campaign ends because every pre-registered screen is exhausted (the screen rule precedes any gate). If the owner wants more, the only lever the
  evidence leaves is a different 8da4w / 4w MMA-loop structure (new kernel family), which the ablations bound at 7 to 10 % of the 8da4w kernel for all staging and which the owner decisions did not order.
- Whether AMDVLK runs the 32-lane fused kernels as wave32 (UNVERIFIED; they were correct, 18 tier passes, but 2.6x slower than the unfused path; rejected on its timed session).
- The very large 4w tile `t128x256 ... cbt` takes about 14 minutes to run its 12-shape microbench job (probably driver compile time; UNVERIFIED); it is not in the final stack. The device query reports 32768 bytes of shared memory; the table 4w kernel uses about 30 KB by my estimate and some 8da4w kernels (the 780M's `afmb1`) more; not looked at.
- New shaders: `fused3sb` (read for shared writes, not in the final stack); candidate 6's `sarc_sdpa_av_coopmat_sweep_t32x32k32g22s32` is a new yaml variant of an existing template (the reviewer read it: race-free); the second set adds copies of the release linear bodies with options (copies reproducible from the release bodies by the generators; the pitch / drain-in-A-buffer code read by me: address changes only, barriers unchanged,
  none of the new variants is in the final profile).
- The golden against `sarc/golden/spirv.json` is pending (no container image); the DIFF set equals the parent build's in every build measured (`c2`, `c3`, `c5`, `c6b`, `final`, `c7`, `c8`, `c9`).
- Submodules of the working copy were fetched from their public GitHub URLs (read-only download) so that R5 exports can pin them.
- The RX 7600's checkout was read (two commits, one generator pair, four generated shader files copied with the names changed to the `7900xtx` prefix; reproducibility of the two bodies from this tree's release bodies verified); nothing was written there.
