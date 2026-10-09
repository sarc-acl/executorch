# STATUS: sarc-1.5-7900xtx-prefill-refine

Updated 2026-10-09 07:40 UTC (`date -u`). Branch `topic/7900xtx-prefill-refine`, parent `90fe4d013`. **Nothing pushed.** **The round is NOT finished**: second-set items 2 and 3 (8da4w and 4w linear kernels, per-K-step synchronisation) come before the final verification and are in progress; the result below is the interim stack (candidates 1, 3, 5, 6).

## Result
Final stack against the pristine parent, one timed session, 7 valid runs per arm and cell, build of commit `a8dd09570` (no local patch; later commits are evidence and documents only):

| cell | pristine parent | final stack | gain | published 2026-09-28 |
|---|---:|---:|---:|---:|
| 1B 4w | 20277.20 | 22260.90 | +9.78 % | 20078 |
| 1B 8da4w | 22505.50 | 24975.60 | +10.98 % | 22261 |
| 3B 4w | 10138.60 | 10722.50 | +5.76 % | 10089 |
| 3B 8da4w | 10395.90 | 11570.60 | +11.30 % | 10396 |
| 8B 4w | 4762.79 | 4899.52 | +2.87 % | 4774 |
| 8B 8da4w | 4982.97 | 5333.33 | +7.03 % | 4971 |

Geometric mean **+7.91 %** (N1's expected range was +20 to +30 %). Recommended configuration: `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_7900XTX_PROFILE=7900xtx-refine5`, AMDVLK ICD.
Evidence: `results/7900xtx/sessions/final/`; full account: `proposal.md`.

| # | candidate | parent | geomean | state |
|---|---|---|---:|---|
| 1 | softmax `r3` | pristine | +1.56 % | gated, bit-identical (21 of 21) |
| 2 | fused attention `fused3sb` | 1 | -13.79 % | rejected on performance (unfused attention 2.6x faster on 3B / 8B) |
| 3 | linear kernel per layer shape (`refine2`) | 1 | +2.64 % | gated, outputs byte-identical (24 of 24) |
| 4 | whole-texel 8da4w staging (`refine3`) | 3 | -0.42 % | gated, not adopted |
| 5 | unfused attention kernels (`refine4`) | 3 | +4.09 % | gated, bit-identical (21 of 21), real-text KL 0 |
| 6 | attn*V t32x32 for head dim 64 (`refine5`) | 5 | +0.67 % | gated, adopted by the rule written beforehand |

Final verification (all on the final build): `verify.sh` 30 of 32 lines identical to `s0-parent-verify` (the 2 differing lines are the dispatched-kernel names), SDPA tiers 39 runs 0 failed 0 mismatches `pairing=ok`,
SDPA output byte-identical to the pristine parent (21 of 21), real-text probe KL 0 (32 prompts x 6 cells), next token SAME everywhere, golden DIFF set equal to the parent build's (14 lines, native glslc: the container golden is pending),
`check.sh --no-build` PASS, `git diff --name-status 90fe4d013 HEAD` outside the change directory = the five dev-zone files named in `proposal.md`, nothing under `sarc/tools` or `sarc/golden`.
Roofs re-measured (igpu-roofline quick + fast, this driver): matrix fp16 / fp32-acc 140.83 TFLOP/s, int8 141.60 TOP/s; the GEMMs run at 49 to 69 % of them.

## Decisions recorded (all answered by the owner; nothing is open)
- **Thermal mask: ratified** (owner 2026-10-08 19:30 UTC, again 2026-10-09 00:56 UTC). Bit 36 is masked (recorded per run, not rejecting); every other temperature bit rejects; clock floor 2670 MHz; raw status words kept.
- **Passive monitor: the rule stays** (owner 2026-10-09 00:56 UTC). No run is taken while another process holds the card; the queue waits and logs the wait; if a monitor appears again, keep waiting and note it here.
- **Cool-start check on the core temperatures only** (owner 2026-10-08 22:13 UTC, repeated 2026-10-09 00:56 UTC): `gtemp_core` (edge and junction, not memory) in `tools/env.sh`, used by `gate.sh` and by every screen / tier / trace wait; own commit `9a61374ed`, synced to the GPU host 2026-10-09 (core 27 C against max 46 C at idle). Affects when a session starts, not which runs are valid.
- **The stop rule does not end the campaign after the port list** (owner 2026-10-08 23:38 UTC; coordinator 2026-10-09 04:26 UTC): items 2 (8da4w linear) and 3 (4w linear) of the second set come before the final verification. The final verification recorded below (stack of candidates 1, 3, 5, 6) was run too early; it is superseded
  by the redo on the new head after items 2 and 3. Builds on this workstation may run during GPU measurements (owner 2026-10-09 02:33 UTC).

## For the reviewer: not checked by me, or open
- Whether AMDVLK runs the 32-lane fused kernels as wave32 (UNVERIFIED; they were correct, 18 tier passes, but 2.6x slower than the unfused path, cause not investigated; the candidate was rejected on its timed session).
- The very large 4w tile `t128x256 ... cbt` takes about 14 minutes to run its 12-shape microbench job (probably driver compile time; UNVERIFIED); it is not in the final stack.
- The golden against `sarc/golden/spirv.json` is pending (no container image); the DIFF set equals the parent build's in every build measured (`c2`, `c3`, `c5`, `c6b`, `final`).
- New shaders: only `fused3sb` (read for shared writes, not in the final stack); the kernels of candidates 3 and 5 are existing dev-zone shaders with unchanged sources, selected by name; candidate 6's `sarc_sdpa_av_coopmat_sweep_t32x32k32g22s32` is a new yaml variant (commit 1976f2c9e, parameters of an existing template; the reviewer read it and found it race-free); the only new C++ is the `7900xtx` block of `Overrides.cpp`.
- The shell tools were written for a two-machine workflow; their first versions had bugs that cost time (a guard pattern, a refusing guard, an invalid attention variant, one failed build): each is recorded where it happened, the superseded outputs are in the artifact directory.
- Submodules of the working copy were fetched from their public GitHub URLs (read-only download) so that R5 exports can pin them.
