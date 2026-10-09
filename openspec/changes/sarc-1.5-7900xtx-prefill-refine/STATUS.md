# STATUS: sarc-1.5-7900xtx-prefill-refine

Updated 2026-10-09 07:40 UTC (`date -u`). Branch `topic/7900xtx-prefill-refine`, parent `90fe4d013`. **Nothing pushed.** Round finished; nothing is running on either machine; the GPU host is idle (no `HOLD` file).

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

## Decision needed from the owner
1. **Thermal-mask rule (ratification).** The rule fixed before the A/A gave `thermal_mask=0xffff` on this card: bit 36 of `indep_throttle_status` is present in every run, so every later run would be invalid. Evidence: set in 98.4 % of the
   in-window samples at the card's highest clock (2824 MHz median); the samples without it are ramp samples. I masked bit 36 only (recorded, not rejected), kept every other temperature bit rejecting and the clock floor (2670 MHz); done after the A/A
   and before any candidate was measured; the full throttle word of every run is kept. Rejecting it makes every measurement of this campaign invalid under the rule as written.
2. **Stop rule.** By the letter it was met early (candidates 1 and 2), but candidate 2 was not fully gated and the port list still had untried items, so the campaign went on; at the end the last gated candidates are 4 (-0.42 %), 5 (+4.09 %), 6 (+0.67 %):
   not two consecutive under 2 %. It ends because every screen of the dev zone's named kernels is complete and passes nothing more by the 3 % rule. If you want the strict reading, the next step is a new GEMM kernel (the 8da4w phase timing points at its
   barriers and LDS stores), which N1 leaves out of this port; say if you want it.
3. **A passive monitor during timed runs.** An `amdgpu_top` of an interactive session of this account blocked the queue for about 65 minutes (21:00 to 22:05 UTC on 2026-10-08), an `nvtop` made `gl.sh` refuse jobs earlier. By rule no run is taken while another
   process holds the card; the tools now wait instead of refusing. If a passive monitor may stay open during timed runs, say so (a change of R6 / `others.sh`; its effect on the timing is UNVERIFIED).

## For the reviewer: not checked by me, or open
- Whether AMDVLK runs the 32-lane fused kernels as wave32 (UNVERIFIED; they were correct, 18 tier passes, but 2.6x slower than the unfused path, cause not investigated; the candidate was rejected on its timed session).
- The very large 4w tile `t128x256 ... cbt` takes about 14 minutes to run its 12-shape microbench job (probably driver compile time; UNVERIFIED); it is not in the final stack.
- The golden against `sarc/golden/spirv.json` is pending (no container image); the DIFF set equals the parent build's in every build measured (`c2`, `c3`, `c5`, `c6b`, `final`).
- New shaders: only `fused3sb` (read for shared writes, not in the final stack); the kernels of candidates 3, 5 and 6 are existing dev-zone shaders with unchanged sources, selected by name; the only new C++ is the `7900xtx` block of `Overrides.cpp`.
- The shell tools were written for a two-machine workflow; their first versions had bugs that cost time (a guard pattern, a refusing guard, an invalid attention variant, one failed build): each is recorded where it happened, the superseded outputs are in the artifact directory.
- Submodules of the working copy were fetched from their public GitHub URLs (read-only download) so that R5 exports can pin them.
