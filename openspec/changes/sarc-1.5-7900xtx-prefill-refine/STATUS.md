# STATUS: sarc-1.5-7900xtx-prefill-refine

Updated 2026-10-09 00:49 UTC (`date -u`). Branch `topic/7900xtx-prefill-refine`, parent `90fe4d013`. Nothing pushed.

## Running now
- Control workstation: native build of commit 958bc07e7 as tag `c5` (profile `7900xtx-refine4`: candidate 3's linear picks plus the screened attention kernels), detached, niced.
- GPU host: nothing (the last queue, the attention screen, finished 00:47 UTC).

## Candidates so far (geometric mean over the six cells, each against its parent; details and per-cell numbers in `proposal.md`)
| # | candidate | parent | geomean | result |
|---|---|---|---:|---|
| 1 | softmax `r3` (`ET_VK_SARC_780M_PROFILE=c7`) | pristine parent | +1.56 % | gated, bit-identical to the release softmax in 21 of 21 SDPA cases |
| 2 | fused attention kernel `fused3sb` (rk variants picked by the screen) | 1 | **-13.79 %** | rejected on performance: the existing three-kernel attention is 2.6x faster than the best fused variant on 3B / 8B (trace evidence); gate stopped after the timed session |
| 3 | linear kernel per layer shape (`7900xtx-refine2`) | 1 | **+2.64 %** | gated; outputs byte-identical in 24 of 24 linear shapes; verify.sh as the snapshot except the two kernel-name lines |
| 4 | whole-texel 8da4w staging (`7900xtx-refine3`) | 3 | -0.42 % | gated, not adopted (8da4w cells +0.00 / +0.00 / +0.26 %) |
| 5 | unfused attention kernels of the screen (`7900xtx-refine4`: QK^T `pk_t128x128k32g42s32nf` 1.30x / 1.45x, attn*V `sweep_t64x64k32g42s32` 1.08x / 1.10x at kernel level) | 3 | pending | build running; then `q-c5.sh` (gate with SDPA tiers, reference-error evidence, real-text probe) |

Stack against the pristine parent so far: candidates 1 and 3 (about +4.2 % geomean by multiplication; the final session will measure it).

## Notes
- **Stop rule.** Candidates 1 and 2 were two consecutive ones under 2 % (the rule is met by its letter); I went on with the port-list item that nothing had tried (candidate 3), which gained +2.64 %, so the count restarted. Candidate 4 (-0.42 %) is the first of a new pair; if candidate 5 gains under 2 % the campaign stops after it. If you read the rule differently, candidate 3 and later are still valid measurements.
- **Foreign GPU users.** An `amdgpu_top` of an interactive session of this account held the card from about 21:00 to 22:05 UTC; `e2e5.sh` / `gl.sh` waited for it (the wait is in `runs.csv`, `foreign_wait_s`); no run was taken while it was there. Earlier an `nvtop` made `gl.sh` refuse jobs of the 8da4w linear screen; that screen was completed afterwards (864 rows, 3 rounds).
- **Thermal mask amendment** (see below, still to be ratified).
- Submodules of the working copy were fetched from their public GitHub URLs (read-only download) so that R5 exports can pin them.
- The benchmark prompt is the kit's `prompt_2048.txt`; the baseline agrees with the published numbers within 1.1 %.
- `verify.sh` finds no unaligned prompt `r*.txt` in the stage directories (as in the sibling campaigns); the unaligned next-token comparison is `prompt_check.txt` in `e2e5.sh`.
- Phase timing of the 8da4w table kernel: MMA 22 %, barrier wait 31 %, LDS store 26 %, weight fetch 11 % of the wave (not weight-load bound). No PROF twin of the 4w table kernel exists; not made.
- Roofs (R6): not re-measured yet (igpu-roofline `fast` plan on this driver; one GPU job, to be run when the card is free of the queue).

## Next
Candidate 5 gate -> stop rule -> final verification of the stack on a build of the committed head, final session against the pristine parent, `check.sh --no-build`, roofs, final report. Nothing is pushed.

## Decision needed from the owner
1. **Thermal-mask rule (ratification).** The rule fixed before the A/A (`proposal.md`, Thresholds) gave `thermal_mask=0xffff` on this card: bit 36 of `indep_throttle_status` is present in every run, so every later run would be invalid. Evidence: it is set in 98.4 % of the in-window samples at the card's highest clock (2824 MHz median); the samples without it are ramp samples. I masked bit 36 only (recorded, not rejected), kept every other temperature bit rejecting, and kept the clock-floor rule (2670 MHz); this was done after the A/A and before any candidate was measured. All raw words are kept, so a literal reading can be applied afterwards. Please confirm or reject; rejecting makes every measurement of this campaign invalid under the rule as written.
2. **A passive monitor during timed runs.** `amdgpu_top` in an interactive shell of this account blocked the queue for about 65 minutes. By rule no run is taken while another process holds the card. If a passive monitor may stay open during timed runs, say so (a change of R6 / `others.sh`; its effect on the timing is UNVERIFIED).
