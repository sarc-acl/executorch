# STATUS: sarc-1.5-7900xtx-prefill-refine

Updated 2026-10-08 18:28 UTC (`date -u`). Branch `topic/7900xtx-prefill-refine`, parent `90fe4d013`. Nothing pushed.

## Running now
- GPU host: `q-c1` (candidate 1, softmax `r3` via `ET_VK_SARC_780M_PROFILE=c7` on the parent binary): gate = timed session against the parent (7 repeats), SDPA tiers (12 passes each), `verify.sh` against the snapshot, ETDump traces; then the SDPA output evidence. Started 18:23 UTC.
- Control workstation: native build of commit 86a7f96c8 as tag `c2` (parent + the `fused3sb` variants; for candidate 2).

## State
| step | result |
|---|---|
| 0 device check | passed (2026-10-08 17:06 UTC) |
| snapshot `s0-parent-verify` | stored (`results/7900xtx/sessions/s0-parent-verify/`); parent lines it contains that are not passes: `correctness rc=1`, `linear 4w rc=1`, `linear 8da4w rc=1`, `pdiff ... 4w buffer FAILED` for 1B / 3B / 8B (the kUnverified 4w tile on buffer storage; texture3d and all 8da4w pass), compared line by line from here on |
| parent SDPA tiers (table kernels) | all / extended / full, 1 pass each: 4 + 8 + 4 passed, 0 failed, 0 mismatches, `pairing=ok` |
| golden | native build: 14 of 53 shipped variants differ from `sarc/golden/spirv.json` (owners 780M / Arc), 0 differ from the native parent build (`results/7900xtx/golden-diff-parent.txt`); pending as in the sibling campaigns |
| baseline | all six cells within 3 % of the published numbers (+0.99 % .. -1.08 %), see `proposal.md` |
| A/A | geomean -0.24 % (inside the band); next token SAME everywhere |
| calibration | clock floor 2670 MHz, 7 repeats, busy ceiling 5 %, thermal mask `0xffef` (see the question below) |

Per-cell table of the parent (A/A, median of 7, tok/s): 1B 4w 20277.2 | 1B 8da4w 22021.5 | 3B 4w 10138.6 | 3B 8da4w 10449.0 | 8B 4w 4785.1 | 8B 8da4w 4995.1.

## Notes
- Submodules of the working copy were fetched from their public GitHub URLs (read-only download; the clone had none) so that R5 exports can pin them.
- The benchmark prompt is the kit's `prompt_2048.txt` ("the" x 2048, sha256 bfce65eb...), as in the published notes; the GPU host's `p2048tok.txt` (sha256 d1e7a8d7...) is not used. The baseline agrees with the published numbers within 1.1 %, which supports the choice.
- `verify.sh` finds no unaligned prompt `r*.txt` in the stage directories (as in the sibling campaigns), so its "unaligned" lines are absent from the snapshot and from every candidate alike; the unaligned next-token comparison is `prompt_check.txt` (1972 tokens) in `e2e5.sh`.
- The fused kernels declare 32-lane subgroups; whether AMDVLK runs them correctly is open until candidate 2's SDPA tiers run (UNVERIFIED).
- The 24 untimed next-token runs of a session (real, check) show `clock_low` under the final floor (cold, no `--warmup`, short window); only their text is compared.

## Next
Candidate 1 gate and timed session -> locate (ETDump families of the parent and candidate 1 come with the gate) -> fused-variant screen (`fused_screen.sh`, 3 rounds, on the parent binary with the `fused3` variants) -> candidate 2 (`fused3sb`, build `c2`) gate and session -> linear screens and candidate 3.

## Decision needed from the owner
1. **Thermal-mask rule (ratification).** The rule fixed before the A/A (`proposal.md`, Thresholds) gave `thermal_mask=0xffff` on this card: bit 36 of `indep_throttle_status` is present in every run, so every later run would be invalid. Evidence: it is set in 98.4 % of the in-window samples at the card's highest clock (2824 MHz median); the samples without it are ramp samples. I masked bit 36 only (recorded, not rejected), kept every other temperature bit rejecting, and kept the clock-floor rule (2670 MHz); this was done after the A/A and before any candidate was measured. All raw words are kept, so a literal reading can be applied afterwards. Please confirm or reject; rejecting makes every measurement of this campaign invalid under the rule as written.
