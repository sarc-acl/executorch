# STATUS: sarc-1.5-orin-prefill-refine

**2026-10-06 00:05 UTC. STOP RULE MET; the closing measurements are running. Four candidates are accepted, in
this order: 1h (SDPA prefill kernels + fp32 softmax, +57.8 % over the parent), 2 (8da4w linear, +2.84 % over the
parent alone), 3 (4w linear tiles, **+0.82 %** over its parent), 4 (softmax with subgroup reductions,
**+0.41 %** over its parent, inside the +-2 % band). Candidates 3 and 4 are two consecutive gated candidates
below +2 %. Candidate 1 (release softmax) stays REJECTED. The branch is not pushed yet.**

All times are UTC from `date -u`.

## Running now

- Device `duck-naughty` (primary), detached, under its gpu-lab lock; `tools/dstat.sh <job>` from the workstation:
  - `chain19b` (`tools/chain19.sh`): `s5-c3` and `s6-c4` done (both `GATE_ACCEPTED`); running since 23:55:
    `s7-final` (`timed.sh`: the final stack `orin-refine5` + `orin_g64` on build `topic13` against the pristine
    parent, timed session and traces), then `s8-noenv` (`noenv_verify.sh`: `verify.sh` on `topic13` with nothing
    selected, against the parent control). Until about 01:45.
  - `chain20` (waiting for `chain19b`): the 41-prompt real-text logits of the final stack, default and tiled
    (the parent's two arms exist), comparison and reference-error rule in `probe/final-g64/`. About 2.5 hours.
- Device `duck-stable`: nothing; not used any more.
- Workstation: nothing.

## Next step

Read `s7-final` and `s8-noenv`; final tables in `proposal.md`; `sarc/tools/check.sh --no-build`; commit; push.
The real-text evidence of `chain20` is added when it ends.

## Second Orin (`duck-stable`): agreement batch, retest, owner decision

- Agreement batch (the 26 configurations of 4w screen 1, one round per device; threshold fixed before it ran in
  `tools/agree.py`: rank correlation >= 0.95 and every ratio within 0.95 to 1.05):
  `results/orin/second-device/agreement-screen1-4w.csv`. Spearman rank correlation 0.9973; 25 of 26 ratios
  second / primary within 0.975 to 1.002; one, `4070ti_t256x256k16g44s32gac`, at 0.8037. Verdict by the
  threshold: **DISAGREE**. I stopped using the device at 18:21 UTC (job `chain-s1` killed). That verdict stands
  as recorded; the threshold was not touched.
- Owner decision of 19:10 UTC (`CAMPAIGN.md`): retest that configuration with at least three rounds per
  device, then use the device for screening in any case. Retest (`tools/chain15.sh`, same build and
  invocation, 3 rounds, both devices idle otherwise; `results/orin/second-device/agreement-retest.csv`):

  | configuration | primary, geomean kernel time (3 rounds) | second device | ratio |
  |---|---:|---:|---:|
  | `base` (the shipped Orin rows) | 7898 / 7901 / 7899 us | 7885 / 7885 / 7888 us | 0.998 |
  | `4070ti_t256x256k16g44s32gac` | 18628 / 18411 / 18618 us | 15057 / 15140 / 15053 us | **0.809** (first batch 0.804) |

  The retest does **not** fall inside 0.95 to 1.05: the difference is reproducible (each device repeats within
  1.2 %), so it is a property of the two devices for this configuration, not noise. As decided by the owner the
  configuration is treated as device-sensitive (it is 2.4 times slower than `base` and cannot be selected) and
  `duck-stable` is used for screening from 19:34 UTC. Unchanged: whatever a screen there ranks first is
  confirmed on `duck-naughty` before it counts; a contradiction ends the use of the device.
  What the outlier has that the other 25 do not: it is the only 1024-thread tile of the batch (256 x 256, grid
  4 x 4). Not investigated further.
- Used for: SDPA screens 5 and 6 (softmax, second and third batch; `results/orin/second-device/sdpa-screen{5,6}.csv`).
- **Contradiction, and the device is not used any more (20:29 UTC).** Screen 6 there ranked the softmax variants
  `orin_g128` (7.24 ms per 1B layer) before `orin_g64` (7.34), both 10 to 11 % faster than `4070ti_nzf` (8.16).
  The confirmation on the primary (SDPA screen 7, 2 rounds, `results/orin/screens/sdpa-screen7.csv`) ranks
  `orin_g64` (8.25 ms) before `orin_g128` (8.36), and both are only 4.4 to 5.6 % faster than `4070ti_nzf`
  (8.74). The direction holds (subgroup reductions are faster on both devices) but the first place and the
  size of the gain do not. As the owner's decision says, I report it and stopped using the device (job
  `chain17` killed; nothing else was queued there). What the two devices differ in, from these runs: the
  kernels bound by memory traffic are faster on `duck-stable` (the same softmax 8.16 against 8.74 ms, QK^T 6.49
  against 6.73, attn*V 3.79 against 3.84), while the linear kernels agree within 0.3 %. A screen of a
  memory-bound kernel on that device does not transfer; a screen of a matrix-bound kernel did.

## Softmax, single-read variants: negative (SDPA screen 4, primary, `results/orin/screens/sdpa-screen4.csv`)

Hypothesis: after candidate 1h the softmax is the largest attention kernel (140 of 1378 ms on 1B 4w); it makes
three passes over a row and loads the row in each, and 3 x 134 MB read + 134 MB written at the DRAM roofs is
8.8 ms per 1B layer against 8.75 measured, so the time would be the traffic. `tools/gen_orin_softmax.py`, first
family: a worker keeps the texels it loaded in pass 1 (local array `l`, or shared memory `s`; 8, 16 or 32
texels per worker; `e`: the exponentials too). Kernel time per layer at S = 2048, ms:

| softmax | 1B | 3B | 8B |
|---|---:|---:|---:|
| `4070ti_nzf` (candidate 1h) | 8.74 | 6.54 | 8.74 |
| local array, 8 / 16 / 32 texels | 11.50 / 9.80 / 9.81 | 8.63 / 7.42 / 7.38 | 11.51 / 9.83 / 9.88 |
| local array + exponentials, 8 / 16 / 32 | 12.37 / 12.37 / 12.70 | 9.30 / 9.30 / 9.54 | 12.41 / 12.34 / 12.41 |
| shared memory, 8 / 16 / 32 texels (8 / 16 / 32 KB) | 16.26 / 22.69 / 40.38 | 12.17 / 16.93 / 30.21 | 16.27 / 22.69 / 40.37 |
| shared memory + exponentials, 8 / 16 / 32 | 19.22 / 24.25 / 40.76 | 14.43 / 18.11 / 30.49 | 19.25 / 24.29 / 40.75 |

Every variant is slower (1.12x to 4.7x the time), so the hypothesis is wrong: the re-reads of a row are not
what the softmax pays for (a row is at most 4 KB and was just read), and the agreement of the traffic sum with
the measured time was a coincidence. Stopped after round 1 (round 2 only for `4070ti_nzf` and the two 8-texel
local variants; they repeat within 0.1 %); the bit comparison was not run. Not a candidate. One thing the
screen shows that matters beyond the softmax: on this device a kernel's time grows steeply with the shared
memory a workgroup declares (64 threads with 8 / 16 / 32 KB: 1.9x / 2.6x / 4.6x).

## Softmax, fewer barriers per row: candidate 4 (`orin_g64`)

What a row costs besides its traffic is 14 barriers (two tree reductions over 64 workers, 7 each) for on
average 4 texels per worker. `tools/gen_orin_softmax.py`, second and third batch (builds `topic11`, `topic12`):
`t` the tree with fewer workers; `z` worker 0 walks the same tree alone (2 barriers per reduction; `z64` has
the pairs and order of `4070ti_nzf`); `f` one barrier per reduction, every worker combines the partial results
in index order; `g` each subgroup reduces without a barrier (`subgroupMax` / `subgroupAdd`), one barrier, every
worker combines the subgroups' results. The number is the workers per row (the workgroup size is fixed in the
shader; the dispatch stays one workgroup per row). Kernel time per 1B layer at S = 2048, ms:

| workers per row | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `t` (tree), second device | | | | 31.8 | 17.9 | 11.0 | 8.16 (`4070ti_nzf`) | | | |
| `z` (worker 0 walks the tree), second device | | | | 32.7 | 19.5 | 14.0 | 14.4 | | | |
| `f` (flat), second device | 144 | 88.1 | 59.6 | 31.4 | 17.5 | 10.6 | 7.74 | 10.9 | 34.0 | |
| `g` (subgroup), second device | | | | | | 10.6 | 7.34 | 7.24 | 11.1 | 20.1 |
| **primary** (screen 7, 2 rounds): `4070ti_nzf` 8.74 | | | | | | | `f` 8.47, **`g` 8.25** | `g` 8.36 | | |

- The time is roughly inverse to the workers per row up to 32 (one thread per row: 144 ms), flat between 64 and
  128, and rises again beyond: a row needs a full subgroup or two, no more. Anything done by one thread while
  the others wait is expensive (`z64` 1.76x).
- Where a row's time goes (second device, exploration only; measurement twins of `4070ti_nzf` that stop early
  and are wrong by construction, `orin_xp0/1/2`): launch of the workgroups 1.02 ms; pass 1 (the read of the
  row from DRAM and the maximum) 4.09; pass 2 (exp, sum) 1.56; pass 3 (exp, divide, store) 1.49; sum 8.16.
  Half of the softmax is the first read; the two reductions are inside the 4.09 and the 1.56.
- On the primary: `orin_g64` 8.25 / 6.18 / 8.25 ms (1B / 3B / 8B) against 8.74 / 6.53 / 8.74: **-5.6 %** of the
  softmax, i.e. 8 of 1378 ms on 1B 4w, 16 of 7137 on 8B 4w: an expected +0.2 to +0.6 % end to end.
- It changes the arithmetic: a row's exponentials are summed in another order (fp32, rounded once on the
  store). Measured on the primary with the gate's test binary (`results/orin/sdpa-error5/`): 0 mismatches in
  all 12 cases, `pairing=ok`; error against the fp32 CPU reference on the production cases, rms / maximum:
  stock 8.55e-5 / 1.713e-3 (1B), 8.69e-5 / 1.408e-3 (3B), 8.70e-5 / 1.587e-3 (8B); `4070ti_nzf` 2.1010e-5 /
  9.135e-4, 2.0676e-5 / 7.828e-4, 2.0566e-5 / 8.911e-4; `orin_g64` 2.1010e-5 / 9.135e-4, 2.0676e-5 / 7.828e-4,
  2.0566e-5 / 8.911e-4. Not larger than the parent's in rms and maximum in 12 of 12 cases; the maximum equals
  `4070ti_nzf`'s in every case and the rms is within 1.2e-4 relative of it. Direct difference from
  `4070ti_nzf`'s output (`diff-orin_g64-vs-4070ti_nzf.csv`): 0.3 to 0.4 % of the fp16 elements differ, by at
  most 1.2e-4 (one fp16 step below 1.0), rms 4.6e-7.
- Candidate 4 = `ET_VK_SARC_SOFTMAX_VARIANT=orin_g64`, judged as an arithmetic change (`gate_sdpa.sh`, the
  reference error above; if a next-token item differs, the rule's logits evidence is owed before it can count).

## Owner decision received (2026-10-05): the softmax-name hook is accepted

Asked here as "Decision needed from the owner" at 08:30 UTC (candidate 1 cannot pass the gate from the dev
zone: one next-token item differs and the reference-error rule fails on one maximum error; the softmax that
removes this needs a hook). Answer, in `CAMPAIGN.md`: option (a). One release-zone edit for the softmax name,
as its own commit; candidate 1h is judged by the unchanged gate and the reference-error rule per case; option
(b) is not granted and candidate 1 stays rejected.

Done: commit `307abb2ed` (`Override::softmax_variant`, 8 lines in `impl/sarc/Select.h` and
`impl/sarc/SdpaCoopmat.cpp`; the `Override` form, no environment variable in release code), diff and checks in
`proposal.md` under "Release-zone hook (owner decision 2026-10-05)". The dev zone names the variant
(`ET_VK_SARC_SOFTMAX_VARIANT`, commit `4718f3e07`). `sarc/tools/check.sh --no-build`: PASS, `test_sarc_select`
on the release tables unchanged (1240 checks, 31 rows). Build `topic6` contains both; candidate 1h is gated on
it (`s4-c1h`), not on the local-patch build `hook4`.

## Candidates

| # | profile / environment | what | reachable from the dev zone | state |
|---|---|---|---|---|
| 1 | `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=orin-refine1` | SDPA prefill kernels (QK^T, attn*V) | yes (`OrinSdpa.cpp`, no hook) | **REJECTED** at `verify-check` (`s2-c1`); evidence session running |
| 2 | `ET_VK_SARC_DEV_PROFILE=orin-lin-refine2` | 8da4w linear: whole-texel weight staging | yes | **GATE_ACCEPTED** (`s3-c2`, plain pass: output bit-identical to the shipped kernel), +2.84 % (8da4w cells +3.8 / +6.9 / +6.7 %) |
| 3 | `orin-refine5` + `4070ti_nzf` (on top of 1h + 2) | 4w linear: column-major weight staging on the 256 x 128 tile (K <= 8192), texel-wise staging on the fp32 tile (K > 8192) | yes | **GATE_ACCEPTED** (`s5-c3`, plain pass: bit-identical to the shipped kernels on the shapes each tile serves), **+0.82 %** over its parent (candidates 1h + 2); below 2 % |
| 4 | the accepted profile + `ET_VK_SARC_SOFTMAX_VARIANT=orin_g64` | softmax: reductions inside the subgroups, 2 barriers per row instead of 14 | through the softmax-name hook | **GATE_ACCEPTED** (`s6-c4`; no next-token item differs; an arithmetic change, reference error reported), **+0.41 %** over its parent (1h + 2 + 3): inside the +-2 % band |
| 1h | candidate 1 + `ET_VK_SARC_SOFTMAX_VARIANT=4070ti_nzf` | fp32 softmax without the zero tail | through the softmax-name hook (owner decision 2026-10-05, commit `307abb2ed`), build `topic6` | **`ACCEPTED (reference-error rule, owner decision 2026-10-04)`** (`s4-c1h`, `GATE_ACCEPTED`, no differing item), +57.8 % |

### Candidate 4, softmax `orin_g64` on top of candidates 1h + 2 + 3: `GATE_ACCEPTED`, +0.41 % (inside the noise band)

Build `topic13` in both arms. Parent arm: `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=orin-refine5
ET_VK_SARC_SOFTMAX_VARIANT=4070ti_nzf`; candidate arm: the same with `ET_VK_SARC_SOFTMAX_VARIANT=orin_g64`.
What the kernel is, its kernel-level screens and its measured error: "Softmax, fewer barriers per row" below.

- SDPA correctness with the candidate environment: 12 passes x tiers extended and full: 0 mismatches,
  `pairing=ok`, softmax kernel `sarc_sdpa_attn_weights_softmax_buffer_half_orin_g64` in all 144 cases
  (`gate_check.py sdpa`: ACCEPT, 0 findings).
- Unmodified `verify.sh`: `gate_check.py verify`: ACCEPT, 0 findings; all four default-vs-tiled next-token
  items SAME.
- Timed session (`results/orin/sessions/s6-c4/`; tok/s, median of 5 valid interleaved runs per arm):

  | cell | parent arm (1h + 2 + 3) | candidate 4 | gain | next token (4 prompts) |
  |---|---:|---:|---:|---|
  | 1B 4w | 1482.98 | 1491.62 | +0.58 % | SAME |
  | 1B 8da4w | 1373.57 | 1385.66 | +0.88 % | SAME |
  | 3B 4w | 628.03 | 629.77 | +0.28 % | SAME |
  | 3B 8da4w | 569.05 | 570.79 | +0.31 % | SAME |
  | 8B 4w | 295.06 | 295.78 | +0.25 % | SAME |
  | 8B 8da4w | 268.66 | 269.15 | +0.18 % | SAME |

  Geomean **+0.41 %**, no cell outside the +-2 % band: by the protocol this is noise, not a gain, and I do not
  claim one. (It is positive in all six cells and matches the ETDump below, but the rule is the rule.) 60 timed
  runs all valid; `gate_check.py session`: ACCEPT, 0 findings (24 of 24 next-token rows SAME); `env-check`:
  ACCEPT. `gate.done`: `GATE_ACCEPTED ... all steps passed`.
- ETDump (warm, ms, parent arm -> candidate; `s6-c4/trace/`): softmax 140.1 -> 132.2 (1B), 183.0 -> 172.8 (3B),
  279.8 -> 264.3 (8B), -5.6 % as at kernel level; total dispatch 1366 -> 1358, 3245 -> 3234, 6924 -> 6910 (4w).
- Recording: the gate passed without a differing next-token item, so the reference-error rule was not needed
  to accept it. Because it changes the arithmetic (the order of a row's sum), its measured error against the
  fp32 reference is reported (not larger than the parent's in 12 of 12 cases, equal to `4070ti_nzf`'s to four
  digits), and the rule's real-text logits comparison is being collected for the final stack (`chain20`).
  The candidate can be dropped without a measurable loss: the stack without it is this session's parent arm.

### Candidate 3, `orin-refine5` (4w linear tiles) on top of candidates 1h + 2: `GATE_ACCEPTED`, +0.82 %

Build `topic13` (`fd44f8011`) in both arms. Parent arm: `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=orin-refine3
ET_VK_SARC_SOFTMAX_VARIANT=4070ti_nzf` (candidates 1h + 2, the accepted stack); candidate arm: the same with
`orin-refine5`, which adds `orin_t256x128k16g42s32bt` (K <= 8192) and `bx_t128x128k32g42s32f32c` (K > 8192) for
the 4w linear layers. So the session measures candidate 3 over its parent directly, and its `verify.sh` run is
the first gate of the whole stack 1h + 2 + 3 together.

- Bit identity (the tiles claim the shipped arithmetic): see "4w linear" below; identical on every shape a tile
  serves.
- Unmodified `verify.sh` with the candidate environment: `gate_check.py verify`: ACCEPT, 0 findings: every item
  equals the parent control, all four default-vs-tiled next-token items SAME.
- Timed session (`results/orin/sessions/s5-c3/`; tok/s, median of 5 valid interleaved runs per arm):

  | cell | parent arm (1h + 2) | candidate 3 | gain | next token (4 prompts) |
  |---|---:|---:|---:|---|
  | 1B 4w | 1472.32 | 1484.06 | +0.80 % | SAME |
  | 1B 8da4w | 1374.50 | 1376.34 | +0.13 % | SAME |
  | 3B 4w | 621.55 | 627.84 | +1.01 % | SAME |
  | 3B 8da4w | 569.05 | 569.21 | +0.03 % | SAME |
  | 8B 4w | 286.43 | 294.97 | +2.98 % | SAME |
  | 8B 8da4w | 268.59 | 268.66 | +0.03 % | SAME |

  Geomean **+0.82 %**: below the 2 % of the stop rule. Only 8B 4w is outside the +-2 % band (repeat spread there
  0.06 %); the other 4w cells gain 0.8 to 1.0 % with a spread of 0.1 to 0.4 %, which the A/A floor (0.09 %)
  resolves but the +-2 % rule calls noise; the 8da4w cells do not use the kernels. 60 timed runs all valid at
  612 MHz; `gate_check.py session`: ACCEPT, 0 findings (24 of 24 next-token rows SAME); `env-check`: ACCEPT.
- Where it comes from (warm ETDump, ms, parent arm -> candidate; `s5-c3/trace/`): the linear GEMM family of the
  4w cells, 667 -> 655 (1B), 1895 -> 1862 (3B), 4862 -> 4654 (8B; 150 of the 208 ms are the K = 14336 shape on
  the fp32 tile); total dispatch 1378 -> 1366, 3277 -> 3245, 7132 -> 6924. Nothing else moves.
- The parent arm of this session is also the first end-to-end measurement of candidates 1h + 2 together:
  1B 1472.3 / 1374.5, 3B 621.6 / 569.1, 8B 286.4 / 268.6 tok/s (4w / 8da4w).

### Candidate 2, `orin-lin-refine2` (8da4w linear, whole-texel weight staging): `GATE_ACCEPTED`

Build `topic6` with `ET_VK_SARC_DEV_PROFILE=orin-lin-refine2` against the pristine parent (so this session
measures candidate 2 alone; its effect on top of candidate 1h is measured by the combined candidate).
Kernel: `sarc_linear_dq8ca_coopmat_zpgtr_orin_bf_t128x128k64g24s32mk32ra` for every shape of the Orin row.

- The kernel claims not to change the arithmetic, so it is shown, not assumed: the raw output of all 12 model
  shapes (test's seeded inputs) is byte-identical to the shipped kernel's, as is the `g42` grid; the shipped
  kernel run twice is identical to itself (`results/orin/screens/bit-bf.txt`, 36 of 36 identical). Sampled
  production-diff with non-zero zero points: ALL PASSED on 1B, 3B, 8B (`screens/pdiff-bf2.txt`).
- Unmodified `verify.sh`: `gate_check.py verify`: ACCEPT, 0 findings: every item equals the parent control.
- Timed session (`results/orin/sessions/s3-c2/`; tok/s, median of 5 valid interleaved runs per arm):

  | cell | parent | candidate 2 | gain | next token parent vs candidate (4 prompts) |
  |---|---:|---:|---:|---|
  | 1B 4w | 891.21 | 891.21 | 0.00 % | SAME |
  | 1B 8da4w | 823.81 | 854.76 | +3.76 % | SAME |
  | 3B 4w | 360.63 | 360.56 | -0.02 % | SAME |
  | 3B 8da4w | 320.45 | 342.59 | +6.91 % | SAME |
  | 8B 4w | 189.82 | 189.84 | +0.01 % | SAME |
  | 8B 8da4w | 170.51 | 181.88 | +6.67 % | SAME |

  Geomean **+2.84 %** over six cells (the three 8da4w cells alone: +5.77 %; the 4w cells do not use the
  kernel and do not move). Repeat spread at most 0.40 %, 60 timed runs all valid at 612 MHz, `gate_check.py
  session`: ACCEPT, 0 findings; `env-check`: ACCEPT. `gate.done`: `GATE_ACCEPTED ... all steps passed`.
- Where the gain comes from (warm ETDump, ms per 2048-token prefill, parent -> candidate; `s3-c2/trace/`):
  the linear GEMM family of the 8da4w cells, 870 -> 778 (1B), 2577 -> 2171 (3B), 6054 -> 5306 (8B), i.e.
  1.12x / 1.19x / 1.14x; total dispatch 2474 -> 2381, 6369 -> 5964, 11990 -> 11244. The 8-bit quantize
  (99 / 252 / 425 ms), attention and everything else are unchanged, and so is every family of the 4w cells.
  In the kernel: the weight fetch falls from 38 to 47 % of a wave to 15 to 24 % (phase timing below).

### Candidate 1h, `orin-refine1` + `ET_VK_SARC_SOFTMAX_VARIANT=4070ti_nzf`: `ACCEPTED (reference-error rule, owner decision 2026-10-04)`

Build `topic6` (`4718f3e07`: the branch with the softmax-name hook `307abb2ed`) against the pristine parent.
What differs from candidate 1: the softmax reduces each row in fp32 and does not write the zero tail; QK^T and
attn*V are candidate 1's kernels.

- Reference error (criterion 1 of the rule), measured before the gate with the gate's test binary
  (`results/orin/sdpa-error2/summary.txt`, five arms, same seeded inputs, 0 mismatches in all 12 cases of every
  arm):

  | production case (S = 2048) | rms error: parent / candidate 1 / candidate 1h | maximum error: parent / candidate 1 / candidate 1h |
  |---|---|---|
  | 1B head configuration | 8.55e-5 / 3.49e-5 / 2.10e-5 | 1.713e-3 / 1.288e-3 / 0.914e-3 |
  | 3B head configuration | 8.69e-5 / 3.54e-5 / 2.07e-5 | 1.408e-3 / **1.570e-3** / 0.783e-3 |
  | 8B head configuration | 8.70e-5 / 3.48e-5 / 2.06e-5 | 1.587e-3 / 1.498e-3 / 0.891e-3 |

  Candidate 1h is not larger than the parent in rms and in maximum error in 12 of 12 cases (candidate 1: 11 of
  12). The fp32 softmax alone (`4070ti_f32`) gives the same numbers as `4070ti_nzf`, element for element; the
  no-zero-tail softmax alone (`4070ti_nz`) the same as the release softmax.
- SDPA correctness: 12 passes x tiers extended and full: 0 mismatches, both cooperative-matrix kernels
  dispatched, `pairing=ok`, and the softmax kernel is `sarc_sdpa_attn_weights_softmax_buffer_half_4070ti_nzf` in
  all 144 cases (`gate_check.py sdpa`: ACCEPT, 0 findings).
- Unmodified `verify.sh`: `gate_check.py verify`: **ACCEPT, 0 findings**. Every item equals the parent control,
  including all four next-token items: `1b 8da4w unaligned: default vs tiled output SAME` this time.
- Timed session (`results/orin/sessions/s4-c1h/`; pristine parent against `topic6` with the candidate
  environment; tok/s, median of 5 valid interleaved runs per arm; `cells.csv` = the original `dev/1.5` numbers):

  | cell | parent | candidate 1h | gain | `cells.csv` | next token parent vs candidate (4 prompts) |
  |---|---:|---:|---:|---:|---|
  | 1B 4w | 890.82 | 1471.26 | +65.2 % | 890.82 | SAME |
  | 1B 8da4w | 822.82 | 1292.93 | +57.1 % | 822.82 | SAME |
  | 3B 4w | 360.50 | 621.36 | +72.4 % | 360.37 | SAME |
  | 3B 8da4w | 320.45 | 511.23 | +59.5 % | 320.30 | SAME |
  | 8B 4w | 189.74 | 286.23 | +50.9 % | 189.74 | SAME |
  | 8B 8da4w | 170.45 | 244.57 | +43.5 % | 170.43 | SAME |

  Geomean **+57.81 %**, every cell far outside the +-2 % band (A/A noise 0.09 %), repeat spread at most
  0.25 %, 60 timed runs all valid (clock 612 MHz in every run), `gate_check.py session`: ACCEPT, 0 findings
  (24 of 24 next-token rows SAME). `env-check`: ACCEPT. `gate.done`: `GATE_ACCEPTED ... all steps passed`.
- Where the gain comes from (warm ETDump of both arms, ms per 2048-token prefill, parent -> candidate;
  `results/orin/sessions/s4-c1h/trace/`):

  | family | 1B 4w | 1B 8da4w | 3B 4w | 3B 8da4w | 8B 4w | 8B 8da4w |
  |---|---|---|---|---|---|---|
  | QK^T | 581 -> 108 | 581 -> 108 | 1487 -> 199 | 1488 -> 198 | 2268 -> 302 | 2268 -> 301 |
  | attn*V | 457 -> 61 | 457 -> 61 | 1194 -> 144 | 1194 -> 144 | 1818 -> 216 | 1818 -> 216 |
  | softmax | 177 -> 140 | 177 -> 140 | 231 -> 183 | 231 -> 183 | 351 -> 280 | 352 -> 280 |
  | linear GEMM | 666 -> 667 | 870 -> 870 | 1894 -> 1895 | 2577 -> 2577 | 4865 -> 4865 | 6055 -> 6063 |
  | everything else | 402 -> 402 | 390 -> 391 | 856 -> 857 | 882 -> 884 | 1474 -> 1474 | 1502 -> 1503 |
  | total dispatch | 2283 -> 1378 | 2475 -> 1570 | 5662 -> 3278 | 6372 -> 3986 | 10776 -> 7137 | 11995 -> 8363 |

  The whole gain is attention: 1215 -> 309 ms on 1B, 2912 -> 526 on 3B, 4437 -> 798 on 8B. Nothing else moves.
- The reference-error rule, as written (`results/orin/probe/refine1-nzf/`: `reference-error-rule.txt`,
  `REFERENCE_ERROR.json`, `compare.csv`, `position/`). The gate found no differing next-token item, so it did
  not need the rule to accept; the owner's decision of 2026-10-05 asks for it all the same, and the candidate
  changes the attention arithmetic, so it is recorded under the rule and not as a plain pass.
  - Criterion 1: met, 12 of 12 cases (table above).
  - Criterion 2, evidence. Logits at the position of the gate's unaligned item (prompt 0 of the set, 1B 8da4w,
    `position/summary.csv`): all four arms pick token 220; top-2 margin parent +0.31 (default) / +0.38 (tiled),
    candidate +0.11 (default) / +0.30 (tiled). Real-text comparison, 41 prompts (32 tile-aligned lengths 64 ..
    2048, 8 unaligned lengths, the gate's unaligned prompt; the sibling campaign's set, unchanged), full
    next-token distribution of the last position, all six cells, parent tiled vs parent default beside it:

    | cell | top-1 differences of 41 (parent's two arms / candidate vs parent) | mean KL, nats | max KL | max abs logit diff | perplexity (parent / candidate) |
    |---|---|---|---|---|---|
    | 1B 4w | 0 / 0 | 0.00055 / 0.00144 | 0.0109 / 0.0099 | 0.73 / 1.20 | 8.70 / 8.93 |
    | 1B 8da4w | 2 / 5 | 0.0618 / 0.0756 | 0.606 / 0.825 | 4.42 / 4.34 | 9.58 / 10.33 |
    | 3B 4w | 0 / 0 | 0.00012 / 0.00032 | 0.0014 / 0.0019 | 0.33 / 0.73 | 3.72 / 3.77 |
    | 3B 8da4w | 1 / 1 | 0.0106 / 0.0238 | 0.152 / 0.408 | 3.73 / 4.00 | 3.63 / 3.69 |
    | 8B 4w | 0 / 0 | 0.00023 / 0.00044 | 0.0027 / 0.0023 | 0.49 / 0.85 | 2.72 / 2.71 |
    | 8B 8da4w | 1 / 4 | 0.0278 / 0.0251 | 0.447 / 0.299 | 2.72 / 4.56 | 2.84 / 2.84 |

    The 8da4w rows are the 4070 Ti campaign's numbers for the same kernels, digit for digit; the 4w rows differ
    from that device's because the 4w linear kernel does.
  - Criterion 3, gross divergence: largest mean KL 0.076 nat (limit 0.5); the top-1 token differs on at most 5
    of 41 prompts (limit: one third). None.
  - Verdict of `tools/ref_error_rule.py`: MET. Differing next-token items: none. (The last line of
    `compare.csv` prints the first decision's test, "outside twice the noise floor"; the second decision
    replaced that test for arithmetic changes. `REFERENCE_ERROR.device.json` is the file as first written on
    the device, with one empty string in its item list, a slip of `chain13.sh`; the rule was re-run on the same
    files without it.)
  - The probe took 4.5 hours of device time, most of it in the two tiled arms (the linear layers run at 25 to
    230 tok/s there). Memory: 1.2 GB available with the 8B model loaded, swap use 222 MB, unchanged in the run.

### Candidate 1, `orin-refine1` (SDPA prefill kernels): gate `s2-c1`, REJECTED

Build `topic4` (`8973debef`, dev zone only) against the pristine parent.

- SDPA correctness: 12 passes x tiers extended and full, 8 + 4 cases each: 0 mismatches, both cooperative-matrix
  kernels dispatched and `pairing=ok` in all 144 cases (`gate_check.py sdpa`: ACCEPT, 0 findings).
- Unmodified `verify.sh`: every item equals the parent control (correctness cases, 24 linear cases, the 12
  production-diff cases shape by shape, decode 31 tokens, default vs tiled SAME on `prompt_check` for both
  schemes and on the unaligned prompt for 4w) **except one**: `1b 8da4w unaligned: default vs tiled output
  DIFFER`. `gate_check.py verify`: REJECT, 1 finding. `gate.done`: `GATE_REJECTED ... step verify-check`.
  This is the item the 4070 Ti's candidate 1 failed on, with the same attention arithmetic and the same 8da4w
  linear kernel.
- Reference-error rule (second owner decision of 2026-10-04), criterion 1, measured with the gate's own test
  binary (`results/orin/sdpa-error1/`, table below under "What the SDPA candidate is expected to hit"; the
  numbers are the same): rms error lower than the parent's in all 12 cases, maximum error lower in the 1B and 8B
  production cases and **11 % higher in the 3B production case (1.570e-3 against 1.408e-3)**. The criterion
  says rms and maximum, every head configuration: NOT MET. The candidate stays rejected; I did not collect the
  41-prompt logits comparison for it, because it cannot change this verdict (it is collected for candidate 1h).
- Measured gain, for the record (`s2-c1`, `gate_rest.sh`: evidence only, `gate.done` stays REJECTED; pristine
  parent against `topic4` with the candidate environment, tok/s, median of 5 valid interleaved runs per arm):

  | cell | parent | `orin-refine1` | gain | next token parent vs candidate (4 prompts) |
  |---|---:|---:|---:|---|
  | 1B 4w | 890.82 | 1466.00 | +64.6 % | SAME |
  | 1B 8da4w | 822.82 | 1290.49 | +56.8 % | SAME |
  | 3B 4w | 360.44 | 620.23 | +72.1 % | SAME |
  | 3B 8da4w | 320.40 | 510.21 | +59.2 % | SAME |
  | 8B 4w | 189.79 | 285.99 | +50.7 % | SAME |
  | 8B 8da4w | 170.48 | 244.28 | +43.3 % | SAME |

  Geomean **+57.5 %**, repeat spread at most 0.43 %, 60 timed runs all valid, `gate_check.py session`: ACCEPT
  (0 findings): parent and candidate print the same next token in all six cells on the timed prompt, the
  real-text prompt, `prompt_check` and the unaligned prompt (24 of 24). The one differing item of the gate is
  the candidate's own default-vs-tiled comparison on the unaligned prompt for 1B 8da4w.
- Decode: that `verify.sh` run read 17.6 tok/s for 1B 4w against the parent control's 19.4. A 3-run decode
  A/B, arms interleaved (`raw/decode-ab1`, 1B, 32 new tokens after the 2048-token prompt), does not confirm it:
  4w parent 19.29 / 19.39 / 19.31, `orin-refine1` 19.07 / 19.15 / 19.15 (-0.9 %); 8da4w parent 11.05 / 11.02 /
  (third run pending), `orin-refine1` 10.99 / 10.97 (-0.5 %); `orin-lin-refine2` 4w 19.17 / 19.16 / 19.41,
  8da4w 10.92 / 10.90 (-1.1 %). All inside the +-2 % band; decode is not what these candidates change.

## SDPA kernels on the Orin (kernel level, `test_llama_microbench --sdpa`, ms per layer at S = 2048)

`results/orin/screens/sdpa-screen{1,2}.csv`: screen 1 = 59 profiles + stock, 1 round; screen 2 = the best of
screen 1 and 12 new Orin QK^T tiles, 2 rounds (the two screens agree within 0.01 ms). Reached from the dev zone
through `impl/sarc_dev/OrinSdpa.cpp` (two `kUnverified` base rows for `tegra orin` that match only while the
profile is `orin-*` and not `orin-lin-*`); the softmax is the release SARC softmax in every row but stock.

| kernels | 1B QK^T / softmax / attn*V | 3B | 8B |
|---|---|---|---|
| stock (the parent) | 36.29 / 10.97 / 28.60 | 53.11 / 8.23 / 42.64 | 70.87 / 10.97 / 56.76 |
| base rows (the 4070 Ti port tiles, mask fill) | 10.05 / 9.03 / 3.84 | 10.04 / 6.79 / 5.59 | 13.38 / 9.03 / 7.39 |
| **`orin-refine1`** | **6.73 / 9.03 / 3.84** | **7.10 / 6.79 / 5.13** | **9.40 / 9.03 / 6.75** |

- QK^T: packed staging wins on this device and direct feed loses (10.3 to 20 ms; the opposite of the 4070 Ti,
  where direct feed won for head_dim 64). K = 64 per chunk beats K = 32 on the same tile (9.40 against 10.27 ms
  on 8B). Best: `4070ti_pk_t128x64k64g42s32nf` for all three head configurations. None of the 12 Orin tiles
  (`tools/gen_orin_qk.py`: K = 64 on other tiles and grids, K = 128) beats it: best 6.82 / 7.28 / 9.66
  (`orin_pk_t64x64k64g22s32nf`); K = 128 tiles 8.3 to 15 ms. Negative result, kept.
- attn*V: head_dim 64 keeps the 64 x 64 tile (3.84 ms; every other tile 4.1 to 10.8); head_dim 128 takes
  `4070ti_ml_t64x128k32g42s32` (5.13 / 6.75 against 5.59 / 7.39). Direct-feed attn*V: 5.99 to 21 ms.
- Against the fresh roofs, 1B per layer: QK^T writes 134 MB (the unmasked half) and does 8.6 GFLOP: 2.3 ms at
  the DRAM write roof plus 0.9 ms at the fp16 -> fp32 matrix roof; measured 6.73. attn*V reads 134 MB: 2.2 ms
  at the read roof plus 0.9 ms; measured 3.84. The softmax reads 134 MB and writes 268 MB (half of it the zero
  tail): 6.8 ms of traffic at the roofs; measured 9.03. After candidate 1 the softmax is the largest of the
  three, and its name is fixed in the release zone.
- fp16-accumulating variants (`dfg`, `dfh`) were screened with the rest and are not pursued (no gain, and they
  would change precision).

### What the SDPA candidate is expected to hit (measured before the gate, `results/orin/sdpa-error-early/`)

The error of the attention block against the fp32 CPU reference, same test binary and seeded inputs, is on the
Orin digit for digit what the 4070 Ti campaign measured, for the stock kernels and for the SARC kernels:

| production case (S = 2048) | rms error, stock / SARC kernels | maximum error, stock / SARC kernels |
|---|---|---|
| 1B head configuration | 8.55e-5 / 3.49e-5 | 1.713e-3 / 1.288e-3 |
| 3B head configuration | 8.69e-5 / 3.54e-5 | 1.408e-3 / **1.570e-3** |
| 8B head configuration | 8.70e-5 / 3.48e-5 | 1.587e-3 / 1.498e-3 |

0 mismatches in all 12 cases of both arms. So the SARC kernels are 2.5 times closer to the reference in rms and
closer in maximum error in two of the three production cases, and 11 % further in the third: the owner's
reference-error rule (criterion 1, every head configuration, rms and maximum) is NOT met by a candidate that
keeps the release softmax, exactly as for the 4070 Ti's candidate 1. That only matters if the gate shows a
next-token DIFFER (the 4070 Ti's did: `1b 8da4w unaligned`). On the 4070 Ti the fp32 softmax variant
(`4070ti_nzf`) met the rule; it needs the softmax-name hook, which is outside the dev zone. I am not looking
for an attention kernel that happens to pass: the gate result is reported as it comes.

## A test fix the reviewer should look at

`backends/vulkan/test/sarc_dev/test_llama_microbench.cpp`, the SDPA pairing check ("a QK^T kernel without mask
fill must run with the truncated SARC softmax"): it compared the softmax kernel name by PREFIX. The Orin cross
build links the event tracer (as the campaign that produced the `cells.csv` numbers did), and then kernel names
are reported as `"kernel_name": "<name>", "operator_id": N`, so the prefix never matched and a correct pairing
read `pairing=BROKEN` with 0 mismatches (`results/orin/sdpa-error-early/qkpk.txt`, build `topic3`: the line
shows `softmax="kernel_name": "sarc_sdpa_attn_weights_softmax_buffer_half"` next to `pairing=BROKEN`). Commit
`417bd7a03` makes it a substring test. The stock softmax name (`sdpa_attn_weights_softmax_...`) does not contain
the SARC name, so a wrong pairing is still reported. No tolerance, case or mismatch rule is touched. Say so if
this should instead be solved by a tracer-free test build.

## 8da4w linear: phase timing and the whole-texel twin

Phase timing of the shipped `t128x128k64g44s32mk32ra` on the Orin (shader clock, share of one wave, the 12
model shapes; `results/orin/phases/prof1-8da4w.csv`): barrier 10 to 15 %, **fetch 38 to 47 %**, MMA 24 to
31 %, shared-memory store 8 to 10 %, epilogue 4 %, prologue + write 5 %. A wave spends more time fetching than
multiplying, as on the other devices.

The shipped kernel fetches every packed-weight texel four times per chunk (2 `texelFetch` per thread, one of
the four 32-bit words kept). `tools/gen_orin_bf.py` generates a twin of the release body in which a staging slot
is a whole texel: one fetch, eight shared-memory words; same values, same shared-memory layout, same MMA order.
Kernel-level screens, all 12 model shapes, geomean of kernel time against the shipped kernel
(`results/orin/screens/screen{1,2}-8da4w.txt`; 2 rounds for the Orin tiles, repeat spread below 0.3 %):

| tile | threads | weight fetches per thread and chunk | speed against shipped |
|---|---:|---:|---:|
| shipped `t128x128k64g44` | 512 | 2 (one word kept of each) | 1.000 |
| half texel `4070ti_bh_..g44` | 512 | 1 | 1.049 |
| whole texel `orin_bf_..k64g44` | 512 | 0.5 (every second thread) | 1.019 |
| plain 256-thread tile `4070ti_..k64g42` | 256 | 4 | 0.775 |
| half texel `4070ti_bh_..k64g42` | 256 | 2 | 0.908 |
| **whole texel `orin_bf_t128x128k64g24`** (`g42`: 1.154) | 256 | 1 | **1.158** |
| whole texel, one staging slice, K = 128 (`orin_bf1_..k128g44`; `g42` 0.964, `g24` 0.924) | 512 | 1 | 0.917 |
| whole texel, one slice, K = 64 (`orin_bf1_..k64g42`) | 256 | 1 | 0.987 |
| 12 other 4070 Ti sweep tiles | | | 0.69 to 0.87 |

- What matters is the number of weight fetches a thread makes per chunk, and that every thread makes the same
  number: one whole texel per thread on a 256-thread tile is +15.8 %; the same staging with half the threads
  idle is +1.9 %.
- The second barrier per chunk that a single staging slice needs costs about 17 % (0.987 against 1.154 on the
  same tile), more than K = 128 gains back. Double buffering stays.
- All 9 whole-texel tiles pass the sampled production-diff on the 1B shapes (non-zero zero points).
- Screen 3 (`screen3-8da4w.txt`, second batch of whole-texel tiles, built to test whether smaller subgroup
  tiles and fewer loads per thread help): no. `t64x128k64g22` 1.021, `t128x128k64g22` 0.994, `t64x128k64g42`
  0.969, `t64x64k64g22` 0.868, `t64x64k128g42` / `g24` 0.70 / 0.68. The 128 x 128 tile on 256 threads stays the
  best (1.158 again). All six pass the sampled production-diff on the 1B shapes.
- Candidate 2 = `orin_bf_t128x128k64g24s32mk32ra` for every shape the Orin row covers.

Where candidate 2's gain comes from (phase timing of its tile, `results/orin/phases/prof3-8da4w.csv`, against
the shipped kernel's above): fetch 15 to 24 % of a wave (shipped: 38 to 47 %), MMA 48 to 59 % (shipped: 24 to
31 %), barrier 10 %, shared-memory store 7 %. The wave now spends half its time multiplying.

## 4w linear: phase timing and screen (negative, one shape excepted)

Phase timing of the Orin 4w tiles (`results/orin/phases/prof2-4w.csv`): `t256x128k16g22s32`: barrier 10 to
11 %, fetch 17 to 18 %, MMA 45 to 49 %, shared-memory store (weight dequantisation) 18 to 20 %;
`t128x128k32g42s32f32` (8B `w2`, K = 14336): barrier 14 to 15 %, fetch 15 to 17 %, MMA 30 to 37 %, store 26 to
38 %. The weight staging has the same four-fold texel fetch as 8da4w, but here the activations are the larger
fetch load (8 texture3d fetches per thread and chunk against 2 weight fetches on the 256 x 128 tile).

4w screen 1 (`results/orin/screens/screen1-4w.txt`, kernel time, all 12 model shapes, 1 round): every existing
subgroup-32 dev tile, 25 of them (the 780M sweep tiles with fp32 accumulation, drain in Ash, column-major and
texel-wise weight staging; the 4070 Ti `ga` tiles). None beats the shipped Orin rows over the 12 shapes: best
0.94x (`bx_t128x128k32g42s32f32c`), `..f32cbt` 0.93x, the 4070 Ti `ga` tiles 0.42 to 0.79x. On the one shape the
Orin serves with fp32 accumulation (8B `w2`, K = 14336, shipped `t128x128k32g42s32f32`: 45.87 ms) two tiles are
faster: texel-wise weight staging `bx_t128x128k32g42s32f32c` 41.17 ms (1.11x) and column-major staging
`t128x128k32g42s32f32cbt` 41.75 ms (1.10x). That shape is 14 % of the 8B 4w prefill, so the end-to-end effect is
about +1 % on one cell. Profile `orin-lin-refine3` = candidate 2 + that tile for K > 8192 (build `topic7`).

4w screen 2 (`results/orin/screens/screen2-4w.txt`, build `topic8`, 2 rounds, primary): 12 staging and grid
variants of the shipped Orin tiles themselves (`tools/gen_orin_q4.py`). Kernel time against the shipped rows
over the 11 shapes with K <= 8192: column-major weight staging on the 256 x 128 K = 16 tile, 4 x 2 subgroup
grid (`orin_t256x128k16g42s32bt`) 1.3 to 1.8 % faster on every shape (1B 2.86 / 0.74 / 11.35 / 10.86 ms against
2.91 / 0.75 / 11.50 / 11.02; 8B `w1_w3` 38.30 against 38.85); the same staging on the shipped 2 x 2 grid 0.8 %,
with texel-wise fetches (`cbt`) 0.9 %; the 2 x 4 and 4 x 4 grids 4 to 37 % slower; the K = 32 tiles 8 to 18 %
slower. (The "geomean vs base" column of the file includes the K = 14336 shape, where a forced fp16 tile is
faster than the shipped fp32 one and is not valid: it fails production-diff there, `screens/pdiff-q4b.txt`.)

Candidate 3 = profile `orin-refine5`: `orin_t256x128k16g42s32bt` on the shapes the table gives to the 256 x 128
tile (K <= 8192, and not the M = 256, N % 2048 != 0 case, which keeps the table's tile), and
`bx_t128x128k32g42s32f32c` for K > 8192. Pre-checks on the primary: the raw output is byte-identical to the
shipped kernels' on the shapes each tile serves (`screens/bit-q4.txt`, `bit-q4b.txt`: the fp16 tile on the 11
shapes with K <= 8192, the fp32 tile on the K = 14336 shape; the shipped kernel run twice is identical to
itself); sampled production-diff ALL PASSED on those shapes. Expected end to end: +0.7 % on the 4w cells from
the fp16 tile and +1 % more on 8B 4w from the fp32 shape.

## Softmax variants on the Orin (kernel level, ms per layer at S = 2048; `results/orin/screens/sdpa-screen3.csv`)

| softmax | 1B | 3B | 8B |
|---|---:|---:|---:|
| release SARC softmax (candidate 1) | 9.03 | 6.79 | 9.04 |
| `4070ti_f32` (fp32 reduction) | 9.80 | 7.36 | 9.81 |
| `4070ti_nz` (no zero tail) | 8.01 | 6.00 | 8.01 |
| `4070ti_nzf` (both; candidate 1h) | 8.75 | 6.55 | 8.74 |

On this device the zero tail is worth 11 % of the softmax (30 % on the 4070 Ti) and the fp32 reduction costs
8.5 %, so candidate 1h is about as fast as candidate 1; its purpose is precision, not speed.

## Baseline and A/A (session `s1-aa`, pristine parent against build `topic1` with no environment)

tok/s, median of 5 valid runs per arm, arms interleaved, record-only clock (`results/orin/sessions/s1-aa/`):

| cell | parent | topic1, no env | ratio | `cells.csv` (dev/1.5) | parent vs `cells.csv` |
|---|---:|---:|---:|---:|---:|
| 1B 4w | 890.05 | 890.82 | 1.0009 | 890.82 | -0.09 % |
| 1B 8da4w | 823.48 | 823.48 | 1.0000 | 822.82 | +0.08 % |
| 3B 4w | 360.44 | 360.37 | 0.9998 | 360.37 | +0.02 % |
| 3B 8da4w | 320.30 | 320.25 | 0.9998 | 320.30 | 0.00 % |
| 8B 4w | 189.68 | 189.72 | 1.0002 | 189.74 | -0.03 % |
| 8B 8da4w | 170.47 | 170.41 | 0.9997 | 170.43 | +0.02 % |

The baseline agrees with `cells.csv` within 0.1 % in every cell (threshold 3 %). A/A geomean +0.01 %, largest
cell difference 0.09 %, repeat spread at most 0.36 %: the noise on this device is far inside the +-2 % band.
60 timed runs, all valid; next token parent vs topic SAME in all six cells on the four prompts (24 of 24).
`gate_check.py session --calibration --require-logs`: ACCEPT, 0 findings.

Clock: the devfreq clock reads 612 MHz (the 15 W cap) as the median of every timed run.
`results/orin/clkmin.json`: device-wide threshold floor(0.97 x 612) = **593 MHz**; no run below it.
Temperature 60 to 65 C at run start (idle 60 C; the fan keeps it there). Memory: 5.6 to 6.6 GB available
before every run; 13 of the 60 timed runs paged something out while they ran (at most 202 pages = 0.8 MB,
3.9 MB in total), with no visible effect on the rates.

## Parent control (`s0-parent-verify`)

Unmodified `sarc/tools/verify.sh --models 1b,3b,8b --schemes 4w,8da4w --pdiff --flat-models`, no environment,
22 of 22 runner calls rc 0, default vs tiled SAME on the check and the unaligned prompt for 1B 4w and 8da4w,
decode 31 tokens. What the pristine parent itself shows on this device that a naive checker calls a failure
(all listed as `PARENT-STATUS` in `results/orin/sessions/s0-parent-verify/verify-check.txt`; the same lines are
in the Orin evidence of `sarc-1.5-4w-port` and `sarc-1.5-8da4w-port`):

| line | parent | why |
|---|---|---|
| `correctness rc=1` | 3 upstream cases FAILED (`linear_q4gsw_M128_K4096_N128`, texture3d / buffer / rank-3 buffer), 41 PASSED | the Orin rows cover only the measured M = 2048 projections; M = 128 runs the upstream fp16 kernel, which fails at K = 4096 |
| `linear 4w rc=1`, `linear 8da4w rc=1` | 24 `unexpected_coopmat` + 24 `fallback_tiled` | texture3d coopmat dispatch reported as unexpected (as everywhere); buffer IO has no Orin row |
| `pdiff <model> <scheme> buffer rc=1`, 6 of 6 | FAILED | buffer IO has no Orin row: "NOT coopmat -- fallback, cannot validate the shader under test"; the upstream 4w buffer kernel also fails numerically at K >= 4096 (5 shapes) |
| `pdiff <model> <scheme> texture3d`, 6 of 6 | ALL PASSED | the model path |

`gate_check.py verify` first rejected the control because it required every correctness and production-diff
case to pass. It now records the status of every correctness case and of every production-diff shape (coopmat
or fallback, PASSED / FAILED / threw, rc, final line) and requires a candidate to EQUAL the control per case;
where the parent passes, the candidate must pass. Re-run on the same files: ACCEPT, 0 findings; the first
verdict is kept as `gate.done.first-check`. 41 unit tests.
SDPA correctness on the parent (recorded only): 0 mismatches in 8 + 4 cases, upstream kernels (`qk_coopmat=NO`).

## Where the time goes (parent, warm ETDump, ms per 2048-token prefill; `results/orin/sessions/s1-aa/trace/`)

| family | 1B 4w | 1B 8da4w | 3B 4w | 3B 8da4w | 8B 4w | 8B 8da4w |
|---|---:|---:|---:|---:|---:|---:|
| linear GEMM | 667 (29 %) | 871 (35 %) | 1895 (33 %) | 2577 (40 %) | 4863 (45 %) | 6054 (50 %) |
| QK^T (stock) | 581 | 581 | 1488 | 1488 | 2268 | 2268 |
| attn*V (stock) | 458 | 457 | 1194 | 1194 | 1817 | 1816 |
| softmax (stock) | 177 | 177 | 231 | 231 | 351 | 351 |
| attention total | 1215 (53 %) | 1215 (49 %) | 2913 (51 %) | 2912 (46 %) | 4436 (41 %) | 4435 (37 %) |
| copy / view / other | 267 | 154 | 581 | 351 | 985 | 574 |
| elementwise | 101 | 101 | 190 | 190 | 363 | 377 |
| 8-bit quantize | - | 101 | - | 254 | - | 425 |
| total dispatch | 2285 | 2477 | 5666 | 6373 | 10773 | 11992 |

Attention, all stock kernels, is half of the 1B and 3B prefill and 37 to 41 % of the 8B one.
Linear kernels in the model against the fresh roofs: 4w `t256x128k16g22s32` 5.74 to 6.21 TFLOP/s = 59 to 64 %
of the fp16 matrix roof (9.716); 4w `t128x128k32g42s32f32` (8B, K = 14336) 5.23 = 54 % of the fp16 -> fp32 roof
(9.722); 8da4w zpgtr 4.39 to 4.85 TOP/s = **22.5 to 24.9 %** of the int8 matrix roof (19.482). 8da4w linear
is 30 % slower than 4w linear in every model, on a device whose int8 matrix rate is twice its fp16 rate.

## Done so far

- Builds (cross image `localhost/et-jetson-cross:jp7.2.1`, GCC 13.3, shaderc v2026.1, 8 jobs, under the desktop
  build lock): `parent` = pristine `6a7cc8cc6`; `topic1` = `1d827a139` (dev zone only: `OrinSdpa.cpp`, the
  `orin-*` profiles). Provenance `.artifacts/orin-prefill-refine/build/<tag>.src.txt`.
- Shipped SPIR-V (`tools/shipped.py`, `build/topic1.shipped.txt`): all 53 shipped variants byte-identical
  between `parent` and `topic1`. Against the golden, 14 variants differ in BOTH builds (the cross image's glslc
  is not the one the goldens were made with; the e2e report documents the same for 11 of 48); none belongs to
  the Orin rows, whose variants all match the golden.
- Device capabilities (`results/orin/vk-caps.txt`): subgroup size 32 only (min = max = 32), shared memory
  49152 bytes, cooperative matrix fp16 16x16x16 / 16x8x16 / 16x8x8 with fp16 or fp32 result, int8 16x16x32 /
  16x8x32. The same shapes the 4070 Ti SDPA kernels use (16x16x16 fp16 -> fp32, subgroup 32).
- Fresh roofs, igpu-roofline `fast`, driver 595.78, run `raw/roof-2026-10-05-fast` (2026-10-05 02:52 to 03:33
  UTC, 2474 s, clocks not pinned: 612 MHz under load, 15 W mode; every roof confirmed with 3 repeats within
  1.4 %; report in `results/orin/roofline/2026-10-05-fast/`): matrix fp16 9.716 TFLOP/s, fp16 -> fp32 9.722,
  int8 19.482 TOP/s; fed from shared memory 9.511 / 8.771 / 17.818; DRAM read 62.1, write 58.0, copy 64.1 GB/s;
  texture2d read from DRAM 20.1 GB/s against 40.1 for texture3d and 62 for a buffer.
  The tool is the fleet copy already on the device (`~/.cache/igpu-roofline/fleet-quick-20260925`, runner
  `c7fba81beb1e`), run from a copy under `~/hmz-sarc-orin/roofline` with `tools/roof_fast.py`: that tree's
  controller without its clock pinning (the original pins the GPU to 1020 MHz with sudo, which is not allowed
  here). The report's own clock field says "unavailable"; the clock was sampled every 10 s beside it.
- First numbers of the parent control (still running): 1B 4w 890.4, 1B 8da4w 822.5, 3B 4w 360.4, 3B 8da4w
  320.2 tok/s (`cells.csv`: 890.8 / 822.8 / 360.4 / 320.3).

## Incidents

- I edited `tools/build-orin.sh` while its first invocation was running from the same file. I stopped that
  invocation before it reached the edited lines (its tree step ran on and completed), made the build resumable
  and added `tools/wsrun.sh`, which runs workstation jobs from a private copy of the tools.
- `chain2` was started while the parent control still had 20 minutes to go; its session would have given up
  after the 900 s lock wait. Killed before any run (`jobs/chain2.status`); restarted as `chain2b`, which waits
  for the control. Killing it by a name pattern also killed my own ssh shell twice; `tools/dkill.sh` now ends a
  job by its recorded session id.
- `nvidia-smi pmon -c 1` hangs on this device (my probe, killed). Not used by any tool.
- With the 8B model loaded the device has about 1.3 GB available and swap use rose from 107 to 155 MB during
  the parent control. Every session records memory and swap counters per run (`logs/<run>.mem`).

## Thresholds, fixed before any measurement

- Baseline: each of the six cells within 3 % of the Orin SARC median in
  `sarc-1.5-e2e-benchmark/results/cells.csv` (890.8 / 360.4 / 189.7 tok/s for 4w, 822.8 / 320.3 / 170.4 for
  8da4w). Otherwise stop and find out why.
- Noise: a difference inside +-2 % is not a gain. The A/A session reports the real floor.
- Normal clock: `calibrate_clock.py` on the baseline and A/A runs: one device-wide threshold,
  floor(0.97 x the lowest per-cell median of the per-run median devfreq clock). A timed run below it is invalid.
- Arithmetic changes (SDPA kernels): the owner's reference-error rule of 2026-10-04 as written (rms and maximum
  error against the fp32 CPU reference not larger than the parent's on every S = 2048 head configuration, all
  tiers 0 mismatches; gross divergence: mean KL <= 0.5 nat and top-1 differences <= one third of the prompts in
  every cell).
- Stop rule: two consecutive gated candidates each below +2 % geomean over their parent.

## What differs from the sibling (4070 Ti) tools

The measuring host is not the build host. `tools/common.sh` serves both sides; the device holds a copy of the
tools, the unmodified `sarc/tools/verify.sh` and the kit prompts under `~/hmz-sarc-orin/executorch/` (same
relative paths), the builds under `~/hmz-sarc-orin/build/<tag>/bundle/`.

| tool | change |
|---|---|
| `build-orin.sh`, `jetson-cross/` | replaces `build-both.sh`: `mktree.sh` tree + the cross recipe of the campaign that produced the Orin rows of `cells.csv` (`reference-tools/jetson-cross`, image `localhost/et-jetson-cross:jp7.2.1`), under the desktop build lock. One runner serves timed and traced runs (ETDump is linked, as in that campaign). Added to the recipe: `vk-caps.cpp` (capability query) |
| `deploy.sh`, `drun.sh`, `dstat.sh`, `pull.sh` | new: copy tools and builds to the device, start a tool there detached with a status file, show status, mirror the results to `.artifacts/orin-prefill-refine/device/` |
| `common.sh` | temperature from `/sys/class/thermal` (zone `gpu-thermal`), clock from devfreq `17000000.gpu`, load from the nvgpu node, power from ina3221 `VDD_IN`; `nvidia-smi` is not used (N/A on a Jetson, and `nvidia-smi pmon` hangs there). No per-process GPU client list exists without root: foreign jobs are found by name (known GPU programs, a running Actions job `Runner.Worker`). Cooling waits also end when the temperature has stopped falling |
| `e2e5.sh` | sampler 0.1 s from sysfs; flat model directory; memory and swap counters before and after every run (`logs/<run>.mem`) |
| `trace.sh` / `trace_analyze.sh` | the ETDump runs happen on the device, the analysis on the workstation |
| `gen_orin_sdpa.py`, `devzone.py` | the Orin SDPA base rows (`impl/sarc_dev/OrinSdpa.cpp`, the Xe2 mechanism, no hook) and `orin-*` profiles in marked blocks of `Overrides.cpp` |
| not carried over | `gen_4070ti_*.py`, the local hook patches, `Containerfile`, `podman-shim.sh` (they stay in the sibling's directory) |
