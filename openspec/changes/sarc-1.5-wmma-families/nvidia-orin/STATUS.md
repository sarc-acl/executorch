# NVIDIA Orin WMMA matrix status

Updated: 2026-10-16 08:22 UTC. Actor: Claude (claude-opus-5-5), lean protocol of the owner decision of
2026-10-15 23:30 UTC. Deadline 2026-10-17 18:00 UTC.

## Now running: family finalists (coordinator decision 2026-10-16 08:20 UTC)

Detached chain `orin-wmma-fin-20261016-1`, state `.artifacts/jobs/orin-wmma-fin-20261016-1/state.json`, launched
08:20 UTC. It runs these steps in order:
1. Cross-build `orin-wmma-fin-20261016-1` of `b28066718b` under the desktop-build lock, with the golden check.
2. On the Orin, on that one binary:
   - control `verify.sh --pdiff` and a six-cell A/A with a fresh clock floor (new binary);
   - then for each finalist: `verify.sh --pdiff`, then screen, then the SDPA tiers (all, extended, full x12;
     peaked, fused x3; `tools/sdpa_check.py`), then a six-cell interleaved session against `orin-fused1`;
   - a production-diff rejection is not timed (lean rule 3).
- Profiles: `orin-fused-nv-family-a` (`kOrinWmmaFamilyA`) and `orin-fused-nv-family-b` (`kOrinRefine5`)
  (`results/finalist-profiles.json`).
  - CPU selector dump (`.artifacts/selector-tests/finalists/`): nv-family-b is byte-identical to the control. So I
    am running its session as well, for the next-token evidence on all three prompts, although the decision lets
    me skip it.
  - nv-family-a differs only in 4w: bt at K = 14336 (M = 256, 1972 unaligned, 2048) and at the M = 256,
    N % 2048 != 0 shapes. At unaligned M = 128, K = 14336, bt does not fit and the table's fp32 tile is used.
- Known risk for nv-family-a: the Orin campaign recorded that a forced fp16 4w tile at K = 14336 fails the
  production diff (`sarc-1.5-orin-prefill-refine/STATUS.md`, 4w screen 2). That
  is the reason for the bx rule. If the screen rejects it, it is recorded as rejected and not timed.

## Matrix state

The finite legal matrix is complete on this member and ready for its single review.
Queue `orin-wmma-m1-20261015` ran from 2026-10-15 23:31 to 2026-10-16 07:44 UTC with status `completed`.
All steps returned rc 0, all with complete instrumentation (37 wrapped calls per verify, childless samplers).
No HOLD, GPU_GONE, ABORTED or foreign process occurred.

- Job state: `.artifacts/jobs/orin-wmma-m1-20261015/state.json`.
- Raw evidence: `.artifacts/target-evidence/orin-wmma-m1-20261015/`. It holds `steps/verify-*/`
  (verify.out, verify/, invocations.jsonl, screen.json, content-check.txt), `steps/session-*/raw/`
  (runs.csv, logs/*.log|.clk|.mem|.others, nexttoken.csv, env.txt) and `floor.json`.
- Recomputed results: `results/matrix-m1/`. They come from `tools/analyze_matrix.py`, which recomputes every
  timed row from its log and clock samples with `runrow.py` against the floor; there are 0 mismatches.

## Matrix (one binary, one session per candidate, control = `orin-fused1` in the same session)

Ratio = control median tok/s / candidate median tok/s, from the first 7 valid runs per arm. A ratio above 1
means the candidate is slower. Family limits: geomean <= 1.03 and every cell <= 1.08.

| candidate | origin | linear kernel (scheme, storage) | correctness screen | 1B 4w | 1B 8da4w | 3B 4w | 3B 8da4w | 8B 4w | 8B 8da4w | geomean | limits |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A/A (control vs control) | | | next token 24/24 SAME | 0.9992 | 1.0007 | 0.9997 | 1.0000 | 1.0006 | 0.9997 | 1.0000 | |
| -01 | 4070 Ti R30 | q4gsw t256x128k16g42s32ga (4w, tex3d) | PASS | 1.1977 | 1.0007 | 1.2313 | 1.0000 | 1.2472 | 1.0000 | 1.1071 | FAIL (geomean, 3 cells) |
| -02 | 4070 Ti R31 | q4gsw t128x128k16g24s32ga (4w, tex3d) | PASS | 1.5085 | 1.0000 | 1.6027 | 1.0000 | 1.6721 | 1.0003 | 1.2622 | FAIL (geomean, 3 cells) |
| -03 | 4070 Ti R32 | q4gsw t128x256k16g42s32ga (4w, buf/tex2d) | PASS | not timed (no timed cell uses buffer IO) | | | | | | | |
| -04 | 4070 Ti R33 | q4gsw t128x128k16g42s32ga (4w, buf/tex2d) | PASS | not timed (buffer IO) | | | | | | | |
| -05 | 4070 Ti R33 | q4gsw t128x128k16g42s32ga (4w, buf/buf) | PASS, but no verify case dispatches it (0 changed cases) | not timed (buffer IO) | | | | | | | |
| -06 | 4070 Ti R34 / Orin R39 | dq8ca zpgtr t128x128k64g44s32mk32ra (8da4w, tex3d) | PASS | 1.0000 | 1.0661 | 1.0013 | 1.1176 | 0.9994 | 1.1011 | 1.0464 | FAIL (geomean, 2 cells) |
| -07 | 4070 Ti R34 | dq8ca zpgtr t128x128k64g44s32mk32ra (8da4w, buf/tex2d) | PASS | not timed (buffer IO) | | | | | | | |
| -08 | Orin R35 | q4gsw bx t128x128k32g42s32f32c (4w, tex3d) | PASS | 1.0024 | 1.0000 | 1.0003 | 1.0000 | 0.9998 | 0.9999 | 1.0004 | within |
| -09 | Orin R36 | q4gsw orin t256x128k16g42s32bt (4w, tex3d) | PASS | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0001 | 1.0000 | within |
| -10 | Orin R37 | dq8ca zpgtr orin_bf t128x128k64g24s32mk32ra (8da4w, tex3d) | PASS | 1.0000 | 1.0007 | 1.0000 | 1.0000 | 1.0000 | 1.0001 | 1.0001 | within |
| -11 | Orin R38 | q4gsw t128x128k16g22s32 (4w, tex3d) | PASS | 1.0420 | 1.0000 | 1.0470 | 1.0000 | 1.0398 | 1.0000 | 1.0212 | within |

Correctness:
- Production diff: every candidate passes the registered rule. No shape threw or failed where the candidate
  changed the kernel.
- The buffer candidates -03, -04 and -07 turn the control's buffer fallbacks (upstream tiled kernels, partly
  threw) into coopmat:PASSED shapes.
- Next tokens: every verify comparison is SAME (6 timed prompts, check and unaligned for 1B), and so is the
  1B decode text. Every timed-session row is SAME as well (24 rows per session, 192 in total).
- No candidate is CORRECTNESS_PENDING.

Dispatched linear kernels (`results/matrix-m1/dispatch.csv`, 48 microbench cases per candidate):
- Control uses Orin bt for 11 of the 12 4w texture3d cases and bx for 8B w2. 8da4w texture3d uses Orin bf
  everywhere. Buffer IO falls back to the upstream tiled kernels.
- -01 changes 11 of the 12 4w texture3d cases; 1B wk/wv at N=512 stays on bt. -02 changes 12 and -11 changes 11.
- -06 changes all 12 8da4w texture3d cases.
- -08, -09 and -10 change no case. These are the kernels the control already dispatches, and their sessions read
  1.000 like the A/A.
- Attention environment is identical in both arms of every session: `ET_VK_SARC_ORIN_SDPA_FUSED` pair, with
  softmax `orin_g64` via the profile prefix.

Uncertainty:
- A/A per-cell deviation is at most 0.07 %.
- The 7-run spread (max - min over median) is at most 0.67 % in any arm.
- The control medians of all 8 sessions vary by at most 0.32 % per cell.
- Clock: all 672 timed runs at 612 MHz. Floor 593.64 MHz exact, 594 as e2e5.sh uses it (`floor.json`).
- 0 invalid timed runs, so no extra pairs were needed.

Memory: 320 8B runs; MemAvailable never below 5704 MB. Swap-out occurred during 72 runs, at most 3755 pages;
it is recorded, not judged (historical protocol).

## Control check and baseline

- Unmodified `verify.sh --pdiff` on the control (`steps/verify-control/verify.out`) equals the Orin campaign's
  final gate `sarc-1.5-orin-fused-port/results/orin/sessions/s5-c1/verify.out` line by line with rates removed:
  34 lines, 0 differing (`results/matrix-m1/control-verify-vs-s5-c1.txt`). The content check is ACCEPT
  (`content-check.txt`, rc 0).
- Baseline: the control medians in every session are within -0.40 % to +0.15 % of the published final-stack
  medians (`s5-c1`: 1656.96 / 1520.42 / 659.58 / 595.35 / 305.49 / 277.24). That is inside 3 %, so there is
  nothing to explain.
- The memory-model runtime change (`cbb7df9b6a`) shows no measurable effect against the historical numbers.

## Reading (this member only; no family-wide claim)

- Orin's tuned best is its own set (-08 bx, -09 bt, -10 bf), which the control already dispatches.
- -11, the 128-row g22 fallback, would cost Orin 2.1 % geomean (max 4.7 %), which is within limits.
- The 4070 Ti's three texture3d linear kernels all exceed the limits on Orin:
  - -01 costs 10.7 % geomean and up to 24.7 % (4w).
  - -02 costs 26.2 % and up to 67.2 % (4w).
  - -06 costs 4.6 % and up to 11.8 % (8da4w).
- So a set made only of the 4070 Ti's linear kernels does not meet the margin on Orin. Whether Orin's kernels
  meet it on the 4070 Ti is that member's evidence, not this run's.
- The buffer entries cannot be timed here, and -05 has no correctness coverage on this device, because no verify
  case uses buffer weights. That evidence is missing; it is not passed.

## Provenance

Inputs: INPUT.md snapshot `f393c3ff7ff4fc41e4651a5109a8509097c501f6`; archive SHA256
`65d1e3ca0eb7e9dffa8e9fbbb1a100bf2cf609868720b9053de67ff8bf5247d8` and NVIDIA manifest SHA256
`ef5d13471421898c8515e9a8356caff64a20aa97432175cbcab7bf3a5cd748b1`, reconfirmed 2026-10-15 23:25 UTC.
Original base `617897184529ff180af491ee0d5d7de778cbb6c6`. Both arms ran on changed base
`cbb7df9b6a7a605d28192fbc2a8b62c6f3611d9c` (memory-model runtime change, owner decision 2026-10-15 03:57 UTC).
The build is `orin-wmma-mm-20261015-1`, golden PASS on 53 shipped variants. Binary hashes and the build-manifest
hash are in every row of `results/matrix-m1/runs.csv`. Payload hashes are in the evidence's
`source-provenance.json`, with orchestration commit `31225c6697`. Profiles are `orin-fused-wmma-NN`
(`results/candidate-profiles.json`). Rules were registered before data in `proposal.md`, section "Lean matrix
protocol". The earlier snapshot `orin-wmma-s0-20261015-1` failed on a tool bug (fixed in `aff996d876`); it is kept
as a record and is not evidence.

## Question for the coordinator

(none)
