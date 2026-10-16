# Xe2 B580 finite linear matrix: status

Updated 2026-10-16T05:00Z. Actor: Claude (claude-opus-5-5), under the owner's lean-finish decision of
2026-10-15 23:30 UTC (deadline 2026-10-17 18:00 UTC). Earlier recovery history: git log of this file.

## Inputs

INPUT.md inventory `f393c3ff7ff4fc41e4651a5109a8509097c501f6`; archive SHA-256
`65d1e3ca0eb7e9dffa8e9fbbb1a100bf2cf609868720b9053de67ff8bf5247d8` and all 18 per-file hashes re-checked
2026-10-15T23:25Z (Xe2 manifest `5904caa4...6bcf048306`). Original model files (hashes in
`.artifacts/preflight/model-hashes.json`); the revised RoPE inputs are kept and not used.

## Intel image-extent caveat (stated once)

ANV reports a 2048 limit per texture3d axis; the RoPE tables in all six models (height 5120), the 8B
activations (width 3584) and verify.sh's 8B production-diff fixtures exceed it. By the owner's decision this is
a pre-existing, documented caveat, identical in both arms; every earlier Intel campaign ran these exact files
and passed its gate (b580-fused s6-final GATE_PASS). It is not a blocker. BLOCKER.md is kept as the record.

## Protocol in force (owner 2026-10-15 23:30 UTC, replaces earlier conflicting rules)

One binary `wmma-v1` (branch commit, all profiles) for control and candidates. Control
`b580-fused1-order-p13`. A/A `s1-aa`: pristine `parent-v2` vs `wmma-v1`, control profile in both arms,
calibrates clock floor (97 % of the lowest valid per-run median) and idle temperature. Control check: unmodified
`verify.sh --pdiff` on `wmma-v1` + control, compared with b580-fused s6-final / order2 g2. Per candidate:
unmodified `verify.sh --pdiff` first (production-diff failure rejects, untimed); then six cells interleaved,
7 valid runs per arm (up to 5 extra pairs), next token on three prompts (a DIFFER alone = CORRECTNESS_PENDING,
still timed). Validity rules applied: rc 0, 2048/0 tokens, >= 5 clock samples, median clock >= 2651 MHz, no
thermal reason, no foreign GPU process, and a foreign engine share in the prefill window at or below
BUSYMAX = 2.97 % (e2e5.sh calibration capped at 5 %, coordinator ruling 2026-10-16 01:15 UTC); a row above it is
invalid (foreign_busy) and kept. Timed sessions wait until the foreign share is <= BUSYMAX over 60 s
(coordinator note 02:15 UTC); load average is recorded per row (`load1`). SDPA tiers not
run (attention identical in both arms). Order: B70-final-profile candidates first.

Queue `tools/queue.sh`, state `.artifacts/queue/wmma-v1/job-state.json`.

**Matrix complete: all six legal candidates measured in all six cells.** Ratios are candidate / same-session
control throughput (valid-run medians, 7 valid runs per arm and cell, BUSYMAX 2.97 %). Bounds: geomean >= 0.97
and every cell >= 0.92. Noise band +-2 %.

| session | candidate | linear names dispatched (verify linear suites; texture3d unless noted) | correctness | 1B 4w | 1B 8da4w | 3B 4w | 3B 8da4w | 8B 4w | 8B 8da4w | geomean | bounds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| s1c-aa | A/A: pristine parent-v2 vs wmma-v1, control both | control (below) | gate s1-aa: pdiff 12/12, = s6-final; next token SAME 18/18 | 1.0000 | 1.0088 | 0.9974 | 1.0000 | 1.0000 | 1.0000 | 1.0010 | in |
| s2 (+s2b 8B 4w) | b70-linear | 4w flib x11 + flw x1; 8da4w xe2bto i x10 + n26 x2 | pdiff 12/12; next token SAME 18/18 | 1.0000 | 0.9914 | 1.0107 | 0.9768 | 1.0201 | 0.9954 | 0.9990 | in |
| s3 | dq-n26 | 4w fli; 8da4w i x10 + n26 x2 | pdiff 12/12; SAME 18/18 | 1.0000 | 1.0000 | 1.0000 | 0.9764 | 1.0012 | 0.9952 | 0.9954 | in |
| s4 | q4-narrow | 4w flw x1 + fli x11; 8da4w control | pdiff 12/12; SAME 18/18 | 1.0000 | 1.0000 | 1.0000 | 0.9966 | 1.0023 | 1.0000 | 0.9998 | in |
| s5 | q4-band | 4w flib x12; 8da4w control | pdiff 12/12; SAME 18/18 | 1.0000 | 1.0000 | 1.0135 | 1.0000 | 1.0201 | 1.0000 | 1.0056 | in |
| s6 | dq-table | 4w fli; 8da4w table t256x64k32g48s16m8 x12 | pdiff 12/12; SAME 18/18 | 1.0000 | 0.8692 | 1.0000 | 0.8151 | 1.0000 | 0.8013 | 0.9100 | **out** (three cells < 0.92) |
| s7 | dq-i | 4w fli; 8da4w xe2bto i x12 | pdiff 12/12; SAME 18/18 | 1.0073 | 1.0000 | 1.0000 | 0.9798 | 1.0012 | 0.9628 | 0.9917 | in |

Control `b580-fused1-order-p13`: 4w `t128x128k16g44s16m8fli` (buffer and texture3d); 8da4w xe2bto `i` x8 and
`n13` x4 per storage. The 4w buffer variant stays `fli` in every arm (buffer projections are M=1, outside the
prefill matrix). Correctness: unmodified `verify.sh --pdiff` per candidate on wmma-v1; with rates and linear
lines removed every candidate's verify.out equals the control's (correctness rc=1 / linear rc=1 are the
pre-existing statuses of s6-final); next token parent vs candidate SAME on prompt_2048, prompt_check and r1304
in all six cells of every session. No candidate is CORRECTNESS_PENDING.

Reading (this member only; no family-wide claim): nothing beats the B580 control outside the noise band except
8B 4w with `flib` (+2.0 % in s2b and s5, bootstrap 95 % [1.017, 1.021]); q4-band is the only candidate whose
geomean exceeds 1 (1.0056). B70's n26 rule costs 3B 8da4w 2.3 to 2.4 % (s2, s3); `i` without n13 costs 8B 8da4w
3.7 %; the release-table 8da4w kernel is 13 to 20 % slower and fails the cell bound. Every other candidate is
within the member bounds. The 1 ms runner timer quantises 1B rates (one step about 1 %), hence many 1.0000.
Absolute control medians differ between sessions by up to 4 % (8B 8da4w 3170 in s2, under desktop activity and
load ~19, vs 3298 in s4-s6); ratios use only same-session controls.

Evidence: `results/matrix-summary.csv` (medians, bootstrap, spreads, valid/invalid counts), `results/runs.csv`
(960 rows of every session incl. superseded s1-aa, `used` marks the 588 analysed rows, `notes` carries the caveat
and BUSYMAX; host/device identity only in `.artifacts/results/runs.csv`), `results/cell-coverage.csv`, per-stage
`analysis.json`, `verify.out`, `gate.done`, `raw/nexttoken.csv` under `.artifacts/stage/`. Recompute with
`analyze_session.py 2.97 <raw dirs>` (s2: `s2-b70-linear/raw s2b-b70-linear/raw-8b-4w`) and `combine_runs.py`.
`matrix.json`'s `whole_model_measurement: BLOCKED_*` fields are the pre-decision record and are superseded here.

`q4-table` and `dq-n13` are identity aliases of the control (CPU selector audit), covered by s1-aa.

## Control check (s1-aa gate, 2026-10-15 23:41-23:52Z)

`wmma-v1` built from `921e8cb355` (BUILD_BOTH_OK, shipped SPIR-V golden PASS, 53 variants). Unmodified
`verify.sh --pdiff` with the control on `wmma-v1`: VERIFY_DONE rc=0, production diff 12/12 ALL PASSED,
default vs tiled SAME on check and unaligned prompts, decode rc=0 31 tokens. With rates removed it equals
b580-fused s6-final line by line except the 8da4w dispatch list, which is expected: s6-final ran `b580-fused1`
(xe2bt), the control `b580-fused1-order-p13` dispatches xe2bto `i` (8 per storage) and `n13` (4 per storage), as
in the manifest. `correctness rc=1` and `linear rc=1` are the same pre-existing statuses as s6-final.
The gate wrapper recorded foreign desktop engine time during the 11-minute gate (`GATE_RECORDED rc=1`);
this does not affect correctness results and the gate timings are not used.

## A/A s1-aa (2026-10-16 00:05-00:41Z): INVALID, preserved

168 rows, 144 timed prefill runs (12 per arm and cell, 7 + 5 extra pairs), 0 valid: 140 `foreign_busy`,
4 `busy_unsampled`. Foreign engine share in the prefill window 0.52 to 2.84 % (median 0.87 %), never zero;
the busiest foreign client in 130 runs was Firefox (pid 1699641), 10 runs ghostty, while the desktop reported
IdleHint=yes. Clocks 2850 MHz, no thermal reason, rc 0. Next token parent vs wmma-v1 SAME in all six cells on
all three prompts (check/unaligned runs are not subject to the busy rule). No calibration was written; these
rows are not timing evidence. Raw: `.artifacts/stage/s1-aa/raw/`.

## A/A s1c-aa (E2E5_OK, finished 01:31Z)

Calibration: idle 40.0 C, clock floor 2651 MHz, BUSYMAX 2.97 % (min(5, 2 x p90)). 7/7 valid runs per arm and
cell, no invalid rows; per-cell spreads 1.3 to 3.5 %; bootstrap 95 % intervals all contain 1.0. Analysis:
`.artifacts/stage/s1c-aa/analysis.json`. Noise band stays the pre-registered +-2 %.

## Gates re-recorded (coordinator note 2026-10-16 02:15 UTC)

Foreign activity is recorded, not asserted, in correctness gates. s1-aa: 0.00 % foreign cycles during the
gate; s2-b70-linear: 2.25 %. Both VERIFY_DONE rc=0 with 12/12 production diff -> GATE_RECORDED rc=0; previous
files kept as `gate.done.foreign-assert`, evidence in `gate.reeval.txt`.

## Invalid and repaired runs

s2-b70-linear 8B 4w: desktop activity of 6 to 24 % between 02:38 and 02:48Z left 1 valid run per arm after 5
extra pairs (rows kept, foreign_busy); the queue re-timed only that cell in s2b-b70-linear (same binary and
environments, E2E5_OK, next token SAME). Other timed prefill rows: s3 to s7 have none invalid; s2 outside
8B 4w has 1B 4w parent 1 (busy_unsampled), 3B 8da4w cand 1, 8B 8da4w parent 1 and cand 2 (foreign_busy); all
kept, each cell still reached 7 valid runs per arm. The 32 `clock_low` rows are check/unaligned next-token runs,
not timed prefill rows; the clock floor does not apply to token comparisons, which used them.

## Baseline against the historical campaign

A/A control arm (wmma-v1, `b580-fused1-order-p13`, s1c-aa) against b580-fused order2 g2 P (the same profile,
2026-10-12, `results/b580/order2/a5.md` of that campaign): 1B 4w -1.45 %, 1B 8da4w -1.75 %, 3B 4w -0.79 %,
3B 8da4w -2.03 %, 8B 4w -0.92 %, 8B 8da4w -1.74 %; all within the 3 % baseline tolerance. The s2 control on 8B
8da4w (3170.3) is -3.9 % under desktop activity and load ~19, explained above; ratios use same-session controls.

## Now

Queue completed 04:24:53Z. No job running. Matrix ready for the single review.

## Question for the coordinator

None.
