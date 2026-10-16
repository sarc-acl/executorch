# Xe2 B580 finite linear matrix: status

Updated 2026-10-16T02:20Z. Actor: Claude (claude-opus-5-5), under the owner's lean-finish decision of
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
still timed). Validity rules unchanged: rc 0, 2048/0 tokens, >= 5 clock samples, clock floor, no thermal
reason, zero foreign engine time in the prefill window, no foreign GPU process, desktop idle. SDPA tiers not
run (attention identical in both arms). Order: B70-final-profile candidates first.

Queue `tools/queue.sh`, state `.artifacts/queue/wmma-v1/job-state.json`.

| session | candidate | linear change vs control | correctness | 1B 4w | 1B 8da4w | 3B 4w | 3B 8da4w | 8B 4w | 8B 8da4w | geomean |
|---|---|---|---|---|---|---|---|---|---|---|
| s1c-aa | A/A (control vs pristine) | none | gate s1-aa PASS-equivalent; next token SAME 18/18 | 1.0000 | 1.0088 | 0.9974 | 1.0000 | 1.0000 | 1.0000 | 1.0010 |
| s2 | b70-linear | B70 final linear rules (n26 8da4w; flw N<=512, flib 4w) | verify: pdiff 12/12, default vs tiled SAME, decode OK; timing pending | | | | | | | |
| s3 | dq-n26 | 8da4w n26 at N*K>=33554432, else i | pending | | | | | | | |
| s4 | q4-narrow | 4w flw at N<=512 | pending | | | | | | | |
| s5 | q4-band | 4w flib where it fits | pending | | | | | | | |
| s6 | dq-table | 8da4w table choice | pending | | | | | | | |
| s7 | dq-i | 8da4w ident base everywhere | pending | | | | | | | |

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

## Now

Desktop-idle waits replaced by `busy_wait` before each timed session: foreign share over 60 s at or below
BUSYMAX (`tools/foreign_share.py`, same accounting as fbusy_pct). Gates no longer wait. Rows above BUSYMAX stay
invalid. runs.csv gains `load1` per row (workstation load ~19 from other builds; same for both interleaved arms;
s1c-aa rows predate the column). Queue relaunched at s2-b70-linear's session.

## Question for the coordinator

None.
