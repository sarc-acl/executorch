# Xe2 B580 finite linear matrix: status

Updated 2026-10-15T23:30Z. Actor: Claude (claude-opus-5-5), under the owner's lean-finish decision of
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
| s1-aa | A/A (control vs pristine) | none | pending | | | | | | | |
| s2 | b70-linear | B70 final linear rules (n26 8da4w; flw N<=512, flib 4w) | pending | | | | | | | |
| s3 | dq-n26 | 8da4w n26 at N*K>=33554432, else i | pending | | | | | | | |
| s4 | q4-narrow | 4w flw at N<=512 | pending | | | | | | | |
| s5 | q4-band | 4w flib where it fits | pending | | | | | | | |
| s6 | dq-table | 8da4w table choice | pending | | | | | | | |
| s7 | dq-i | 8da4w ident base everywhere | pending | | | | | | | |

`q4-table` and `dq-n13` are identity aliases of the control (CPU selector audit), covered by s1-aa.

## Now

Building `wmma-v1`, then the queue runs unattended (timed units wait for an idle desktop).

## Question for the coordinator

None.
