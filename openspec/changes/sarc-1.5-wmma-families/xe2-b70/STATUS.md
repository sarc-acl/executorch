# Xe2 B70 finite matrix status

2026-10-15 23:30 UTC. Actor: Claude (claude-opus-5-5). Protocol: owner decision 2026-10-15 23:30 UTC
(lean finish; deadline 2026-10-17 18:00 UTC). Earlier preparation and the earlier owner question about
revised inputs are superseded by that decision; their records stay below in git history and under the
artifact root.

**Running:** detached job `lean-01` (`tools/lean_queue.sh wmma-b70-topic-02`), status file
`/home/doremy/wmma-b70/.artifacts/jobs/lean-01.json`, progress in `.artifacts/queue.status`.

## Configuration

- One binary for both arms and every profile: exported-commit build `wmma-b70-topic-02` (commit `2fcc1c7e9`,
  base `6178971845` + the dev-only `b70-wmma` selector in `impl/sarc_dev/Overrides.cpp`; spirv golden PASS,
  53 shipped variants). No rebuild.
- Control = the device's final profile `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=b70-fused1-order-i-n26-w16m`.
  Candidates add only `ET_VK_SARC_B70_WMMA_PROFILE=<name>`; attention (fused node, softmax, QK/AV), decode and
  fp linear are the control's in both arms.
- Inputs: the ORIGINAL exported models and tokenizers, hashes in `.artifacts/preflight/original-input-admission.json`
  (8B hashes equal the RoPE audit's source hashes). Revised RoPE inputs are kept, not used.
- **Caveat (stated once, owner decision 2026-10-15 23:30 UTC):** ANV reports a 2048 image extent; the RoPE tables
  of all six models, the 8B activations and verify.sh's 8B production-diff fixtures use texture3d images over that
  extent. Pre-existing, documented, identical in both arms; every earlier Intel campaign ran these files and passed
  the gate (b70-fused-port GATE_PASS, xe2-prefill-refine pdiff 12/12). Not a blocker. The marker
  `RESOURCE_LIMIT_BLOCKED` was moved to `.artifacts/superseded/owner-2026-10-15T2330-original-inputs/`.
- Protocol: control `verify.sh --pdiff` once, compared line by line (rates removed) with the historical final gate
  of the final profile (`order-2/b70` stage `g1`); one seven-repeat A/A session (topic-02 vs topic-02, control in
  both) that calibrates the clock floor. Per candidate: `verify.sh --pdiff` (production-diff failure rejects
  without timing), seven valid interleaved runs per arm and cell against the same-session control, next tokens on
  the timed, real-text and unaligned prompts, dispatched names from ETDump. SDPA tiers not run (attention
  identical in both arms). Thresholds as preregistered in `proposal.md`, repeats fixed at 7 by the owner.
- Order: profiles that appear in B580's final profile first (dq-n13, q4-table, dq-i, dq-table), then B70's own
  picks (q4-k16, q4-k32, dq-n26).

## Matrix

Cand/control > 1 means the candidate is faster. Limits: six-cell geomean slowdown <= 3 %, no cell > 8 % slower
(slowdown = control tok/s / candidate tok/s - 1). Small evidence: `results/matrix-summary.csv`,
`results/matrix-dispatch.csv`; raw rows: `.artifacts/runs.csv` and `.artifacts/stage/<session>/raw/runs.csv`.

<!-- matrix begin -->
| Session | Candidate linear kernels (dispatched, cand arm) | Correctness | Cand/control tok/s, cells 1B 4w, 1B 8da4w, 3B 4w, 3B 8da4w, 8B 4w, 8B 8da4w | Six-cell geomean cand/control | Within 3 %/8 % |
|---|---|---|---|---:|---|
| s1-aa | - | CONTROL_OK | 1.000 / 1.000 / 1.004 / 0.995 / 1.000 / 1.000 | 0.9998 | True |
| b70-wmma-dq-n13 | - | PASS_SCREEN (pdiff 12/12, next tokens SAME) | 1.000 / 1.000 / 1.000 / 0.995 / 1.002 / 0.998 | 0.9991 | True |
| b70-wmma-q4-table | - | PASS_SCREEN (pdiff 12/12, next tokens SAME) | 0.980 / 1.000 / 0.977 / 1.000 / 0.977 / 1.005 | 0.9897 | True |
| b70-wmma-dq-i | - | PASS_SCREEN (pdiff 12/12, next tokens SAME) | 1.000 / 1.000 / 1.000 / 0.995 / 1.003 / 0.959 | 0.9928 | True |
| b70-wmma-dq-table | - | NOT_STARTED | - | - | - |
| b70-wmma-q4-k16 | - | NOT_STARTED | - | - | - |
| b70-wmma-q4-k32 | - | NOT_STARTED | - | - | - |
| b70-wmma-dq-n26 | - | NOT_STARTED | - | - | - |
<!-- matrix end -->

No family-wide acceptance is claimed from this member's results.

## Question for the coordinator

(none)

## Per-file reviewed input hashes

| Input file | SHA256 |
|---|---|
| LEGALITY.md | 0d44a9fc58e27a5e912486baec81f2d6618195027707ad4e42b6d21c40f6bbda |
| PROFILES.md | f266487d038b46f195a5f2f42c555fd05299d479289b609148ab9c28f4a1cf4a |
| README.md | c7d2d969372a480904005d26acf58f4e6f5d1be281ec9e70c4f6e87e63c7b224 |
| REVIEW.md | a8fd4b0c68ad002a20f1b8490b86cd93e62f44e1e1f2743a8ffe2c5ebd16895f |
| STATUS.md | f642fe626ed6a8ec1b3736fc78610e3aaa73bb0023f9c37a77677d991b187b2c |
| VALIDATION.md | 0cff1dda430d49f94ab70103c3a046a21a53ae8528f36ad0705558b1eb423611 |
| generate.py | 11e6bed5304b053b6603e067dfdabb765063fdb75082de65e5be4c6ac874f97c |
| inventory.csv | f89656186b8fca434270f303c965789c515a8fe05a358910d97c7c774ed2b5a7 |
| legality-matrix.csv | b3945596c0e7ecf92538f8ea607cce755dd3591308b4add0aad0ad7a5573808c |
| manifests/nvidia-ampere-ada.json | ef5d13471421898c8515e9a8356caff64a20aa97432175cbcab7bf3a5cd748b1 |
| manifests/rdna3-radv.json | edcfa39ea7d8fac9c907a6a901754b96aa2d2894302b9c8ad623b90c2780d923 |
| manifests/xclipse.json | 2fe20db84ffba914961bda2c13f3c2a51b6857de7dbf8925f9aac2c84d0e089f |
| manifests/xe2.json | 5904caa4c1977f256d58b4c1c111e7fdcb2e54bc312676b40c90da6bcf048306 |
| packets/7900xtx-radv.md | 7f8483cc3674fd7b981131ba8332f648ddcf44ee9ef7a99e8f63328ebb36f0c5 |
| packets/COMMON.md | e6e321e80f2522ebfc68947f55a0b15f3e64404d3d528746c51ad9e5deb51cc7 |
| packets/m51-private.md | ec3f2280f5f432a5a2b55971ce27ac5ebd852782d1f0e159e0ec8ae456ad1ade |
| packets/rx7600.md | 57ea9506763c594721c1b7ea09b445cbbf130526ca9eaa0b900ab45e7ceb4789 |
| source-hashes.csv | c4a7380cf0b810473b63223a245f248528b81c4adbc6fc1fd455b8d7a6762ace |
