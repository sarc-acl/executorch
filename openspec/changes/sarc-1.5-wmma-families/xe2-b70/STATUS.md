# Xe2 B70 finite matrix status

2026-10-15 23:30 UTC. Actor: Claude (claude-opus-5-5). Protocol: owner decision 2026-10-15 23:30 UTC
(lean finish; deadline 2026-10-17 18:00 UTC). Earlier preparation and the earlier owner question about
revised inputs are superseded by that decision; their records stay below in git history and under the
artifact root.

**Matrix complete** (2026-10-16 02:52 UTC); nothing is running. Jobs `lean-01` (matrix, 23:29 to 02:31 UTC) and
`redispatch-01` (dispatch names, 02:33 to 02:52 UTC) both completed with exit code 0. Waiting for the single review of the finished matrix.

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
| b70-wmma-dq-n13 | 4w: sarc_linear_q4gsw_coopmat_sweep_t128x128k16g82s16m8flib, sarc_linear_q4gsw_coopmat_sweep_t128x128k32g84s16m8flw; 8da4w: sarc_dev_linear_dq8ca_coopmat_zpg_xe2bto_t128x128k64g84s16m8i, sarc_dev_linear_dq8ca_coopmat_zpg_xe2bto_t128x128k64g84s16m8n13 | PASS_SCREEN (pdiff 12/12, next tokens SAME) | 1.000 / 1.000 / 1.000 / 0.995 / 1.002 / 0.998 | 0.9991 | True |
| b70-wmma-q4-table | 4w: sarc_linear_q4gsw_coopmat_t128x128k16g44s16m8fli; 8da4w: sarc_dev_linear_dq8ca_coopmat_zpg_xe2bto_t128x128k64g84s16m8i, sarc_dev_linear_dq8ca_coopmat_zpg_xe2bto_t128x128k64g84s16m8n26 | PASS_SCREEN (pdiff 12/12, next tokens SAME) | 0.980 / 1.000 / 0.977 / 1.000 / 0.977 / 1.005 | 0.9897 | True |
| b70-wmma-dq-i | 4w: sarc_linear_q4gsw_coopmat_sweep_t128x128k16g82s16m8flib, sarc_linear_q4gsw_coopmat_sweep_t128x128k32g84s16m8flw; 8da4w: sarc_dev_linear_dq8ca_coopmat_zpg_xe2bto_t128x128k64g84s16m8i | PASS_SCREEN (pdiff 12/12, next tokens SAME) | 1.000 / 1.000 / 1.000 / 0.995 / 1.003 / 0.959 | 0.9928 | True |
| b70-wmma-dq-table | 4w: sarc_linear_q4gsw_coopmat_sweep_t128x128k16g82s16m8flib, sarc_linear_q4gsw_coopmat_sweep_t128x128k32g84s16m8flw; 8da4w: sarc_linear_dq8ca_coopmat_zpg_t256x64k32g48s16m8 | PASS_SCREEN (pdiff 12/12, next tokens SAME) | 1.010 / 0.883 / 1.000 / 0.839 / 1.000 / 0.804 | 0.9188 | False |
| b70-wmma-q4-k16 | 4w: sarc_linear_q4gsw_coopmat_sweep_t128x128k16g82s16m8flib; 8da4w: sarc_dev_linear_dq8ca_coopmat_zpg_xe2bto_t128x128k64g84s16m8i, sarc_dev_linear_dq8ca_coopmat_zpg_xe2bto_t128x128k64g84s16m8n26 | PASS_SCREEN (pdiff 12/12, next tokens SAME) | 1.000 / 1.000 / 1.000 / 1.000 / 1.002 / 1.000 | 1.0003 | True |
| b70-wmma-q4-k32 | 4w: sarc_linear_q4gsw_coopmat_sweep_t128x128k16g82s16m8flib, sarc_linear_q4gsw_coopmat_sweep_t128x128k32g84s16m8flw; 8da4w: sarc_dev_linear_dq8ca_coopmat_zpg_xe2bto_t128x128k64g84s16m8i, sarc_dev_linear_dq8ca_coopmat_zpg_xe2bto_t128x128k64g84s16m8n26 | PASS_SCREEN (pdiff 12/12, next tokens SAME) | 1.000 / 1.000 / 1.000 / 1.000 / 1.002 / 1.000 | 1.0003 | True |
| b70-wmma-dq-n26 | 4w: sarc_linear_q4gsw_coopmat_sweep_t128x128k16g82s16m8flib, sarc_linear_q4gsw_coopmat_sweep_t128x128k32g84s16m8flw; 8da4w: sarc_dev_linear_dq8ca_coopmat_zpg_xe2bto_t128x128k64g84s16m8i, sarc_dev_linear_dq8ca_coopmat_zpg_xe2bto_t128x128k64g84s16m8n26 | PASS_SCREEN (pdiff 12/12, next tokens SAME) | 1.010 / 1.000 / 1.008 / 0.990 / 1.000 / 1.005 | 1.0021 | True |
<!-- matrix end -->

No family-wide acceptance is claimed from this member's results.

## Results summary (B70 member only)

- Control check: control `verify.sh --pdiff` on topic-02 is line-identical to the historical final gate `g1`, rates
  removed (`.artifacts/stage/s1-aa/verify-vs-historical.txt`, VERIFY_SAME, 34 lines). The A/A next tokens are SAME in
  all six cells on all three prompts. The calibrated clock floor is 2457 MHz (the historical B70 value was also 2457).
- Baseline: the A/A control medians against the published final-profile references (`proposal.md`) are 1B 4w +0.00 %,
  3B 4w +0.39 %, 8B 4w +0.86 %, 1B 8da4w +0.00 %, 3B 8da4w +0.49 %, 8B 8da4w -0.00 %. All eight same-session
  controls lie between -0.47 % and +1.02 % of the published values, inside the 3 % rule.
- Validity: `.artifacts/runs.csv` has 864 rows: 672 timed prefill runs (8 sessions x 84) and 192 next-token check
  runs. All 672 timed runs are valid, with no extra pairs, no foreign GPU process and no thermal reason. 24 check
  runs (14 on `prompt_check.txt`, 10 on `r1304.txt`; dq-n13 3, q4-table 6, dq-i 2, dq-table 6, q4-k16 2, q4-k32 3,
  dq-n26 2, s1-aa 0) are below the 2457 MHz floor (`clock_low`). They are kept and are not used for timing; the
  token comparison ignores the clock by design (`e2e5.sh` `toklog`).
- `results/candidate-coverage.csv` is the pre-run manifest mapping; its `readiness` column predates the matrix.
  The screened outcome is the table above. The `a_a_consistent` key in `evidence.json` is meaningful only for
  `s1-aa`.
- Correctness: every profile passes production diff 12/12 with next tokens SAME. The only line of each candidate's
  verify output that differs from the control's is the linear kernel-name line, with reviewed manifest kernels.
  dq-n26 and q4-k32 are line-identical to the control and dispatch the control's kernels: they are the final
  profile's own picks.
- Member margins (point estimates): six of seven profiles are within 3 % geomean and 8 % per cell of the
  same-session tuned control. dq-table fails, with slowdowns of 13.3 % (1B 8da4w), 19.2 % (3B 8da4w) and 24.4 %
  (8B 8da4w); its geomean slowdown is 8.8 %. q4-table costs 2.0 to 2.4 % on the three 4w cells (geomean 1.0 %).
  dq-i costs 4.3 % on 8B 8da4w (geomean 0.7 %). dq-n13, q4-k16, q4-k32 and dq-n26 are inside the +-2 % noise band
  everywhere.
- Uncertainty: 20,000 bootstrap resamples per arm (seed 7321), 95 % intervals in each session's `evidence.json`.
  Two intervals reach a limit. dq-table's geomean slowdown is [8.5 %, 9.3 %], over the 3 % limit. dq-i's 8B 8da4w
  slowdown is [3.8 %, 4.8 %], under the 8 % limit. The 1B intervals are coarse, because the runner's 1 ms timer
  quantises a roughly 0.1 s prefill (L19).
- Dispatch evidence: `results/matrix-dispatch.csv`. The first collection stopped early in 5 sessions: `cool_start`
  returned status 1 under `set -e`, which is now fixed in `host.sh`. Those partial collections are preserved in
  `.artifacts/superseded/dispatch-cool-start-status/` and were collected again in full. In q4-table and q4-k32 all
  12 traced runs had completed; only the name extractor was wrong (it read `name` instead of
  `delegate_debug_id_str`). Both were re-extracted on the CPU, and their `dispatch.done` was written by the actor.
- Property signature: the queried cooperative-matrix shapes, subgroup and shared-memory limits are in
  `.artifacts/preflight/cooperative-matrices.json` and `vulkan-device0.json`, recorded before the matrix.

## Xe2 family finalists (coordinator decision 2026-10-16 05:20 UTC)

Registered as dev-zone profiles in the `b70-wmma` block of `impl/sarc_dev/Overrides.cpp`, selected with
`ET_VK_SARC_B70_WMMA_PROFILE` on top of the unchanged control:

- `xe2-family-a`: 4w `sarc_linear_q4gsw_coopmat_sweep_t128x128k16g82s16m8flib` for every 4w prefill shape;
  8da4w `sarc_dev_linear_dq8ca_coopmat_zpg_xe2bto_t128x128k64g84s16m8i` for every 8da4w prefill shape.
- `xe2-family-b`: the same 4w; 8da4w n13 where K*N >= 24,000,000 (the B580 final rule), i otherwise.
- Shapes a kernel cannot serve (for example buffer IO for k16g82, which has only a texture3d variant) keep the
  control's choice, as in the matrix. The CPU selection fixture (`tools/test_profiles.cpp`, built with host g++)
  passes all 9 profiles, 12 storage variants and 7 family cases.
- Running: detached job `final-01` (`tools/final_queue.sh wmma-b70-topic-03`), status file
  `/home/doremy/wmma-b70/.artifacts/jobs/final-01.json`. It does one build of HEAD with the golden check, then for
  each finalist: unmodified `verify.sh --pdiff`, 12 passes each of the SDPA tiers all, extended and full, a
  seven-repeat interleaved session against the matrix control with next tokens on the three prompts, dispatched
  names, and `gate_check.py --sdpa` against the control's verify output. The warm-trace item of `gate_check.py`
  is not part of the coordinator's gate and is not run, so `gate.txt` reports it as FAIL (missing); that item is
  missing evidence, not a pass.

## Question for the coordinator

B70 matrix complete and reviewed (2026-10-16); which family finalists should get the full gate (verify, SDPA tiers, D3 evidence) on this device?

Answered 2026-10-16 03:15 UTC: matrix accepted as the B70 member evidence; Xe2 finalists are named after the B580 matrix completes. Parked; no GPU work on this device until the finalist list is appended to the task file.

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
