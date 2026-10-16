# rdna3-780m status

Updated 2026-10-16 11:10 UTC. Actor: Claude (owner decision 2026-10-15 23:30 UTC, lean finish).
Stage: finite RDNA3 RADV linear matrix on the Radeon 780M. Hard limit 2026-10-17 18:00 UTC.
**The matrix is complete: all 19 legal candidates gated and timed. Nothing is running. Ready for the one
review of the finished matrix.** No family acceptance is claimed from this member alone; RX 7600 and
RX 7900 XTX evidence is external and still missing.

Every result here carries the monitoring caveat: own-user GPU-client accounting plus device-wide busy
sampling; other users' DRM clients are not readable
(`own_user_plus_device_busy_other_users_unobservable`, owner decision 2026-10-15 03:57 UTC).

## Result

Chain `tools/matrix.sh`, tag `rdna3-780m-screen-20261015a`, ran 2026-10-15 23:50 to 2026-10-16 10:52 UTC,
state `completed` (`<artifact-root>/jobs/rdna3-780m-screen-20261015a/state.json`). Every arm used one
binary: build `rdna3-780m-matrix-20261015a` (commit `4f9aba6b4`, `llama_main` `e155b975...`,
`libllama_runner.so` `f8b35251...`, `test_llama_microbench` `a36237ea...`; 53-variant golden PASS); the
control and candidate arms differ only in `ET_VK_SARC_780M_WMMA_PROFILE` (`results/matrix/*/STAGE.md`).
Attention setup is identical in both arms (`ET_VK_SARC_DEV_PROFILE=780m-final`; the WMMA selector returns
the control's choice for every op other than the profile's linear op).

- Control gate: unmodified `verify.sh --pdiff` on `780m-final` is **identical line by line (rates removed)**
  to the campaign's final gate (`results/matrix/ctrl-verify/`), including its known `linear <scheme> rc=1`.
- Control against the published final profile (round 3, session B, `780m-final` arm): 1B 4w 3835.21 vs
  3835.21, 1B 8da4w 3764.71 vs 3757.80 (+0.18 %), 3B 4w 1458.69 vs 1457.65 (+0.07 %), 3B 8da4w 1409.50 vs
  1414.36 (-0.34 %), 8B 4w 639.20 vs 639.00, 8B 8da4w 628.61 vs 628.41: all within 3 %.
- A/A (control vs control): geomean 1.0008 (95 % CI 0.9980 to 1.0015), cells 0.9993 to 1.0062, spread at most
  0.83 %, 0 invalid rows. Clock floor 2716 MHz (97 % of 2800, `results/matrix/floor.txt`).
- Every candidate passed the production diff (12 of 12) and every next token equals the control's on
  `prompt_2048`, `prompt_check` and `r1304` in all six cells; no `verify.sh` line other than the linear
  kernel lists differs from the control's. No candidate is CORRECTNESS_PENDING or rejected for correctness.
- Margin on this member (every r >= 0.92, geomean >= 0.97): **FAIL for rdna3-12** (RX 7600 4w tile,
  4w cells 0.30 to 0.34), **rdna3-13** (RX 7600 8da4w tile, 8da4w cells 0.36 to 0.41) and **rdna3-17**
  (7900 XTX 8da4w sweep tile, 8da4w cells 0.906 to 0.932, geomean 0.9606). PASS for the other 16.
- Dispatch changed against the control (microbench linear lists in `verify.out`) for 06, 12, 13, 16, 17, 18,
  19, 02, 14 and, for buffer IO only, 15. Of the passing ones, three cells fall outside the +-2 % band, all
  slower: rdna3-16 1B 4w 0.9727, rdna3-02 8B 4w 0.9762, rdna3-14 8B 8da4w 0.9696. No candidate is faster
  than the control outside the noise band on any cell.
- 01, 03, 04, 05, 07, 08, 09, 10, 11 (780M rows) dispatch exactly the control's linear kernels, and 15
  dispatches them on the texture3d production path: these ten sessions are control vs control in effect;
  their geomeans lie between 0.9971 and 1.0008.

Validity: every session has 7 valid timed runs per arm and cell (rdna3-07 needed one extra pair: an 8B 4w
control run at 2308 MHz, `clock_low`). The 9 to 14 invalid rows per candidate session are next-token check
runs (`prompt_check`, `r1304`: no `--warmup`, short windows) marked `clock_low` against the floor
calibrated on timed runs, or with fewer than 5 clock samples; they are kept, are not used for any median, and
their output text is what the next-token comparison reads. One `foreign_gpu_client` row (rdna3-06, a check
run) and freewait waits up to 195 s (rdna3-06, -07, -08) show the other session's GPU use being caught and
waited out. `own_procs_after` = 1 (the watcher) on every row: no sampler left behind.

Not recorded: end-to-end dispatched kernel names of the llama_main runs (no ETDump trace in this lean
protocol); the linear names come from the microbench production shapes of `verify.sh`. Top-2 logits were
not needed (no next-token difference).

## Session tool changes (run-local, `tools/session.sh`, copied from the 780M campaign's `e2e5.sh`)

| change | why |
|---|---|
| 7 valid runs per arm (`--reps 7`), at most 3 extra pairs | owner decision 23:30 UTC |
| `--clkmin` required: 0 in the A/A, then the A/A floor | registered floor rule |
| a watcher started with the session records foreign GPU activity (guard-pattern process outside the session, or an own-user DRM client outside it that appears or whose engine time grows); every run waits until 60 s have passed without any (`freewait_s`); after 3 h the run is marked `gpu_not_free` | owner: wait for 60 s without a foreign client before each timed run |
| cooling ends at idle + 5 C, or after three 5 s readings without a fall, or at 120 s (`cool_end`) | registered cooling rule |
| lock wait 2 h instead of 15 min | another session uses the GPU from time to time |
| extra row columns: session, candidate, commit, binary sha256, driver, monitoring_visibility, freewait_s, cool_end, own_procs_after | row provenance |

Validity predicates, thermal mask `0xFFFFE670` and the monitor are unchanged. The mask's bit table was
checked against the running host kernel's own source (`smu13_driver_if_v13_0_4.h` from
`kernel-6.12.0-211.65.1.el10_2.src.rpm`, signature OK): bits 4, 5, 6 THM, 9, 10 PROCHOT, 0 to 3 power,
7, 8, 11, 12 current; `gpu_metrics` format 2.1, `throttle_status` at byte 108. Smoke test (1B 4w, one pair,
rdna3-06): valid rows, next tokens SAME, no samplers left after the session.

## Matrix

r = candidate / same-session control, medians of 7 valid runs; margin: every r >= 0.92 and geomean >= 0.97.
One member's result is not a family acceptance.

<!-- matrix table begin -->
| candidate | linear kernel | correctness | 1b-4w | 1b-8da4w | 3b-4w | 3b-8da4w | 8b-4w | 8b-8da4w | geomean (95 % CI) | margin |
|---|---|---|---|---|---|---|---|---|---|---|
| aa | control vs control |  | 1.0000 | 1.0000 | 0.9993 | 1.0062 | 0.9994 | 1.0000 | 1.0008 (0.9980-1.0015) | PASS |
| rdna3-06 | sarc_dev_780m_x_linear_dq8ca_coopmat_zpg_t256x64k64g48s32afmb1 | pdiff PASS; next token SAME | 1.0000 | 0.9963 | 1.0000 | 1.0076 | 0.9997 | 1.0000 | 1.0006 (0.9974-1.0014) | PASS |
| rdna3-12 | sarc_dev_rx7600_x_linear_q4gsw_coopmat_t256x128k32g28s32f32cbtap4bp12 | pdiff PASS; next token SAME | 0.3365 | 1.0000 | 0.3154 | 1.0014 | 0.2989 | 0.9997 | 0.5627 (0.5622-0.5632) | FAIL |
| rdna3-13 | sarc_dev_rx7600_x_linear_dq8ca_coopmat_zpg_t256x64k64g48s32pa6pb4csha | pdiff PASS; next token SAME | 1.0000 | 0.4069 | 0.9993 | 0.3868 | 0.9994 | 0.3615 | 0.6200 (0.6198-0.6215) | FAIL |
| rdna3-16 | sarc_linear_q4gsw_coopmat_sweep_t128x128k32g42s32f32cbt | pdiff PASS; next token SAME | 0.9727 | 1.0000 | 1.0007 | 0.9973 | 0.9997 | 0.9997 | 0.9950 (0.9947-0.9966) | PASS |
| rdna3-17 | sarc_linear_dq8ca_coopmat_zpg_sweep_t256x64k32g24s32 | pdiff PASS; next token SAME | 1.0019 | 0.9283 | 1.0007 | 0.9060 | 0.9991 | 0.9324 | 0.9606 (0.9585-0.9611) | FAIL |
| rdna3-18 | sarc_linear_dq8ca_coopmat_zpg_sweep_t128x64k32g22s32 | pdiff PASS; next token SAME | 1.0000 | 1.0000 | 1.0007 | 1.0090 | 0.9997 | 0.9861 | 0.9992 (0.9956-0.9998) | PASS |
| rdna3-19 | sarc_linear_dq8ca_coopmat_zpg_sweep_t64x64k32g22s32 | pdiff PASS; next token SAME | 0.9981 | 0.9963 | 1.0007 | 1.0042 | 0.9994 | 0.9997 | 0.9997 (0.9979-1.0007) | PASS |
| rdna3-02 | sarc_dev_780m_x_linear_q4gsw_coopmat_t256x128k32g24s32f32cbt | pdiff PASS; next token SAME | 0.9871 | 1.0000 | 0.9915 | 1.0104 | 0.9762 | 0.9997 | 0.9941 (0.9901-0.9947) | PASS |
| rdna3-14 | sarc_linear_dq8ca_coopmat_zpg_t128x64k32g42s32 | pdiff PASS; next token SAME | 0.9981 | 1.0000 | 1.0000 | 0.9925 | 1.0006 | 0.9696 | 0.9934 (0.9929-0.9947) | PASS |
| rdna3-15 | sarc_linear_dq8ca_coopmat_zpg_t128x64k32g42s32_buffer_texture2d_half | pdiff PASS; next token SAME | 1.0000 | 1.0000 | 1.0007 | 0.9877 | 0.9994 | 0.9994 | 0.9979 (0.9972-1.0030) | PASS |
| rdna3-01 | sarc_dev_780m_x_linear_q4gsw_coopmat_t256x128k32g18s32f32cbt | pdiff PASS; next token SAME | 1.0000 | 1.0000 | 0.9993 | 0.9911 | 1.0000 | 1.0000 | 0.9984 (0.9980-1.0018) | PASS |
| rdna3-03 | sarc_dev_780m_x_linear_q4gsw_coopmat_t256x128k32g28s32f32cbt | pdiff PASS; next token SAME | 1.0000 | 1.0000 | 1.0000 | 0.9979 | 1.0000 | 1.0003 | 0.9997 (0.9990-1.0011) | PASS |
| rdna3-04 | sarc_dev_780m_x_linear_q4gsw_coopmat_t128x128k32g24s32f32cbt | pdiff PASS; next token SAME | 1.0000 | 1.0018 | 1.0000 | 0.9945 | 0.9984 | 1.0006 | 0.9992 (0.9984-1.0017) | PASS |
| rdna3-05 | sarc_dev_780m_x_linear_q4gsw_coopmat_t128x256k32g42s32f32cbt | pdiff PASS; next token SAME | 1.0019 | 1.0000 | 1.0000 | 0.9993 | 0.9997 | 0.9997 | 1.0001 (0.9992-1.0006) | PASS |
| rdna3-07 | sarc_dev_linear_dq8ca_coopmat_zpg_bt_t128x64k32g22s32 | pdiff PASS; next token SAME | 1.0000 | 1.0018 | 1.0000 | 1.0041 | 0.9991 | 1.0000 | 1.0008 (0.9991-1.0014) | PASS |
| rdna3-08 | sarc_dev_linear_dq8ca_coopmat_zpg_bt_t128x64k32g22s32_buffer_texture2d_half | pdiff PASS; next token SAME | 1.0000 | 1.0000 | 0.9993 | 0.9993 | 0.9997 | 0.9997 | 0.9997 (0.9925-1.0044) | PASS |
| rdna3-09 | sarc_linear_q4gsw_coopmat_t128x128k32g42s32f32c | pdiff PASS; next token SAME | 0.9981 | 1.0000 | 1.0000 | 0.9898 | 1.0006 | 0.9997 | 0.9980 (0.9975-1.0022) | PASS |
| rdna3-10 | sarc_linear_q4gsw_coopmat_t128x128k32g42s32f32c_buffer_texture2d_half | pdiff PASS; next token SAME | 1.0000 | 1.0000 | 1.0007 | 0.9823 | 1.0006 | 0.9994 | 0.9971 (0.9964-1.0037) | PASS |
| rdna3-11 | sarc_linear_q4gsw_coopmat_t128x128k32g42s32f32c_buffer_buffer_half | pdiff PASS; next token SAME | 1.0000 | 1.0018 | 1.0000 | 1.0014 | 1.0003 | 1.0003 | 1.0006 (0.9995-1.0013) | PASS |
<!-- matrix table end -->

## Earlier preparation (reused, not repeated)

Reviewed inventory `f393c3ff7f` (archive sha256 `65d1e3ca...`, RDNA3 manifest `edcfa39e...`, 18 file hashes
re-confirmed 2026-10-15 23:2x UTC). Parent build, export audits, runtime model metadata (context 2560, group
128, embedding group 32), selector build and CPU selector checks: see `results/*.json` and git history of
this file.

## Question for the coordinator

(none)
