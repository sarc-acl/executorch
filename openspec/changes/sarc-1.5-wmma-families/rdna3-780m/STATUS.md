# rdna3-780m status

Updated 2026-10-15 23:55 UTC. Actor: Claude (owner decision 2026-10-15 23:30 UTC, lean finish).
Stage: finite RDNA3 RADV linear matrix on the Radeon 780M. Hard limit 2026-10-17 18:00 UTC.

Every result here carries the monitoring caveat: own-user GPU-client accounting plus device-wide busy
sampling; other users' DRM clients are not readable
(`own_user_plus_device_busy_other_users_unobservable`, owner decision 2026-10-15 03:57 UTC).

## Running now

Detached chain `tools/matrix.sh`, tag `rdna3-780m-screen-20261015a`, state file
`<artifact-root>/jobs/rdna3-780m-screen-20261015a/state.json`. One binary for every arm: build
`rdna3-780m-matrix-20261015a` (commit `4f9aba6b4`, `llama_main` sha256 `e155b975...`, 53-variant golden PASS),
which contains the control and all 19 candidate profiles.

1. `ctrl-verify`: unmodified `verify.sh --models 1b,3b,8b --schemes 4w,8da4w --pdiff` on the control
   (`ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=780m-final`), compared line by line (rates removed) with
   the campaign's final gate (`r3b-final-dev15-head4/verify.out`, same environment).
2. `aa`: control vs control, six cells, 7 valid runs per arm; clock floor = 97 % of the lowest per-run median
   clock of its valid runs (rule registered in `proposal.md` before the data).
3. Per candidate, in the order below: `verify.sh --pdiff` with the candidate environment (a production-diff
   failure rejects it untimed), then control vs candidate, six cells, interleaved, 7 valid runs per arm,
   next token on `prompt_2048`, `prompt_check` and `r1304`; summary, this table, a local commit.

Order (owner rule 4): candidates in another member's final profile first (06, 12, 13, 16, 17, 18, 19), then
those from other members' table layers (02, 14, 15), then the 780M-only rows (01, 03, 04, 05, 07, 08, 09, 10,
11). The 780M-only rows reselect the kernel the control already dispatches wherever their source route
applies, so on this device they are expected to time as control vs control.

Projection: about 50 min per candidate (5 min `verify.sh`, about 45 min session), about 17 h in all; finishes
before the hard limit. Nothing above 48 h is queued.

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
(no session finished yet)
<!-- matrix table end -->

## Earlier preparation (reused, not repeated)

Reviewed inventory `f393c3ff7f` (archive sha256 `65d1e3ca...`, RDNA3 manifest `edcfa39e...`, 18 file hashes
re-confirmed 2026-10-15 23:2x UTC). Parent build, export audits, runtime model metadata (context 2560, group
128, embedding group 32), selector build and CPU selector checks: see `results/*.json` and git history of
this file.

## Question for the coordinator

(none)
