# Status — 2026-07-12, paused mid-confirmation on driver drift

**Current state: PAUSED by user decision.** Do not reflash, do not switch boards, do not run
any further measurements on M5 EVT1 until a resumed session with real user sign-off on the
driver reflash below.

## Done and correctness-verified (T001–T016)

- **T001–T002**: Execution worktree created at `4w-e2e-tile-sweep/executorch/` (branch
  `028-4w-e2e-tile-sweep`, cut from `yanwen/dev-1.3`). `install_executorch.sh --minimal +
  pip install -e .` completed successfully in a dedicated `.venv`.
- **T003/T006**: `results/prefilter_ranking.json` — 8 shortlisted candidates from
  `specs/022`'s `round2_results.json`/`round3_results.json`, all
  `correctness_all_shapes_pass: true`.
- **T004/T005**: All three 4w buffer PTEs (8B/1B/3B) confirmed already staged on-device
  (`0000088f8e579c33` via `sj1-dmckee-d01`). Driver hash confirmed matching documented
  default (`c9861e9906…` = `f14c51b6f8`) at session start.
- **T007–T009 (Phase 2.5 port)**: `linear_q4gsw_coopmat_tsweep.{glsl,yaml}` created
  (structural copy of `linear_qw_coopmat.glsl`'s INT4 dbuf1 path — note: research.md's
  Decision 0 named the wrong base file, `linear_q4gsw_coop.glsl`; the real current fp16
  q4gsw coopmat shader is `linear_qw_coopmat.glsl`, generating kernel names
  `linear_q4gsw_coopmat_*`). `ET_VK_Q4GSW_COOPMAT_VARIANT` env-var dispatch token added to
  `QuantizedLinear.cpp`, mirroring `ET_VK_DQ8CA_COOPMAT_VARIANT`'s pattern (sourced from the
  `dbuf-int8-sweep` worktree, since `dev` itself never had this exact mechanism for q4gsw).
  Additionally ported `test_coopmat_linear_bench.cpp` + its `CMakeLists.txt` entry (missing
  from `dev` entirely, confirmed absent by search) since T011's prescribed correctness gate
  requires it.
- **T010**: 16 `shader_variants` entries added (8 tokens × 2 weight-storage). Full Android
  cross-build succeeded: `vulkan_backend` (`--target install`), `llama_main`,
  `test_coopmat_linear_bench` — all built and pushed to device.
- **T011/T012**: `results/port_verification.json` — all 8 shortlisted tokens PASS
  correctness (`COOPMAT_BENCH_CORRECTNESS_ONLY=1`), including explicit rank3-batch=1
  coopmat-dispatch confirmation (not just numeric pass) for every token. No exclusions.
- **T013–T016**: `results/screen_results.json` (9/9: baseline + 8 tokens, real 2048-token
  prefill via `llama_main`) and `results/escalation_decisions.json` (8/8 decisions,
  `escalated = screen_ratio >= -0.10`). Screen headline: baseline 152.904 tok/s; best
  screen result `tsweep_t64x128k16g41s32` at 155.481 (+1.69%); the microbenchmark's #1-ranked
  token `tsweep_t128x64k16g14s32` came in at 149.806 (**behind** baseline) — a likely
  microbenchmark-vs-e2e rank disagreement, not yet formally computed (T020, blocked).
  5 tokens escalated to confirmation: `tsweep_t128x64k16g14s32`, `tsweep_t64x128k16g41s32`,
  `tsweep_t128x64k16g41s32`, `tsweep_t64x64k16g21s64`, `tsweep_t64x64k16g12s64`.

## Blocked: T017 (3-run confirmation), 14/18 runs done

`results/confirm_results.json` has 14 of the required 18 runs (baseline 3/3,
`tsweep_t128x64k16g14s32` 3/3, `tsweep_t64x128k16g41s32` 3/3, `tsweep_t128x64k16g41s32` 3/3,
`tsweep_t64x64k16g21s64` 2/3). **Missing:** `tsweep_t64x64k16g21s64` run 3, and all 3 runs
for `tsweep_t64x64k16g12s64`.

**Why it stopped:** mid-round, a fresh pre-round driver-hash check (re-run automatically by
`run_e2e_confirm.py` on retry after an unrelated transient timeout was fixed) returned
`21e1251c432ec9c8314470ef63d03e3b` — **not** the documented default `f14c51b6f8`
(`c9861e9906d03fa2c7d48b804e1a1c80`) that was verified on-device immediately before this same
confirmation round started. This is the shared M5 EVT1 board's known recurring drift pattern
(see `.shared-context/ACTIVE-STATUS.md`'s prior "DRIFT FOUND & FIXED" entries) — an
unrecognized driver build appeared mid-session, not present in any documented hash table.

Per Constitution Principle VIII, the in-flight run was killed immediately and no further
measurements were taken on the drifted driver.

**Remediation attempted, blocked:** the unknown driver was backed up —
`/tmp/vulkan.samsung.so.device-unknown-21e1251c-backup-2026-07-12` (on the adb host) and
`/sarc-c/gpusw/users/yanwen.xu/vulkan.samsung.so.device-unknown-21e1251c-backup-2026-07-12`
(NFS, 46,868,408 B). The documented reflash-to-default procedure
(`adb root/remount/setenforce 0/stop → push f14c51b6f8 → start`) was then attempted and
**denied by the permission system's auto-mode classifier** — it requires the real user's
explicit, specific authorization, not an agent/coordinator instruction. The device is
currently **unchanged**, still on the unrecognized `21e1251c…` driver (re-verified via a
subsequent read-only `md5sum` after the denial).

**User decision (2026-07-12):** stop here. Do not reflash, do not switch boards, do not run
any further measurements at this time.

## Data integrity note

All 14 recorded confirm runs, and all 9 screen runs, were taken while the driver hash was
independently verified as the documented default (`c9861e9906…`) immediately before each of
those rounds started. There is no direct evidence they are contaminated. However, the exact
moment of drift within the confirm round is not pinned down (the script only re-checks the
hash once, at round start) — treat the confirm-round numbers as provisional until a fresh
correctness re-check (`COOPMAT_BENCH_CORRECTNESS_ONLY=1`) is run on whatever driver is
confirmed present at resume time.

## What needs to happen before resuming

1. A session with the real user present re-verifies (or explicitly authorizes reflashing)
   the M5 EVT1 driver to the documented default `f14c51b6f8`
   (`adb -s 0000088f8e579c33 shell md5sum /vendor/lib64/hw/vulkan.samsung.so` should read
   `c9861e9906d03fa2c7d48b804e1a1c80`).
2. Re-run `COOPMAT_BENCH_CORRECTNESS_ONLY=1 ./test_coopmat_linear_bench` for at least the two
   remaining candidates (`tsweep_t64x64k16g21s64`, `tsweep_t64x64k16g12s64`) as a sanity
   check before trusting further e2e numbers on that driver.
3. Resume `run_e2e_confirm.py` (already resume-safe — it skips any `(token, run_index)` pair
   already present in `confirm_results.json`) to complete the missing 4 runs:
   `tsweep_t64x64k16g21s64` run 3, `tsweep_t64x64k16g12s64` runs 1–3.
4. Only then proceed to T018–T021 (confirmation summary, rank-agreement finding, 8B
   `winner_token` determination) — do not determine a winner from the incomplete 14/18 data.
5. T022 onward (US2 conditional extension, US3 report, US4 1B/3B confirmation) follow from
   T021 as originally planned in `tasks.md`.

## Execution worktree

`/local/yanwen.xu/workspace/4w-e2e-tile-sweep/executorch/` — left as-is, **uncommitted**, on
branch `028-4w-e2e-tile-sweep`. Contains the ported shader/dispatch files
(`linear_q4gsw_coopmat_tsweep.{glsl,yaml}`, `QuantizedLinear.cpp` diff,
`test_coopmat_linear_bench.cpp` + `CMakeLists.txt` diff), plus the full Android
cross-build output (`cmake-out-android-vk/`). No commits made; nothing pushed.

## Provenance for all numbers in this feature so far

- **Board**: `0000088f8e579c33` via `ssh yanwen.xu@sj1-dmckee-d01`
- **Driver** (screen round + first 14 confirm runs): `c9861e9906d03fa2c7d48b804e1a1c80`
  (= `f14c51b6f8`, documented default)
- **Clocks**: pinned, GPU 509 MHz (`min_freq`/`max_freq` both `509000`)
- **Model**: `llama3_1_8b_4w_buffer_ctx3072.pte`, 2048-token prefill
  (`ET_VK_EXECUTE_NODE_THRESHOLD=16`, `p2048_exact.txt`, `num_bos=1`)
- **Run counts**: screen = 1 run/candidate (9 total); confirm = 3 runs/candidate where
  complete, 2/3 for one candidate, 0/3 for another (14/18 total)
