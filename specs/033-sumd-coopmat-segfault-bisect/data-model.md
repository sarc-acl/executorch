# Data Model: SUMD Driver Bisect for the Coopmat-Dispatch Segfault Regression

**Feature**: `033-sumd-coopmat-segfault-bisect` | **Date**: 2026-07-21

Two entities, matching spec.md's Key Entities, plus a small supporting record for the
backup/restore protocol (FR-003) that `specs/032` didn't need. No persistence layer — all are
rows/sections in the single bisect-trace document (`results/`), not a database.

## Bisect Step

One SUMD commit under test.

| Field | Type | Notes |
|---|---|---|
| `commit_sha` | string (full 40-char SHA) | The SUMD `main` commit tested |
| `commit_date` | ISO 8601 date | For context/ordering |
| `bisect_role` | enum: `endpoint-good` \| `endpoint-bad` \| `bisect-step` \| `skip-adjacent-probe` | Distinguishes the two range-endpoint re-confirmations (User Story 1) from interior `git bisect` steps and from manual adjacent-commit probes forced by a skip run (FR-009) |
| `build_outcome` | enum: `success` \| `build-failed` \| `flash-failed` | `build-failed`/`flash-failed` both imply `verdict = skip` |
| `driver_hash_post_flash` | string (md5) | Captured via `md5sum /vendor/lib64/hw/vulkan.samsung.so` immediately after flashing (FR-005) — confirms the flash took |
| `driver_hash_pre_test` | string (md5) | Same command, immediately before running the bench — must equal `driver_hash_post_flash`; a mismatch means drift happened mid-step and the step must be re-run, not trusted |
| `verdict` | enum: `good` \| `bad` \| `skip` | `good` iff the bench prints `Completed 16 test cases` with exit 0; `bad` iff the process crashes (signal-indicating exit code, e.g. 139) or stdout stops before the completion line (FR-007); `skip` iff `build_outcome != success` or a hang exceeds the bounded timeout while the device stays otherwise responsive |
| `skip_reason` | string \| null | Required when `verdict = skip` (e.g. "GpuRt Too many users, retry-with-stripped-LD_LIBRARY_PATH also failed", or "test hung >N min, device still responsive after recovery") |
| `crash_evidence` | object \| null | Required when `verdict = bad` — see Crash Evidence below; null otherwise |
| `pre_step_backup` | object \| null | See Driver Backup Record below — present whenever the device's pre-flash driver wasn't already this study's own | |

**Validation rules** (from Functional Requirements):
- `verdict` is derived, never independently set: `skip` if `build_outcome != success` or a
  bounded-timeout hang; else `bad` if the bench crashes/doesn't complete; else `good` (FR-007).
- Every row's device is implicitly M5 EVT1 serial `0000088f8e579c33` (FR-002) — a study-wide
  invariant, not a per-row field, since no other device is ever valid.
- A row where `driver_hash_post_flash != driver_hash_pre_test` is not a terminal record — drift
  happened mid-step and the step must be re-run (this is the M5 EVT1-specific analogue of
  `specs/032`'s clock-pin-verification invariant).
- `crash_evidence` is required and non-null for every `bad` row (FR-008); `pre_step_backup` is
  required and non-null for every row where the device wasn't already on a driver this study
  flashed (FR-003).

### Crash Evidence (sub-object of Bisect Step, present iff `verdict = bad`)

| Field | Type | Notes |
|---|---|---|
| `tombstone_path` | string \| null | Path to the pulled tombstone file (`results/tombstones/<sha>.txt`); null if no tombstone was written (fallback case) |
| `signal` | string \| null | e.g. `SIGSEGV (SEGV_MAPERR)`, from the tombstone header; null if no tombstone |
| `fault_frames` | list\<string\> | The top backtrace frames, at minimum whichever are inside `vulkan.samsung.so` (from the tombstone, if present) |
| `exit_code` | int | Process exit code (e.g. 139 for a raw SIGSEGV) — always present, tombstone or not |
| `last_console_line` | string | Last line printed before the crash (e.g. which test case name was in progress) — always present |

### Driver Backup Record (sub-object of Bisect Step, present iff a non-study driver was found)

| Field | Type | Notes |
|---|---|---|
| `found_hash` | string (md5) | The hash discovered on-device immediately before this step's flash |
| `backup_path` | string | NFS path the `.so` was copied to before overwriting (FR-003) |
| `restored_after_step` | bool | Must be `true` in the final record — a step is not complete until this is confirmed |

## Culprit Commit

The single first-bad commit `git bisect` converges on (User Story 2), derived from the full set
of Bisect Step rows, not measured independently.

| Field | Type | Notes |
|---|---|---|
| `commit_sha` | string | The identified first-bad commit |
| `author` | string | From `git log` metadata |
| `commit_date` | ISO 8601 date | |
| `subject` | string | Commit subject line |
| `last_good_sha` | string | The immediate parent/predecessor Bisect Step with `verdict = good` |
| `last_good_driver_hash` | string | For the hash delta (FR-012) |
| `first_bad_driver_hash` | string | For the hash delta (FR-012) |
| `diff_summary` | string | What the commit's diff changed, and why that plausibly causes the observed coopmat-dispatch segfault (FR-011) — populated by reading the commit's actual diff, permitted here since Rule 0 is lifted (unlike `specs/032`, which stopped at the row above this one) |

**Relationship**: `Culprit Commit` is a computed view over the ordered set of `Bisect Step` rows —
it always has exactly one `last_good` predecessor and is itself the first row with `verdict = bad`
in commit-chronological order among the steps `git bisect` actually narrowed to (SC-002).
`diff_summary` is the one field with no `specs/032` analogue, added because source-reading is
permitted for this study.
