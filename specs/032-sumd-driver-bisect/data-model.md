# Data Model: SUMD Driver Bisect for the 8da4w-Slower-Than-4w Regression

**Feature**: `032-sumd-driver-bisect` | **Date**: 2026-07-16

Two entities, matching spec.md's Key Entities. No persistence layer — both are rows in the single
bisect-trace document (`results/`), not a database.

## Bisect Step

One SUMD commit under test.

| Field | Type | Notes |
|---|---|---|
| `commit_sha` | string (full 40-char SHA) | The SUMD `main` commit tested |
| `commit_date` | ISO 8601 date | For context/ordering |
| `bisect_role` | enum: `endpoint-old` \| `endpoint-new` \| `bisect-step` \| `skip-adjacent-probe` | Distinguishes the two range-endpoint checks (User Story 1) from interior `git bisect` steps and from manual adjacent-commit probes forced by a skip run (FR-009) |
| `build_outcome` | enum: `success` \| `build-failed` \| `crashed-on-device` | `build-failed`/`crashed-on-device` both imply `verdict = skip` |
| `driver_version_string` | string (md5 hash) | Captured via `md5sum /vendor/lib64/hw/vulkan.samsung.so` after flashing (FR-005) — never `logcat \| grep SUMD`, which is documented as unreliable for identifying the active build (dumps build-ancestry commit hashes, not the active build's own); null if `build_outcome != success` |
| `prefill_4w_tok_s` | float \| null | Single-rep, 2048-token prefill, Llama 3.2 1B, release/1.3 vanilla |
| `prefill_8da4w_tok_s` | float \| null | Same conditions, 8da4w quant mode |
| `verdict` | enum: `good` \| `bad` \| `skip` | `good` iff `prefill_8da4w_tok_s > prefill_4w_tok_s` (FR-008, strict comparison, no tie-break — see spec Clarifications); `skip` iff `build_outcome != success` |
| `skip_reason` | string \| null | Required when `verdict = skip` (e.g. "GpuRt Too many users, retry-with-stripped-LD_LIBRARY_PATH also failed" or "VK_ERROR_DEVICE_LOST on flash-verify run") |
| `clock_pin_verified` | bool | Result of the sysfs readback check (FR-006) — a step whose pin didn't verify is not a valid measurement and must be re-taken, not recorded with `clock_pin_verified = false` as final |

**Validation rules** (from Functional Requirements):
- `verdict` is derived, never independently set: `skip` if `build_outcome != success`, else
  `good`/`bad` from the strict prefill comparison (FR-008).
- Every row's device is implicitly M41 serial `00000a34cdd4abd3` (FR-003) — this is a study-wide
  invariant, not a per-row field, since no other device is ever valid.
- A row with `clock_pin_verified = false` is not a terminal record for that commit — the step
  must be re-measured before a verdict is trusted (FR-006).

## Culprit Commit

The single first-bad commit `git bisect` converges on (User Story 2), derived from the full set
of Bisect Step rows, not measured independently.

| Field | Type | Notes |
|---|---|---|
| `commit_sha` | string | The identified first-bad commit |
| `author` | string | From `git log` metadata (metadata only — no source content read, per Rule 0) |
| `commit_date` | ISO 8601 date | |
| `subject` | string | Commit subject line |
| `last_good_sha` | string | The immediate parent/predecessor Bisect Step with `verdict = good` |
| `last_good_driver_version` | string | For the version-string delta (FR-011) |
| `first_bad_driver_version` | string | For the version-string delta (FR-011) |

**Relationship**: `Culprit Commit` is a computed view over the ordered set of `Bisect Step` rows
— it always has exactly one `last_good` predecessor and is itself the first row with `verdict =
bad` in commit-chronological order among the steps `git bisect` actually narrowed to (SC-002).
