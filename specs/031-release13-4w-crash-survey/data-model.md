# Phase 1 Data Model: Release/1.3 Crash Survey on M5 EVT1 (4w + 8da4w, Floating + Pinned)

This feature has no runtime database or persistent service state — "data model" here means the
shape of the records this survey collects and reports, matching the spec's Key Entities.
**Extended** (2026-07-14) from the original `4w`-only, floating-only shape to add `quant_scheme`,
`clocks`, and `node_threshold` fields below (marked *Extension*).

## Benchmark Attempt

One (model, quant_scheme, clocks, rep_index, node_threshold) combination, run against either the
vanilla `release-1.3/executorch` worktree's `llama_main_rel1.3` runner or (only when vanilla is
confirmed to crash on that exact cell) `release13-node-threshold/executorch`'s
`llama_main_nodethresh` runner.

| Field | Type | Notes |
|---|---|---|
| `model` | enum {1B, 3B, 8B} | Llama 3.2 1B / Llama 3.2 3B / Llama 3.1 8B, `ctx3072`/texture |
| `quant_scheme` | enum {4w, 8da4w} *(Extension — was hardcoded 4w originally)* | |
| `clocks` | enum {floating, pinned} *(Extension — was hardcoded floating originally)* | floating = devfreq range `255000`–`980000`; pinned = `509000`/`509000` |
| `rep_index` | int, 1–3 | Attempt number within this (model, quant_scheme, clocks) cell's sequence |
| `node_threshold` | enum {none, 32, 64}, *(Extension)* | `none` = vanilla `llama_main_rel1.3`; `32`/`64` = `llama_main_nodethresh` with `ET_VK_EXECUTE_NODE_THRESHOLD` set accordingly |
| `driver_md5_pre` | string | `md5sum /vendor/lib64/hw/vulkan.samsung.so`, checked immediately before this attempt |
| `clock_range_pre` | (min_freq, max_freq) | sysfs readback immediately before this attempt; expected `(255000, 980000)` if `clocks=floating`, `(509000, 509000)` if `clocks=pinned` |
| `outcome` | enum {completed, crashed} | completed = runner printed its JSON stats line; crashed = device dropped off `adb` / re-enumerated as bootloader |
| `prefill_tok_s` | float, nullable | Only present when `outcome = completed` |
| `decode_tok_s` | float, nullable | Only present when `outcome = completed` |
| `prompt_tokens` | int, nullable | Sanity check — must read 2048 |
| `generated_tokens` | int, nullable | Sanity check — must read 1023 (+1 prefill-boundary token = 1024 total, per `specs/029` Finding 3) |
| `crash_event_id` | reference, nullable | Only present when `outcome = crashed`; links to the Crash Event record |

**Validation rules**: `prefill_tok_s`/`decode_tok_s`/`prompt_tokens`/`generated_tokens` are
mutually exclusive with `crash_event_id` — exactly one side is populated depending on `outcome`.
An attempt with `outcome = completed` but `prompt_tokens != 2048` is not counted toward this
model's summary (it means the coherence/config check itself failed, not a clean measurement) and
must be re-attempted rather than silently included.

## Crash Event

Created whenever a Benchmark Attempt's outcome is `crashed`.

| Field | Type | Notes |
|---|---|---|
| `attempt_ref` | reference | The Benchmark Attempt that triggered this event |
| `time_to_drop_s` | float | Wall-clock seconds from run start to the device disappearing from `adb devices` |
| `usb_state_observed` | enum {bootloader, absent, other} | What `lsusb`/`fastboot devices` showed while investigating |
| `recovery_method` | enum {fastboot_reboot, escalated_unrecovered} | Per spec Edge Cases — only `fastboot_reboot` is attempted automatically; anything else stops the survey |
| `recovery_time_s` | float | Wall-clock seconds from issuing `fastboot reboot` to `sys.boot_completed=1` |
| `driver_md5_post` | string | Re-verified after recovery, before the next attempt |
| `clock_range_post` | (min_freq, max_freq) | Re-verified after recovery, before the next attempt |

## Model Row (report unit)

Derived, not separately collected — aggregates one (model, quant_scheme, clocks) cell's
up-to-3-plus-retries Benchmark Attempts (12 cells total post-extension: 3 models × 2 quant
schemes × 2 clock policies).

| Field | Type | Derivation |
|---|---|---|
| `model` | enum {1B, 3B, 8B} | |
| `quant_scheme` | enum {4w, 8da4w} *(Extension)* | |
| `clocks` | enum {floating, pinned} *(Extension)* | |
| `node_threshold_used` | enum {none, 32, 64} *(Extension)* | Which config's attempts are counted toward this row's `completed_count`/median/CoV — a cell may have crashed attempts under `none` and/or a higher threshold before landing on the value that actually worked; only the working config's completed attempts count |
| `completed_count` | int, 0–3 | count of attempts with `outcome = completed` under `node_threshold_used` |
| `crashed_count` | int, 0+ | count of attempts with `outcome = crashed`, across **all** configs tried for this cell (vanilla + any threshold values that didn't work) — not capped at 3, since retries don't consume a rep slot |
| `prefill_tok_s_median` | float, nullable | median of `prefill_tok_s` over completed attempts under `node_threshold_used`; null if `completed_count = 0` |
| `prefill_cov_pct` | float, nullable | stddev/mean × 100 over completed attempts' `prefill_tok_s`; null if `completed_count < 2` |
| `decode_tok_s_median` | float, nullable | median of `decode_tok_s` over completed attempts under `node_threshold_used`; null if `completed_count = 0` |
| `decode_cov_pct` | float, nullable | stddev/mean × 100 over completed attempts' `decode_tok_s`; null if `completed_count < 2` |
| `crash_annotation` | string | e.g. `"0/3 crashed"`, `"vanilla: 7/7 crashed; t64: 0/3 crashed"` — always present and states which config(s) were tried, even when the final `crashed_count` under `node_threshold_used` is 0 |

**State transitions**: none — these are one-shot measurement records, not long-lived stateful
entities. The only "transition" in this feature is the board's own Android ⇄ bootloader state
during a Crash Event, which is operational (tracked via `usb_state_observed`/`recovery_method`),
not part of the reported data model.
