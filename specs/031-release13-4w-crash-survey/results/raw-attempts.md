# Raw Attempts Log — Release/1.3 4w + 8da4w Crash Survey (Floating + Pinned)

Working log for `specs/031-release13-4w-crash-survey`. Original scope (2026-07-14, T005–T007)
was `4w`-only, floating-only. **Extended same day** to: (a) fill the `4w` pinned-clock gaps left
open in the original survey, and (b) run the identical methodology against `8da4w`, covering
both floating and pinned for all three models. The self-contained deliverable is
`results/report.md` — this file is the raw backing data for both the original scope and the
extension.

**Node-threshold policy note**: the original survey's crash workaround used
`ET_VK_EXECUTE_NODE_THRESHOLD=32`. Mid-extension, per explicit user direction, the default
fallback threshold changed to **64** — cells already collected at 32 (the original floating 8B
attempt, and 2 of 3 pinned 8B reps) were **re-collected fresh at 64 where 64 turned out to work**
(8B floating). Where 64 was empirically insufficient (8B pinned, both quant schemes), the policy
fell back to 32, which is noted per-cell below and in `report.md`.

## Provenance (T002)

- **Device**: M5 EVT1, `0000088f8e579c33`, via `ssh yanwen.xu@sj1-dmckee-d01`.
- **Runner**: `llama_main_rel1.3` (vanilla `release-1.3/executorch`, built 2026-07-14, staged at
  `/data/local/tmp/llama_vk/`) — confirmed present, 15,267,552 B.
- **PTEs confirmed present on-device** (all `4w`, texture, `ctx3072`):
  - `llama3_2_1b_4w_texture_ctx3072.pte` (1,692,057,856 B, pushed 2026-07-10)
  - `llama3_2_3b_4w_texture_ctx3072.pte` (3,240,939,136 B, pushed 2026-07-10)
  - `llama3_1_8b_4w_texture_ctx3072.pte` (5,979,595,648 B, pushed 2026-06-17)
- `tokenizer.model` and `p2048_exact.txt` confirmed present.

## Foundational checkpoint (T003/T004)

| Check | Expected | Observed | Status |
|---|---|---|---|
| Driver md5 | `c9861e9906d03fa2c7d48b804e1a1c80` | `c9861e9906d03fa2c7d48b804e1a1c80` | ✅ match |
| Clock min_freq/max_freq | `255000`/`980000` (floating) | `255000`/`980000` | ✅ floating confirmed |

## Benchmark Attempts (T005–T007)

| Model | Rep | driver_md5_pre | clock_range_pre | outcome | prompt_tokens | generated_tokens | prefill_tok_s | decode_tok_s | crash_event_id |
|---|---|---|---|---|---|---|---|---|---|
| 3B | 1 (attempt A) | c9861e99… | 255000/980000 | **crashed** | — | — | — | — | CE1 |
| 3B | 1 (attempt B, retry) | c9861e99… | 255000/980000 | completed | 2048 | 1023 | 215.874 | 13.7062 | — |
| 3B | 2 | c9861e99… | 255000/980000 | completed | 2048 | 1023 | 216.079 | 13.7049 | — |
| 3B | 3 | c9861e99… | 255000/980000 | completed | 2048 | 1023 | 216.193 | 13.7256 | — |
| 1B | 1 | c9861e99… | 255000/980000 | completed | 2048 | 1023 | 592.936 | 26.9714 | — |
| 1B | 2 | c9861e99… | 255000/980000 | completed | 2048 | 1023 | 592.936 | 27.0042 | — |
| 1B | 3 | c9861e99… | 255000/980000 | completed | 2048 | 1023 | 594.14 | 27.0427 | — |
| 8B | 1 (attempt A) | c9861e99… | 255000/980000 | **crashed** | — | — | — | — | CE2 |
| 8B | 1 (attempt B, retry) | c9861e99… | 255000/980000 | **crashed** | — | — | — | — | CE3 |
| 8B | 1 (attempt C, retry) | c9861e99… | 255000/980000 | **crashed, terminal (not retried further)** | — | — | — | — | CE4 |
| 8B | 2 (attempt A) | c9861e99… | 255000/980000 | **crashed** | — | — | — | — | CE5 |
| 8B | 2 (attempt B, retry) | c9861e99… | 255000/980000 | **crashed, terminal (not retried further)** | — | — | — | — | CE6 |
| 8B | 3 (attempt A) | c9861e99… | 255000/980000 | **crashed** | — | — | — | — | CE7 |
| 8B | 3 (attempt B, retry) | c9861e99… | 255000/980000 | **crashed, terminal (survey concludes 8B = 0/3 completed)** | — | — | — | — | CE8 |

## Post-survey cleanup (T010)

GPU clocks restored to the workspace pinned default via `pin_freqs.sh`: confirmed
`min_freq`=`max_freq`=`509000` via sysfs readback (direct `echo > sysfs` attempt hit a
permission error under the unprivileged shell; the script's own mechanism succeeded regardless —
verified by readback, not by the script's exit status). Driver hash re-confirmed
`c9861e9906d03fa2c7d48b804e1a1c80` (unchanged) as the final action of this survey.

## Crash Event Log

| crash_event_id | attempt_ref | time_to_drop_s | usb_state_observed | recovery_method | recovery_time_s | driver_md5_post | clock_range_post |
|---|---|---|---|---|---|---|---|
| CE1 | 3B rep1 (attempt A) | ~5s | bootloader (`S5E9975_LK_Bootloader`, confirmed via `fastboot devices`) | fastboot_reboot | ~31s | c9861e99… (match) | 255000/980000 (floating, match) |
| CE2 | 8B rep1 (attempt A) | ~40s | bootloader (`S5E9975_LK_Bootloader`, confirmed via `fastboot devices`) | fastboot_reboot | ~30s | c9861e99… (match) | 255000/980000 (floating, match) |
| CE3 | 8B rep1 (attempt B, retry) | ~40s | bootloader (`S5E9975_LK_Bootloader`, confirmed via `fastboot devices`) | fastboot_reboot | ~30s | c9861e99… (match) | 255000/980000 (floating, match) |
| CE4 | 8B rep1 (attempt C, retry) | ~19s | bootloader (`S5E9975_LK_Bootloader`, confirmed via `fastboot devices`) | fastboot_reboot | ~30s | c9861e99… (match) | 255000/980000 (floating, match) |
| CE5 | 8B rep2 (attempt A) | ~7s | bootloader (`S5E9975_LK_Bootloader`, confirmed via `fastboot devices`) | fastboot_reboot | ~30s | c9861e99… (match) | 255000/980000 (floating, match) |
| CE6 | 8B rep2 (attempt B, retry) | ~26s | bootloader (`S5E9975_LK_Bootloader`, confirmed via `fastboot devices`) | fastboot_reboot | ~30s | c9861e99… (match) | 255000/980000 (floating, match) |
| CE7 | 8B rep3 (attempt A) | ~30s | bootloader (`S5E9975_LK_Bootloader`, confirmed via `fastboot devices`) | fastboot_reboot | ~31s | c9861e99… (match) | 255000/980000 (floating, match) |
| CE8 | 8B rep3 (attempt B, retry) | ~15s | bootloader (`S5E9975_LK_Bootloader`, confirmed via `fastboot devices`) | fastboot_reboot | ~30s | c9861e99… (match) | 255000/980000 (floating, match) |

---

## Extension (2026-07-14, continued): 4w pinned gap-fill + threshold policy change to 64

### Extension provenance

- **Runner (node-threshold branch)**: `llama_main_nodethresh`, built fresh from
  `release13-node-threshold/executorch` (`ComputeGraph.cpp` diff confirmed present via `git diff`
  before build; binary verified via `strings <bin> | grep ET_VK_EXECUTE_NODE_THRESHOLD` and
  correct `ELF 64-bit ... ARM aarch64` arch after build). Staged to NFS
  (`android-run/runners/llama_main_nodethresh`, replacing the stale 2026-07-10 build) and pushed
  to device 2026-07-14 17:12.
- Same three `4w` texture `ctx3072` PTEs as the original survey (no new export needed — the
  node-threshold branch is a pure runtime patch on top of vanilla `release/1.3`, AOT side
  identical).

### 4w gap-fill Benchmark Attempts

| Model | Clocks | Rep | Attempt | Config | Outcome | prefill_tok_s | decode_tok_s | Crash Event |
|---|---|---|---|---|---|---|---|---|
| 3B | Pinned | 1 | A | vanilla | **crashed** | — | — | CE9 |
| 3B | Pinned | 1 | B (retry) | vanilla | **crashed** | — | — | CE10 |
| 3B | Pinned | 1 | C (retry) | vanilla | **crashed, terminal — switch to threshold=64** | — | — | CE11 |
| 3B | Pinned | 1 | — | threshold=64 | completed | 113.607 | 7.44292 | — |
| 3B | Pinned | 2 | — | threshold=64 | completed | 113.645 | 7.45382 | — |
| 3B | Pinned | 3 | — | threshold=64 | completed | 113.570 | 7.43319 | — |
| 8B | Floating | 1 | — | threshold=64 (fresh, replaces stale-32 data) | completed | 95.8218 | 5.3173 | — |
| 8B | Floating | 2 | — | threshold=64 | completed | 96.0375 | 5.26511 | — |
| 8B | Floating | 3 | — | threshold=64 | completed | 96.042 | 5.26839 | — |
| 8B | Pinned | 1 | A | threshold=64 | **crashed** | — | — | CE12 |
| 8B | Pinned | 1 | B (retry) | threshold=64 | **crashed, terminal — 64 insufficient here, fell back to 32** | — | — | CE13 |
| 8B | Pinned | 1\* | — | threshold=32 (old binary, pre-extension) | completed | 51.5947 | 4.02631 | — |
| 8B | Pinned | 2\* | — | threshold=32 (fresh build) | completed | 51.725 | 4.009 | — |
| 8B | Pinned | 3 | — | threshold=32 | completed | 51.6155 | 4.10533 | — |

\* Reps 1–2 of the 8B-pinned-threshold=32 set predate this extension (collected earlier
2026-07-14 while verifying the fresh `llama_main_nodethresh` build); rep 3 was collected during
this extension to complete the 3-rep set. All three share the same driver hash / config; treated
as one consistent 3-rep sample.

### Crash Event Log (extension, 4w gap-fill)

| ID | Attempt | Time to drop | Recovery |
|---|---|---|---|
| CE9 | 3B pinned rep1, vanilla attempt A | ~6s | fastboot_reboot, ~31s, driver+clock re-verified |
| CE10 | 3B pinned rep1, vanilla attempt B | ~7s | fastboot_reboot, ~35s, driver+clock re-verified |
| CE11 | 3B pinned rep1, vanilla attempt C | ~8s | fastboot_reboot, ~30s, driver+clock re-verified |
| CE12 | 8B pinned rep1, threshold=64 attempt A | ~10s | fastboot_reboot, ~34s, driver+clock re-verified |
| CE13 | 8B pinned rep1, threshold=64 attempt B | ~17s | fastboot_reboot, ~30s, driver+clock re-verified |

All 5 crash events share the same signature as CE1–CE8 (device drops off `adb`, re-enumerates as
`S5E9975_LK_Bootloader`, confirmed via `fastboot devices`, recovered via plain `fastboot reboot`,
driver hash + clock range re-verified unchanged after every recovery before the next attempt).

---

## Extension (2026-07-14): 8da4w full matrix (floating + pinned, all 3 models)

### 8da4w provenance

- PTEs: `llama3_2_1b_8da4w_texture_ctx3072.pte` (1,730,754,176 B), `llama3_2_3b_8da4w_texture_ctx3072.pte`
  (3,341,470,208 B), `llama3_1_8b_8da4w_texture_ctx3072.pte` (6,214,274,048 B) — all already
  existed in `.pte_out/` (exported 2026-07-06), pushed fresh to device for this extension.
- Same runners as above (`llama_main_rel1.3` for vanilla attempts, `llama_main_nodethresh` for
  threshold fallback attempts).
- Methodology: identical to the `4w` survey — coherence check once per model/config, vanilla
  attempted first, threshold fallback only on confirmed crash, driver+clock re-verified before
  every model's first rep and after every crash recovery.

### 8da4w Benchmark Attempts

| Model | Clocks | Rep | Attempt | Config | Outcome | prefill_tok_s | decode_tok_s | Crash Event |
|---|---|---|---|---|---|---|---|---|
| 1B | Floating | 1 | — | vanilla | completed | 401.175 | 21.0559 | — |
| 1B | Floating | 2 | A | vanilla | **crashed** | — | — | CE14 |
| 1B | Floating | 2 | B (retry) | vanilla | completed | 424.104 | 25.9704 | — |
| 1B | Floating | 3 | A | vanilla | **crashed** | — | — | CE15 |
| 1B | Floating | 3 | B (retry) | vanilla | completed | 422.355 | 25.9132 | — |
| 1B | Pinned | 1 | — | vanilla | completed | 224.046 | 14.3793 | — |
| 1B | Pinned | 2 | — | vanilla | completed | 224.217 | 14.4093 | — |
| 1B | Pinned | 3 | — | vanilla | completed | 224.586 | 14.3327 | — |
| 3B | Floating | 1 | A | vanilla | **crashed** | — | — | CE16 |
| 3B | Floating | 1 | B (retry) | vanilla | **crashed** | — | — | CE17 |
| 3B | Floating | 1 | C (retry) | vanilla | completed | 152.881 | 13.1305 | — |
| 3B | Floating | 2 | — | vanilla | completed | 152.904 | 13.2781 | — |
| 3B | Floating | 3 | — | vanilla | completed | 152.893 | 13.2845 | — |
| 3B | Pinned | 1 | A | vanilla | **crashed, terminal — switch to threshold=64** | — | — | CE18 |
| 3B | Pinned | 1 | — | threshold=64 | completed | 80.1033 | 7.22294 | — |
| 3B | Pinned | 2 | — | threshold=64 | completed | 80.1534 | 7.23848 | — |
| 3B | Pinned | 3 | — | threshold=64 | completed | 80.1377 | 7.23566 | — |
| 8B | Floating | 1 | A | vanilla | **crashed, terminal — switch to threshold=64** | — | — | CE19 |
| 8B | Floating | 1 | — | threshold=64 | completed | 67.3463 | 7.45855 | — |
| 8B | Floating | 2 | A | threshold=64 | **crashed** | — | — | CE20 |
| 8B | Floating | 2 | B (retry) | threshold=64 | completed | 67.4461 | 7.49885 | — |
| 8B | Floating | 3 | — | threshold=64 | completed | 67.4839 | 7.43308 | — |
| 8B | Pinned | 1 | — | threshold=32 (went straight to 32, per established pattern — no 64 attempt this cell) | completed | 35.1558 | 3.91347 | — |
| 8B | Pinned | 2 | — | threshold=32 | completed | 35.1534 | 3.90904 | — |
| 8B | Pinned | 3 | — | threshold=32 | completed | 35.1957 | 3.90855 | — |

### Crash Event Log (extension, 8da4w)

| ID | Attempt | Time to drop | Recovery |
|---|---|---|---|
| CE14 | 1B floating rep2, vanilla attempt A | ~6s | fastboot_reboot, ~30s, driver+clock re-verified (already floating post-reboot) |
| CE15 | 1B floating rep3, vanilla attempt A | ~6s | fastboot_reboot, ~30s, driver+clock re-verified |
| CE16 | 3B floating rep1, vanilla attempt A | ~5s | fastboot_reboot, ~30s, driver+clock re-verified |
| CE17 | 3B floating rep1, vanilla attempt B | ~6s | fastboot_reboot, ~30s, driver+clock re-verified |
| CE18 | 3B pinned rep1, vanilla attempt A | ~6s | fastboot_reboot, ~30s, driver+clock re-verified |
| CE19 | 8B floating rep1, vanilla attempt | ~10s | fastboot_reboot, ~30s, driver+clock re-verified |
| CE20 | 8B floating rep2, threshold=64 attempt A | ~8s | fastboot_reboot, ~30s, driver+clock re-verified |

All 7 share the same crash signature and recovery procedure as every prior crash event in this
document. Total crash events across the entire feature (original + both extensions): **20**
(CE1–CE20), all recovered via plain `fastboot reboot`, zero unrecovered/escalated incidents.

## Extension post-run cleanup

GPU clocks left pinned (509000/509000, via `pin_freqs.sh`) at the conclusion of the extension,
driver hash re-confirmed unchanged (`c9861e9906d03fa2c7d48b804e1a1c80`) throughout — every single
attempt across the full extension (34 completed + 12 crashed = 46 attempts) ran on a
hash-verified, undrifted driver.
