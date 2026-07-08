# M5 EVT1 E2E Results — LLaMA 3.2 3B

Status as of 2026-07-06. Workload: 2048-token prefill / 1024-token decode,
pinned clocks (verified), driver `f14c51b6f8` (verified). 3-run means with
CoV; dispatch confirmed via a separate ETDump run before each e2e capture.

| Config | Prefill tok/s (mean, CoV) | Decode tok/s (mean, CoV) | Dispatch | Prior finding | Comparison |
|---|---|---|---|---|---|
| `4w` (linear coopmat -- **restored, see UPDATE below**) | **218.26** (0.822%) | 6.911 (1.548%) | `linear_q4gsw_coopmat` **confirmed** -- direct throughput A/B (on 1B) + `specs/016` microbenchmark (all 3 models), not ETDump alone | `quant-dev`'s 128x64-tile figure: 213.9 tok/s | +2.0% vs prior, genuine coopmat both sides |
| `8da4w` (linear coopmat -- **restored**) | **200.91** (0.088%) | 6.748 (1.348%) | `linear_dq8ca_q4gsw_coopmat` **confirmed** (same evidence) | **T-tiled baseline: 79.83 tok/s (0.21% CoV), `specs/018-m5-8da4w-t-tiled-baseline`** | **2.52x speedup vs T-tiled baseline** (200.91 / 79.83) |
| `4w` + SDPA-coopmat (full-stack, `ET_VK_SDPA_COOPMAT=1`) | **333.97** (0.43%) | 6.69 | `sdpa_compute_attn_weights_coopmat`/`sdpa_compute_out_coopmat` -- confirmed genuine dispatch (bind-time capture matches ETDump on 1B; same build/mechanism); no crash | same doc, 512-prefill only, no exact tok/s | **Resolved 2026-07-06** -- see UPDATE below; previously blocked, now measured |
| `8da4w` + SDPA-coopmat (full-stack) | **286.31** (1.55%) | 6.447 (0.42%) | Same confirmed SDPA dispatch as above, combined with `8da4w` linear | **T-tiled baseline: 79.83 tok/s (0.21% CoV), `specs/018-m5-8da4w-t-tiled-baseline`** | **3.59x speedup vs T-tiled baseline** (286.31 / 79.83) |

## UPDATE (2026-07-06): previously-blocked SDPA row resolved -- was host-side OOM, not a GPU crash

The `VK_ERROR_DEVICE_LOST` blocker below is **not** a GPU/driver defect.
Once M5 EVT1 was available again, the identical config (`ET_VK_SDPA_COOPMAT=1`,
2048 prefill + full 1024 decode) was retried directly and did not crash at
all, at any `--max_new_tokens` from 64 to 1024. The *original* crash's
proper cause was found separately on 8B (same session): a genuine Android
OOM kill (confirmed via `dmesg`), caused by this workstream's own
accumulated on-device files (~29GB of staged PTEs/`.etdp` traces)
depleting available RAM. After cleanup and switching to `--warmup=false`,
this row's 3-rep headline completed cleanly. Full writeup:
`research.md` Decision 8, `.specify/memory/gotchas.md` G11.

## SDPA note (superseded by the UPDATE above -- kept for the historical record): blocked by a GPU device-lost crash, not measured

Unlike 1B (where `ET_VK_SDPA_COOPMAT=1` succeeded and gave a real +41%
prefill speedup via an unclear mechanism, see workspace `open-questions.md`
Q12), the identical config on 3B at the full 2048-prefill/1024-decode
workload crashes with `VK_ERROR_DEVICE_LOST`
(`vkQueueWaitIdle` returned -4) partway through decode. Per this feature's
watchdog policy, this was recorded as `blocked_reason` and not silently
retried at a shorter prefill/decode length. The device recovered cleanly
immediately afterward (a follow-up coherence check passed). No SDPA-coopmat
number is reported for 3B.

## UPDATE (2026-07-06): the CORRECTION below is itself superseded -- coopmat genuinely dispatches

Same reversal as `1b-results.md`: a direct wall-clock A/B against the
genuine `ET_VK_FORCE_TILED_LINEAR` kill switch (run on 1B, same build) and
`specs/016-m5-linear-sdpa-microbench`'s independent microbenchmark (own
kernel-name capture + SPIR-V + correctness, covering 3B directly) both
show coopmat genuinely dispatches for `linear_q4gsw`/`linear_dq8ca_q4gsw`
on this build. ETDump's per-event kernel-name field, which the CORRECTION
below relied on, is unreliable in the full LLaMA graph context -- a
tooling bug, not a dispatch bug. `dispatch_status = confirmed` restored.
Full writeup: `research.md` Decision 7's reversal, `open-questions.md`
Q11's "二次反转" addendum.

## CORRECTION (2026-07-06, SUPERSEDED BY THE UPDATE ABOVE -- kept for the historical record)

Same finding as `1b-results.md`: re-verification (independent binary + original
`llama_main_etdump_spec015`) found both 3B linear schemes dispatch 100% `_tiled`,
not `_coopmat`, despite the C++ eligibility gate passing. Root cause open --
`.shared-context/report-for-human/open-questions.md` Q11. The tok/s numbers are
real hardware measurements of the tiled path, not evidence of coopmat/WMMA speedup.

## Important note: both PTEs were re-exported (`research.md` Decision 6)

Same root-cause as 1B: the pre-existing `llama3_2_3b_4w_buffer_ctx3072.pte`
(dated 2026-06-17) was internally `Texture3D` despite its name, produced
with a broken export mechanism. Re-exported with
`backend.vulkan.storage_override: buffer`; dispatch confirmed clean
(196/196 coopmat calls for both schemes = 28 layers x 7 linear ops).

## `4w` vs `8da4w` at this model size

Unlike 1B (where `8da4w` was slower than `4w`), the two schemes are much
closer at 3B (200.91 vs 218.26, `8da4w` still slower but by a smaller
relative margin: -7.9% vs 1B's -8.6%). Consistent direction with 1B (both
show `4w` faster on this repo's current shader), still the *opposite*
direction from the one existing `8da4w`-beats-`4w` data point at 8B.
Deferred to the consolidated report until 8B's own numbers exist.
