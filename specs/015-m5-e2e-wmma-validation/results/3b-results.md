# M5 EVT1 E2E Results — LLaMA 3.2 3B

Status as of 2026-07-06. Workload: 2048-token prefill / 1024-token decode,
pinned clocks (verified), driver `f14c51b6f8` (verified). 3-run means with
CoV; dispatch confirmed via a separate ETDump run before each e2e capture.

| Config | Prefill tok/s (mean, CoV) | Decode tok/s (mean, CoV) | Dispatch | Prior finding | Comparison |
|---|---|---|---|---|---|
| `4w` (linear coopmat) | **218.26** (0.822%) | 6.911 (1.548%) | `linear_q4gsw_coopmat` 196/196 confirmed | `quant-dev`'s 128x64-tile figure: 213.9 tok/s | **Directional** -- 218.26 vs 213.9 is +2.0%, same direction as 1B's comparison |
| `8da4w` (linear coopmat) | **200.91** (0.088%) | 6.748 (1.348%) | `linear_dq8ca_q4gsw_coopmat` 196/196 confirmed | **None** -- no prior M5 EVT1 `8da4w` baseline for 3B | **New measurement** |
| SDPA-coopmat | *(pending -- User Story 3)* | | | | |

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
