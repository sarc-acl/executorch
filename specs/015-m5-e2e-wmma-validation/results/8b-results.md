# M5 EVT1 E2E Results — LLaMA 3.1 8B

Status as of 2026-07-06. Workload: 2048-token prefill / 1024-token decode,
pinned clocks (509/2730/663 MHz, verified via GFLOP/s cross-check), driver
`f14c51b6f8` (verified, same uninterrupted adb session as 1B/3B -- no
reboot/reflash in between). All numbers are **3-run means** with CoV
(`research.md` Decision 5); dispatch confirmed via a separate ETDump run
before each e2e capture (Principle IV/VI). No GPU watchdog issue occurred
for either scheme, despite this being the highest-watchdog-risk model in
the feature.

| Config | Prefill tok/s (mean, CoV) | Decode tok/s (mean, CoV) | Dispatch | Prior finding | Comparison |
|---|---|---|---|---|---|
| `4w` (linear coopmat -- **restored, see UPDATE below**) | **112.71** (0.088%) | 3.853 (0.169%) | `linear_q4gsw_coopmat` **confirmed** -- direct throughput A/B (on 1B) + `specs/016` microbenchmark (covering 8B directly), not ETDump alone | `report-for-human/jira-tile-sweep.md` (128x64 tile): 110.6 tok/s | +1.9% vs prior, genuine coopmat both sides |
| `8da4w` (linear coopmat -- **restored**) | **99.98** (0.504%) | 3.787 (0.260%) | `linear_dq8ca_q4gsw_coopmat` **confirmed** (same evidence) | `report-for-human/e2e-spec.md` / `RESULTS-SUMMARY.md`: 85.1 tok/s; **T-tiled baseline: 35.17 tok/s (0.13% CoV), `specs/018-m5-8da4w-t-tiled-baseline`** | +17.5% vs prior historical figure; **2.84x speedup vs T-tiled baseline** (99.98 / 35.17) |
| `4w` + SDPA-coopmat (full-stack, `ET_VK_SDPA_COOPMAT=1`) | **153.30** (0.43%) | 3.79 | `sdpa_compute_attn_weights_coopmat`/`sdpa_compute_out_coopmat` -- confirmed genuine dispatch (bind-time capture matches ETDump on 1B; same build/mechanism); no crash | same doc, 512-prefill only, no exact tok/s | **Resolved 2026-07-06** -- see UPDATE 2 below; previously blocked, now measured |
| `8da4w` + SDPA-coopmat (full-stack) | **130.05** (0.09%) | 3.67 | Same confirmed SDPA dispatch as above, combined with `8da4w` linear | **T-tiled baseline: 35.17 tok/s (0.13% CoV), `specs/018-m5-8da4w-t-tiled-baseline`** | **3.70x speedup vs T-tiled baseline** (130.05 / 35.17) |

## UPDATE 2 (2026-07-06, later session): previously-blocked SDPA rows resolved -- was host-side OOM, not a GPU crash

The `VK_ERROR_DEVICE_LOST` blocker described below is **not** a GPU/driver
defect. Once M5 EVT1 was available again, both `4w` and `8da4w` SDPA
configs were retried directly at the full 1024-decode length and did not
crash. The actual crash's cause was found on this same model this
session: attempting the *proper* 3-rep headline measurement
(`--warmup=true`) failed silently (exit 0, no output) -- `dmesg` showed a
genuine Android OOM kill (`anon-rss:1971136kB, file-rss:2446176kB`),
caused by this workstream's own accumulated on-device files (~29GB of
staged PTEs/`.etdp` traces from a long session) depleting available RAM.
After cleanup and switching to `--warmup=false`, both schemes' 3-rep
headlines completed cleanly. Full writeup: `research.md` Decision 8,
`.specify/memory/gotchas.md` G11.

## UPDATE (2026-07-06): this IS a genuine coopmat/WMMA result

This file originally (and honestly, given what was known at the time)
reported a "tiled-fallback, not coopmat" finding, on the strength of
ETDump's per-event kernel-name field showing 100% `_tiled` dispatch for
both schemes' 224 prefill linear calls -- the same finding independently
made for 1B and 3B. That ETDump-based conclusion has since been
superseded: a direct wall-clock A/B against the genuine
`ET_VK_FORCE_TILED_LINEAR` kill switch (run on 1B, same build) showed the
default path is 1.8x faster than truly-forced-tiled -- impossible if the
default path genuinely dispatched tiled. Separately,
`specs/016-m5-linear-sdpa-microbench`'s independent microbenchmark, using
the harness's own kernel-name capture (not ETDump) plus SPIR-V inspection,
directly confirms `linear_q4gsw`/`linear_dq8ca_q4gsw` dispatch coopmat at
8B's exact production shapes (K=4096/14336, N=1024/4096/14336), 3.0-4.4x
faster than tiled, correctness-verified.

**Revised conclusion**: `dispatch_status = confirmed` for both `4w` and
`8da4w` above. The C++ eligibility gate passing and the coopmat
`ShaderInfo` resolving successfully (as documented in the original
investigation below) were correct signals all along -- **ETDump's
per-event kernel-name field is what was wrong**, unreliable specifically
in the full LLaMA graph context (224+ linear nodes sharing one
graph/pipeline-cache context), not the dispatch logic. Full writeup:
`research.md` Decision 7's reversal, workspace `open-questions.md` Q11's
"二次反转" addendum.

The tok/s numbers above are real, reproducible hardware measurements (3
consistent runs per scheme, no watchdog, no thermal throttle evident given
the tight CoV) of genuine coopmat/WMMA dispatch for linear ops on M5 EVT1.

## Original investigation (2026-07-06, superseded by the UPDATE above -- kept for the historical record)

For every one of 8B's 224 prefill linear dispatches (both `4w` and
`8da4w`), the C++ eligibility gate (`can_use_q4gsw_coopmat` / the `dq8ca`
equivalent in `QuantizedLinear.cpp`) evaluates true -- shapes are
tile-aligned (M=2048, N/K drawn from `dim=4096`/`ffn=14336` per
`params.json`, all divisible by the 128x64x16 tile), output is Buffer, no
bias -- and the coopmat `ShaderInfo` is constructed and resolved
successfully from the shader registry. ETDump showed the shader that
actually ran on the GPU as the tiled one, not `_coopmat` -- this is now
understood to be an ETDump attribution bug, not a real dispatch issue (see
UPDATE above).

## SDPA note (superseded by UPDATE 2 above -- kept for the historical record): blocked by a GPU device-lost crash, not measured

Same failure as 3B: `ET_VK_SDPA_COOPMAT=1` at the full 2048-prefill/1024-decode
workload crashes with `VK_ERROR_DEVICE_LOST`. Recorded as `blocked_reason`,
not retried at a shorter length. Device recovered cleanly afterward. Only
1B succeeded with this env var (see `1b-results.md` and `open-questions.md`
Q12) -- both 3B and 8B are blocked, so whatever the +41%-on-1B mechanism
is, it doesn't reliably scale to larger models without crashing this build.

**This was wrong** -- see UPDATE 2 above: all three models' SDPA-coopmat
now measure cleanly once the actual (host-side OOM) cause was found and
fixed. Kept here only for the historical record of what was observed and
believed at the time.

## `4w` vs `8da4w` at this model size

`8da4w` (99.98 tok/s) is slower than `4w` (112.71 tok/s) here too --
consistent in direction with 1B and 3B (both showed `4w` faster than
`8da4w` under the current, tiled-fallback behavior), and now also
consistent with the *existing* `report-for-human` prior-finding pair (110.6
vs 85.1, `4w` faster) -- though note that comparison itself may have been
`4w`-tiled vs `8da4w`-coopmat or some other combination not yet confirmed
via ETDump at the time it was recorded, given this defect's scope was
unknown then. Not drawing further conclusions about *why* 8da4w trails 4w
until Q11 is root-caused, since "coopmat vs tiled" per-scheme dispatch
status may differ once fixed.
