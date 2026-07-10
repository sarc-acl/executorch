# M5 EVT1 E2E Results — LLaMA 3.2 1B

Status as of 2026-07-06. Workload: 2048-token prefill / 1024-token decode,
pinned clocks (509/2730/663 MHz, verified via GFLOP/s cross-check), driver
`f14c51b6f8` (verified). All numbers are **3-run means** with CoV
(`research.md` Decision 5); dispatch confirmed via a separate ETDump run
before each e2e capture (Principle IV/VI).

| Config | Prefill tok/s (mean, CoV) | Decode tok/s (mean, CoV) | Dispatch | Prior finding | Comparison |
|---|---|---|---|---|---|
| `4w` (linear coopmat -- **restored, see UPDATE below**) | **583.70** (0.271%) | 14.273 (0.366%) | `linear_q4gsw_coopmat` **confirmed** -- direct throughput A/B + `specs/016` microbenchmark, not ETDump alone (see UPDATE) | `quant-dev`'s 128x64-tile figure: 565.3 tok/s (`report-for-human/jira-tile-sweep.md`) | **Directional** -- +3.3% over prior, consistent direction, genuine coopmat both sides |
| `8da4w` (linear coopmat -- **restored**) | **533.44** (0.536%) | 13.745 (0.680%) | `linear_dq8ca_q4gsw_coopmat` **confirmed** (same evidence) | **T-tiled baseline: 222.30 tok/s (0.28% CoV), `specs/018-m5-8da4w-t-tiled-baseline`** | **2.40x speedup vs T-tiled baseline** (533.44 / 222.30) |
| `4w` + SDPA-coopmat (full-stack, `ET_VK_SDPA_COOPMAT=1`) | **812.59** (0.02%, prior session) / **769.35** (6.87%, fresh re-measurement 2026-07-06) | 14.155 (0.14%) / 13.60 (fresh) | **Confirmed directly** -- `ET_VK_DEBUG_ENCODE_DISPATCH` bind-time capture matches ETDump exactly (16/16 `sdpa_compute_attn_weights_coopmat`/`sdpa_compute_out_coopmat`), see UPDATE 2 below | `report-for-human/session-2026-06-23-sdpa-wmma-findings.md`: 763 tok/s (combined WMMA stack, not SDPA-isolated) | Real +41% vs the env-var-off baseline (577); dispatch now directly confirmed, not inferred |
| `8da4w` + SDPA-coopmat (full-stack) | **723.00** (0.27%, 2026-07-06) | 12.83 | Same confirmed SDPA dispatch mechanism as above, combined with `8da4w` linear | **T-tiled baseline: 222.30 tok/s (0.28% CoV), `specs/018-m5-8da4w-t-tiled-baseline`** | **3.25x speedup vs T-tiled baseline** (723.00 / 222.30) |

## UPDATE 2 (2026-07-06, later session): SDPA dispatch directly confirmed; fresh re-measurement flags a variance caveat

`ET_VK_DEBUG_ENCODE_DISPATCH` (built earlier, run on-device for the first
time this session) captured the bind-time `kernel_name` for a short 1B
SDPA-coopmat run and compared it directly against the same run's `.etdp`:
both show `sdpa_compute_attn_weights_coopmat_buffer_buffer_half` /
`sdpa_compute_out_coopmat_buffer_buffer_half`, 16/16, in exact agreement.
This is now **direct** evidence, not an inference from Q11's linear
finding -- `dispatch_status = confirmed` for SDPA/1B.

A fresh 3-rep headline re-measurement this same session (`--warmup=false`,
for consistency with the memory-pressure fix applied to 3B/8B -- see
`research.md` Decision 8 and `.specify/memory/gotchas.md` G11) gave
769.35 tok/s mean, but with **6.87% CoV** (range 718.3-823.8 tok/s) --
notably higher than every other config measured this session (all <2%)
and than this same config's own prior-session figure (812.4-812.7 tok/s,
CoV ~0.03%). Not yet attributed to a specific cause; flagged rather than
silently averaged over. Both numbers are retained in the table above
rather than picking one, since neither has been invalidated -- they were
measured under different methodology (`warmup=true` vs `false`) and
different points in a long device session.

## SDPA note: real speedup, likely genuine coopmat (Q12, re-evaluated 2026-07-06)

`ET_VK_SDPA_COOPMAT=1` gives a real, reproducible, immediately-reversible
+41% prefill speedup on 1B (577 -> 812 tok/s, confirmed via an A-B-A-B
alternating test in one uninterrupted adb session, clock pin re-verified
unchanged at 509MHz throughout). ETDump showed the attention shader
dispatched as still `_tiled` by name in both cases -- but per the UPDATE
above, ETDump's per-event kernel-name field is now known unreliable in
this full-graph context, so that observation is no longer strong evidence
of anything. `specs/016`'s independent SDPA microbenchmark (own
kernel-name capture, not ETDump) confirms `sdpa_compute_attn_weights_coopmat`/
`sdpa_compute_out_coopmat` genuinely dispatch on this build with a 75-82%
shader-level speedup -- consistent in direction with this e2e +41%
(smaller, as expected from Amdahl dilution at the full-model level). Most
likely explanation: this IS genuine coopmat SDPA, mis-attributed by the
same ETDump bug as the linear case. See workspace `open-questions.md` Q12
for the full A/B methodology, ruled-out confounders
(`ET_VK_EXECUTE_NODE_THRESHOLD`, clock-pin drift), and this re-evaluation.

## UPDATE (2026-07-06): the CORRECTION below is itself superseded -- coopmat genuinely dispatches

The "CORRECTION" section below (based on ETDump's per-event kernel-name
field) is **wrong**. Two independent pieces of evidence, neither relying
on ETDump's per-event field, now show coopmat genuinely dispatches for
these numbers:

1. **Direct wall-clock A/B on this exact e2e path**: `ET_VK_FORCE_TILED_LINEAR=1`
   (a real, source-confirmed kill switch) vs default, A-B-A-B alternating
   on this same 1B/`4w` PTE and prompt: default 576.7/577.1 tok/s,
   genuinely-forced-tiled 321.0/321.3 tok/s. Default is 1.8x faster than
   forced-tiled, and 321 matches the historical T-tiled baseline (312.7)
   closely -- if the default path really dispatched tiled, it could not
   be 1.8x faster than a build that is *actually* forced to tiled.
2. **`specs/016-m5-linear-sdpa-microbench`'s independent microbenchmark**
   on this same build: `linear_q4gsw`/`linear_dq8ca_q4gsw` dispatch
   coopmat 3.04x/4.16x faster than tiled at production shapes, confirmed
   via the harness's own kernel-name capture (not ETDump) plus SPIR-V
   inspection and existing correctness coverage.

**Revised conclusion**: `dispatch_status = confirmed` for both rows above.
ETDump's per-event kernel-name field is unreliable in the full LLaMA graph
context -- a tooling bug, not a dispatch bug. Full writeup: `research.md`
Decision 7's reversal, workspace `open-questions.md` Q11's "二次反转"
addendum. The SDPA row's `+41%` finding (Q12) likely has the same
explanation, though not independently re-confirmed the way linear was.

## CORRECTION (2026-07-06, SUPERSEDED BY THE UPDATE ABOVE -- kept for the historical record)

Both rows above originally claimed `linear_q4gsw_coopmat`/`linear_dq8ca_q4gsw_coopmat`
"confirmed" dispatch. Re-verification during 8B's dispatch-confirm step (T031-T034)
found this was wrong: ETDump's actual per-kernel breakdown for both 1B schemes shows
100% `_tiled` dispatch, not `_coopmat`. This was re-checked independently with the
original `llama_main_etdump_spec015` binary and a freshly-rebuilt diagnostic binary,
both agreeing. The eligibility gate (`can_use_q4gsw_coopmat` in `QuantizedLinear.cpp`)
evaluates to true (shape-aligned, buffer output, no bias) and the coopmat `ShaderInfo`
is successfully constructed and resolved from the shader registry -- yet the shader
that actually executes on the GPU is the tiled one. Root cause not yet located; full
investigation and exclusion list in workspace `.shared-context/report-for-human/open-questions.md`
Q11. The tok/s numbers above are real, reproducible hardware measurements (3-run
means, matching across two independent binaries) -- they just measure the tiled path,
not coopmat/WMMA, so they are **not** evidence of this workstream's coopmat speedup on
M5 EVT1 for linear ops. Same finding holds for 3B and (per `8b-results.md`) 8B.

## Notable finding this model surfaced (export mechanism -- separate from the correction above)

Both PTEs used here were exported (1B/`4w`: **re**-exported; 1B/`8da4w`:
newly exported) using this repo's actual `backend.vulkan.storage_override:
buffer` mechanism -- **not** `.shared-context/scripts/export_quant.sh`'s
documented `ET_VK_FORCE_BUFFER` env var, which does not exist anywhere in
this repo's source and silently produces an internally-`Texture3D` PTE
despite the "buffer" filename. See `research.md` Decision 6 for the full
story (this was caught by User Story 1's dispatch-confirm step before any
number was trusted, and is why 3B/8B's existing `4w` "buffer" PTEs must
also be re-exported, not reused). This bug is independent of, and was fixed
before, the tiled-fallback finding above.

## `4w` vs `8da4w` at this model size

`8da4w`'s prefill (533.44) is *slower* than `4w`'s (583.70) on 1B, the
opposite of the direction the existing `8da4w` 8B point of comparison
showed (`8da4w` beating `4w` at 8B, per `report-for-human/e2e-spec.md`).
Not yet enough evidence to call this a real model-size-dependent effect
vs. noise/measurement variance -- flagged here for the consolidated report
(User Story 4) to address once 3B/8B data exists for both schemes.
