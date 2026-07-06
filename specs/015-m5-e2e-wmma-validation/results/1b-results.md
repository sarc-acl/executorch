# M5 EVT1 E2E Results — LLaMA 3.2 1B

Status as of 2026-07-06. Workload: 2048-token prefill / 1024-token decode,
pinned clocks (509/2730/663 MHz, verified via GFLOP/s cross-check), driver
`f14c51b6f8` (verified). All numbers are **3-run means** with CoV
(`research.md` Decision 5); dispatch confirmed via a separate ETDump run
before each e2e capture (Principle IV/VI).

| Config | Prefill tok/s (mean, CoV) | Decode tok/s (mean, CoV) | Dispatch | Prior finding | Comparison |
|---|---|---|---|---|---|
| `4w` (linear coopmat) | **583.70** (0.271%) | 14.273 (0.366%) | `linear_q4gsw_coopmat` 112/112 confirmed | `quant-dev`'s 128x64-tile figure: 565.3 tok/s (`report-for-human/jira-tile-sweep.md`) | **Directional** -- this repo's shader has `specs/014`'s extra changes on top of the same tile; 583.70 vs 565.3 is a +3.3% difference, consistent in direction (not a regression), not a claimed reproduction |
| `8da4w` (linear coopmat) | **533.44** (0.536%) | 13.745 (0.680%) | `linear_dq8ca_q4gsw_coopmat` 112/112 confirmed | **None** -- no prior M5 EVT1 `8da4w` baseline exists for 1B (only 8B has one, per `data-model.md`) | **New measurement**, not a reproduction |
| SDPA-coopmat | *(pending -- User Story 3, T038-T040)* | | | | |

## Notable finding this model surfaced

Both PTEs used here were exported (1B/`4w`: **re**-exported; 1B/`8da4w`:
newly exported) using this repo's actual `backend.vulkan.storage_override:
buffer` mechanism -- **not** `.shared-context/scripts/export_quant.sh`'s
documented `ET_VK_FORCE_BUFFER` env var, which does not exist anywhere in
this repo's source and silently produces an internally-`Texture3D` PTE
despite the "buffer" filename. See `research.md` Decision 6 for the full
story (this was caught by User Story 1's dispatch-confirm step before any
number was trusted, and is why 3B/8B's existing `4w` "buffer" PTEs must
also be re-exported, not reused).

## `4w` vs `8da4w` at this model size

`8da4w`'s prefill (533.44) is *slower* than `4w`'s (583.70) on 1B, the
opposite of the direction the existing `8da4w` 8B point of comparison
showed (`8da4w` beating `4w` at 8B, per `report-for-human/e2e-spec.md`).
Not yet enough evidence to call this a real model-size-dependent effect
vs. noise/measurement variance -- flagged here for the consolidated report
(User Story 4) to address once 3B/8B data exists for both schemes.
