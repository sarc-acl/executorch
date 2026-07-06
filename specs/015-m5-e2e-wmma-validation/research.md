# Research: M5 EVT1 End-to-End WMMA Validation

## Decision 1: Export from this repo's own venv; reuse existing `4w` PTEs, export only `8da4w`

**Decision**: The four `4w` `.pte`s already in `.pte_out/`
(`llama3_1_8b_4w_{texture,buffer}_ctx3072.pte`,
`llama3_2_1b_4w_buffer_ctx3072.pte`, `llama3_2_3b_4w_buffer_ctx3072.pte`)
are reused as-is. The three missing `8da4w` buffer PTEs (1B, 3B, 8B) are
exported fresh, using `.shared-context/scripts/export_quant.sh 8da4w 128
buffer` run from **this repo's own venv** (`quant-perf-optimization/
executorch/.venv`), not the `quant-dev` worktree `export-pte.md`'s
examples `cd` into.

**Grounding**: `export-pte.md` documents export as pure-Python AOT
(quantization scheme + graph construction) with no dependency on the
Vulkan runtime/shader code this workstream has been changing -- "no NDK,
no glslc, no Vulkan SDK needed... those are only for building runtime
binaries." Confirmed directly: `python -c "import
executorch.extension.llm.export.export_llm"` succeeds in this repo's own
`.venv` (editable-installed, dated 2026-06-30). Storage type
(`ET_VK_FORCE_BUFFER`) is the only export-time knob that matters for
coopmat eligibility; it is independent of which worktree runs the export.

**Rationale**: Avoids re-exporting four multi-GB `.pte` files that already
exist and are already known-good (the 4w buffer ones were almost certainly
what produced the correctness-validated coopmat dispatch in `specs/014`'s
own T009 run, which read production shapes from a live model context).

**Alternatives considered**: Re-exporting all six from scratch (rejected --
no reason to believe the existing 4w PTEs are stale, since export doesn't
embed shader code; re-exporting would only cost device-independent
CPU/RAM time for no new information). Exporting from the `quant-dev`
worktree per `export-pte.md`'s literal examples (rejected -- unnecessary
cross-worktree dependency when this repo's own venv already works).

## Decision 2: Build and push this repo's own `llama_main`, never the `_origcm` runners

**Decision**: All e2e measurement uses `cmake-out-android-vk/examples/
models/llama/llama_main` (already built in this repo, reflecting the
current `vulkan_backend` with the 128x64 tile and all three `specs/014`
shader changes) plus a freshly-built ETDump variant via
`build_etdump_android.sh`. The `llama_main_origcm`/`llama_main_etdump_origcm`
runners referenced in `.shared-context/instruction-for-ai/commands.md`'s
example commands are explicitly NOT used.

**Grounding**: Per workspace-root `CLAUDE.md`, `_origcm` runners were built
in the `.tmp-origcm` worktree, pinned at a different, older commit ("our
coopmat", pre-dbuf4) -- a different, independently-evolved codebase from
this repo, per the same reasoning already established in `specs/014`'s own
research.md Decision 1 (why `quant-dev`'s numbers aren't this repo's
baseline either). Using an `_origcm` runner would silently measure the
wrong shader entirely.

**Rationale**: The whole point of this feature is measuring *this repo's*
code on M5 EVT1; any prebuilt runner from a different worktree defeats
that purpose regardless of how convenient the `commands.md` examples make
it look.

**Alternatives considered**: Using `_origcm` runners for speed (rejected --
would measure the wrong code, silently).

## Decision 3: Sequence 1B → 3B → 8B, report incrementally

**Decision**: Per explicit user instruction during planning, work proceeds
in strict model order 1B, then 3B, then 8B -- for both linear (`4w`/
`8da4w`) and SDPA-coopmat -- with each model's results published
(`results/<model>-results.md`) as soon as that model's measurements
complete, not held back for a single final report.

**Grounding**: 1B has the fewest transformer layers and lowest per-token
compute of the three models, so it carries the lowest risk of the known
GPU-watchdog issue that previously blocked 8B/3B at 2048-token prefill
(`.shared-context/report-for-human/session-2026-06-23-sdpa-wmma-findings.md`,
jira `#001`). Validating the full measurement pipeline (export → build →
deploy → dispatch-confirm → measure) on the lowest-risk model first, and
reporting it immediately, means a watchdog recurrence on 8B/3B doesn't
block the user from seeing any results.

**Rationale**: Matches this workstream's own established practice (every
prior feature proves its mechanism on one configuration before scaling --
`001`/`004`/`007`/`008`/`009` all did this), specialized here to the
user's explicit ordering and incremental-reporting instruction.

**Alternatives considered**: Grouping by scheme (`4w` for all 3 models,
then `8da4w`, then SDPA) -- rejected per the user's explicit instruction to
sequence by model (risk order), not by scheme.

## Decision 4: Re-verify driver identity at the start of this feature, not trust `specs/014`'s end-of-session state

**Decision**: Before any dispatch-confirmation or e2e run, re-check
`/vendor/lib64/hw/vulkan.samsung.so`'s md5 against the known-good table in
`flash-sumd-driver.md`, even though `specs/014`'s session ended with it
confirmed on known-good `f14c51b6f8`.

**Grounding**: Constitution Principle VIII: "never assume a prior
session's driver is still there" -- the board is shared and this exact
scenario (drift between sessions) already happened once in `specs/014`'s
own session.

**Rationale**: Cheap to check (one `md5sum`), and the cost of skipping it
and being wrong (a silent miscompile, per the Q9 precedent) is severe.

**Alternatives considered**: Trusting the last-known state (rejected --
directly contradicts Principle VIII and this exact workstream's own recent
history).
