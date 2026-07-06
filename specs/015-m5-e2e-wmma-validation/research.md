# Research: M5 EVT1 End-to-End WMMA Validation

## Decision 1: Export from this repo's own venv; reuse existing `4w` PTEs, export only `8da4w`

**Decision**: The three `4w` **Buffer**-storage `.pte`s already in
`.pte_out/` (`llama3_1_8b_4w_buffer_ctx3072.pte`,
`llama3_2_1b_4w_buffer_ctx3072.pte`, `llama3_2_3b_4w_buffer_ctx3072.pte`
-- one per model) are reused as-is. Matching `Texture3D` exports also
exist for the same three models (`.pte_out/` has six `4w` files total,
confirmed by direct `ls`), but this feature never uses them -- coopmat/
WMMA dispatch requires `Buffer` storage, and no task stages, pushes, or
runs a texture PTE. **Correction (found during `/speckit-analyze`)**: an
earlier draft of this decision undercounted the existing files as "four"
and omitted `llama3_2_1b`/`llama3_2_3b`'s texture variants from its own
list -- the count is fixed here; it never affected which files this
feature actually uses (always the three Buffer ones), only this
document's own bookkeeping accuracy. The three missing `8da4w` buffer
PTEs (1B, 3B, 8B) are exported fresh, using
`.shared-context/scripts/export_quant.sh 8da4w 128 buffer` run from
**this repo's own venv** (`quant-perf-optimization/executorch/.venv`),
not the `quant-dev` worktree `export-pte.md`'s examples `cd` into.

**Grounding**: `export-pte.md` documents export as pure-Python AOT
(quantization scheme + graph construction) with no dependency on the
Vulkan runtime/shader code this workstream has been changing -- "no NDK,
no glslc, no Vulkan SDK needed... those are only for building runtime
binaries." Confirmed directly: `python -c "import
executorch.extension.llm.export.export_llm"` succeeds in this repo's own
`.venv` (editable-installed, dated 2026-06-30). Storage type
(`ET_VK_FORCE_BUFFER`) is the only export-time knob that matters for
coopmat eligibility; it is independent of which worktree runs the export.

**Rationale**: Avoids re-exporting three multi-GB `.pte` files that already
exist and are already known-good (the 4w buffer ones were almost certainly
what produced the correctness-validated coopmat dispatch in `specs/014`'s
own T009 run, which read production shapes from a live model context).

**Alternatives considered**: Re-exporting all six `4w` files (three Buffer
+ three Texture3D) from scratch (rejected -- no reason to believe the
existing PTEs are stale, since export doesn't embed shader code, and the
Texture3D half is never used by this feature regardless; re-exporting
would only cost device-independent CPU/RAM time for no new information).
Exporting from the `quant-dev` worktree per `export-pte.md`'s literal
examples (rejected -- unnecessary cross-worktree dependency when this
repo's own venv already works).

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

## Decision 5 (added during `/speckit-analyze`): 3-run mean + CoV per configuration; verify the clock pin bound, don't just command it

**Decision**: Every e2e prefill/decode capture is **3 repeated runs**,
reporting the mean and coefficient of variation (CoV), not a single-shot
run. Before any of those runs, `pin_freqs.sh` is run once per session and
its effect is verified via a GFLOP/s-or-tok/s cross-check against an
equivalently-pinned microbenchmark (constitution Principle VII) -- not
just trusted because the pin command exited successfully.

**Grounding**: This feature's own first plan/tasks draft specified a
single run per configuration and never invoked `pin_freqs.sh` at all --
found by `/speckit-analyze` as a CRITICAL gap against Principle VII (which
explicitly requires pin verification, not just pinning) and a HIGH gap
against this workstream's own established e2e methodology:
`.shared-context/report-for-human/e2e-spec.md` states its headline 4w/
8da4w numbers are "3-run means" with CoV reported (e.g. "4w: 79.3 (CoV
0.05%)"), and the `results_ctx3072/logs/*_rep{1,2,3}.log` naming
convention throughout `report-for-human/`'s archives confirms this has
been the actual practice, not a one-off.

**Rationale**: Constitution Principle VII's own rationale names the exact
failure mode this closes: a prior session on this same board reported a
~980MHz DVFS-boost number as if it were the intended 509MHz pin (Q10),
caught only by a GFLOP/s cross-check, not by the pin command appearing to
succeed. A single-run capture is likewise exactly the failure mode
Principle IV's tier-1 discipline (iteration count + stddev) already
guards against at the microbenchmark tier; there is no reason tier-2 e2e
numbers should be held to a lower bar than this workstream already holds
tier-1 numbers to, especially given `e2e-spec.md` shows 3-run reporting
was already the norm before this feature existed.

**Alternatives considered**: Single-run capture, citing time cost
(rejected -- a full 2048-prefill/1024-decode run is the expensive part
regardless; 3 reps roughly triples wall-clock time per configuration but
is what this workstream's own prior numbers were actually built on, and
reporting a number this workstream wouldn't otherwise trust defeats the
feature's purpose). Skipping pin verification and trusting the command
(rejected -- directly contradicts Principle VII and the Q10 precedent).

## Decision 6 (found during implementation, US1): the venv was non-editable AND `ET_VK_FORCE_BUFFER` doesn't exist in this repo -- every existing `4w` "buffer" PTE was actually Texture3D internally

**Decision**: Fixed this repo's venv (`pip install -e . --no-build-isolation`
-- it had been installed non-editable, physically copying a stale
2026-06-30 snapshot into `site-packages` instead of linking to live repo
source, per `build.md`'s own documented gotcha). Re-exported all `4w`
buffer PTEs using `backend.vulkan.storage_override: buffer` in `config.yaml`
(equivalently `--vulkan-storage-override=buffer` on the CLI) -- **not**
`export-pte.md`'s documented `ET_VK_FORCE_BUFFER` env var, which does not
exist anywhere in this repo's Python source (confirmed by
`grep -rl ET_VK_FORCE_BUFFER` across the whole tree -- zero hits outside
this feature's own docs). `storage_override` is this repo's own,
already-implemented mechanism (`extension/llm/export/partitioner_lib.py`
`get_vulkan_partitioner(storage_override=...)`, added for
`specs/006-e2e-storage-comparison`).

**Grounding**: User Story 1's dispatch-confirmation step (the entire reason
this feature does US1 before trusting any number) caught this directly.
The pre-existing `llama3_2_1b_4w_buffer_ctx3072.pte` (dated 2026-06-30)
produced an ETDump trace where `linear_q4gsw_tiled_texture3d_texture2d_half`
dispatched 112/112 times and every other op in the main graph (rms_norm,
binary_mul, sigmoid, rotary_embedding, view) showed `_texture3d_half` --
only SDPA (which has its own separate buffer-forcing logic) showed
`_buffer`. Re-exporting with the current (editable) venv but still via
`export_quant.sh`'s `ET_VK_FORCE_BUFFER=1` produced byte-for-byte the same
result -- proving the env var itself does nothing here, not that the venv
staleness was the (sole) cause. Only exporting via `storage_override:
buffer` in `config.yaml` produced a PTE where the export log's "Operators
included in this Vulkan partition" no longer shows the flood of
`TensorRepr(TEXTURE_3D) -> TensorRepr(BUFFER)` transitions the broken
exports logged, and the resulting ETDump trace shows
`linear_q4gsw_coopmat_buffer_texture2d_half` dispatching all 112/112 times,
every other main-graph op as `_buffer_half`, and total leaf GPU time
dropping from ~6.5-6.7ms to 3.67ms (prefill tok/s 303-312 -> 553.8).

**Rationale**: `.shared-context/instruction-for-ai/export-pte.md` is
written from and for the `quant-dev` worktree, which has its own
env-var-based storage-override wrapper around the partitioner that this
repo never had (this repo instead kept the original, more direct
`--vulkan-storage-override` CLI flag / `backend.vulkan.storage_override`
config field it was presumably forked from, before `quant-dev` added its
own env-var convenience layer on top). Per constitution Principle X,
`export-pte.md` was read first, but its literal recipe still produced
silently-wrong PTEs here -- the lesson isn't "don't read the docs first,"
it's that a *cross-worktree* doc's example commands can be actively
misleading in a way `research.md` Decision 2 already flagged for runner
binaries, and this finding extends that same caution to the *export*
step, not just the *build/run* step.

**Consequence**: every `4w` PTE this feature was going to reuse "as-is"
(spec FR-001, this document's own superseded Decision 1) must instead be
re-exported with the corrected mechanism before any dispatch-confirm or
e2e capture is trusted -- there is no shortcut where some of the six
pre-existing `4w` files happen to be fine and others don't; all were
produced the same (broken) way and must be treated as suspect until
re-exported and re-verified via ETDump, per model, before use.

**Alternatives considered**: Assuming the stale-venv fix alone would
resolve it, without also fixing the storage mechanism (rejected --
directly disproven by the byte-identical re-export result using the fixed
venv but the old `ET_VK_FORCE_BUFFER` mechanism). Continuing to use
`export_quant.sh` with a patched env var name (rejected -- simpler and more
maintainable to use this repo's own already-existing, already-tested
`--vulkan-storage-override` mechanism directly than to patch a
cross-worktree script to match this repo's actual code).
