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

## Decision 7: Linear coopmat dispatch (`4w`/`8da4w`) does not actually fire on M5 EVT1 -- every checked Configuration falls back to tiled

**Finding (2026-07-06, during US2 8B dispatch-confirm, T031-T034)**: The
`1b-results.md`/`3b-results.md` claims of "`linear_q4gsw_coopmat` N/N
confirmed" (published earlier in this feature) are **wrong**. Re-checking
with the original `llama_main_etdump_spec015` binary (the same one used to
produce those claims) shows the actual per-kernel ETDump breakdown is 100%
`linear_q4gsw_tiled_buffer_texture2d_half` / `linear_dq8ca_q4gsw_tiled_buffer_texture2d_half`
for every linear Configuration checked: 1B `4w` (112/112), 1B `8da4w`
(112/112), 3B `4w` (196/196), 8B `4w` (224/224). Not one shows the
`_coopmat` kernel family for the bulk of prefill's linear dispatches (the
single `linear_q4gsw_coop_*` call per run is the unrelated GEMV/lm-head
path, M=1, not gated by `can_use_q4gsw_coopmat` at all).

**Investigation** (full write-up: workspace
`.shared-context/report-for-human/open-questions.md` Q11): added temporary
`fprintf` diagnostics to `can_use_q4gsw_coopmat`/`pick_linear_qw_shader`
(reverted after use, never committed), rebuilt the Android ETDump runner,
and confirmed the C++ eligibility gate evaluates **true** for every one of
8B's 224 prefill linear dispatches (shapes are tile-aligned, output is
Buffer, dtype half, no bias), and the constructed kernel name
(`linear_q4gsw_coopmat_buffer_texture2d_half`) resolves successfully via
`VK_KERNEL_FROM_STR` (no exception -- the shader registry has a real,
distinct, non-aliased `.spv` for this exact combination, confirmed via
`md5sum` against the tiled variant's `.spv`). Despite this, the shader
that actually executes on the GPU (per ETDump) is the tiled one. Ruled
out: stale/incomplete export (an independently-pre-existing, pre-session
`ctx2304` 8B PTE shows the identical pattern), shape misalignment, bias
presence, `ET_VK_EXECUTE_NODE_THRESHOLD`, and a duplicate-registration
collision between `et_vk.linear_q4gsw.default` (coopmat-aware, this is
what the AOT `patterns/quantized_linear.py` actually emits) and the
legacy, tiled-only `et_vk.q4gsw_linear.default`. The `dq8ca` (8da4w)
variant shows the identical symptom despite using the device's native
subgroup size (64, not the `q4gsw` coopmat shader's forced 32), which
weakens (but doesn't rule out) a subgroup-size/pipeline-creation
hypothesis for that specific shader.

**Root cause: not yet located.** This needs Vulkan-API-level
instrumentation (validation layers, or a `VK_CHECK` around the actual
pipeline-creation/binding call) beyond what a source-level read of
`QuantizedLinear.cpp` can resolve, and is out of this feature's scope to
fully root-cause. Logged as workspace `open-questions.md` Q11 for
follow-up.

**Consequence for this feature**: every already-published `4w`/`8da4w`
tok/s number (1B, 3B) has been corrected in place (`1b-results.md`,
`3b-results.md`) to say "tiled fallback, not coopmat" rather than
retracted -- the throughput numbers themselves are real, reproducible
hardware measurements (matched across two independently-built binaries),
they simply are not evidence of this workstream's coopmat/WMMA speedup on
M5 EVT1 for linear ops. `data-model.md`'s `dispatch_status` column is
corrected to `fallback` for every linear Configuration measured so far.
8B's own results (`8b-results.md`) will report the same honestly from the
start, not as a later correction. SDPA-coopmat (User Story 3, `ET_VK_SDPA_COOPMAT`)
is a separate opt-in code path and is not known to share this defect --
that still needs its own dispatch-confirm check once enabled, per plan.

**Alternatives considered**: Silently leaving the "confirmed" claims as
originally published and only fixing 8B going forward (rejected -- this
workstream's Principle I/VI require correctness over convenience, and an
inconsistent record across 1B/3B/8B would misrepresent what was actually
measured). Pausing the feature entirely until Q11 is root-caused (rejected
for now -- the e2e tok/s numbers are still real, useful measurements of
this build's *actual* current behavior, and the SDPA user story is
independent of this defect; root-causing Q11 is logged as follow-up work,
not a blocker for finishing this feature's measurement scope).

**Post-completion lead (2026-07-06, not yet acted on)**: diffing
`QuantizedLinear.cpp`/`linear_qw_coopmat.glsl` against the `quant-dev`
worktree traced the current shader's fp16-accumulate + flattened-dbuf1-loop
+ vectorized-dequant + 128x64-retile to commit `133044739`, whose own
message states these changes are "`pending` hardware validation ... this
commit preserves the work, it does not claim it works." Separately,
`.shared-context/report-for-human/jira-tile-sweep.md` (source of this
feature's 110.6/213.9/565.3 `4w` Prior-Finding numbers) states its own
128x64 result was measured via a `.tmp-origcm`-worktree-only
`ET_VK_Q4GSW_COOPMAT_VARIANT` toggle, not this repo's production
`can_use_q4gsw_coopmat` dispatch path, and that "the production q4gsw
coopmat shader still ships dbuf1 (128x128)" -- i.e. this feature's
comparison baseline was never itself confirmed working through the code
path being tested here.

## Decision 7 REVERSED (2026-07-06): coopmat genuinely dispatches; ETDump's per-event kernel-name attribution is the actual bug

**This decision's headline finding above is wrong.** Two independent
pieces of evidence, neither relying on ETDump's per-event kernel-name
field, converge on the opposite conclusion:

1. **Direct wall-clock A/B test on the exact e2e path.** Using the
   genuinely-functional `ET_VK_FORCE_TILED_LINEAR=1` kill switch (confirmed
   real by reading `can_use_q4gsw_coopmat`'s source), an A-B-A-B alternating
   test on 1B/`4w` (same PTE, same prompt, same session) measured: default
   (no override) 576.7/577.1 tok/s prefill; genuinely-forced-tiled
   321.0/321.3 tok/s. Default is **1.8x faster** than forced-tiled, and
   321 tok/s matches the historical T-tiled baseline (312.7) closely. If
   the default path were truly dispatching tiled (as ETDump claimed), it
   could not be 1.8x faster than forced-tiled on the identical code path.
2. **`specs/016-m5-linear-sdpa-microbench`'s independent microbenchmark**,
   on this same build/hardware, using the harness's own kernel-name
   capture (not ETDump's per-event field) plus SPIR-V inspection: both
   `linear_q4gsw` and `linear_dq8ca_q4gsw` genuinely dispatch coopmat at
   production K/N shapes, 3.04x/4.16x faster than tiled respectively,
   correctness-verified.

**Revised conclusion**: the e2e tok/s numbers already published in
`1b-results.md`/`3b-results.md`/`8b-results.md` (583.70, 218.26, 112.71,
etc.) are genuine coopmat/WMMA results, not tiled fallback as this
Decision originally concluded. **ETDump's per-event kernel-name
attribution is unreliable specifically in the full LLaMA graph context**
(224+ linear nodes sharing one graph/pipeline-cache context) -- it is a
tooling/instrumentation bug, not a dispatch bug. This does not fully
close Q11: *why* ETDump's attribution is wrong in this context is still
unverified (candidates: pipeline-cache key collision, a GPU query-pool
index/dispatch-ID mapping error at scale) and would need Vulkan validation
layers or a `VK_CHECK`-level trace to pin down -- but the practical
question this feature cares about ("did coopmat actually run for the
reported numbers") is now answered: yes. See workspace `open-questions.md`
Q11's "二次反转" addendum for the full writeup, and Q12 for the parallel
re-evaluation of the SDPA env-var finding (likely the same attribution bug,
not a mysterious non-coopmat speedup).

**Consequence**: `1b-results.md`/`3b-results.md`/`8b-results.md`,
`data-model.md`, and `results/m5-e2e-validation-report.md` are updated
with an "UPDATE" section superseding their "CORRECTION" sections --
the original "confirmed" claims were closer to right than the
"tiled fallback" correction that followed them, just for the wrong
reason (the original claims never independently verified dispatch either;
they happened to be right by luck, not by a trustworthy method). The
`dispatch_status` for every linear Configuration is restored to
`confirmed`, now on firmer evidence (direct throughput A/B +
`specs/016`'s independent microbenchmark) than the original ETDump-only
claim ever had.

The user was informed of the original "not yet acted on" lead above and
declined to pursue the revert-and-retest experiment; it turned out not to
be necessary -- the shader works fine as-is, per the evidence above.

## Decision 8 (2026-07-06, once M5 EVT1 was free again): the `ET_VK_DEBUG_ENCODE_DISPATCH` diagnostic finally ran; `VK_ERROR_DEVICE_LOST` was host-side OOM, not a GPU crash; all 6 full-stack (linear+SDPA) configs now measured

Two threads left open by Decision 7's reversal are addressed here: (1)
actually running the `ET_VK_DEBUG_ENCODE_DISPATCH` diagnostic (built but
never executed as of that Decision) to get direct print-vs-ETDump
ground truth, and (2) the `VK_ERROR_DEVICE_LOST` crash blocking SDPA on
3B/8B.

**Diagnostic run (Q11/G6)**: with the M5 EVT1 driver freshly re-verified
(`f14c51b6f8`, reflashed after finding the device on an unrecognized
build the teammate had left on it) and this repo's `cmake-out-android-vk-etdump`
binary confirmed to carry the diagnostic (`strings | grep ENCODE_DISPATCH`
-> 2 matches, `PICK_SHADER` -> 3 matches), a short (`--max_new_tokens=4`)
capture on 1B/`4w` linear compared the `[ENCODE_DISPATCH]` stderr print
(read directly from `shader_.kernel_name` at the exact bind+log call site)
against `analyze_etdump_shaders.py`'s own per-kernel breakdown of the same
run's `.etdp`. **Result: they agreed exactly** -- 112/112
`linear_q4gsw_coopmat_buffer_texture2d_half` in both. A second capture
with `ET_VK_SDPA_COOPMAT=1` showed the same agreement for SDPA:
`sdpa_compute_attn_weights_coopmat_buffer_buffer_half` /
`sdpa_compute_out_coopmat_buffer_buffer_half`, 16/16 in both the stderr
ground truth and ETDump's own analysis. **The misattribution did not
reproduce.** This directly elevates SDPA/1B from "likely coopmat" (Q12's
inference from the linear finding) to **confirmed** by direct bind-time
evidence, independent of ETDump. It does not, however, locate *why* the
original misattribution happened during this feature's US2 dispatch-confirm
step (Decision 7) -- Q11's root cause remains open; today's finding only
adds that the bug is not persistent/deterministic across sessions on this
build, which narrows out "ETDump is fundamentally broken for this graph
shape" as an explanation without replacing it with a confirmed mechanism.
(Incidental finding, not a bug: `linear_q4gsw_coop_*`/`sdpa_compute_*_coop_*`
-- note "coop", not "coopmat" -- are a real, separate decode-only
(`M=1`) GEMV/subgroup-cooperative shader family, self-gated to a no-op at
prefill; seeing both `_coop_` and `_coopmat_` names in one capture is
expected, not a naming collision or attribution bug.)

**`VK_ERROR_DEVICE_LOST` root cause (Q12)**: bisecting `--max_new_tokens`
(64/256/512/1024) on 3B with `ET_VK_SDPA_COOPMAT=1` found **no crash at
any length**, including the full documented 1024-decode crash point --
directly contradicting the earlier `blocked_reason`. The same held for
8B (both `4w` and `8da4w`) at a smoke-test tier. When the *proper* 3-rep
headline measurement was then attempted with `warmup=true` (matching this
feature's established methodology) on 8B, it failed silently (exit 0, no
`PyTorchObserver` line) -- `dmesg` showed why: a genuine Android **OOM
kill** (`Out of memory: Killed process ... llama_main_etdu ...
anon-rss:1971136kB, file-rss:2446176kB`), not a Vulkan/GPU error at all.
`/proc/meminfo` showed `MemAvailable` down to ~0.6-1.5GB out of 11.19GB
total -- caused by this session's own accumulation of ~29GB of staged
PTEs and `.etdp` traces in `/data/local/tmp/llama_vk`, none of it cleaned
up between runs. Deleting already-consumed `.etdp`/log files and PTEs not
immediately needed, then switching to `--warmup=false` (which avoids
running the full pipeline twice back-to-back, halving peak transient
memory), let every remaining config complete cleanly. **The original
`VK_ERROR_DEVICE_LOST` finding was real (it did happen, and is worth
keeping as history) but its cause was host-side memory pressure from
this workstream's own on-device file accumulation, not a driver/GPU
defect** -- a materially different, and much less alarming, explanation
than "the GPU hung." See `.specify/memory/gotchas.md` G11.

**Full-stack (linear WMMA + SDPA WMMA together) e2e, all 6 configs, 3-run
means, `warmup=false`, pinned clocks re-verified, driver re-verified**:

| Model | Scheme | Prefill tok/s (mean, CoV) | Decode tok/s (mean) | Note |
|---|---|---|---|---|
| 1B | `4w` | 769.35 (6.87%) | 13.60 | High CoV vs. every other row here (all <2%) -- see caveat below |
| 1B | `8da4w` | 723.00 (0.27%) | 12.83 | |
| 3B | `4w` | 333.97 (0.43%) | 6.69 | |
| 3B | `8da4w` | 286.31 (1.55%) | 6.45 | Previously blocked; resolved (see above) |
| 8B | `4w` | 153.30 (0.43%) | 3.79 | Previously blocked; resolved (see above) |
| 8B | `8da4w` | 130.05 (0.09%) | 3.67 | Previously blocked; resolved (see above) |

**1B/`4w` CoV caveat**: this row's 6.87% CoV (rep range 718.3-823.8 tok/s)
is an outlier against every other config measured this session (and
against this same config's own prior-session figure in
`results/raw/1b_4w_e2e.log`, ~812.4-812.7 tok/s, CoV ~0.03%). Two
methodology differences from the tight prior-session number: this
session's reps used `--warmup=false` (all six rows above do, for
consistency with the OOM fix), and ran after a long sequence of other
back-to-back device activity (unlike a fresh-session baseline). Not yet
attributed to a specific cause (thermal drift despite pinned clocks,
residual memory/cache pressure from the same accumulation that caused the
OOM above, or genuine run-to-run noise at this model's very high tok/s
where absolute timing noise is a larger fraction of a shorter wall-clock
run) -- flagged here rather than silently averaged over, per Principle
VII's discipline on floating-clock throttle variance (this is pinned, but
the same "don't just trust a mean" caution applies once CoV is this far
outside the pattern of every sibling measurement).
