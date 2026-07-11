# Research: End-to-End tok/s Report — Texture, Buffer, and WMMA Across 4w/8da4w

## Decision 1: Rank-3 blocker resolution mechanism

**Decision**: Resolve the rank-3 blocker with a **narrow relaxation of
`can_use_q4gsw_coopmat()`'s guard** (`QuantizedLinear.cpp:192-196`), from
`graph->dim_of(output) > 2` to a check that rejects only when a genuine
batch is present -- i.e. when the product of all output dims *before* the
trailing two is `!= 1`. Not a graph-shape (reshape/squeeze) fix.

**Rationale**: Verified directly from code (constitution Principle VI), not
assumed:
- `quantized_linear_global_wg_size` (`QuantizedLinear.cpp:88-132`), which
  sizes the dispatch grid for both the `4w` and `8da4w` coopmat kernels,
  reads only `utils::val_at(-1, out_sizes)` (N) and `val_at(-2, out_sizes)`
  (M) -- rank-agnostic by construction. It never reads a batch dim.
- The shaders themselves (`linear_qw_coopmat.glsl`, `linear_dq8ca_qw_coopmat.glsl`)
  bind `output_sizes`/`input_sizes` as `ivec4` UBOs but only ever read `.x`
  (the innermost dim: N for output, K for input) plus the `out_N_arg` spec
  constant. `.y`/`.z`/`.w` (which would carry a batch dim, if any) are never
  read. The M row index comes entirely from `gl_WorkGroupID`/`tileID`, which
  the C++ side already sizes from the trailing two dims only.
- `resize_linear_qw_node` (`QuantizedLinear.cpp:26-54`) already has a working
  rank-3 branch (`mat1_sizes.size() != 2`) that preserves the leading batch
  dim in the resized output shape -- this path is already exercised whenever
  the *tiled* fallback runs today, since the real model's activations are
  already rank-3 (`003`'s finding). Nothing here needs to change.
- For a `Buffer`-storage tensor, a contiguous `[1, M, N]` layout is
  bit-identical to `[M, N]` when the leading dim is exactly 1 (its stride
  contributes zero offset). Since every real per-model shape in this
  workstream has batch `== 1` (`003`, confirmed from raw ETDump event args,
  never squeezed), the existing 2D dispatch grid already covers 100% of the
  data with no missing "slice."

The guard's original comment ("batched (rank > 2) outputs would silently
miscompute all slices beyond the first") is correct for an actual batch > 1
and stays enforced by the relaxed check -- only the specific, verified-safe
case of a size-1 leading dim is newly allowed through.

**Alternatives considered**:
1. **Graph-shape fix** (insert squeeze/unsqueeze around every q4gsw/
   dq8ca-q4gsw linear call site, e.g. in a pass or wrapper). Rejected: higher
   scope and risk than a single guard-condition change for the same safety
   guarantee -- would touch graph-construction code in multiple places
   instead of one eligibility function, with no additional correctness
   benefit.
2. **Hardcode "rank == 3" as the allowed exception** instead of the general
   "leading dims collapse to 1" check. Rejected in favor of the more general
   form: it costs nothing extra, and correctly handles a hypothetical rank-4
   case the same way without a second special-case later.

**Risk classification**: Contained, single-function, well-understood change
-- meets the FR-009/`006`-precedent bar for "a contained, well-understood
fix." Per FR-009 and constitution Principle V, applying it (at implementation
time, not during this planning phase) requires explicit user authorization
and an inline comment at the change site naming what was relaxed and why,
matching the comment style already used at `007`'s wiring fix
(`QuantizedLinear.cpp`'s new `linear_q4gsw()`, currently uncommitted).

## Decision 2: New correctness coverage required for the relaxed guard

**Decision**: The guard relaxation needs **one new, small, rank-3 (batch=1)
correctness check** comparing the coopmat path's output against the CPU/
tiled reference, added alongside the guard change -- it cannot rely solely on
citing existing coverage the way `007`'s registration fix could.

**Rationale**: Grepped every existing q4gsw/8da4w-coopmat correctness and
benchmark test (`test_coopmat_linear_bench.cpp`, `test_fpa_q4gsw_linear.cpp`,
`test_q4gsw_linear.cpp`): all construct exclusively rank-2 (`{M, K}`/`{N, K}`)
shapes. No existing test exercises a rank-3 input/output through either
coopmat shader. Since this is a genuinely new shape class newly reaching the
shader (unlike `007`'s fix, which only changed *which registration* already-
covered 2D shapes went through), constitution Principle I's bar ("no coopmat
[dispatch] change is done until it passes... correctness tests... at small,
tile-aligned shapes") requires new, not just cited, coverage.

**Alternatives considered**: Relying on the e2e smoke-check alone (coherent,
non-degenerate output at real model scale, `006`'s existing bar). Rejected as
the sole signal -- Principle I's tier is the small, tile-aligned, CPU-reference
comparison; the e2e smoke-check is a necessary second layer (catches gross
breakage) but not a substitute for a numerical correctness check, consistent
with how `006` and `007` both layered the two.

## Decision 3: Composition of a "WMMA-eligible export"

**Decision**: A WMMA-eligible export for a given (model, scheme) is produced
by combining three already-scoped ingredients, with **no additional new
flag**:
1. `--vulkan-storage-override buffer` (`006`, already committed).
2. Decision 1's guard relaxation (new, this feature).
3. `007`'s `linear_q4gsw` registration fix for `4w` only -- `8da4w`'s
   `et_vk.linear_dq8ca_q4gsw.default` was already correctly registered to
   `linear_dq8ca_q4gsw` pre-`007` (confirmed from the current diff: only the
   `linear_q4gsw` registration line is new).

No `ET_VK_FORCE_TILED_LINEAR` env var is set for these captures -- unset is
the natural-dispatch state (`007`'s precedent): once eligibility passes,
`pick_linear_qw_shader`/`pick_linear_dqa_qw_shader` route to the coopmat
kernel automatically. There is no separate "enable coopmat" switch to add.

**Rationale**: This is exactly the three-part recipe spec.md's Assumptions
section already names; this decision just confirms each part's mechanism is
real (grep-verified above) and that nothing else is required.

**Alternatives considered**: None -- this is confirmation of an
already-scoped design, not an open choice.

## Decision 4: ETDump-based dispatch verification (User Story 1 / FR-003)

**Decision**: Reuse `002`'s existing ETDump capture pipeline (`--etdump_path`
on the standard LLaMA runner build, already wired with
`EXECUTORCH_BUILD_DEVTOOLS`/`etdump`/`flatccrt`) and its `kernel_name` field
extraction (per `002`'s `data-model.md`, one row per `(kernel_name, shape)`)
to confirm dispatch. A configuration passes FR-003 when its captured
per-op `kernel_name` for each linear op contains `_coopmat` (matching
`linear_q4gsw_coopmat_*`/`linear_dq8ca_q4gsw_coopmat_*`, the same naming
`007` already confirmed at the microbenchmark tier); if the tiled kernel name
appears instead, the configuration fails the dispatch check and reports no
WMMA number (FR-007).

**Rationale**: This is precisely what constitution Principle VI's
model-level clause requires ("MUST capture an ETDump trace and confirm...
that the WMMA/coopmat kernel dispatched") and what `003` already did once
(reading kernel names directly from raw ETDump event args) -- no new
capture mechanism needs to be built, only applied per-configuration here.

**Alternatives considered**: Trusting the eligibility gate's return value
alone (i.e., assuming dispatch because `can_use_q4gsw_coopmat()` should
return true). Rejected -- this is the exact failure mode Principle VI and
`007`'s own discovery (the dead `linear_q4gsw` registration silently routing
everywhere except the coopmat path, invisible from the gate logic alone)
warn against.

## Decision 5: Reuse of `006`'s `Texture3D`/`Buffer` e2e numbers

**Decision**: Read `006`'s per-configuration `Texture3D`/`Buffer`
prefill/decode tok/s directly from
`specs/006-e2e-storage-comparison/results/e2e-storage-comparison-report.md`'s
per-configuration table -- no re-capture, no re-derivation, matching FR-001
(already locked in spec.md's Assumptions).

**Carried-forward risk, stated explicitly rather than re-litigated**: `006`
found and documented real session-to-session prefill variance on
`rocky-ryzen` unrelated to storage type (its own same-session control:
`llama-3.2-3b`/`4w` recaptured at 355.5±22.5 tok/s vs. its original
388.4±3.93 tok/s capture of the identical `.pte`; decode was unaffected).
Since this feature's own WMMA capture happens in yet another new session,
any prefill delta this report shows between the WMMA arm and `006`'s
`Texture3D`/`Buffer` numbers inherits the same cross-session caveat --
FR-006's "consistent with / diverges from" language for prefill specifically
MUST be read with this in mind (a single-digit-percent prefill difference is
not automatically a storage/dispatch-arm effect). Decode has no such
precedent of cross-session drift and can be compared more directly. This is
not a new decision -- spec.md's own Assumptions already chose reuse over
re-capture; this decision records the known consequence rather than
silently forgetting `006`'s own hard-won lesson.

**Alternatives considered**: Re-capturing a same-session `Texture3D`/`Buffer`
control alongside the new WMMA numbers (as `006` itself suggested as the
"fully rigorous version" it didn't have time for). Rejected for this
feature: spec.md's Assumptions already made this call explicitly (reuse,
not re-measure) to bound device time; revisiting it here would contradict
an already-validated spec decision.

## Decision 6: E2E capture methodology for the WMMA arm

**Decision**: Identical procedure to `001`/`006`: 5 repeated runs per
configuration (first 2 discarded only for the one config already known to
show GPU warm-up drift, `llama-3.2-1b`/`4w`, matching precedent), no
concurrent GPU load, same fixed prompt/`--temperature 0`/2048-prefill/
1024-decode workload, `prefill_tokens_per_sec`/`decode_tokens_per_sec`/
`prefill_tokens`/`decode_tokens`/`num_runs`/`variance`/`run_metadata` JSON
shape (unchanged since `001`). A smoke-check (completes without crashing,
`generated_tokens` matches request, output coherent/non-degenerate) gates
every WMMA export before any timing is trusted, exactly `006`'s Principle-I
enforcement (FR-002's dispatch check is the *additional* gate this feature
adds on top, not a replacement for the smoke-check).

**Rationale**: Reuse, not reinvention -- `006` already validated this exact
methodology satisfies constitution Principle IV (statistically sound
model-level benchmarking) for this precise comparison shape (three arms
instead of two changes nothing about the per-arm capture procedure).

**Alternatives considered**: None new -- carried forward from `001`/`006`.

## Decision 7: Report verdict logic

**Decision**: Per configuration, the report states the `Texture3D`/`Buffer`/
WMMA prefill and decode tok/s triple, the relative WMMA-vs-Buffer and
WMMA-vs-Texture3D percentage differences, and one of:
- **consistent** with `007`'s microbenchmark finding for that scheme (`4w`:
  WMMA meaningfully faster; `8da4w`: WMMA meaningfully slower), and (for
  `8da4w` only) whether it's also consistent with `008`'s finding that the
  shipped config stays roughly at parity with tiled once tuned (a directional
  cross-check, not a numeric one -- `008`'s config 5 finding is explicitly
  unreachable in production per FR-008/spec Assumptions, so the *shipped*
  config's own tiled-vs-coopmat gap, not config 5's, is the relevant
  `008` data point).
- **diverges** -- named explicitly, with the observed direction/magnitude,
  never averaged away (edge case in spec.md).
- **blocked/failed** -- rank-3 fix inapplicable to that configuration, or
  FR-003's dispatch check failed -- with the specific reason (FR-007).

One closing statement per scheme (not per configuration) answers "does WMMA
help this device's real token generation rate" (SC-003), consistent with
`007`'s per-scheme framing (`4w` vs `8da4w` diverge in direction, so a single
blended answer would misrepresent both, mirroring why `007`'s own report
led with per-scheme numbers and only footnoted a blended one).

**Rationale**: Directly operationalizes FR-005/FR-006/FR-007/SC-003 using
the same "lead with per-scheme, name divergences, never blend away opposite
effects" structure `007`'s report already established as this workstream's
convention.

**Alternatives considered**: A single blended overall verdict across both
schemes. Rejected -- `007` explicitly found this misleading for this exact
data shape (`4w` and `8da4w` move in opposite directions) and this feature
inherits the same shape.

## Decision 8: `--vulkan-force-fp16` silently defeated `--vulkan-storage-override buffer` for every real per-layer linear op (found during implementation, fixed with authorization)

**Finding**: After applying Decision 1's rank-3 fix, `006`'s already-existing
"Buffer" `.pte` exports still dispatched every per-layer linear op
(`wq`/`wk`/`wv`/`wo`/`w1_gate`/`w3_up`/`w2_down`) through the **tiled**
kernel, confirmed via ETDump kernel-name inspection
(`linear_q4gsw_tiled_texture3d_texture2d_half` -- `_texture3d_` is
`graph->storage_type_of(output)`, read directly, not assumed). A completely
fresh re-export with the identical `--vulkan-storage-override buffer` flag
showed the same result, ruling out a stale/wrong file.

**Root cause** (found via direct code tracing, constitution Principle VI):
`backends/vulkan/_passes/tag_memory_meta_pass.py`'s
`constrain_op_arg_repset()` unconditionally intersected every op argument's
valid-storage set down to `utils.ANY_TEXTURE` (zero valid buffer layouts)
whenever `force_fp16` was set -- *before* `006`'s `default_storage`
preference was ever consulted later in the same pass. Since every op's
output is also an argument to whatever consumes it next, this constraint
propagates backward through virtually the entire graph via the
downstream-user BFS trace, and `make_tensor_repr()`'s
`preferred_storage == BUFFER and self.buffer_is_valid()` branch can never
fire once `buffer_is_valid()` is already `False`. This predates `006`
entirely -- it was introduced by an unrelated upstream commit (`e4aba1e658`,
"buffer implementation of rotary positional embeddings", Nov 2025,
`pytorchbot`), from back when far fewer ops had buffer+fp16 support, and was
never revisited when `006` added `storage_type_override` months later. The
two mechanisms had simply never been checked together via ETDump until now
-- `006`'s own smoke-checks and e2e captures completed successfully and
looked reasonable precisely because *nothing about storage actually
changed*, so there was nothing to crash or look wrong.

**Consequence for prior features**: `--vulkan-force-fp16` is required for
every export in this workstream (`001`: raw fp16 export is rejected by the
Vulkan partitioner, `--vulkan-force-fp16` is the only way to run fp16 on
Vulkan at all). This means `006`'s "Buffer storage export" never actually
put the per-layer linear ops in `Buffer` storage, and `004`'s "storage
switch is basically free" finding was comparing `Texture3D` against
`Texture3D` -- not a storage-type finding at all. This does not get
silently retconned here; it is recorded as a real, load-bearing correction
this feature's own verification uncovered, per FR-009/constitution
Principle VI's mandate to verify with tools rather than trust prior
code-level reasoning.

**Fix applied** (proposed, then authorized by explicit user instruction --
"apply the fix and re-verify"): in `constrain_op_arg_repset()`, replace the
unconditional `utils.ANY_TEXTURE` with a choice keyed on `self.default_storage`:

```python
if self.force_fp16:
    fp16_repset = (
        utils.ANY_STORAGE
        if self.default_storage == VkStorageType.BUFFER
        else utils.ANY_TEXTURE
    )
    op_repsets.try_constrain_with_arg_repset(arg_i, fp16_repset)
```

`utils.ANY_BUFFER`/`utils.ANY_STORAGE` already existed as constants (unused
by this call site before). This mirrors `006`'s own fix in spirit --
restoring the *possibility* of honoring a preference, not forcing a new
default -- and is provably scoped: the branch only executes when
`force_fp16` is set (every other export path is untouched by construction),
and within it, `default_storage`'s own default remains `TEXTURE_3D`, so any
caller not passing `storage_type_override=BUFFER` gets the exact same
`ANY_TEXTURE` behavior as before.

**Verification** (Principle VI -- confirmed via tools, not assumed):
1. **Safety property**: re-exported `llama-3.2-1b`/`4w` with
   `--vulkan-force-fp16` and *no* storage override -- ETDump shows the
   identical kernel-name/count profile as before the fix (112x
   `linear_q4gsw_tiled_texture3d_texture2d_half`, same for every other
   kernel), confirming byte-for-byte-equivalent dispatch behavior for the
   default (no-override) path.
2. **Fix efficacy**: re-exported `llama-3.2-1b`/`4w` with the storage
   override -- ETDump now shows all 112 per-layer linear dispatches as
   `linear_q4gsw_coopmat_buffer_texture2d_half` (genuine coopmat, `_buffer_`
   confirmed), plus the entire rest of the graph (`rms_norm`, `view`,
   `binary_add`/`mul`, `rotary_embedding`, sdpa) now buffer-backed too. The
   one remaining non-coopmat linear dispatch is the GEMV (`M=1`) case,
   expected and unaffected per `003`'s own classification (no WMMA-capable
   GEMV kernel exists at all, independent of storage).
3. **Correctness spot-check**: ran the fixed `Buffer`+coopmat export with a
   factual prompt ("The capital of France is") -- output: "Paris. The
   capital of the United Kingdom is London. The capital of the United
   States is Washington," -- coherent and correct, not just non-crashing.

**Alternatives considered**:
1. Leave `force_fp16` hardcoded to `ANY_TEXTURE` and find some other way to
   force Buffer storage (e.g. a separate, narrower override bypassing this
   pass entirely). Rejected -- would fight the existing preference
   mechanism instead of fixing the actual conflict, and risks diverging
   further from upstream's own pass structure.
2. Make `force_fp16` always prefer `ANY_STORAGE` unconditionally (drop the
   `default_storage` check entirely). Rejected -- would change default
   (no-override) behavior for every existing caller of `force_fp16`, which
   is exactly the regression `006`'s own precedent (and this fix) commits
   to avoiding.

**Scope note**: this fix's blast radius is provably limited to the
`force_fp16 AND storage_type_override=BUFFER` combination -- a combination
that, per the finding above, has never actually worked as intended before,
so this can only newly enable correct behavior, not regress an
already-working one.
