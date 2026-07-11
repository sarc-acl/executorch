# Contract: E2E tok/s Report Data Formats

## Guard relaxation (production code, `QuantizedLinear.cpp`)

`can_use_q4gsw_coopmat()`'s rank check changes from:

```cpp
if (graph->dim_of(output) > 2) { return false; }
```

to (exact form finalized at implementation time, per research.md Decision 1):

```cpp
// only reject a genuine batch (>1); a size-1 leading dim is safe -- see
// inline comment/citation at implementation
```

- MUST NOT change behavior for any already-passing rank-2 or genuine-batch
  (>1) case -- this is the safety property research.md Decision 1 depends
  on, exactly mirroring `006`'s own "default behavior provably unchanged"
  bar for its storage-override fix.
- MUST ship with an inline comment naming why a size-1 leading dim is safe
  (constitution Principle V/VI), matching the comment style already used at
  `007`'s `linear_q4gsw()` fix.
- Applying this change requires explicit user authorization at
  implementation time (FR-009) -- this contract records the proposed shape
  of the change, not a pre-authorized diff.

## WMMA-eligible export command

No new CLI flag. Combines three already-existing/proposed pieces
(research.md Decision 3):

```bash
python -m examples.models.llama.export_llama \
  <same args as 006 used for this model/scheme> \
  --vulkan-storage-override buffer \
  --output_name specs/009-e2e-tokrate-report/results/pte/<model>_<scheme>.pte
```

run against a build with the guard relaxation and `007`'s `linear_q4gsw`
registration fix applied, `ET_VK_FORCE_TILED_LINEAR` unset.

## ETDump dispatch-check capture

Reuses `002`'s existing `--etdump_path` mechanism verbatim -- no new capture
code. Output: one `.etdump` file per configuration under
`results/etdump/<model>_<scheme>.etdump`, parsed for each linear op's
`kernel_name` (per `002`'s `data-model.md` Kernel Invocation row shape).

- A configuration's `dispatch_status` is `confirmed` only if every measured
  linear op's `kernel_name` contains `_coopmat`.
- Any tiled/coop (non-`_coopmat`) kernel name for a linear op flips the
  whole configuration to `fallback` -- no WMMA tok/s number is reported for
  it (FR-003/FR-007).

## E2E capture output

Same `e2e` JSON object shape as `001`/`005`/`006` (`prefill_tokens_per_sec`,
`decode_tokens_per_sec`, `prefill_tokens`, `decode_tokens`, `num_runs`,
`variance`, `run_metadata`) -- no new schema. Stored under
`results/raw/<model>_<scheme>_wmma.json`.

## `results/e2e-tokrate-report.md`

Rules a consumer can depend on:

- Every one of the 6 configurations appears in the Blocked/Failed section or
  contributes both its prefill and decode rows to the main table -- never
  partially or silently absent (FR-007/SC-004).
- No `wmma_tok_s` value appears for a configuration whose `dispatch_status`
  is not `confirmed` (FR-003, SC-001).
- Every `Texture3D`/`Buffer` value is cited as coming from `006`'s report
  verbatim, not re-measured (FR-001).
- Every prefill row carries the inherited cross-session caveat note
  (research.md Decision 5); decode rows do not.
- Two per-scheme summary verdicts (`4w`, `8da4w`) appear before the detailed
  table, each stating whether e2e WMMA helps, and whether that agrees with
  `007`'s (and, for `8da4w`, `008`'s) prior finding (FR-006, SC-003) -- no
  single blended-across-schemes number (research.md Decision 7).
