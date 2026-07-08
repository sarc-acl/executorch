# Quickstart: End-to-End tok/s Report — Texture, Buffer, and WMMA Across 4w/8da4w

Real device work, like every prior tier-2 feature in this workstream --
build, export, and GPU capture on the `rocky-ryzen` MiniPC.

## Prerequisites

- `006-e2e-storage-comparison` is complete (its `Texture3D`/`Buffer` e2e
  numbers are this feature's comparison baseline -- reused, not re-captured).
- `007-wmma-improvement-microbench` is complete (its `linear_q4gsw`
  registration fix, currently uncommitted, is required for `4w`; its
  microbenchmark finding is this feature's per-scheme cross-check).
- `008-8da4w-parameter-sweep` is complete (its shipped-config tuning finding
  is this feature's `8da4w` cross-check).
- `002-etdump-shader-profiling`'s ETDump capture build/flow works on this
  machine (`--etdump_path`, `EXECUTORCH_BUILD_DEVTOOLS` already wired).
- Nothing else CPU/GPU-heavy running before any capture.

## 1. Apply the rank-3 guard relaxation (requires explicit authorization)

Per research.md Decision 1 and FR-009: propose the exact diff to
`can_use_q4gsw_coopmat()` in `QuantizedLinear.cpp`, get explicit user
authorization, then apply it with an inline comment documenting why a
size-1 leading dim is safe (contracts/e2e-tokrate-report-schema.md).

**Verify the safety property first**: confirm every existing rank-2 coopmat
correctness/benchmark case (`test_coopmat_linear_bench.cpp`,
`test_fpa_q4gsw_linear.cpp` -- Buck-only, skip if Buck isn't installed,
`test_q4gsw_linear.cpp`) still passes unchanged.

## 1b. Apply the `force_fp16`/storage-override pass fix (also requires explicit authorization)

**Found during implementation, not part of the original plan** (research.md
Decision 8): `--vulkan-storage-override buffer` alone does *not* achieve
`Buffer` storage for the per-layer linear ops -- `--vulkan-force-fp16`
(required for every export in this workstream) unconditionally forces every
op argument to `ANY_TEXTURE` in `tag_memory_meta_pass.py`'s
`constrain_op_arg_repset()`, *before* the storage override is ever
consulted. **Do not skip this step** -- without it, step 5's export below
will still dispatch the tiled kernel despite the override flag, and steps
1/2's rank-3 fix will never actually be exercised by the real model (the
storage gate fails first). Fix: make the `force_fp16` branch pick
`utils.ANY_STORAGE` instead of `utils.ANY_TEXTURE` when
`self.default_storage == VkStorageType.BUFFER`. Verify via ETDump on one
config both ways: (a) no override -> kernel names unchanged from before the
fix (safety property), (b) with `--vulkan-storage-override buffer` -> every
per-layer linear kernel name now contains `_coopmat` *and* `_buffer_`.

## 2. Add the new rank-3 correctness check

Per research.md Decision 2: a small, tile-aligned, batch=1, rank-3 shape
through both the `4w` and `8da4w` coopmat kernels, compared against the CPU/
tiled reference. This is new coverage, not a citation of existing tests.

## 3. Confirm `007`'s wiring fix is applied

`Q4gswLinear.cpp`/`QuantizedLinear.cpp`'s current uncommitted diff (routing
`et_vk.linear_q4gsw.default` through `add_linear_qw_node`) must be present in
the build used for every `4w` export in this feature.

## 4. Rebuild

Standard Reference Build Recipe (constitution, MiniPC/Linux preset), plus
whatever CMake option this environment already uses for ETDump
(`002`'s existing recipe -- `EXECUTORCH_BUILD_DEVTOOLS=ON`, `etdump`/
`flatccrt` linked into the LLaMA runner).

## 5. Produce one WMMA-eligible export first (prove dispatch before scaling)

**Do not reuse `006`'s existing `.pte` files** -- they were exported before
step 1b's fix and never actually achieved `Buffer` storage either (same
blocker). Every configuration needs a fresh export against the fixed build:

```bash
python -m examples.models.llama.export_llama \
  --model llama3_2 \
  -c /home/doremy/checkpoints/llama3_2_1b/original/consolidated.00.pth \
  -p /home/doremy/checkpoints/llama3_2_1b/original/params.json \
  -t /home/doremy/checkpoints/llama3_2_1b/original/tokenizer.model \
  -kv --use_sdpa_with_kv_cache -qmode 4w --group_size 32 \
  --max_seq_length 3072 --max_context_length 3072 \
  -V --vulkan-force-fp16 --vulkan-storage-override buffer \
  -o <output-dir>
```
(`--model llama3_1` + `/home/doremy/archive/llama3_1_8b/original/` for the
8B model; `--model llama3_2` + `.../llama3_2_3b/original/` for 3B. Output is
always named `<--model>.pte` -- rename immediately, per `001`'s precedent.)

## 6. Smoke-check it, then capture ETDump and confirm dispatch (US1)

Build a second tree with `EXECUTORCH_ENABLE_EVENT_TRACER=ON` (mirroring
`002`'s recipe exactly -- a separate build so tracer overhead never
contaminates step 8's timing) and its own `examples/models/llama` configure
(`002`'s finding: the event-tracer flag must be passed to that sub-configure
explicitly, it isn't inherited).

```bash
./cmake-out-vk-etdump/examples/models/llama/llama_main \
  --model_path <config>.pte \
  --tokenizer_path <same as export> --prompt_file <shared 2048-token prompt> \
  --num_bos 1 --temperature 0 --max_new_tokens 1 --seq_len 3072 \
  --etdump_path <config>_prefill.etdump
```

Load the `.etdump` with `executorch.devtools.Inspector`, extract each
event's embedded JSON (`kernel_name` field, same convention `002`
established), and check every linear op's dispatched kernel name for
`_coopmat`. **Do not proceed to e2e timing for this configuration until
confirmed** (FR-003). The GEMV-shaped (`M=1`) linear dispatch is expected to
stay non-coopmat (`003`: no WMMA-capable GEMV kernel exists) -- don't treat
that one as a failure.

## 7. If dispatch is confirmed, repeat steps 5-6 for the remaining 5 configurations

Any configuration where the rank-3 fix doesn't apply, export fails, or the
smoke-check fails is recorded as `export_status: blocked` with the actual
reason -- never retried with a silent workaround (matching `006`'s
precedent).

## 8. Capture e2e for every configuration that passed its dispatch check

Same 5-repeated-run, no-concurrent-load procedure as `001`/`006`, against
the **non**-instrumented `cmake-out-vk` build (not `cmake-out-vk-etdump`).
`--warmup true` adds a full discarded run before the timed one -- budget
roughly double the untimed inference time per rep (e.g. ~150s/rep observed
for `llama-3.2-3b`, ~185-230s/rep for `llama-3.1-8b`), not just the timed
portion, when estimating total capture time.

## 9. Compare and generate the report

```bash
python specs/009-e2e-tokrate-report/scripts/compare_e2e_tokrate.py \
  --wmma-raw-dir specs/009-e2e-tokrate-report/results/raw \
  --storage-comparison-report specs/006-e2e-storage-comparison/results/e2e-storage-comparison-report.md \
  --out specs/009-e2e-tokrate-report/results/e2e-tokrate-report.md
```

(`007`'s and `008`'s findings are cited as hardcoded, sourced constants in
the script rather than re-parsed from their reports, since both reports'
headline figures are prose, not a stable machine-parseable format --
`006`'s report table is parsed directly since its row format is stable.)

## 10. Sanity-check

- Every one of the six configurations appears exactly once -- either in the
  main table (both phases) or in the Blocked/Failed section with a reason.
- No `wmma_tok_s` value appears for a configuration whose dispatch check
  didn't pass.
- Every prefill row is annotated with the cross-session caveat inherited
  from `006` (research.md Decision 5); no prefill divergence is reported as
  a confirmed storage/dispatch effect without that caveat attached.
- Two per-scheme (not one blended) top-line verdicts appear, each stating
  whether e2e WMMA helps and whether that agrees with `007`'s (and, for
  `8da4w`, `008`'s) finding.
