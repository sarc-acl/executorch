# Quickstart: End-to-End Texture3D vs. Buffer Storage Comparison

Real device work, like `001`/`004` — build, export, and GPU capture on the
`rocky-ryzen` MiniPC.

## Prerequisites

- `001-minipc-baseline-benchmarks` is complete (its `Texture3D` `.pte` files
  and e2e `results/raw/<model>_<scheme>.json` exist — the comparison point).
- `004-linear-storage-comparison` is complete (the microbenchmark-level
  finding this feature checks against).
- Nothing else CPU/GPU-heavy running before any capture.

## 1. Restore the dead-code storage check and add the CLI flag

Apply research.md Decision 1's fix to `backends/vulkan/utils.py` /
`backends/vulkan/_passes/tag_memory_meta_pass.py`, and Decision 2's new
`--vulkan-storage-override` flag to `export_llama_lib.py` /
`partitioner_lib.py`.

**Verify the safety property before doing anything else**: export one
configuration *without* the new flag and confirm its `.pte` is
byte-identical (or at least behaviorally identical — same e2e numbers
within noise) to `001`'s existing `.pte` for that configuration. If this
check fails, stop — it means the fix changed default behavior, which
research.md Decision 1 explicitly says must not happen.

## 2. Export one configuration at Buffer storage (prove the mechanism first)

```bash
python -m examples.models.llama.export_llama \
  <same args 001 used for llama-3.2-1b_4w> \
  --vulkan-storage-override buffer \
  --output_name specs/006-e2e-storage-comparison/results/pte/llama-3.2-1b_4w.pte
```

Note: no `_buffer` suffix needed — this feature's `.pte`s live under their
own `results/pte/` directory, already distinct from `001`'s `Texture3D`
`.pte`s, so the plain `<model>_<scheme>.pte` name is unambiguous (this is
what was actually used, despite the original task list calling for a
`_buffer` suffix).

## 3. Smoke-check it

```bash
./cmake-out-vk/examples/models/llama/llama_main \
  --model_path specs/006-e2e-storage-comparison/results/pte/llama-3.2-1b_4w.pte \
  --tokenizer_path <same as 001> --prompt_file <same as 001> \
  --num_bos 1 --temperature 0 --max_new_tokens 32 --seq_len 3072
```

Expected outcome: completes without crashing; `generated_tokens` matches
what was requested; generated text is coherent, not degenerate (not a
single token repeated the whole way) — per research.md Decision 3, this is
NOT a token-for-token match against the `Texture3D` variant.

## 4. If the smoke-check passes, repeat steps 2-3 for the remaining 5 configurations

Watch specifically for `lm_head`-related allocation failures (research.md
Decision 4) — if one occurs, record it as `export_status: blocked` with the
actual error, do not retry with a workaround silently.

## 5. Capture e2e for every configuration that passed its smoke-check

Same 5-repeated-run, no-concurrent-load procedure as `001`, against each
`.pte`.

**Lesson learned (found during T015 self-review)**: comparing this
feature's `Buffer` numbers directly against `001`'s existing `Texture3D`
numbers conflates storage type with cross-session variance — a same-session
recapture of `001`'s own `llama-3.2-3b/4w` `.pte` showed prefill mean/stdev
noticeably different from `001`'s original capture of the *identical* file,
despite storage type being unchanged. If reproducing this comparison later,
budget time for at least one same-session `Texture3D` recapture as a control
before trusting any cross-session prefill diff.

## 6. Compare and generate the report

```bash
python specs/006-e2e-storage-comparison/scripts/compare_e2e_storage.py \
  --buffer-raw-dir specs/006-e2e-storage-comparison/results/raw \
  --texture3d-baseline-dir specs/001-minipc-baseline-benchmarks/results/raw \
  --microbench-report specs/004-linear-storage-comparison/results/storage-comparison-report.md \
  --out specs/006-e2e-storage-comparison/results/e2e-storage-comparison-report.md
```

## 7. Sanity-check

- Every one of the six configurations appears in the report — either
  measured or explicitly blocked/failed with a reason (never silently
  absent).
- No timing number appears for a configuration whose smoke-check didn't
  pass.
- The report explicitly says whether each measured configuration agrees
  with or diverges from `004`'s microbenchmark-level finding.
