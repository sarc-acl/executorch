# Contract: SDPA Coopmat E2E Validation Data Formats

## ETDump dispatch-check capture

Reuses `002`/`009`'s existing `--etdump_path` mechanism verbatim -- no new
capture code. Command shape:

```bash
ET_VK_SDPA_COOPMAT=1 ./cmake-out-vk-etdump/examples/models/llama/llama_main \
  --model_path specs/009-e2e-tokrate-report/results/pte/<model>_<scheme>.pte \
  --tokenizer_path <model's tokenizer.model> \
  --prompt_file specs/001-minipc-baseline-benchmarks/results/prompts/shared_2048.txt \
  --num_bos 1 --temperature 0 --max_new_tokens 1 --seq_len 3072 \
  --etdump_path specs/011-sdpa-coopmat-e2e/results/etdump/<model>_<scheme>.etdump
```

- A configuration's `qk_dispatch_status`/`av_dispatch_status` are
  `confirmed` only if every measured occurrence of
  `sdpa_compute_attn_weights_*`/`sdpa_compute_out_*` in the trace contains
  `_coopmat`.
- Any `_tiled` kernel name for either position flips that configuration to
  `fallback` -- no e2e number is reported for it (FR-002/FR-007).

## E2E capture output

Same `e2e` JSON object shape as `001`/`006`/`009`
(`prefill_tokens_per_sec`, `decode_tokens_per_sec`, `prefill_tokens`,
`decode_tokens`, `num_runs`, `variance`, `run_metadata`) -- no new schema.
Stored under `results/raw/<model>_<scheme>_rep{1..5}.log`, matching `009`'s
naming convention exactly.

## `results/sdpa-coopmat-e2e-report.md`

Rules a consumer can depend on:

- Every one of the 6 configurations appears in the Blocked/Failed section
  or contributes both its prefill and decode rows to the main table --
  never partially or silently absent (FR-007/SC-004).
- No SDPA-coopmat-enabled tok/s value appears for a configuration whose
  dispatch status is not `confirmed` for both shaders (FR-002, SC-001).
- Every baseline (`009`) value is cited as coming from that report
  verbatim, not re-measured (FR-004).
- Every prefill row carries the inherited cross-session caveat note
  (research.md Decision 6); decode rows do not.
- One overall verdict statement appears, stating whether enabling SDPA
  coopmat helps real e2e tok/s and whether that agrees with `010`'s prior
  microbenchmark-level finding (FR-006, SC-003) -- any per-configuration
  divergence named explicitly, not averaged away (research.md Decision 5).
