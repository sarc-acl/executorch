# Quickstart: SDPA Coopmat E2E Validation

Real device work on the `rocky-ryzen` MiniPC, reusing `009`'s exports and
capture tooling almost entirely -- this feature adds one env var to an
otherwise-identical procedure.

## Prerequisites

- `009`'s six `Buffer`-storage `.pte` exports still exist under
  `specs/009-e2e-tokrate-report/results/pte/`.
- `009`'s `cmake-out-vk` (standard) and `cmake-out-vk-etdump`
  (event-tracer-enabled) build trees still exist and are current --
  confirmed both during planning AND during implementation (`git status`
  on `backends/vulkan/runtime/` showed no changes either time) that no
  rebuild was ever needed for this feature.
- `010` is complete (its 66.8%-average microbenchmark finding is this
  feature's cross-check).
- Nothing else CPU/GPU-heavy running before any capture.

## 1. Confirm dispatch for one configuration first (US1, MVP)

Already spot-checked during planning for `llama-3.2-1b`/`4w` (research.md
Decision 1) -- both `sdpa_compute_attn_weights_coopmat`/
`sdpa_compute_out_coopmat` confirmed dispatching 16 times each (matching
its 16 layers). Repeat formally, saving the trace (this exact command is
what was actually run -- no changes needed from planning):

```bash
ET_VK_SDPA_COOPMAT=1 ./cmake-out-vk-etdump/examples/models/llama/llama_main \
  --model_path specs/009-e2e-tokrate-report/results/pte/llama-3.2-1b_4w.pte \
  --tokenizer_path /home/doremy/checkpoints/llama3_2_1b/original/tokenizer.model \
  --prompt_file specs/001-minipc-baseline-benchmarks/results/prompts/shared_2048.txt \
  --num_bos 1 --temperature 0 --max_new_tokens 1 --seq_len 3072 \
  --etdump_path specs/011-sdpa-coopmat-e2e/results/etdump/llama-3.2-1b_4w.etdump
```

Load with `executorch.devtools.Inspector`, confirm every
`sdpa_compute_attn_weights_*`/`sdpa_compute_out_*` kernel name contains
`_coopmat` (contracts/sdpa-coopmat-e2e-schema.md).

## 2. Repeat for the remaining 5 configurations

Any configuration whose dispatch doesn't confirm is recorded as
`fallback` with the actual kernel name -- never silently excluded from the
six-configuration count.

## 3. Capture e2e for every dispatch-confirmed configuration

Identical to `009`'s own procedure, plus the one env var:

```bash
ET_VK_SDPA_COOPMAT=1 ./cmake-out-vk/examples/models/llama/llama_main \
  --model_path specs/009-e2e-tokrate-report/results/pte/<model>_<scheme>.pte \
  --tokenizer_path <model's tokenizer.model> \
  --prompt_file specs/001-minipc-baseline-benchmarks/results/prompts/shared_2048.txt \
  --num_bos 1 --temperature 0 --warmup true \
  --max_new_tokens 1024 --seq_len 3072 \
  > specs/011-sdpa-coopmat-e2e/results/raw/<model>_<scheme>_rep{N}.log
```

5 reps, no concurrent GPU load (`ps aux`/`free -h` clean before each), same
budget-per-rep expectations `009` already documented (`--warmup true`
roughly doubles the untimed inference time per rep).

## 4. Compare and generate the report

```bash
python specs/011-sdpa-coopmat-e2e/scripts/compare_sdpa_e2e.py \
  --sdpa-raw-dir specs/011-sdpa-coopmat-e2e/results/raw \
  --baseline-report specs/009-e2e-tokrate-report/results/e2e-tokrate-report.md \
  --out specs/011-sdpa-coopmat-e2e/results/sdpa-coopmat-e2e-report.md
```

## 5. Sanity-check

- Every one of the six configurations appears -- either in the main table
  (both phases) or the Excluded/not-collected section with a reason.
- No SDPA-coopmat-enabled number appears for a configuration whose
  dispatch check didn't pass for both shaders, or whose e2e capture didn't
  reach the full 5 reps.
- Every prefill row carries the cross-session caveat inherited from `006`
  via `009` (research.md Decision 6).
- One overall verdict statement appears, stating whether it agrees with
  `010`'s prior finding; any per-configuration divergence is named
  explicitly, not averaged away.

## Handling an early-stopped collection

If data collection is stopped (e.g. by explicit user request) before all
six configurations complete their 5 reps, `compare_sdpa_e2e.py` does not
treat this as a blocking error -- it renders the report from whatever
configurations *did* reach 5 reps, and lists every other configuration
explicitly in the Excluded/not-collected section with the actual rep count
found (e.g. "2/5 reps"), rather than requiring all six before producing any
report at all. This is what actually happened in this feature's own run:
`llama-3.1-8b`'s two configurations were excluded this way, while the
other four configurations' full report was still produced (FR-007,
SC-004).
