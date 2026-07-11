# Research: MiniPC No-WMMA Baseline Benchmarks

All unknowns from the plan's Technical Context are resolved below; there are no
remaining `NEEDS CLARIFICATION` markers.

## Decision 1: How to exclude the coopmat/WMMA dispatch path for a controlled baseline

**Decision**: Add one small, off-by-default runtime toggle (an environment
variable, e.g. `ET_VK_FORCE_TILED_LINEAR=1`) checked at the top of
`can_use_q4gsw_coopmat()` in `backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.cpp`,
returning `false` immediately when set. Unset (the default), behavior is
byte-for-byte unchanged from today.

**Rationale**: This branch already has coopmat/WMMA dispatch merged
(commits `8c76ba4f5`, `5426101bf`), so "default behavior without WMMA" cannot
mean "just run HEAD" — it needs a controlled exclusion. A runtime toggle keeps
the baseline and the future WMMA-enabled comparison on the *exact same*
build/binary, which isolates the one variable that matters (dispatch path)
and avoids reintroducing toolchain/driver/ccache-state differences as a
confound. This also matches existing precedent in this exact codebase area:
`test_coopmat_linear_bench.cpp` already reads `COOPMAT_BENCH_CORRECTNESS_ONLY`
from the environment for an analogous benchmark-only control-flow override.
It satisfies constitution Principle III (explicit, testable gate) and
Principle I (zero behavior change when unset, so no new correctness risk).

**Alternatives considered**:
- *Build/run from a pre-coopmat git commit.* Rejected — compares two
  different binaries built at two different times, reintroducing exactly the
  measurement noise Principle IV warns against, and this branch's coopmat
  commits are interleaved with unrelated shader-consolidation work, so
  "pre-coopmat" is not a single clean commit to pin.
- *A compile-time CMake flag (two build directories).* Rejected — heavier
  than needed for a benchmark-only control; the env-var achieves the same
  isolation with a single build.

**Correction found during implementation**: at the decode regime (M=1), the
dispatch code never reaches the coopmat eligibility check at all —
`pick_linear_qw_shader`/`pick_linear_dqa_qw_shader` in `QuantizedLinear.cpp`
branch on `is_gemv_case` (M==1) *before* calling `can_use_q4gsw_coopmat()`,
and for `is_gemv_case=true` always select a GEMV-specialized kernel
(`q4gsw_linear_gemv_coop_*` for 4w, `linear_dq8ca_q4gsw_coop_*` for 8da4w —
confirmed by running the actual microbenchmark; see
`results/microbench_raw.log`). That kernel is a software-cooperative GEMV
algorithm, not the hardware coopmat/WMMA path, and it is dispatched
regardless of `ET_VK_FORCE_TILED_LINEAR`. So the toggle only has an effect at
the *prefill* regime; decode's "no-WMMA" baseline is simply today's only
decode dispatch path for these schemes, toggle or not. This does not change
what this feature measures (decode was never going to involve coopmat either
way) but a future WMMA-enabled comparison feature should not expect the
toggle to change decode numbers.

## Decision 2: Producing the six `.pte` exports

**Decision**: Use the existing `export_llama` CLI
(`examples/models/llama/export_llama_lib.py`) with `-V/--vulkan`,
`-qmode 4w` or `-qmode 8da4w`, and an explicit `--group_size` chosen as a
multiple of 32 (the coopmat K-tile size used by `linear_qw_coopmat`/
`linear_dq8ca_qw_coopmat`), plus `--max_seq_len` set to at least 3072
(2048 prefill + 1024 decode).

**Rationale**: `-qmode`'s accepted values already include exactly `4w` and
`8da4w` (`export_llama_lib.py`, `_qmode_type`) — no new export capability is
needed, confirming the spec's assumption. Picking a 32-aligned `--group_size`
now costs nothing for this tiled-only baseline but keeps the same `.pte`
directly reusable by the future WMMA-enabled comparison feature (FR-008/US3
require the Baseline Report, and by extension its exports, to be reusable).

**Alternatives considered**: leaving `--group_size` at whatever default is
picked without checking coopmat alignment — rejected because it would force a
silent re-export later, undermining the "reusable baseline" goal.

**Correction found during implementation**: `-d fp16` is rejected by
`get_vulkan_partitioner()` with `AssertionError: Vulkan backend does not
support non fp32 dtypes at the moment` — the export-level dtype override must
stay `fp32` (or be omitted); actually running fp16 on Vulkan requires the
separate `--vulkan-force-fp16` flag instead. The correct export invocation is
`-qmode {4w,8da4w} --group_size 32 --max_seq_length 3072
--max_context_length 3072 -V --vulkan-force-fp16` (no `-d` flag at all —
also note the real flag names are `--max_seq_length`/`--max_context_length`,
not `--max_seq_len`).

## Decision 3: Measuring e2e tokens/sec

**Decision**: Use the existing `examples/models/llama/main.cpp` (`llama_main`)
runner unmodified, via `--prompt_file <2048-token prompt> --num_bos 1
--temperature 0 --warmup true --max_new_tokens 1024 --seq_len 3072`, and read
`prefill_token_per_sec` / `decode_token_per_sec` directly from the
`PyTorchObserver {...}` JSON line it prints (from `Stats`/`stats.h`).

**Rationale**: `stats.h` already computes exactly the two metrics FR-002
requires, and this is the same runner/metric pair used to produce every other
Llama performance number already published in this repo's README — no new
measurement code, and the result is directly comparable to prior art.
`--warmup true` runs an internal discarded warmup pass before the timed
generation in the same process invocation, which is what satisfies FR-005's
"explicit warmup" requirement at the e2e tier (multiple repeats of the whole
command still supply the required variance across runs). `--num_bos 1` is
required to match how the prompt file's token count was verified (Decision
4) — the runner's default (`--num_bos 0`) does not prepend a BOS token, which
would silently shift the prompt token count by one.

**Alternatives considered**: a custom Vulkan-only timing wrapper around the
graph executor — rejected as duplicating `stats.h` for no benefit, and it
would produce a number that isn't comparable to the rest of the project's
Llama benchmarks.

## Decision 4: Constructing the fixed 2048-token prompt

**Decision**: Build one prompt file by encoding a fixed source passage with
the shared Llama 3 tokenizer and adjusting (trim/repeat) until it encodes to
exactly 2048 tokens (`bos=True`, matching `--num_bos 1` at runtime); store it
under this feature's `results/prompts/` directory.

**Rationale**: FR-002 fixes the prefill size at exactly 2048 tokens.

**Correction found during implementation**: this file is shared across all
three models, not one-per-model as originally planned here — `tokenizer.model`
is byte-identical (verified by `md5sum`) across the Llama 3.1 8B, 3.2 3B, and
3.2 1B checkpoints, so a single `results/prompts/shared_2048.txt` produces
exactly 2048 prompt tokens for all three (confirmed empirically via
`llama_main`'s reported `prompt_tokens` for each model). The original
per-model-file plan was a reasonable precaution before checkpoints were in
hand, but was unnecessary once verified.

**Alternatives considered**: reusing the README's existing "prompt length of
64" convention — rejected; the spec's clarification explicitly fixes this
feature's e2e sizes at 2048/1024, not 64.

## Decision 5: Sourcing real per-model GEMM/GEMV shapes for the microbenchmark tier

**Decision**: Derive each model's actual linear-layer shapes (prefill
M=2048, decode M=1; N/K from that model's own `dim`, `hidden_dim`, `n_heads`,
`n_kv_heads`, `vocab_size`) from the `params.json` shipped with each
downloaded checkpoint, at the point the microbenchmark cases are authored
(tasks/implementation phase) — not hardcoded in this plan.

**Rationale**: The real Llama 3.1 8B / 3.2 3B / 3.2 1B `params.json` files
come from the downloaded checkpoints, not from this repository (only a small
synthetic `demo_config.json` test config is checked in). Hardcoding
publicly-known config values here risks silently drifting from whatever
checkpoint is actually used for the export in Decision 2.

**Alternatives considered**: hardcoding known Llama dimensions directly into
this plan — rejected as a stale-data risk; the checkpoint's own `params.json`
is the authoritative source and should be read directly when needed.

## Decision 6: Microbenchmark harness extension

**Decision**: Add one new benchmark source to `backends/vulkan/test/custom_ops/`
reusing the existing `BenchmarkResult`/`ValueSpec`/`TestCase` machinery
(`utils.h`/`utils.cpp`), parameterized by the per-model shape lists from
Decision 5, and dispatched through the tiled shader only — either via
Decision 1's toggle, or (matching `test_coopmat_linear_bench.cpp`'s existing
convention) by choosing the Texture3D/Half output storage that already routes
to the tiled path instead of the coopmat one.

**Rationale**: `test_coopmat_linear_bench.cpp` already demonstrates this
exact pattern (same op, storage-type selects tiled vs. coopmat dispatch) for
synthetic Llama-3.1-8B-shaped configs; this feature mainly swaps in the three
real models' actual shapes across both the prefill and decode regimes rather
than building new benchmarking infrastructure.

**Alternatives considered**: none — this is a direct, low-risk extension of
prior art already in the codebase.
