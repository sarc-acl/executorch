# Contract: M5 EVT1 Linear + SDPA Coopmat Microbenchmark Data Formats

## Linear harness output (`test_coopmat_linear_bench`, extended `kShapes`)

Same per-case output format the binary already produces (per-case line:
kernel name, dispatch time mean/stdev, GFLOP/s, correctness PASS/FAIL) --
no format change, only more shape rows (1B/3B added alongside the
existing 8B rows) each tagged with a model label so the aggregation step
can group by model without guessing from K/N alone.

Two capture runs per model/scheme/op: `ET_VK_FORCE_TILED_LINEAR=1` (tiled)
and default (coopmat) -- both on the same binary, same shape, same
process invocation pattern already established this session.

## SDPA harness output (new `test_sdpa_coopmat_bench` build target)

Per `specs/010`'s existing header comment in this file: one case pair
(`ET_VK_SDPA_COOPMAT` unset / set) per target model, at that model's real
`head_dim`/`num_heads`/`num_kv_heads` and the fixed 2048-token prefill
workload, timed via the GPU query-pool, isolating only the
`sdpa_compute_attn_weights_*`/`sdpa_compute_out_*` dispatches (excluding
KV-cache-update/softmax, which are unaccelerated and identical either
way) -- unchanged from how `specs/010` already built and ran it on
MiniPC; only the target platform (M5 EVT1 Android arm64, not MiniPC
Linux) and the CMake build wiring (Decision 2) are new.

## SPIR-V inspection output: `results/spirv/<kernel_name>.dis.txt`

Plain `spirv-dis` output for each distinct coopmat kernel variant actually
observed dispatching in either capture (linear:
`linear_q4gsw_coopmat_buffer_*_half`, `linear_dq8ca_q4gsw_coopmat_buffer_*_half`;
SDPA: `sdpa_compute_attn_weights_coopmat`, `sdpa_compute_out_coopmat`).
Reused from `specs/007`/`010`'s existing citations if the compiled shader
is unchanged since (same SPIR-V bytes, confirmed via `md5sum`); freshly
captured otherwise. A companion one-line verdict records whether
`OpCooperativeMatrixLoadKHR`/`OpCooperativeMatrixMulAddKHR` were found.

## `results/linear-coopmat-microbench-report.md`

Structure a consumer can rely on (mirrors `specs/007`'s
`wmma-improvement-report.md` exactly, labeled M5 EVT1):

1. One time-weighted overall `4w` and `8da4w` speedup figure at the top.
2. The full 42-row case table (Model, Scheme, Op, Tiled (us), Coopmat
   (us), Speedup %, Significance, Dispatch, Correctness columns), sorted
   by model/scheme/op.
3. An Excluded section, present even if empty, per FR's exclusion rules --
   never silently dropped from the 42-case count.
4. A direct one-line comparison against `specs/007`'s MiniPC overall
   figures, per SC-004.

## `results/sdpa-coopmat-microbench-report.md`

Structure a consumer can rely on (mirrors `specs/010`'s
`sdpa-coopmat-microbench-report.md` exactly, labeled M5 EVT1):

1. A dispatch + correctness verification summary first, before any
   performance table (constitution Principle I).
2. One overall average speedup figure across valid models.
3. The 3-row (or fewer) per-model comparison table (Model, head_dim,
   num_heads, num_kv_heads, Tiled (us), Coopmat (us), Speedup,
   Significance columns).
4. An Excluded/Blocked section, present even if empty -- never silently
   dropped from the 3-model count (FR-006 / Edge Cases).
5. A direct one-line comparison against `specs/010`'s MiniPC overall
   figure, per SC-004.
