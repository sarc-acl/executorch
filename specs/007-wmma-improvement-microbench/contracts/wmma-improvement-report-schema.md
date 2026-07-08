# Contract: WMMA Improvement Data Formats

## Harness output: `RESULT` CSV line (unchanged format, reused as-is)

```
RESULT,<model>,<scheme>,<regime>,<storage>,<op>,<M>,<K>,<N>,<mean_us>,<stddev_us>,<iterations>,<kernel>
```

Identical to `004`'s format -- no changes to the harness or its output. This
feature's new capture uses the same binary with `ET_VK_FORCE_TILED_LINEAR`
unset; only rows with `regime=prefill`, `storage=buffer`, and `op != lm_head`
are used (see `research.md` Decision 3 for the `lm_head` exclusion).

## SPIR-V inspection output: `results/spirv/<kernel_name>.dis.txt`

Plain `spirv-dis` output for each distinct Buffer/Buffer coopmat kernel
variant actually observed in the WMMA capture (at most two:
`linear_q4gsw_coopmat_buffer_buffer_half`,
`linear_dq8ca_q4gsw_coopmat_buffer_buffer_half`). A companion one-line
verdict file or header comment records whether
`OpCooperativeMatrixLoadKHR`/`OpCooperativeMatrixMulAddKHR` were found.

## `results/wmma-improvement-report.md`

Structure a consumer can rely on:

1. **One time-weighted overall improvement figure** at the very top (FR-008,
   SC-005) -- computed per `research.md` Decision 6, stated as a plain
   percentage (e.g. "WMMA is ~X% faster than tiled, time-weighted across
   measured prefill linear ops").
2. **The full 42-row case table** (`data-model.md`'s WMMA Comparison Case),
   sorted by `model`, `scheme`, `op`, including each row's `dispatch_status`,
   `correctness_verified`, and `significance` columns -- no result is
   presented without its iteration count and stdev (FR-003, SC-003).
3. **An Excluded / Out-of-Scope section**, present even if empty (stating
   "none" explicitly): `lm_head` (Decision 3), decode/GEMV (FR-006), and any
   case failing dispatch or correctness verification (FR-004, FR-007) --
   never silently omitted (FR-009, SC-001).
4. **A correctness-verification summary**: which kernel families were
   SPIR-V-inspected, the instructions found, and which existing correctness
   test(s) cover them (FR-007, SC-004) -- so a reader never mistakes an
   unverified number for a validated one.
