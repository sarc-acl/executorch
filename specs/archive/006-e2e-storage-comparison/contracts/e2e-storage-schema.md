# Contract: E2E Storage Comparison Data Formats

## New export CLI flag

```
python -m examples.models.llama.export_llama ... --vulkan-storage-override buffer
```

- Omitting the flag MUST produce byte-for-byte the same export behavior as
  today (default `texture3d`, matching the current hardcoded preference) —
  this is the core safety property research.md Decision 1 depends on.
- The flag accepts exactly `texture3d` or `buffer`; any other value is a
  usage error, not silently ignored.

## `results/pte/<model>_<scheme>_buffer.pte`

Naming convention: `<model>_<scheme>_buffer.pte`, distinct from `001`'s
`<model>_<scheme>.pte` (implicitly Texture3D) — never overwrites `001`'s
files.

## E2E capture output

Same `e2e` JSON object shape as `001`/`005` (`prefill_tokens_per_sec`,
`decode_tokens_per_sec`, `prefill_tokens`, `decode_tokens`, `num_runs`,
`variance`, `run_metadata`) — no new schema introduced (research.md
Decision 5).

## `results/e2e-storage-comparison-report.md`

Rules a consumer can depend on:

- Every one of the six configurations appears exactly once, either in the
  measured comparison table or in an explicit "blocked/failed" list with a
  stated reason — never silently absent (FR-006/SC-004).
- No `Buffer` configuration's timing appears unless its
  `smoke_check_status` is `pass` (FR-002) — a `fail` or `not_run` status
  MUST NOT have an accompanying tok/s number.
- Each measured configuration states `microbenchmark_consistency`
  (`consistent`/`diverges`) explicitly — SC-003.
