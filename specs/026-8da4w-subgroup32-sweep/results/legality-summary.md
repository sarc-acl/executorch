# User Story 1: SUBGROUP_SIZE=32 legality across 5 tile shapes

**Result: the historical `vkCreateComputePipelines` crash did NOT reproduce at any of
the 5 tested shapes** (`16×16/K16/1×1`, `64×64/K16/2×1`, `128×32/K16/1×2` — `025`'s
winner shape, `128×64/K32/2×2` — the shipped shape, `64×128/K16/4×1`), on driver
`c9861e9906d03fa2c7d48b804e1a1c80` (`f14c51b6f8`), board `xgpusw-debug08`. Every shape
compiled and dispatched genuine coopmat (kernel-name confirmed) with no crash.

This generalizes `025`'s T014 finding (one shape) and this session's earlier
`sg32test` probe (also one shape) to a proper spread — narrowing the axis to a
correctness/performance question, not a compile-legality question. See
`correctness_matrix.json`/`correctness-summary.md` for what actually varies by shape.
