# 1B/3B follow-up validation of the shipped `8da4w` tile/loop winner

Follow-up to `dev-branch-production-validation.md`'s open item ("validate the same
winner on 1B/3B models"). **Result: confirmed win on both models**, same direction as
8B's +12.5%.

## Setup

- `dev` (`yanwen/dev-1.3`) @ `c1aa3eb81` (includes the shipped fix, commit `42aabb4e0`),
  `llama_main` freshly rebuilt this session (`cmake-out-android-vk`, linked
  2026-07-13 17:22).
- Device: M5 EVT1 primary board (`0000088f8e579c33` @ `sj1-dmckee-d01`). Driver was
  found on an unrecognized build (md5 `3880e697df8753a0d4a8ec3b394430a7`, matching
  neither documented hash) at session start — backed up to NFS
  (`vulkan.samsung.so.device-unknown-3880e697-backup-2026-07-13`) and reflashed to the
  documented default `f14c51b6f8`; re-verified exact md5 match
  (`c9861e9906d03fa2c7d48b804e1a1c80`) before any measurement, per user authorization
  (this session's driver state was flagged and confirmed with the user first — see
  session transcript; the secondary board `xgpusw-debug08` was also found drifted but
  intentionally left untouched per explicit user instruction).
- Clocks pinned 509/2730/663 MHz. Coherence-checked first for both models (short
  prompt, non-garbage, greedy decode).
- Workload: 2048-token prefill (`p2048_exact.txt`, `num_bos=1`) + 1024-token decode,
  `ET_VK_EXECUTE_NODE_THRESHOLD=16`, `--warmup=true`, `--ignore_eos --temperature=0`.
  PTEs: `llama3_2_{1b,3b}_8da4w_buffer_ctx3072.pte` (buffer storage, coopmat-eligible).

## Results (prefill tok/s, 3-run mean)

| Model | Pre-fix baseline (`specs/015`, same dbuf4/128×64/2×2/s64 config) | Post-fix (this session) | Delta |
|---|---|---|---|
| 1B `8da4w` | 723.00 (0.27% CoV) | **785.30** (786.78 / 790.12 / 779.00, CoV 0.59%) | **+8.6%** |
| 3B `8da4w` | 286.31 (1.55% CoV) | **320.35** (321.66 / 319.75 / 319.65, CoV 0.29%) | **+11.9%** |
| 8B `8da4w` (for reference, from `dev-branch-production-validation.md`) | 131.24 | 147.65 | +12.5% |

The pre-fix baseline numbers are reused from `specs/015` rather than re-measured
same-session, because between that measurement and this fix landing, `dev`'s `8da4w`
shader configuration did not change (still dbuf4 loop / 128×64/K32/2×2/s64 tile — the
same config `dev-branch-production-validation.md`'s own 8B "before" number, 131.24,
matches `specs/015`'s 8B pre-fix number of 130.05 within run-to-run noise, confirming
continuity). Decode tok/s is unaffected on all three models (~13.3/6.5, matching prior
figures within noise), as expected — decode is M=1 GEMV and doesn't dispatch the
tile-swept coopmat GEMM.

## Conclusion

The `specs/027` winner (`tsweep_t64x32k32g12s64`, dbuf2, 64×32/K32/1×2/s64) is a real
end-to-end win on **all three target models**, not just the 8B shape it was
originally validated on. Magnitude varies by model (+8.6% / +11.9% / +12.5% for
1B/3B/8B) but the direction is consistent — no model regresses. This closes
`dev-branch-production-validation.md`'s open follow-up item; no further action needed
before treating this as the validated production default across the full model
lineup.
