# RDNA3 Discrete GPU (RX 7900 XTX) — Release/1.3 Baseline, Buffer Storage

**Date**: 2026-07-22
**Device / driver / commit**: same as `report.md` (texture) — RX 7900 XTX via `xraytracing02`, RADV
`2025.Q2.1 (LLPC)` / Vulkan 1.4.304, `release/1.3` @ `e2f18eb23`.
**Runner**: same binary as the texture run, `runners/rel1.3_linux_x86/llama_main` — storage type is a
`.pte`-embedded property, not a build-time flag, so no rebuild was needed.
**`.pte`**: the 6 pre-existing buffer-storage files at
`/sarc-c/gpusw/users/yanwen.xu/android-run/models/<model>_<quant>_buffer_ctx3072.pte` (no rel1.1/rel1.2
suffix). Coherence-checked individually first — all 6 loaded and produced sane 48-token output, confirming
release/1.3 compatibility before the timed sweep.
**Clocks**: floating only, same reason as the texture run (`NR` pinned — no passwordless sudo on this host).

## Results

### 4w

| Model | Prefill tok/s (median ± CoV, n=3) | Decode tok/s (median ± CoV, n=3) | Crash notes |
|---|---|---|---|
| 1B | 4830.19 ± 0.59% | 296.866 ± 1.72% | None |
| 3B | 1894.54 ± 0.76% | 135.174 ± 0.30% | None |
| 8B | 901.01 ± 0.07% | 85.464 ± 0.30% | None |

### 8da4w

| Model | Prefill tok/s (median ± CoV, n=3) | Decode tok/s (median ± CoV, n=3) | Crash notes |
|---|---|---|---|
| 1B | 9615.02 ± 3.53% | 269.850 ± 1.22% | None (crash-wise) — **decode collapses into `!!!!...` repetition, see Correctness note** |
| 3B | 3930.90 ± 1.24% | 123.342 ± 0.08% | None (crash-wise) — same `!!!!` collapse |
| 8B | 2062.44 ± 0.12% | 78.457 ± 0.30% | None; stays coherent |

All 18 reps (6 cells × 3 reps) completed with `rc=0`, zero crashes. Total sweep wall time ≈ 5.5 minutes.

## Correctness note

Same affected models as the texture run (1B and 3B break on 8da4w, 8B doesn't) — but the **failure mode is
different from texture**: instead of garbled Unicode/foreign-character sub-words, buffer storage collapses
into pure `!!!!!!!!...` repetition for the rest of the 1024-token decode. Storage type changes the *shape*
of whatever numerical issue is happening, not which models are affected — this narrows the likely cause
toward the 8da4w dequantization path itself (shared logic between storage backends) rather than something
specific to the texture image-sampling path. Still out of scope to root-cause here; numbers are valid
throughput measurements regardless (only exit code + presence of the stats line gates validity, not text
quality — same convention as the texture run and the M41 report's own 8da4w caveat).

## Texture vs. buffer comparison (this device)

| Model/quant | Texture prefill / decode | Buffer prefill / decode | Buffer vs texture |
|---|---|---|---|
| 1B/4w | 4841.61 / 276.786 | 4830.19 / 296.866 | ≈same prefill, buffer **+7.3% decode** |
| 1B/8da4w | 9225.23 / 252.717 | 9615.02 / 269.850 | buffer +4.2% prefill, +6.8% decode |
| 3B/4w | 1865.21 / 131.898 | 1894.54 / 135.174 | ≈same, buffer slightly ahead both |
| 3B/8da4w | 3764.71 / 120.623 | 3930.90 / 123.342 | buffer +4.4% prefill, +2.3% decode |
| 8B/4w | 867.43 / 84.756 | 901.01 / 85.464 | buffer +3.9% prefill, ≈same decode |
| 8B/8da4w | 2021.72 / 78.469 | 2062.44 / 78.457 | buffer +2.0% prefill, ≈same decode |

Buffer storage is consistently at or slightly ahead of texture on this GPU (0–7% depending on cell),
never behind — mirroring the mild "buffer-IO edge" pattern this project's cross-device T-vs-B study
already found on RDNA (see memory `tvb-storage-cross-device`), not the "2× weight cliff" once
(mis-)attributed to Adreno before that was found to be a floating-clock artifact.

## Reproduce

Identical to `report.md`'s Reproduce section, except `--model_path` points at the `_buffer_ctx3072.pte`
variant of each model/quant. Raw logs:
`/sarc-c/gpusw/users/yanwen.xu/android-run/results_rdna3_dgpu_buffer/*_rep{1,2,3}.log`, driver script
`run_sweep_buffer.sh` in the same directory.
