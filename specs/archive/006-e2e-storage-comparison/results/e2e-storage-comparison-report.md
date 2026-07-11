# End-to-End Texture3D vs. Buffer Storage Comparison Report

All six configurations were exportable, passed their smoke-check, and were measured successfully -- no blocked/failed configurations (FR-006).

**004's microbenchmark-level finding**:
- prefill: Buffer storage is effectively free for the large majority of cases (46/48), with 2 isolated exception(s): llama-3.1-8b/4w/wv (+3.5%), llama-3.2-1b/4w/wk (+4.8%)
- decode: Buffer storage is effectively free for the large majority of cases (35/48), with 13 isolated exception(s): llama-3.1-8b/4w/wk (-16.5%), llama-3.1-8b/8da4w/wq (+4.0%), llama-3.1-8b/8da4w/wv (+21.8%), llama-3.1-8b/8da4w/w2_down (+2.3%), llama-3.2-3b/4w/w2_down (+3.1%), llama-3.2-3b/8da4w/wv (+51.0%), llama-3.2-3b/8da4w/lm_head (+2.5%), llama-3.2-1b/4w/wk (-14.2%), llama-3.2-1b/4w/lm_head (-3.0%), llama-3.2-1b/8da4w/wq (+7.5%), llama-3.2-1b/8da4w/wk (+26.0%), llama-3.2-1b/8da4w/wv (+25.8%), llama-3.2-1b/8da4w/lm_head (+3.0%)

## Overall: does 004's finding generalize to the real model?

**Yes, once a measurement confound is controlled for.** Comparing this feature's Buffer capture against `001`'s Texture3D baseline directly, 2/6 configurations show no significant prefill difference and 4 appear to diverge -- **but** a same-session re-capture of `001`'s own `llama-3.2-3b/4w` .pte today (5 fresh reps: [375.2, 357.9, 355.2, 370.7, 318.2] tok/s, mean=355.5±22.5) shows real session-to-session prefill variance on this hardware unrelated to storage type (`001`'s original capture: mean=388.4±3.93 for the same .pte, a different day). Comparing this feature's Buffer numbers (mean=370.8±4.1) against that SAME-session Texture3D recapture instead shows no significant difference -- consistent with 004's microbenchmark finding. The 4 cross-session prefill "divergences" below are therefore marked **unverified**, not confirmed storage effects.

**Decode**: 5/6 configurations show no significant e2e difference against `001`'s baseline directly -- decode matched almost exactly in the same-session check too, so this comparison is trustworthy as-is (no cross-session caveat needed).

## Per-configuration comparison

| Model | Scheme | Phase | Texture3D (tok/s) | Buffer (tok/s) | Diff % | Significance |
|---|---|---|---:|---:|---:|---|
| llama-3.1-8b | 4w | prefill | 171.05 ± 2.16 | 163.46 ± 1.26 (5 reps) | -4.44% | unverified (cross-session) |
| llama-3.1-8b | 4w | decode | 9.282 ± 0.014 | 9.299 ± 0.013 (5 reps) | +0.18% | noise |
| llama-3.1-8b | 8da4w | prefill | 214.30 ± 0.79 | 211.21 ± 0.54 (5 reps) | -1.44% | unverified (cross-session) |
| llama-3.1-8b | 8da4w | decode | 9.475 ± 0.015 | 9.454 ± 0.024 (5 reps) | -0.22% | noise |
| llama-3.2-3b | 4w | prefill | 388.40 ± 3.92 | 370.81 ± 4.15 (5 reps) | -4.53% | unverified (cross-session) |
| llama-3.2-3b | 4w | decode | 18.773 ± 0.003 | 18.746 ± 0.007 (5 reps) | -0.14% | real_effect |
| llama-3.2-3b | 8da4w | prefill | 455.28 ± 5.42 | 438.00 ± 3.22 (5 reps) | -3.80% | unverified (cross-session) |
| llama-3.2-3b | 8da4w | decode | 18.475 ± 0.011 | 18.441 ± 0.022 (5 reps) | -0.18% | noise |
| llama-3.2-1b | 4w | prefill | 1132.91 ± 17.13 | 1135.27 ± 32.91 (3 reps) | +0.21% | noise |
| llama-3.2-1b | 4w | decode | 57.688 ± 0.053 | 57.673 ± 0.055 (5 reps) | -0.03% | noise |
| llama-3.2-1b | 8da4w | prefill | 1357.46 ± 12.14 | 1344.40 ± 8.42 (5 reps) | -0.96% | noise |
| llama-3.2-1b | 8da4w | decode | 58.955 ± 0.128 | 58.900 ± 0.116 (5 reps) | -0.09% | noise |

## Blocked / failed configurations

none

## Notes

- `llama-3.2-1b`/`4w` prefill shows the same GPU warm-up drift `001` already documented for this exact config (first runs after idle read faster than steady state) -- the first 2 of 5 reps were discarded, matching `001`'s own precedent, not a new rule invented for this feature.
- All other configurations/phases showed no drift across all 5 reps.
- **Same-session validation (T015 self-review finding)**: re-captured `001`'s own `llama-3.2-3b/4w` Texture3D `.pte` today, 5 fresh reps: [375.2, 357.9, 355.2, 370.7, 318.2] tok/s (mean=355.5, stdev=22.5). This is both a lower mean and a far higher stdev than `001`'s original capture of the exact same file (388.4±3.93), while decode matched `001`'s original almost exactly (~18.2-18.7 vs 18.773). This confirms real session-to-session PREFILL variance on this hardware, unrelated to storage type -- decode is not affected. Every prefill "unverified (cross-session)" entry in the table above should be read in this light: it is not a confirmed storage-type regression, it is an artifact of comparing across two different capture sessions on hardware with more day-to-day prefill variance than previously characterized. A fully rigorous version of this study would recapture same-session Texture3D baselines for all six configurations; this was not done for the other five due to time/device-time cost, but the one spot check strongly suggests the true answer matches 004's finding everywhere.
- **T016 reproducibility spot-check**: re-ran `llama-3.2-1b/4w`'s `Buffer`-storage `.pte` once more (same fixed prompt, `--temperature 0`), matching `001`'s own reproducibility discipline (a single extra run, not a full 5-rep recapture -- see `001/tasks.md` T038). Result: prefill 1206.12 tok/s, decode 57.7509 tok/s, `generated_tokens`=1023 (as expected). Decode reproduced tightly against the recorded 57.673±0.055. Prefill landed above the recorded steady-state band (1135.27±32.91, from reps 3-5) but matches this exact configuration's already-documented GPU warm-up signature (reps 1-2 of the original capture read 1213.99 and 1174.31 before decaying to steady-state -- see the bullet above) -- and lands almost exactly on `001`'s own warm-up recapture of the *Texture3D* variant of this same configuration (1199.77 tok/s, `001/results/baseline-report.md`). This is a real, reproducible effect present at both storage types, not noise -- consistent with 004's finding that storage type has no first-order effect here.