# specs/041 — dbuf4 tile sweep, microbenchmark-scored, two drivers

**Date:** 2026-08-09 · **Device:** M51 (primary) · **Status:** measurement complete, see `results/`

## Why

The dbuf4 workstream (six tsweep shader families adding *loop structure* as a
sweep dimension on top of specs/036's tile geometry) has so far been scored only
by **e2e `llama_main` tok/s** — specs/036's `measure_android.py` runs the
microbenchmark purely as a correctness gate. Two gaps followed:

1. **No kernel-level ranking existed.** The one completed dbuf4 e2e result
   (`sweep_summary_..._dq8ca_dbuf4.json`, `+30.92%`) is not usable as-is: its
   `noise_floor_cov` is 12.4%, `remeasure_pending` is non-empty, and the seed
   token needed to separate loop-structure effect from tile effect is itself
   pending. It was also collected on the **secondary** board (debug08).
2. **Driver sensitivity was unmeasured** — is a tile ranking a property of the
   kernel or of the shader compiler?

## What was run

For each dbuf4 tile token, `test_llama_microbench` at the 12 real Llama linear
dispatch shapes (llama-3.1-8b / 3.2-3b / 3.2-1b, prefill M=2048), under two
drivers.

| | |
|---|---|
| Board | primary M51, `adb -s 0000088f8e579c33` @ `sj1-dmckee-d01` |
| Clocks | PinnedMax — GPU 980000, MIF 5333000, INT 934000, verified by sysfs readback |
| Driver A | `f14c51b6f8` (known-good), md5 `c9861e9906d03fa2c7d48b804e1a1c80` |
| Driver B | `e0da99c1d1` "spal,pal_common: fix ResourceTracker not reaching HW via decorators" (Keonjoo Lee, 2026-07-21), md5 `15de5298506fedada6f39f9015d934a2` |
| Candidates | 48 `q4gsw` + 112 `dq8ca` dbuf4 tokens already present in the tsweep yamls |
| Harness | `test_llama_microbench` @ `6cbd286cbf661eed6648d557cde58f26` |
| Driver | `scripts/microbench_sweep.py` |

## Scope decisions, each verified on device rather than assumed

- **Prefill only.** At M=1 the linear op takes the `is_gemv` short-circuit and
  dispatches `linear_*_coop`, *not* the tsweep coopmat variant — decode rows come
  back `dispatch=not_applicable`, `kernel=linear_q4gsw_coop_...`, identical for
  every token. Sweeping decode would measure the same kernel 160 times.
- **Buffer only, one scheme per invocation.** A `q4gsw` token only changes
  4w+buffer; a `dq8ca` token only changes 8da4w+buffer; `texture3d` is the tiled
  baseline. With `--scheme`/`--storage`/`--regime` an invocation drops from 96
  cases to 12 — an ~8× wall-clock cut over a 160-token sweep.
- **Correctness gate kept on every token.** The prior e2e sweep saw 62/112 dq8ca
  tiles fail correctness; publishing GFLOP/s for a kernel computing garbage is
  this sweep's main exposure.
- **Ranked on `kernel_gflops`** (derived from `kernel_us`, the linear shader
  alone), not the case-level mean. For 8da4w the case mean also covers the
  activation `quantize_and_pack` dispatch, which understates the linear shader
  and does so unevenly across tiles.

## Harness changes required (`test_llama_microbench.cpp`)

1. `--scheme=` / `--storage=` / `--regime=` filters, threaded into **both**
   `generate_linear_perf_cases()` and `generate_correctness_cases()` — the
   correctness matrix is a separate generator and is easy to miss, in which case
   the gate keeps running all schemes and the saving evaporates.
2. **The RESULT line grew a trailing `kernel` field** (full dispatched shader
   name). This is a correctness requirement, not cosmetics: `r.variant` is
   `kernel_class()`, which collapses every shader to `coopmat`/`coop`/`tiled`,
   and `r.dispatch` is derived from `r.variant`. Without the full name an
   unrecognized `ET_VK_*_COOPMAT_VARIANT` token would silently fall back to the
   default kernel and still produce a plausible-looking ranking. The sweep driver
   asserts the requested token appears in `kernel` on **every** row.
   Appended last, after the specs/021 fields, so it is backward compatible:
   `.shared-context/scripts/aggregate_microbench_results.py` gates on
   `len(parts) < 13` (a minimum, not an exact count) and indexes at most
   `parts[14]`, so it ignores the new trailing field rather than breaking.

## Reading the results

- Ranking is **microbench-only, not a production pick.** specs/026 vs specs/027
  showed the 8da4w microbench ranking nearly inverting against e2e (the
  microbench pick lost 2.7% e2e).
- Measured at `--group-size=128`, matching the exported PTEs. See the harness
  bug below for why the original `kGroup = 32` invalidated a first attempt.
- Not comparable to `specs/036-portable-device-sweep/results/sweep_summary_*_dbuf4.json`
  — those are e2e tok/s on the secondary board.

## Harness bug found mid-run: `kGroup = 32` silently deleted 44% of the sweep

The first driver-A pass was run and then **discarded**. Raw data kept as evidence
in `results/superseded-group32/`.

`test_llama_microbench` hardcoded `constexpr int64_t kGroup = 32`, its comment
calling that "the perf sweep's real-export group_size". That is wrong: the export
recipe uses `quantization.group_size=128` (`setup/README.md:247,429`;
specs/036 `protocol.md` — "this box's buffer ptes: 128"; and both existing dbuf4
sweep summaries record `group_size 128`). The 32 is almost certainly the
**embedding** group (`embedding_quantize: 4,32`, `setup/README.md:277`), a
different tensor.

The consequence is not a mislabel, it is missing data. `QuantizedLinear.cpp:344`
bails to the tiled kernel unless `group_size % tile_k == 0`:

| group_size | q4gsw tokens able to dispatch | dq8ca | total |
|---|---|---|---|
| 32 (old) | 26/48 | 63/112 | **89/160** |
| 128 (correct) | 48/48 | 112/112 | **160/160** |

So 71 of 160 tokens could never run their own kernel — and not at random: exactly
the `tile_k ∈ {64,128}` subspace. Verified directly on device, same token, only
the group size changed:

```
group_size=32   kernel: linear_q4gsw_tiled_buffer_texture2d_half              dispatch: fallback_tiled  1796 GFLOP/s
group_size=128  kernel: linear_q4gsw_coopmat_tsweep_dbuf4_t128x32k64g22s32... dispatch: confirmed       2565 GFLOP/s
```

Fixed by replacing the constant with a `--group-size=` flag defaulting to 128, and
by recording `group_size` + `binary_md5` in every jsonl record so a results file
is self-describing (nothing in the old records distinguished them).

### Latent inconsistency worth recording (not fixed here)

`tile_constraints.derive` accepts a tile if **either** `group_size % k == 0`
**or** `k % group_size == 0`, while the runtime requires only the first. That is
why the offline legality check called all 160 tokens legal at group 32 while 71
of them fell back at dispatch. It does not bite at `group_size=128` — every
`tile_k ∈ {16,32,64,128}` divides 128 — and tightening it would change the legal
universe specs/036 swept over, so it is left alone and recorded here.

## Incidental findings

- The primary M51 was found on an **unrostered driver** (`f6c04984b5503763c727755bf7dd54ca`,
  56,344,784 B), not the documented known-good. Backed up to NFS as
  `vulkan.samsung.so.device-unknown-f6c04984-backup-2026-08-09` before reflashing.
- `/sys/kernel/gpu/{min,max}_freq` **do not exist** on this device; the nodes are
  `gpu_{min,max}_clock`. This corroborates the single-source note in
  `m51-tui/src/clocks.rs` and means `pin_freqs.sh`'s two GPU writes *and its own
  verification step* are inert. Pin state must be confirmed via
  `/sys/class/devfreq/23400000.sgpu/*` and MIF/INT `cur_freq`.
