# order-2 / b70: the blocked workgroup order as a production candidate (8da4w) and on 4w, Arc Pro B70

Arc Pro B70 (BMG G31), card `b70-0` only, `xe`, kernel 7.2.9-200.fc44, Mesa ANV 26.2.4; 2026-10-12 16:30 to 19:47 UTC.
ExecuTorch `exp/order2-b70` `9fc118736` (kernel-level builds `o2`, `o3`) and `exp/order2-b70-candidate` `e8c0cb1cc` (build
`cand2` = the campaign's final head `c3d67a821` + `candidate.patch`; gate and model level); box `exp/order2-b70`.
**Experiment and candidate, not a campaign result.** Record with the pre-registrations: `LOG.md`; tables: `tables/`.
VERIFIED = in the file named; INFERRED = a reading. M = 2048, prefill. `inc` = the incumbent kernel through
`b70-fused1`. Ratios are kernel time (or tok/s) from unrounded values; "band" = the shape's noise band.

## Part A: 8da4w

**Candidate (owner's decision of 2026-10-12 18:20): the dev profile `b70-fused1-order-i-n26-w16m`** = the final
configuration `b70-fused1` with the 8da4w kernel replaced by the generated `xe2bto` family: part **`i`** (remap
prologue alone) on every shape, part **`n26`** (block rule, **nb_max = 26**) **where the weight tensor is 16 MiB or
more** (K x N / 2 >= 2^24 bytes: the two 29.4 MB shapes of Llama 3.1 8B; nothing in 1B or 3B).
**A3 as written gives `i` alone**: every tested nb_max (13, 21, 26) is slower than `i` beyond the band on four to
six shapes. The A2 table shows the pattern the decision names, so it applies.

### Kernel table, old against new, every production shape (VERIFIED `tables/summary/a2-a3.md`, `a2.tsv`; three sittings, median (min..max))

| shape | K x N | W MB | `inc` us | `i` / `inc` | candidate runs | candidate / `inc` | candidate / `i` | band % |
|---|---|---:|---|---|---|---|---|---:|
| 8B w1_w3 | 4096 x 14336 | 29.4 | 2252.5..2258.8 | 0.9833 (0.9815..0.9838) | `n26` (5 x 23) | **0.9078** (0.9059..0.9094) | 0.9233 (0.9230..0.9244) | 0.47 |
| 8B w2 | 14336 x 4096 | 29.4 | 2561.3..2562.7 | 0.9970 (0.9963..0.9984) | `n26` (2 x 16) | **0.9260** (0.9260..0.9264) | 0.9293 (0.9276..0.9295) | 1.10 |
| 8B wq_wo | 4096 x 4096 | 8.4 | 601.8..603.5 | 0.9811 (0.9804..0.9829) | `i` | 0.9811 | 1 | 1.57 |
| 8B wk_wv | 4096 x 1024 | 2.1 | 165.2..165.3 | 0.9827 (0.9827..0.9830) | `i` | 0.9827 | 1 | 1.50 |
| 3B w1_w3 | 3072 x 8192 | 12.6 | 888.8..889.8 | 0.9871 (0.9861..0.9875) | `i` | 0.9871 | 1 | 1.79 |
| 3B w2 | 8192 x 3072 | 12.6 | 943.5..944.6 | 0.9956 (0.9953..0.9962) | `i` | 0.9956 | 1 | 1.39 |
| 3B wq_wo | 3072 x 3072 | 4.7 | 318.5..318.6 | 0.9834 (0.9833..0.9840) | `i` | 0.9834 | 1 | 0.13 |
| 3B wk_wv | 3072 x 1024 | 1.6 | 127.2..127.3 | 0.9824 (0.9820..0.9824) | `i` | 0.9824 | 1 | 0.04 |
| 1B w1_w3 | 2048 x 8192 | 8.4 | 629.5..629.8 | 0.9851 (0.9843..0.9851) | `i` | 0.9851 | 1 | 0.78 |
| 1B w2 | 8192 x 2048 | 8.4 | 659.6..661.9 | 0.9960 (0.9940..1.0009) | `i` | 0.9960 | 1 | 0.97 |
| 1B wq_wo | 2048 x 2048 | 2.1 | 146.4..146.5 | 0.9851 (0.9844..0.9858) | `i` | 0.9851 | 1 | 0.14 |
| 1B wk_wv | 2048 x 512 | 0.5 | 45.9..46.1 | 0.9875 (0.9853..0.9932) | `i` | 0.9875 | 1 | 0.69 |

The blocked parts on every shape, ratio to `i` (median): what A3 saw.

| shape (blocks under n13 / n21 / n26) | `n13` | `n21` (= round(F / 3)) | `n26` |
|---|---:|---:|---:|
| 8B w1_w3 (9 x 13 / 6 x 19 / 5 x 23) | 0.9228 | 0.9275 | 0.9233 |
| 8B w2 (3 x 11 / 2 x 16 / 2 x 16) | 0.9726 | 0.9288 | 0.9293 |
| 8B wq_wo (3 x 11 / 2 x 16 / 2 x 16) | 0.9813 | 0.9961 | 0.9942 |
| 8B wk_wv (1 / 1 / 1) | 1.0291 | 1.0307 | 1.0291 |
| 3B w1_w3 (5 x 13 / 4 x 16 / 3 x 22) | 1.0166 | 1.0428 | 1.0092 |
| 3B w2 (2 x 12 / 2 x 12 / 1) | 1.0286 | 1.0277 | 1.0147 |
| 3B wq_wo (2 x 12 / 2 x 12 / 1) | 1.0491 | 1.0417 | 1.0274 |
| 3B wk_wv (1 / 1 / 1) | 1.0266 | 1.0196 | 1.0266 |
| 1B w1_w3 (5 x 13 / 4 x 16 / 3 x 22) | 0.9551 | 0.9587 | 0.9565 |
| 1B w2 (2 x 8 / 1 / 1) | 1.0172 | 1.0028 | 1.0034 |
| 1B wq_wo (2 x 8 / 1 / 1) | 1.0429 | 1.0393 | 1.0422 |
| 1B wk_wv (1 / 1 / 1) | 1.0023 | 1.0057 | 1.0023 |
| one 8B layer (2 w1_w3 + w2 + 2 wq_wo + 2 wk_wv), against `i` / against `inc` | 0.9499 / 0.9376 | 0.9414 / 0.9295 | **0.9386 / 0.9266** |

- `i` alone: one 8B layer 0.9873 of `inc`. The candidate profile, from the same medians (INFERRED): 0.9382 to 0.9393 of
  `i`, **0.9254 to 0.9276 of `inc`**.
- The block computation costs by itself: shapes launched in one block run up to 4 % slower under a blocked part
  than under `i` (8B wk_wv + 2.9 to 3.1 %, 3B wk_wv + 2.0 to 2.7 %, 1B wq_wo + 3.9 to 4.2 %). That is why no nb_max
  qualifies and why the predicate is the candidate. Order-1's 4 % on 8B wq_wo with two blocks is not there.
- Counters, once per variant (VERIFIED `tables/summary/k1.md`): 8B w1_w3 memory bytes / 6.4 and sampler hit 22.5 to
  89.6 % under `n26`; 8B w2 / 1.83, hit 24.7 to 54.3 %. Bytes rise 1.9 times on 1B w1_w3 under `n26` while its time
  falls 4.4 %: bytes do not order the times.
- Sampled production-diff before any timing: `inc`, `i`, `n13`, `n21`, `n26` x 1B, 3B, 8B: 15 of 15 ALL PASSED
  (`tables/c2-20261012T1719Z/correctness.tsv`).

### Gate (VERIFIED `tables/g1/gate.txt`, `verify.out`, `verify_diff.txt`)

**`GATE_PASS`**: the fused-port campaign's `gate.sh --sdpa` (its tools unedited, its thresholds: clock 2457 MHz, 7
runs), candidate profile on build `cand2` against the campaign's parent. 35 PASS lines, no FAIL: production-diff
12 of 12 (1B, 3B, 8B x 4w, 8da4w x buffer, texture3d), correctness 28 of 28 and 4 of 4 rank-3 cases PASSED (`rc=1`,
equal to the parent control, as in the campaign's gate: `verify.out` prints `[correctness] FAILED` for the rank-3
8da4w cases that take the non-coopmat fallback), next token SAME in six cells on three prompts, default
against tiled SAME, decode rc 0, SDPA tiers 12 passes each with 0 mismatches, `E2E5_OK`. **`verify_diff.txt` reads
`VERIFY_DIFFERENT (1 line)`** (the campaign's final gate: `VERIFY_SAME`): the line is the 8da4w kernel list, 10 x
`xe2bto ...i` + 2 x `xe2bto ...n26` per storage instead of 12 x `xe2bt`; that is the change itself, and the gate does
not decide on that file.

### Model level (VERIFIED `tables/model/model.md`, `model.tsv`; build `cand2`, three pairwise interleaved sessions, 7 valid runs per arm, all `E2E5_OK`)

F = `b70-fused1` (the campaign's final configuration), I = `b70-fused1-order-i`, C = the candidate profile.

| 8da4w | prefill tok/s F | I | C | I / F | C / F | C / I | decode I / F, C / F, C / I |
|---|---|---|---|---:|---:|---:|---|
| Llama 3.1 8B | 4623.0 (4571.4..4708.1) | 4665.1 (4612.6..4675.8) | 4853.1 (4807.5..4876.2) | 1.0091 | **1.0498** | **1.0403** | 1.0009, 1.0019, 1.0028 |
| Llama 3.2 3B | 9990.2 (9941.8..9990.2) | 10088.7 (10039.2..10138.6) | 10039.2; 10088.7 (9706.2..10189.1) | 1.0099 | 1.0049 | 1.0000 | 1.0060, 0.9940, 0.9985 |
| Llama 3.2 1B | 24381.0; 24674.7 (24381.0..25924.1) | 24674.7 (24381.0..25284.0) | 24674.7 (23272.7..24975.6) | 1.0120 | 1.0000 | 1.0000 | 1.0029, 0.9971, 0.9970 |

(Medians of each arm in its two sessions, one value where equal; spread over both. Each ratio from the session
that holds both arms.) **8B prefill + 5.0 % over the final configuration, + 4.0 % over `i` alone; 3B and 1B: the
candidate runs `i` there, at most 1 %, inside the campaign's +- 2 % band; decode unchanged** (0.994 to 1.006).
Prefill time has 1 ms resolution: tok/s moves in steps of 1.2 % (1B), 0.5 % (3B), 0.2 % (8B).

### Deliverable

`candidate.patch` (7 files: `gen_xe2.py`, the three generated glsl files, rows in `Xe2Linear.cpp`, profile blocks in
`Overrides.cpp`, candidate golden file). Applies to `c3d67a821`; build `cand2` there: `BUILD_BOTH_OK`, sarc golden
PASS, **1572 of 1572 pre-existing SPIR-V files byte-identical**, candidate golden PASS (12 variants, the B580
build's hashes). With neither part selected (`--order=`) the generator's output is byte-identical to today's.

## Part B: 4w (kernel level)

Incumbent `sarc_linear_q4gsw_coopmat_sweep_t128x128k16g82s16m8flib`: grid (N tiles, M tiles), N-fastest; a workgroup
is 16 hardware threads (F = 128 in flight, INFERRED), holds 64 K bytes of weights and 256 K bytes of fp16
activations. T0: only 8B w1_w3 (14.0 W of memory traffic) and 8B w2 (8.5 W) re-read their weights; every 1B and
3B shape is cached (`tables/t0-20261012T1635Z/counters.tsv`).

| 8B shape | best variant (blocks) | against `ident` (s1 / s2 / s3) | band % | **verdict** | against `inc` |
|---|---|---|---:|---|---|
| w1_w3 K4096 x N14336 | `n21` (6 x 19) | 0.9283 / 0.9280 / 0.9330 | 0.29 | **GAIN** (6.7 to 7.2 %) | 0.9841 / 0.9830 / 0.9885 |
| w2 K14336 x N4096 | `n11` (3 x 11) | 0.9713 / 0.9703 / 0.9730 | 0.28 | **PARTLY** (2.7 to 3.0 %) | 1.0420 / 1.0418 / 1.0416 |
| wq_wo, wk_wv | any | 0.9987 to 1.0029 | 0.22, 0.13 | no change | 1.0547 to 1.0606 |

(VERIFIED `tables/summary/b2.md`; widths 43, 28, 21, 11; every width from 11 to 38 is GAIN on w1_w3.)
**But `ident` is 5.6 to 7.3 % slower than the incumbent on all four shapes**: the remap prologue costs this kernel
more than the order wins. Over one 8B layer every variant is slower than the incumbent (`n11` 1.013 to 1.014,
`n21` 1.016 to 1.019, `ident` 1.062 to 1.063). B3 (one sitting, `tables/summary/b3.md`): on 1B and 3B `n21` is 0 to
1.3 % slower than `ident` and `ident` 3.5 to 6.9 % slower than `inc`; no shape gains. Model-level number: **not
reached** (part A's dev profile cannot select the 4w experiment rows). No gate, no candidate for 4w.
Patch published 16:49 (the task exchange directory), after build and production-diff (6 of 6 ALL PASSED).

## What is not known

- Why the same prologue gains 1 to 2 % on the 8da4w kernel and costs 3.5 to 7.3 % on the 4w kernel, and why the
  block computation costs up to 4 % on one-block shapes: no shader dump of these kernels was taken (not a step
  of this task). A 4w order kernel without that cost was not tried. **Not ISA-verified.**
- 1B w1_w3 (8.4 MB) is 4.4 % faster under `n26` and runs `i` in the candidate: no weight-size threshold selects it
  without the two 12.6 MB shapes of 3B, one of which is slower beyond its band. A predicate on another quantity
  was not tried. Whether 16 MiB is right for shapes between 12.6 and 29.4 MB: no production shape lies there.
- Whether nb_max 26 is the best: 13, 21, 26 only; on 8B w1_w3 13 and 26 are equal (the width does not follow F
  there), on 8B w2 two blocks of 16 beat three of 11.
- F = 128 for the 4w kernel and the cache size are arithmetic, not measured.
- Model level: one session per pair (the campaign's count); 1B and 3B ratios are at the tool's resolution.
- Counters of `i` on 3B wq_wo / w2: the counter tool did not separate the two cases. B3 is one sitting.
- The gate's `verify_diff` differs from the parent control in the kernel list (above).

## For the owner: one decision pending (`LOG.md`, State block)

- **The B580 task published its own generator mechanism for the per-shape profile (`0002`, 18:30) after I had
  written mine (18:25) and before my gate; I did not see it.** Two implementations of that profile exist; the
  kernels are the same. A: accept `candidate.patch` as it is; B: rebase on `0002`, rebuild, gate and sessions again.

- `candidate.patch` holds three things of mine on top of the B580's generator patch (its `git am` conflicts on this
  tree; applied by its README's fallback): the `b70-fused1-order-<part>` profiles, one generated block in
  `fused_variants_b70()` so that they keep the fused attention variants, and the mixed profile with its predicate.
- The second golden file (`order-spirv.json`) is the B580 patch's arrangement; this card's TASK.md has no entry on
  it. `sarc/golden/spirv.json` is untouched.
- Rules: no foreign GPU process, no invalidated pass, no kernel `xe` / `drm` message, `observation_paranoid` 0
  throughout, services untouched, nothing pushed (`LOG.md` entry 9). Reviewed once: APPROVED (`LOG.md` entry 10).
