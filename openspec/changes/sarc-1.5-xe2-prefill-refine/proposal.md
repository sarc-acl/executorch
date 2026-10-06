# sarc-1.5-xe2-prefill-refine

Dev zone only (2026-10-04 to 2026-10-06). Nothing here is promoted and no release-zone or upstream file is touched. Branch
`topic/xe2-prefill-refine`, forked from the head of `topic/780m-prefill-refine` at `6a7cc8cc6` (the parent);
no PR. Device tag `xe2`. Day-by-day detail and every table are in `STATUS.md`.

## Why

On Intel Xe2 `dev/1.5` runs the Llama prefill with the SARC cooperative-matrix kernels for the linear layers
only; attention is stock. The e2e evidence put attention at 30 to 40 % of a 2048-token prefill and the two
linear kernels at 37 to 39 % (4w) and 22 to 25 % (8da4w) of the matrix roofs. This change measures how much of
that can be recovered from the dev zone on the Arc Pro B70.

## What

All under `backends/vulkan/runtime/graph/ops/{glsl,impl}/sarc_dev/`, `backends/vulkan/test/sarc_dev/` and this
directory. Shared dev-zone files (`impl/sarc_dev/Overrides.cpp`, ten sweep yamls) are changed by appended
blocks between `xe2 begin` / `xe2 end` comments only. The one line that is not an append is in
`test_llama_microbench.cpp`: the `NO_MASK_FILL` pairing check now also recognises the Xe2 tile tokens
(`...m8nf`), so the check covers the Xe2 kernels; nothing was loosened.

- `impl/sarc_dev/Xe2Sdpa.cpp`: `kUnverified` SDPA base rows for `bmg g21` / `bmg g31`, active only while
  `ET_VK_SARC_DEV_PROFILE` names an `xe2-*` profile. No hook and no release-table row is needed to reach the
  SDPA kernels. `impl/sarc_dev/Xe2Linear.cpp` registers the Xe2 linear families.
- SDPA kernels at the shapes Xe2 exposes (MMA 8x16x16, fp16 x fp16 -> fp32, subgroup 16): Xe2 variants of the
  780M `pk` / `sweep` / `ml` families and new families `sarc_sdpa_qk_coopmat_xe2`, `..._xe2c`,
  `sarc_sdpa_av_coopmat_xe2` with a fragment-contiguous shared-memory layout.
- Linear: `sarc_dev_linear_dq8ca_coopmat_zpg_xe2bt` (8da4w, texel-wise weight staging for Xe2 tiles),
  `sarc_dev_linear_q4gsw_coopmat_xe2bx` and `..._xe2s` (4w texel-wise and split staging; screened out, kept).
- From the parameter search: two 4w variants of the release body in the dev sweep family
  (`sarc_linear_q4gsw_coopmat_sweep_t128x128k16g82s16m8flib`, `..._t128x128k32g84s16m8flw`), one QK^T variant
  (`sarc_sdpa_qk_coopmat_xe2c_t64x128k32g82s16m8nf`) and two attn*V variants with subgroup size 32
  (`sarc_sdpa_av_coopmat_xe2_t{64x64,128x64}k64g44s32m8`). The thousands of variants the search measured were
  never written to the tree: they exist only in the sweep builds, and the record is the seeded configuration
  lists and the result rows under `results/xe2/sweep/`.
- Measurement only: shader-clock phase twins, rms / maximum error against the fp32 reference in the SDPA
  correctness cases, and two softmax variants (`sarc_dev_sdpa_attn_weights_softmax_xe2`, `..._xe2sg`) that can
  only be dispatched through `tools/hook-sdpa-softmax.patch`, a local release-zone patch that is not committed
  as source and was applied only in the builds `hook1` / `hook2`.
- Everything generated comes from `tools/gen_xe2.py`, which reads the release bodies and never writes them.

**Final profile: `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=xe2-refine5`** (candidates 1, 2 and 5):

| op | kernel | shapes |
|---|---|---|
| SDPA QK^T | `sarc_sdpa_qk_coopmat_pk_t128x64k32g44s16m8nf` | tile-aligned prefill |
| SDPA attn*V | `sarc_sdpa_av_coopmat_xe2_t128x64k32g44s16m8` | head_dim >= 128 (3B, 8B) |
| SDPA attn*V | `sarc_sdpa_av_coopmat_sweep_t64x64k32g44s16m8` | head_dim 64 (1B) |
| softmax | `sarc_sdpa_attn_weights_softmax` (release, truncated) | follows from the SDPA rows |
| 8da4w linear | `sarc_dev_linear_dq8ca_coopmat_zpg_xe2bt_t128x128k64g84s16m8` | all shapes the tile fits |
| 4w linear | `sarc_linear_q4gsw_coopmat_sweep_t128x128k16g82s16m8flib` (the shipped tile on a subgroup grid of 8 x 2, band drain) | all shapes the tile fits, except the next row |
| 4w linear | `sarc_linear_q4gsw_coopmat_sweep_t128x128k32g84s16m8flw` (K = 32, grid 8 x 4, IMG_W) | output of at most 512 columns (1B wk / wv) |

`xe2-refine1` is candidate 1 alone and `xe2-refine2` candidates 1 and 2 (the winner before the parameter
search). `xe2-refine3`, `xe2-refine4` and `xe2-refine6` are candidates 3, 4 and 6: gate passed, no gain outside
the noise band, not part of the final profile.

## Method

- Host `fedora-gpu-eval`, ANV, Mesa 26.2.3. Card `b70-0` (guest PCI `0000:01:00.0`, `ETVK_DEVICE_INDEX=0`) for
  every session, gate, reference-error measurement, full kernel measurement and reported number. The second
  Arc Pro B70 of the VM (guest PCI `0000:02:00.0`, `ETVK_DEVICE_INDEX=1`) ran cheap-mode screens of the
  parameter search only (owner decision 2026-10-05), after a test with acceptance fixed beforehand; while
  `b70-0` measured anything else the second card was idle (a lock in the tools enforces it). Nothing was run
  on the B580. Builds with `sarc/tools/build.sh` in
  `localhost/et-vk-build:rocky10`, built on this host from `tools/Containerfile`; every build exports its
  commit from the object store and its 53 shipped SPIR-V variants match `sarc/golden/spirv.json`.
- End to end: `tools/e2e5.sh` (kit protocol: fresh `llama_main` per run, `--warmup`, `prompt_2048.txt`, one
  new token, temperature 0, arms interleaved, median of the first 5 valid runs per arm), clock, throttle
  reasons, energy and temperature sampled every 10 ms. Valid = rc 0, 2048 prompt tokens, 0 generated tokens,
  no foreign DRM client of the card, median clock >= 2505 MHz, no thermal throttle reason. Differences from
  the 780M protocol and their reasons are in `STATUS.md` ("How this host differs"). 722 timed runs in twelve
  sessions, one rejected (`clock_low`) and replaced.
- Gate per candidate: `tools/gate.sh` / `gate_sdpa.sh` (unmodified `sarc/tools/verify.sh --models 1b,3b,8b
  --schemes 4w,8da4w --pdiff` under the candidate environment, the timing session, next token parent vs
  candidate on `prompt_2048.txt`, `prompt_check.txt` and `r1304.txt` in all six cells, warm ETDump of both
  arms; SDPA candidates add 12 passes each of tiers all / extended / full), then a logits probe on 35
  real-text windows per cell and `tools/decide.py`. Pristine `dev/1.5` does not return `rc=0` everywhere on
  this device (table in `STATUS.md`), so the checker compares those status lines with the parent control
  `s0-parent-verify` instead of requiring them; everything else is absolute.
- Roofs: igpu-roofline plan `fast` on `b70-0`, 2026-10-04 21:54 to 22:17 UTC, this driver
  (`results/xe2/roofline/xe2-fast-20261004/`).
- Parameter search (owner decision 2026-10-04, "how large parameter spaces are searched"): see "Parameter
  search" below.
- Stop rule: two consecutive gated candidates under 2 % geomean over their parent. Candidates 3 and 4 met it
  before the search; candidates 5 and 6, which came out of the search, are under 2 % as well (four in a row).

## Results

Baseline and A/A (`s1-aa`): the re-measured parent agrees with `sarc-1.5-e2e-benchmark/results/cells.csv`
within 0.6 % in every cell; parent against the topic build with no environment: geomean -0.12 %, every cell
inside +-0.6 %.

Each candidate against its parent (the previous candidate), same session, gain in tok/s:

| candidate (profile) | 1B 4w / 8da4w | 3B 4w / 8da4w | 8B 4w / 8da4w | geomean | result |
|---|---|---|---|---:|---|
| 1 `xe2-refine1`: SDPA QK^T and attn*V | **+49.57** / **+49.55** % | **+51.80** / **+55.60** % | **+35.58** / **+40.07** % | +46.86 % | `ACCEPTED (reference-error rule, owner decision 2026-10-04)` (`s2-c1`) |
| 2 `xe2-refine2`: 8da4w texel-wise staging, 128 x 128 K = 64 tile | 0.00 / **+9.00** % | -0.36 / **+16.82** % | +0.16 / **+17.32** % | +6.88 % | `GATE_PASS`, bit-identical (`s3-c2`) |
| 3 `xe2-refine3`: fragment-contiguous ColumnMajor QK^T | 0.00 / +1.01 % | +0.36 / +0.47 % | -0.16 / +0.22 % | +0.32 % | `GATE_PASS`, bit-identical (`s4-c3`), not adopted |
| 4 `xe2-refine4`: 8da4w 64-column tile where N >= 4K | 0.00 / -1.98 % | +0.36 / 0.00 % | +0.32 / +0.66 % | -0.11 % | `GATE_PASS`, bit-identical (`s5-c4`), not adopted |
| 5 `xe2-refine5`: 4w tiles of the search (over `xe2-refine2`) | +1.75 / 0.00 % | +1.84 / +0.93 % | **+2.31** / -0.22 % | +1.10 % | `GATE_PASS`, bit-identical (`s7-c5`), **adopted** |
| 6 `xe2-refine6`: attention tiles of the search (over `xe2-refine5`) | +0.89 / 0.00 % | +0.74 / +0.95 % | +0.33 / +0.66 % | +0.59 % | `GATE_PASS`, bit-identical (`s9-c6`), not adopted |

Bold = outside the +-2 % band. Candidate 5 is adopted because one cell is outside the band, the two other 4w
cells move the same way (+1.75, +1.84 %), the 8da4w cells (whose kernels it does not touch) stay inside +-1 %,
and the trace shows the 4w linear time falling by 2.5 to 3.6 % with nothing else moving. Candidate 6 was gated
twice: the first gate (`s8-c6`, +0.95 % geomean, every cell +0.8 to +1.1 %) ended `GATE_FAIL` on one
next-token item that a message of the campaign's own guard, written into a run log, had caused (the two logs
are otherwise identical and the logits probe is bit-identical); the guard was fixed and the second gate
(`s9-c6`) passed. In neither session is any cell outside the band, so by the protocol it is not a gain, as for
candidates 3 and 4, although the trace shows what the kernels save (0.7 to 1.1 % of a prefill).

**Candidate 1 is not a plain pass.** Its gate ends `GATE_FAIL` on the next token of 8B 8da4w on
`prompt_2048.txt` and on `prompt_check.txt` (parent vs candidate); everything else passes. It replaces
attention kernels that accumulate in fp16 by kernels that accumulate in fp32, so it was decided under the
owner's second decision of 2026-10-04: its rms and maximum error against the fp32 CPU reference are not larger
than the parent's on every S = 2048 head configuration (rms 2.8e-5 against 1.3e-4), the 35-window real-text
comparison stays far below the gross-divergence limits (largest mean KL 0.075 nat, at most 1 of 35 top-1
differences per cell), and the two differing items are listed, not waived. On `prompt_check.txt` the parent
itself is a three-way near-tie (top-2 margin 0.148). On `prompt_2048.txt` (2048 identical tokens) the move is
not small: KL 0.975 nat, largest logit change 6.06. Full numbers: `STATUS.md`, `results/xe2/probe/s2-c1/`,
`results/xe2/sdpa-error/`.

### Final profile against the parent, measured directly (session `s10-final5`)

Pristine parent build, no environment, against the build of `ff29c08ef` with `xe2-refine5`; `b70-0`, second
card idle; tok/s, median of 5 valid runs per arm, 60 timed runs, none rejected. The parent numbers equal the
original `dev/1.5` numbers of this device (`cells.csv`) within 0.6 %:

| cell | parent | final profile | gain | `cells.csv` | final vs `cells.csv` | before the search (`xe2-refine2`, `s6-final`) | `xe2-refine6` (`s11-final6`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1B 4w | 11770.10 | **17964.90** | +52.63 % | 11702.9 | +53.51 % | 17504.30 | 18123.90 |
| 1B 8da4w | 12337.30 | **20480.00** | +66.00 % | 12412.1 | +65.00 % | 20686.90 | 20686.90 |
| 3B 4w | 4864.61 | **7529.41** | +54.78 % | 4864.61 | +54.78 % | 7393.50 | 7613.38 |
| 3B 8da4w | 5264.78 | **9615.02** | +82.63 % | 5251.28 | +83.10 % | 9615.02 | 9752.38 |
| 8B 4w | 2432.30 | **3368.42** | +38.49 % | 2438.10 | +38.16 % | 3292.60 | 3407.65 |
| 8B 8da4w | 2727.03 | **4481.40** | +64.33 % | 2737.97 | +63.68 % | 4481.40 | 4520.97 |
| geomean over the parent of its session | | | **+59.23 %** | | | +57.47 % | +60.72 % |

The same comparison was measured a second time (`s12-final5`, 60 timed runs, none rejected) after the reviewer
found that the timing runner, when started outside a gate, had skipped its guard (`tools/e2e5.sh` called
`gpu_begin` one line before loading it: no pair lock and no coordinator hold for `s10-final5` and
`s11-final6`; nothing else was running and no hold was set, so the numbers stand, but the guarantee was not in
force). With the runner corrected and tested through its entry point (`tools/test_e2e5_guard.sh`):

| cell | parent | final profile | gain | final vs `cells.csv` |
|---|---:|---:|---:|---:|
| 1B 4w | 11770.10 | **17964.90** | +52.63 % | +53.51 % |
| 1B 8da4w | 12412.10 | **20686.90** | +66.67 % | +66.67 % |
| 3B 4w | 4864.61 | **7529.41** | +54.78 % | +54.78 % |
| 3B 8da4w | 5251.28 | **9570.09** | +82.24 % | +82.24 % |
| 8B 4w | 2435.20 | **3379.54** | +38.78 % | +38.61 % |
| 8B 8da4w | 2737.97 | **4481.40** | +63.68 % | +63.68 % |
| geomean | | | **+59.23 %** | |

The two sessions agree within one step of the 1 ms timer in every cell.

Next token SAME in 16 of 18 comparisons; the two that differ are candidate 1's items above (8B 8da4w on
`prompt_2048.txt` and `prompt_check.txt`), as in every session against the stock parent. These sessions are
timing and traces only; the kernels were gated in `s2-c1`, `s3-c2` and `s7-c5`. (1B 8da4w reads 20480.00 here
and 20686.90 in `s6-final` and `s11-final6`: one step of the 1 ms timer, 100 ms here and 99 ms there.)

Files: `results/xe2/sessions/<session>/{STAGE.md,env.txt,runs.csv,summary.csv,nexttoken.csv,gate.txt,decision.txt,verify.out,verify/,trace/,sdpa-correctness/,decode/}`,
`results/xe2/probe/<session>/`.

## Where the gain comes from

Warm ETDump of both arms of `s10-final5`, ms per prefill (`sessions/s10-final5/trace/families.csv`):

| cell | arm | total | linear GEMM | QK^T | attn*V | softmax | 8-bit quantize | other |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1B 4w | parent | 165.4 | 59.1 | 29.8 | 34.6 | 16.4 | | 25.5 |
| 1B 4w | final | 106.0 | 56.9 | 4.8 | 6.1 | 12.7 | | 25.4 |
| 1B 8da4w | parent | 156.7 | 45.5 | 29.1 | 34.1 | 16.3 | 10.9 | 20.8 |
| 1B 8da4w | final | 91.8 | 36.4 | 4.8 | 6.1 | 12.6 | 11.0 | 20.9 |
| 3B 4w | parent | 411.4 | 171.6 | 76.1 | 84.9 | 22.0 | | 56.9 |
| 3B 4w | final | 262.7 | 168.2 | 9.3 | 11.0 | 17.0 | | 57.1 |
| 3B 8da4w | parent | 381.0 | 136.3 | 72.2 | 81.0 | 21.6 | 23.6 | 46.2 |
| 3B 8da4w | final | 206.5 | 100.2 | 8.9 | 10.6 | 16.6 | 23.7 | 46.5 |
| 8B 4w | parent | 829.5 | 442.6 | 117.0 | 134.6 | 33.5 | | 101.9 |
| 8B 4w | final | 597.0 | 437.9 | 14.5 | 16.2 | 25.9 | | 102.5 |
| 8B 8da4w | parent | 737.0 | 348.0 | 108.2 | 125.9 | 32.6 | 42.3 | 80.1 |
| 8B 8da4w | final | 448.7 | 272.6 | 13.4 | 15.3 | 25.0 | 42.2 | 80.2 |

- **Attention (candidate 1).** QK^T and attn*V go from stock tiled kernels to cooperative-matrix kernels, 6 to
  8x less time in each; together they are 94 % (1B) to 101 % (8B) of the whole saving of candidate 1 in the 4w cells (`s2-c1`,
  `s6-final`; there the 8B 4w linear time was 9 ms, 2 %, higher in the candidate arm; not investigated). Kernel level (S = 2048, ms per
  layer, 8B): QK^T 3.38 -> 0.41, attn*V 3.95 -> 0.48. Within the port, packed staging with a ColumnMajor K
  load halves QK^T against the release-style scalar fp16 staging (0.83 -> 0.41), and the 128-row
  fragment-layout tile takes attn*V from 0.57 to 0.48 for head_dim 128 (`results/xe2/screens/screen1-sdpa`,
  `screen2-sdpa`). The truncated softmax that comes with the SDPA rows saves another 3.7 to 7.6 ms per prefill.
- **8da4w linear (candidate 2).** Linear GEMM 1.25x (1B), 1.36x (3B), 1.28x (8B). Shader-clock phase timing of
  the two tiles on 1B wq / wo (`results/xe2/phases/`): 128176 cycles per wave against 190448, with weight
  fetch 36672 against 68640 and barrier 20344 against 35104; each packed-weight texel is fetched once and there
  are half as many K chunks (K = 64 against 32).
- **4w linear (candidate 5).** Linear GEMM 1.036x (1B), 1.029x (3B), 1.025x (8B) in `s7-c5`
  (58.9 -> 56.9, 173.1 -> 168.2, 449.4 -> 438.2 ms), which is what the kernel-level confirmation of the search
  gave (1.026 to 1.031x layer-weighted). The tile, K chunk and staging are those of the shipped kernel; what
  changed is the subgroup grid (8 x 2 instead of 4 x 4: a subgroup owns a 16 x 64 strip instead of a 32 x 32
  block) with the drain staged one band at a time. Why that is faster was not separated further by phase
  timing. The 4w kernel still gains least, which is why the 4w cells, and 8B 4w (73 % linear), gain least.

Percent of the freshly measured roofs, linear kernels in the model (`sessions/s10-final5/trace/gemm.csv`,
flops-weighted over all linear dispatches of a prefill):

| | parent | final profile | roof |
|---|---|---|---|
| 4w, fp16 matrix | 64.6 to 67.5 TFLOP/s, 37.3 to 39.0 % | 65.3 to 70.0 TFLOP/s, 37.7 to 40.4 % (`s12-final5`: 65.4 to 70.2, 37.7 to 40.5 %) | 173.3 TFLOP/s |
| 8da4w, int8 matrix | 82.1 to 87.6 TOP/s, 22.8 to 24.3 % | 104.9 to 115.3 TOP/s, 29.1 to 32.0 % | 359.9 TOP/s |

The SDPA kernels have no in-model rate in the trace tables. From the screen-2 kernel times and the dense
product size 2 x heads x S x S x head_dim (my derivation, not a tool output, and it counts the masked half of
QK^T as work): QK^T 57 (1B), 83 (3B), 84 (8B) TFLOP/s and attn*V 45, 68, 72 TFLOP/s, i.e. 25 to 47 % of the
fp16 -> fp32 matrix roof (179.9 TFLOP/s).

## Parameter search (review follow-up; owner decision 2026-10-04)

Done for all four kernel families between 2026-10-05 05:33 and 2026-10-06 14:15 UTC (32 h 42 min; the first
17 h on one card, then split over both). Tools: `tools/sweep*.py`, `sweep_*.sh`, `build-sweep.sh`,
`card_test.sh`; evidence: `results/xe2/sweep/{4w,8da4w,av,qk,cardtest}/`; per-family detail in `STATUS.md`.

**Legal spaces and static pruning.** A configuration is one compile-time parameter set of one shader body.
Analytic pruning (`sweep.py count`): workgroup of at most 1024 invocations; the subgroup tile a whole number
of 8 x 16 MMA tiles (the only shapes the device exposes); tile sizes dividing every production shape; the flag
exclusions the bodies enforce; the staging-integrality rules of each body. Exact pruning: every configuration
is compiled with the build's glslc, and its shared memory is read from the SPIR-V (limit 46000 bytes, below
the B70's 49152; a tile over the device limit can hang the GPU instead of failing).

| family | parameters | analytically legal | drawn / compile / fit | measured | full enumeration would take |
|---|---|---:|---|---|---|
| 4w linear | body (3), M, N, K, subgroup grid x / y, subgroup size, shared-memory layout, IMG_A, IMG_W, drain, accumulator | 338448 | 3000 / 2851 / 2067 | **sample of 2000** (seed 20261005), 965 neighbours in three rounds, 77 full x 2 confirmations | 53 days in the cheap mode alone |
| 8da4w linear | body (4), M, N, K, grid x / y, subgroup size, five `zpgtr` flags | 44670 | 3000 / 2333 / 2093 | **sample of 2000** (same seed), 323 neighbours, 3 incumbents + 26 neighbours, 19 confirmations | 9 days in the cheap mode alone |
| attn*V | family (3), M, N, K, grid x / y, subgroup size | 1131 | 1131 / 1131 / 1102 | **all 1102** (2 h), 15 confirmations | enumerated |
| QK^T | family (4), M, N, K, grid x / y, subgroup size, NO_MASK_FILL | 4536 | 4536 / 4536 / 2828 | **all 2828** (4 h), 14 confirmations | enumerated |

The two attention spaces fit inside a day, so they were enumerated rather than sampled; no run was projected
beyond 48 hours.

**Procedure, as prescribed.** (1) A cheap screening mode (the 1B shapes, one per shape class, for the linear
kernels; the 1B and 3B head configurations for attention; timing only, one process per configuration) was
validated against the full measurement (all three models, twice) on 60 configurations per family: Spearman
rank correlation 0.963 to 1.000 (4w), 0.984 to 1.000 (8da4w), 0.998 to 0.999 (attn*V), 0.9999 (QK^T) per
(model, shape), and at least 0.995 for the layer-weighted score. The working threshold was 0.9; for 4w it was
not written down before the measurement, for the others it was. (2) The uniform seeded sample, or the whole
space, was screened; the device's table kernel was re-measured every 50 configurations as a drift monitor
(spread 0.06 to 1.5 %). (3) Parameter importance (share of the variance of log time explained by each
parameter alone) and pair interactions (joint share minus the additive share) from the sample alone. (4)
Refinement: every unmeasured legal one-parameter neighbour of the best 20 by score and the best 5 of each shape
class, repeated while a round moved a best by more than 2 % (rule in `tools/sweep_rounds.sh`; 4w ran three
rounds, 8da4w one). (5) Correctness of everything near the top (a separate process: linear numeric cases;
SDPA extended tier), then the best 10 per shape class rebuilt and measured with the full measurement twice
against the incumbent kernels. One addition of mine, stated as such: for 8da4w the kernels already in use beat
the whole sample, so the three incumbent tiles and their neighbours were added as configurations; the sample
and its refinement are unchanged by it.

**Parameter-importance table** (variance share of the layer-weighted score, uniform sample or whole space;
`importance-sample.csv` per family has every level and the per-class tables, `interactions-sample.csv` the
pairs):

| parameter | 4w | 8da4w | attn*V (head_dim 64 / 128) | QK^T |
|---|---:|---:|---:|---:|
| subgroup grid, y | 31.7 % | 37.0 % | 40.8 / 35.4 % | 30.8 % |
| subgroup grid, x | 14.7 % | 12.3 % | 11.4 / 15.9 % | 15.4 % |
| body / family | 7.7 % | 0.7 % | 2.8 / 2.1 % | 3.2 % |
| accumulator (4w) | 6.4 % | | | |
| tile M | 5.2 % | 0.6 % | 16.0 / 11.8 % | 3.5 % |
| tile N | 1.4 % | 5.9 % | 0.0 / 1.5 % | 3.4 % |
| NO_MASK_FILL (QK^T) | | | | 4.6 % |
| K per chunk | 0.9 % | 0.9 % | 0.8 / 0.1 % | 0.6 % |
| shared-memory layout (4w) | 1.0 % | | | |
| subgroup size | 0.7 % | 0.0 % | 0.7 / 0.3 % | 0.2 % |
| IMG_A, IMG_W, drain (4w); the five `zpgtr` flags (8da4w) | 0.3, 0.0, 0.2 % | 0.5 to 1.3 % each | | |
| largest pair interactions | body x N 3.6 %, body x K 1.7 %, grid x x grid y 1.7 % | M x grid y 2.6 %, K x grid y 2.2 %, M x N 2.1 % | M x grid y 5.0 %, grid x x grid y 2.6 % | grid x x grid y 8.9 %, M x grid y 5.4 % |

In every family the subgroup grid explains about half of the variance: a grid of 1 in either direction (few,
large subgroup tiles) costs a factor of 2 to 5, and a tile of 256 rows a factor of 2.6 to 3.9. These main
effects describe how to avoid a slow kernel, not where the best one is: the best configurations sit in a
corner the level medians alone would not pick (4w: M = 128, K = 16, 256 threads), and they differ from the
incumbents by one or two parameters. For 8da4w the body has a small share only because 95 % of the legal
space is the `zpgtr` body with its flags; the other three bodies are 2.5 to 4 times faster at the median.

**Outcome per family** (kernel level, full measurement twice, correctness `ok`):

| family | best found against the kernel in use | what was done with it |
|---|---|---|
| 4w | sample: nothing faster (best 0.951x of the shipped tile). Refinement: grid 8 x 2 + band drain 1.02 to 1.06x on every shape but 1B wk / wv; K = 32 tile 1.18x on 1B wk / wv only; 128 x 256 tiles up to 1.09x on four shapes (1B wq / wo and w2, 8B wk / wv and w2) and 0.69 to 0.92x on every 3B shape | candidate 5 = the first two. The 128 x 256 tiles are not used: no statable shape property separates where they win from where they collapse |
| 8da4w | sample 0.966x of the shipped tile, refinement 1.012x; the candidate 2 tile is 1.20x and none of its neighbours is faster | no candidate; the candidate 2 tile is the best configuration found |
| attn*V | `xe2` body, K = 64, subgroup size 32: 1.13x (1B), 1.20 / 1.22x (3B / 8B) of candidate 1's kernels. The eight fastest configurations by time (`sweep` body, K = 64, up to 1.74x) fail correctness and are excluded | candidate 6 |
| QK^T | `xe2c` 64 x 128, grid 8 x 2: 1.03 / 1.12 / 1.12x of candidate 1's kernel (candidate 3's kernel: 1.01 / 1.07 / 1.07x) | candidate 6 |

**Second card.** One identical batch of 31 arms was screened on `b70-0` alone, on the second card alone and
on both at once, with the acceptance written down first (Spearman >= 0.95 for the score and >= 0.90 per shape
class, card-to-card and alone-against-together; both at once at most 1.5 times the time of one): Spearman
0.9996 to 1.0000 everywhere, median time ratios 0.9997 to 1.0005, both at once 1.07 times the time of one card,
i.e. 1.86 times the throughput (`results/xe2/sweep/cardtest/`). Screens were then split by row position; the
second card's times are scaled per shape by the ratio of the two cards' table-kernel times (0.997 to 1.003)
before they join the first card's rows, each row carries its card, and the second card's own file is kept.

## Failed and rejected candidates

No candidate failed its gate for a correctness reason. Screened out before a gate, all kept in the tree as
sweep variants with their result files (`results/xe2/screens/README.md`):

- 8da4w: every other tile shape (0.26 to 0.88x); texel-wise staging on a subset of the threads (0.56 to 0.95x).
- 4w: every other tile shape (0.14 to 0.90x), K = 32 chunks (0.50x), texel-wise staging (0.80 to 0.85x), a
  band-at-a-time drain, split staging (first body 15 % slower, revised body 1.002x on the shipped geometry and
  0.80 to 0.95x on larger tiles). Five screens, no 4w candidate.
- Softmax: a single-read variant and one with subgroup reductions, measured through the local hook builds
  only: 0.79 ms per layer against 0.80. Both pass SDPA correctness with 0 mismatches. No gain, so no hook is
  proposed.
- SDPA: subgroup tiles larger than 32 x 16 lose; the straight port (`xe2-sdpa0`) is correct but half as fast
  as candidate 1's QK^T.
- From the search: the 4w 128 x 256 tiles (see above); every 8da4w configuration of the sample and its
  refinement (at best 1.012x of the shipped tile, 0.78 to 0.90x of the candidate 2 tile); the `sweep`-body
  attn*V tiles with K = 64, which are the fastest by time and fail the extended correctness tier (8 of the 54
  checked; `results/xe2/sweep/av/sw1-results.csv`, rows `corr64` / `corr128`).
- Gated without a gain outside the noise band, not adopted: candidates 3, 4 and 6 (`xe2-refine3`, `-4`, `-6`).

Superseded runs (kept with reasons under `results/xe2/superseded/`): the first A/A attempt (780M throttle rule
and 0.1 s sampler rejected every run), the first roofline attempt (guard false positive on an operator shell
command), and the first build of the 8da4w incumbents (`sw2i-8da4w-id-collision`: I numbered its
configurations in a range the neighbour build already used; repeated with distinct ids). Recorded as it ended
and not superseded: `s8-c6`, the first gate of candidate 6 (`GATE_FAIL` on an item caused by a guard message),
and `s10-final5` / `s11-final6`, whose runner had skipped its guard (repeated as `s12-final5`).
Interruptions of the search, none of which lost a measured row (`STATUS.md`): a false positive of the new
two-card guard stopped one queue for 4 minutes; the host was rebooted by someone else at 2026-10-06 05:57 UTC
and both queues were restarted 14 minutes later; both cards paused once for 20 minutes (02:32 to 02:52 UTC), which I take to be the coordinator's hold being used.

## What is left (prefill with the final profile, `s10-final5`)

| cell | linear GEMM | softmax | QK^T + attn*V | 8-bit quantize | other |
|---|---:|---:|---:|---:|---:|
| 1B 4w, 106 ms | 54 % | 12 % | 10 % | | 24 % |
| 8B 4w, 597 ms | 73 % | 4 % | 5 % | | 17 % |
| 1B 8da4w, 92 ms | 40 % | 14 % | 12 % | 12 % | 23 % |
| 8B 8da4w, 449 ms | 61 % | 6 % | 6 % | 9 % | 18 % |

## Limits (what stops further progress)

- **The parameter spaces of the existing bodies are used up.** After 2000 sampled and about 1000 refined
  configurations the best 4w kernel is 3 % faster than the shipped one, and no 8da4w configuration beats the
  candidate 2 tile; the two attention spaces were enumerated completely and their best tiles are worth under
  1 % of a prefill. Further gains need new kernel code, not other parameters.
- **4w linear** is the largest remaining item (54 to 73 % of a 4w prefill) and sits at 37.7 to 40.4 % of the
  fp16 matrix roof. Its phase split on the shipped tile is MMA 37 to 40 %, barrier 22 to 23 %, fetch 22 to
  26 %, shared-memory store 13 to 15 %; texel-wise and split staging did not help, and the grid change of
  candidate 5 is the only thing that moved it. What would close the gap to the fed roof is not known from
  this data.
- **8da4w linear** is at 29 to 32 % of the int8 roof; after candidate 2 fetch is still 27 to 33 % of a wave and
  MMA 25 to 29 %.
- **Softmax** (4 to 14 %) is now larger than QK^T or attn*V. Its shader name is fixed in the release zone
  (`impl/sarc/SdpaCoopmat.cpp`); the two variants built behind a local hook were not faster, so reads, exp
  count and barrier count are not where it spends its time. The owner's decision of 2026-10-05 would allow
  the hook to be committed, but no variant here justifies it.
- **Non-kernel work** (elementwise, copies, RMSNorm, 8-bit activation quantize) is 17 to 35 % of a prefill and
  was not touched.
- **Resolution of the end-to-end protocol.** The timer step is 1 ms, about 1 % of a 1B prefill with the final
  profile, and the band is +-2 %: kernel changes worth under about 2 % of a prefill (candidates 3, 4 and 6, and
  two of the three 4w cells of candidate 5) cannot be shown end to end, only in the trace.
- Hardware: only 8x16x16 (fp16) and 8x16x32 (int8) MMA shapes; the card runs power-limited (`pl2`) in every
  loaded run, the 4w cells at 2517 to 2750 MHz.

## Not done

- No phase timing of the candidate 5 kernel against the shipped one; the reason the 8 x 2 grid is faster is
  inferred from the search, not measured in the kernel. Nor was it investigated why the 128 x 256 tiles
  collapse on every 3B shape.
- The search covers the parameters of the existing bodies. New bodies were only tried in the hand-chosen
  screens before it (texel-wise and split staging).
- Decode was measured only as an A/B per candidate (candidate 1 is 0.7 to 2.0 % slower than the parent in all
  six cells, inside each cell's repeat range but in one direction; candidates 5 and 6: 0.995 to 1.013x).
- The cheap-mode threshold for 4w was not written down before its validation (0.9 was the working figure).
- `nvtop` (started before the campaign) held a DRM file of the card through the sessions before the reboot
  with zero engine cycles and zero GPU memory; it is recorded per session as an idle monitor.

## Awaiting B580 confirmation

The kernels of the final profile: candidate 1 (reference-error rule), candidate 2 and candidate 5. All Xe2
variants keep shared memory under 46000 bytes (the B70 reports `maxComputeSharedMemorySize` 49152; the B580's
value was not read here), use workgroups of at most 1024 invocations and the MMA shapes / subgroup size 16 the
shipped Intel rows already use, and do not depend on the amount of device memory. (Candidate 6, not adopted,
would also need subgroup size 32.) Nothing was run on the B580.

## Checks on the branch

`bash sarc/tools/check.sh --no-build` at the head, 2026-10-06 18:00 UTC:

```
== 1 zone rule vs origin/release/1.5
== 2 twin wrappers
== 3 test_sarc_select
test_sarc_select: PASS (1240 checks, 31 rows, 0 candidates, dev zone absent, unverified off)
[sarc_dev] overrides active: unverified=1 variant= dq8ca_variant=
test_sarc_select: PASS (1559 checks, 35 rows, 215 candidates, dev zone linked, unverified on)
check.sh: PASS
```

`git diff --name-status 6a7cc8cc6 HEAD` touches only `backends/vulkan/runtime/graph/ops/glsl/sarc_dev/` (29
files), `.../impl/sarc_dev/` (3), `backends/vulkan/test/sarc_dev/` (1) and this directory. No release source,
`verify.sh`, `check.sh`, `build.sh`, golden, tolerance or prompt is changed; no release-zone hook was
committed (none is needed). `spirv_golden.py` reports the 53 shipped variants unchanged for every build,
including the last one (`topic9`) and every sweep build.
