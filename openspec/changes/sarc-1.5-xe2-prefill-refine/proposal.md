# sarc-1.5-xe2-prefill-refine

Dev zone only (2026-10-05). Nothing here is promoted and no release-zone or upstream file is touched. Branch
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
- Measurement only: shader-clock phase twins, rms / maximum error against the fp32 reference in the SDPA
  correctness cases, and two softmax variants (`sarc_dev_sdpa_attn_weights_softmax_xe2`, `..._xe2sg`) that can
  only be dispatched through `tools/hook-sdpa-softmax.patch`, a local release-zone patch that is not committed
  as source and was applied only in the builds `hook1` / `hook2`.
- Everything generated comes from `tools/gen_xe2.py`, which reads the release bodies and never writes them.

**Winner: `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=xe2-refine2`** (candidates 1 and 2):

| op | kernel | shapes |
|---|---|---|
| SDPA QK^T | `sarc_sdpa_qk_coopmat_pk_t128x64k32g44s16m8nf` | tile-aligned prefill |
| SDPA attn*V | `sarc_sdpa_av_coopmat_xe2_t128x64k32g44s16m8` | head_dim >= 128 (3B, 8B) |
| SDPA attn*V | `sarc_sdpa_av_coopmat_sweep_t64x64k32g44s16m8` | head_dim 64 (1B) |
| softmax | `sarc_sdpa_attn_weights_softmax` (release, truncated) | follows from the SDPA rows |
| 8da4w linear | `sarc_dev_linear_dq8ca_coopmat_zpg_xe2bt_t128x128k64g84s16m8` | all shapes the tile fits |
| 4w linear | unchanged: `sarc_linear_q4gsw_coopmat_t128x128k16g44s16m8fli` | |

`xe2-refine1` is candidate 1 alone; `xe2-refine3` and `xe2-refine4` are candidates 3 and 4 (gate passed, no
measurable gain, not part of the winner).

## Method

- Host `fedora-gpu-eval`, card `b70-0` only (guest PCI `0000:01:00.0`, `ETVK_DEVICE_INDEX=0`), ANV, Mesa
  26.2.3. Nothing was run on the second B70 or on the B580. Builds with `sarc/tools/build.sh` in
  `localhost/et-vk-build:rocky10`, built on this host from `tools/Containerfile`; every build exports its
  commit from the object store and its 53 shipped SPIR-V variants match `sarc/golden/spirv.json`.
- End to end: `tools/e2e5.sh` (kit protocol: fresh `llama_main` per run, `--warmup`, `prompt_2048.txt`, one
  new token, temperature 0, arms interleaved, median of the first 5 valid runs per arm), clock, throttle
  reasons, energy and temperature sampled every 10 ms. Valid = rc 0, 2048 prompt tokens, 0 generated tokens,
  no foreign DRM client of the card, median clock >= 2505 MHz, no thermal throttle reason. Differences from
  the 780M protocol and their reasons are in `STATUS.md` ("How this host differs"). 362 timed runs in six
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
- Stop rule: two consecutive gated candidates under 2 % geomean over their parent. Candidates 3 and 4 met it.

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

Bold = outside the +-2 % band.

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

### Winner against the parent, measured directly (session `s6-final`)

Pristine parent build, no environment, against the head build with `xe2-refine2`; tok/s, median of 5 valid
runs per arm. The parent numbers equal the original `dev/1.5` numbers of this device (`cells.csv`) within
0.6 %:

| cell | parent | winner | gain | `cells.csv` | winner vs `cells.csv` |
|---|---:|---:|---:|---:|---:|
| 1B 4w | 11770.10 | 17504.30 | +48.72 % | 11702.9 | +49.57 % |
| 1B 8da4w | 12412.10 | 20686.90 | +66.67 % | 12412.1 | +66.67 % |
| 3B 4w | 4864.61 | 7393.50 | +51.99 % | 4864.61 | +51.99 % |
| 3B 8da4w | 5264.78 | 9615.02 | +82.63 % | 5251.28 | +83.10 % |
| 8B 4w | 2435.20 | 3292.60 | +35.21 % | 2438.10 | +35.05 % |
| 8B 8da4w | 2734.31 | 4481.40 | +63.90 % | 2737.97 | +63.68 % |

Geomean +57.47 %. Next token SAME in 16 of 18 comparisons; the two that differ are candidate 1's items above.
This session is timing and traces only; the kernels were gated in `s2-c1` and `s3-c2`.

Files: `results/xe2/sessions/<session>/{STAGE.md,env.txt,runs.csv,summary.csv,nexttoken.csv,gate.txt,decision.txt,verify.out,verify/,trace/,sdpa-correctness/,decode/}`,
`results/xe2/probe/<session>/`.

## Where the gain comes from

Warm ETDump of both arms of `s6-final`, ms per prefill (`sessions/s6-final/trace/families.csv`):

| cell | arm | total | linear GEMM | QK^T | attn*V | softmax | 8-bit quantize | other |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1B 4w | parent | 165.0 | 58.8 | 29.9 | 34.5 | 16.4 | | 25.4 |
| 1B 4w | winner | 108.1 | 59.0 | 4.8 | 6.1 | 12.7 | | 25.5 |
| 1B 8da4w | parent | 156.8 | 45.5 | 29.2 | 34.0 | 16.3 | 10.9 | 20.9 |
| 1B 8da4w | winner | 91.8 | 36.4 | 4.8 | 6.1 | 12.6 | 11.0 | 20.9 |
| 3B 4w | parent | 410.9 | 171.2 | 76.0 | 84.9 | 21.9 | | 56.9 |
| 3B 4w | winner | 268.5 | 173.7 | 9.4 | 11.0 | 17.0 | | 57.4 |
| 3B 8da4w | parent | 381.0 | 136.3 | 72.2 | 81.1 | 21.6 | 23.6 | 46.3 |
| 3B 8da4w | winner | 206.5 | 100.1 | 9.0 | 10.6 | 16.6 | 23.7 | 46.5 |
| 8B 4w | parent | 829.4 | 443.0 | 116.8 | 134.4 | 33.5 | | 101.7 |
| 8B 4w | winner | 611.6 | 452.0 | 14.5 | 16.3 | 25.9 | | 102.8 |
| 8B 8da4w | parent | 737.2 | 348.1 | 108.2 | 126.1 | 32.5 | 42.2 | 80.0 |
| 8B 8da4w | winner | 448.8 | 272.7 | 13.4 | 15.3 | 25.0 | 42.2 | 80.2 |

- **Attention (candidate 1).** QK^T and attn*V go from stock tiled kernels to cooperative-matrix kernels, 6 to
  8x less time in each; together they are 94 % (1B) to 101 % (8B) of the whole saving in the 4w cells. (In 8B
  4w the linear time is 9 ms, 2 %, higher in the winner arm, here and in `s2-c1`; not investigated.) Kernel level (S = 2048, ms per
  layer, 8B): QK^T 3.38 -> 0.41, attn*V 3.95 -> 0.48. Within the port, packed staging with a ColumnMajor K
  load halves QK^T against the release-style scalar fp16 staging (0.83 -> 0.41), and the 128-row
  fragment-layout tile takes attn*V from 0.57 to 0.48 for head_dim 128 (`results/xe2/screens/screen1-sdpa`,
  `screen2-sdpa`). The truncated softmax that comes with the SDPA rows saves another 3.7 to 7.6 ms per prefill.
- **8da4w linear (candidate 2).** Linear GEMM 1.25x (1B), 1.36x (3B), 1.28x (8B). Shader-clock phase timing of
  the two tiles on 1B wq / wo (`results/xe2/phases/`): 128176 cycles per wave against 190448, with weight
  fetch 36672 against 68640 and barrier 20344 against 35104; each packed-weight texel is fetched once and there
  are half as many K chunks (K = 64 against 32).
- The 4w linear kernel is unchanged, which is why the 4w cells gain least, and 8B 4w (74 % linear) least of all.

Percent of the freshly measured roofs, linear kernels in the model (`sessions/s6-final/trace/gemm.csv`,
flops-weighted over all linear dispatches of a prefill):

| | parent | winner | roof |
|---|---|---|---|
| 4w, fp16 matrix | 64.5 to 67.8 TFLOP/s, 37.2 to 39.1 % | 63.2 to 67.6 TFLOP/s, 36.5 to 39.0 % | 173.3 TFLOP/s |
| 8da4w, int8 matrix | 82.1 to 87.7 TOP/s, 22.8 to 24.4 % | 104.8 to 115.3 TOP/s, 29.1 to 32.0 % | 359.9 TOP/s |

The SDPA kernels have no in-model rate in the trace tables. From the screen-2 kernel times and the dense
product size 2 x heads x S x S x head_dim (my derivation, not a tool output, and it counts the masked half of
QK^T as work): QK^T 57 (1B), 83 (3B), 84 (8B) TFLOP/s and attn*V 45, 68, 72 TFLOP/s, i.e. 25 to 47 % of the
fp16 -> fp32 matrix roof (179.9 TFLOP/s).

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

Superseded runs (kept with reasons under `results/xe2/superseded/`): the first A/A attempt (780M throttle rule
and 0.1 s sampler rejected every run) and the first roofline attempt (guard false positive on an operator
shell command).

## What is left (prefill with the winner)

| cell | linear GEMM | softmax | QK^T + attn*V | 8-bit quantize | other |
|---|---:|---:|---:|---:|---:|
| 1B 4w, 108 ms | 55 % | 12 % | 10 % | | 24 % |
| 8B 4w, 612 ms | 74 % | 4 % | 5 % | | 17 % |
| 1B 8da4w, 92 ms | 40 % | 14 % | 12 % | 12 % | 23 % |
| 8B 8da4w, 449 ms | 61 % | 6 % | 6 % | 9 % | 18 % |

## Limits

- **4w linear** is the largest remaining item and sits at 36 to 39 % of the fp16 matrix roof. Its phase split
  is MMA 37 to 40 %, barrier 22 to 23 %, fetch 22 to 26 %, shared-memory store 13 to 15 %. None of the staging
  or tile changes tried moved it; what would is not known from this data.
- **8da4w linear** is at 29 to 32 % of the int8 roof; after candidate 2 fetch is still 27 to 33 % of a wave and
  MMA 25 to 29 %.
- **Softmax** is now larger than QK^T or attn*V. Its shader name is fixed in the release zone
  (`impl/sarc/SdpaCoopmat.cpp`); the smallest hook is in `tools/hook-sdpa-softmax.patch`, but neither variant
  built for it is faster, so reads, exp count and barrier count are not where it spends its time.
- **Non-kernel work** (elementwise, copies, RMSNorm, 8-bit activation quantize) is 17 to 35 % of a prefill and
  was not touched.
- The timer resolution is 1 ms: about 1 % of a 1B prefill with the winner.
- Hardware: only 8x16x16 (fp16) and 8x16x32 (int8) MMA shapes, subgroup 16; the card runs power-limited
  (`pl2`) in every loaded run, the 4w cells at 2517 to 2750 MHz.

## Not done

- The sampled parameter search of the owner's decision of 2026-10-04 (2000 to 3000 random configurations,
  parameter-importance table) was not run. The linear and SDPA tile spaces here were searched with twelve
  hand-chosen screens of 4 to 48 configurations. There is therefore no parameter-importance table, and the
  statement "no 4w tile is faster" covers the configurations screened, not the space.
- Decode was measured only as an A/B (candidate 1 is 0.7 to 2.0 % slower than the parent in all six cells,
  inside each cell's repeat range but in one direction; not investigated).
- `nvtop` (pid 1952, started before the campaign) held a DRM file of the card through every session with zero
  engine cycles and zero GPU memory; it is recorded per session as an idle monitor.

## Awaiting B580 confirmation

Both kernels of the winner: candidate 1 (reference-error rule) and candidate 2. All Xe2 variants keep shared
memory under 46000 bytes (the B70 reports `maxComputeSharedMemorySize` 49152; the B580's value was not read
here), use workgroups of at most 1024 invocations and the MMA shapes / subgroup size the shipped Intel rows
already use, and do not depend on the amount of device memory.

## Checks on the branch

`sarc/tools/check.sh --no-build` passes at the head. `git diff --name-status 6a7cc8cc6 HEAD`: only dev-zone
files and this directory. No release source, `verify.sh`, `check.sh`, `build.sh`, golden, tolerance or prompt
is changed; `spirv_golden.py` reports the 53 shipped variants unchanged for every build.
