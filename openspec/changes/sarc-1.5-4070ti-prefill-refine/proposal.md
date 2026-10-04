# sarc-1.5-4070ti-prefill-refine

Dev zone only (2026-10-04). Branch `topic/4070ti-prefill-refine`, forked from `topic/780m-prefill-refine` at
`6a7cc8cc6`. That commit with no profile (the shipped NVIDIA rows) is the parent of every comparison. Nothing is
promoted; no release-zone or upstream file is edited. Day-to-day state and incidents are in `STATUS.md`.

## Why

Push the 2048-token Llama prefill on the RTX 4070 Ti SUPER (Vulkan backend, driver 615.71.09) as far as it
goes, measured against the parent in the same session.

## Outcome in one paragraph

Attention is where the time is on this card, and SDPA prefill kernels for it are worth **+42 % geomean**
(+31 % to +50 % per cell), measured with the full protocol. That candidate is **not accepted**: it fails one
next-token item of the unmodified `verify.sh`, and under the owner's near-tie rule of 2026-10-04 it is outside
twice the noise floor in 4 of 6 cells. The numbers suggest the cause is the parent's fp16 attention arithmetic
rather than an error in the candidate, which is closer to the fp32 reference than the parent is; that
judgement is the owner's and is open. The linear kernels were swept and nothing beats the shipped tiles. Two
further SDPA kernels were written and measured (a softmax without the zero tail, a fused attention kernel);
both need a hook outside the dev zone.

## What is in the tree

All under `backends/vulkan/runtime/graph/ops/{glsl,impl}/sarc_dev/`, `backends/vulkan/test/sarc_dev/` and this
directory. Every shader file of this campaign carries `4070ti` in its name; `impl/sarc_dev/Overrides.cpp` is
changed only inside `// >>> 4070ti <id>` .. `// <<< 4070ti <id>` blocks. The generators in `tools/gen_4070ti_*.py`
read the release sources and never write them.

| family (glsl/sarc_dev) | what | generator |
|---|---|---|
| `sarc_sdpa_qk_coopmat_4070ti`, `_pk`, `sarc_sdpa_av_coopmat_4070ti`, `_ml` | the 780M campaign's SDPA dev twins at subgroup 32, 26 tiles | `gen_4070ti_sdpa.py` |
| `sarc_sdpa_qk_coopmat_4070ti_df`, `sarc_sdpa_av_coopmat_4070ti_df` | **new**: direct-feed QK^T and attn*V (no shared-memory staging), fp32 / grouped fp16 / fp16 accumulation, 33 variants | `gen_4070ti_df.py` |
| `sarc_sdpa_attn_weights_softmax_4070ti` | **new**: LLM softmax that zeroes the masked tail only as far as attn*V reads it | `gen_4070ti_softmax.py` |
| `sarc_sdpa_fused_coopmat_4070ti` | **new**: fused prefill attention, the score matrix is never written | `gen_4070ti_fused.py` |
| `sarc_linear_q4gsw_coopmat_4070ti`, `sarc_linear_dq8ca_coopmat_zpgtr_4070ti` | sweep tiles of the two linear families (release bodies), 14 + 13 | `gen_4070ti_lin.py` |
| `sarc_linear_dq8ca_coopmat_zpgtr_4070ti_bh` | **new**: zpgtr with half-texel weight staging (twin body) | `gen_4070ti_bh.py` |
| `sarc_dev_prof_4070ti_q4gsw`, `sarc_dev_prof_4070ti_dq8ca_zpgtr` | phase-timing twins, measurement only | `gen_4070ti_prof.py` |

Profiles (`ET_VK_SARC_DEV_PROFILE`): `4070ti-refine1` (SDPA), `4070ti-refine2` (8da4w `bh`), `4070ti-refine3`
(`refine2` + 4w drain in Ash), and one screening profile per SDPA tile (`4070ti-qk-*`, `4070ti-av-*`).

`test_llama_microbench --sdpa-correctness-only` gained a reported `[sdpa-error]` line (max and rms error against
its fp32 reference) and `ET_VK_SDPA_DUMP_DIR`; neither changes a verdict.

## What cannot be reached from the dev zone (hooks that would be needed)

These are measured through local patches in `tools/` that are applied to an archived source tree at build time,
recorded in the build provenance, and **not** applied to the branch.

1. **SDPA rows for this device** (`tools/local-hook-nvidia-sdpa.patch`, 13 lines). The dev profile only replaces
   an existing table choice, and the SDPA hooks in `impl/sarc/SdpaCoopmat.cpp` (spec constants, truncated
   softmax) are enabled by a table row. Smallest hook: two `kUnverified` rows in `impl/sarc/table_nvidia.cpp`,
   or a way for a dev profile to mark a device as having SDPA rows. Every SDPA result here depends on it.
2. **Softmax name** (`tools/local-hook-nvidia-sdpa-softmax.patch`, + 9 lines). `sdpa_softmax_shader_name()` in
   `impl/sarc/SdpaCoopmat.cpp` returns a fixed name. Smallest hook: let the dev override name the softmax.
3. **Fused node** (`tools/local-hook-fused-sdpa.patch`, + 150 lines in `impl/SDPA.cpp`). LLM-mode SDPA is three
   nodes; a fused kernel needs one node bound to q, k_cache, v_cache and out, and the three-node path must
   remain for decode and unaligned prompts. A release hook would be
   `sarc::try_add_fused_sdpa(graph, q, k_cache, v_cache, input_pos, out)` in `sdpa_impl`, in the style of
   `sarc::try_add_q4gsw_coopmat`. The local patch instead reduces the three nodes to one junk workgroup each
   when the fused kernel applies; that is good enough to measure and not good enough to ship.

## Method

- Host gpu-dev-4004, RTX 4070 Ti SUPER, NVIDIA 615.71.09. Builds with the unmodified `sarc/tools/build.sh` in
  `localhost/et-vk-build:rocky10` through a `podman` -> `docker` shim, always from a `git archive` of a commit
  (`tools/build-both.sh`), never from the working copy. All nine builds: `spirv_golden.py` PASS, 53 shipped
  variants unchanged.
- End to end: `tools/e2e5.sh` (the kit `e2e.sh` protocol: fresh `llama_main` per run, `--warmup`,
  `prompt_2048.txt`, one new token, arms interleaved, cool between runs), 5 valid runs per arm and cell, clock,
  busy, power and temperature sampled every 20 ms with `nvidia-smi`. A run counts only with rc 0, 2048 prompt
  tokens, 0 generated tokens, no GPU process of another owner and a median clock of at least 2502 MHz
  (`results/4070ti/clkmin.json`, from the A/A session). One GPU job at a time under the gpu-lab lock. No build
  ran during a timed session.
- Gate: `tools/gate.sh` / `tools/gate_sdpa.sh`, judged on the content of the result files by
  `tools/gate_check.py` (36 unit tests in `tools/test_gate_check.py`). The owner decision of 2026-10-04 on
  near-ties is implemented as `--near-tie <NEAR_TIE.json>`: the evidence file is only written when the
  comparison is within twice the noise floor, and an acceptance that used it is labelled as such.
- 1B timer quantisation: a 1B prefill is 72 to 104 ms and the runner's timer is 1 ms, so 1B cells move in steps
  of 1.0 to 1.4 %. The warm ETDump dispatch totals are given alongside.

## Baseline, A/A, roofs

Baseline re-measured (`s1-aa`, pristine parent against the topic build without environment):
1B 19692 / 20898, 3B 8715 / 9660, 8B 4481 / 5007 tok/s (4w / 8da4w), within 0.5 % of
`sarc-1.5-e2e-benchmark/results/cells.csv` in every cell. A/A geomean -0.12 %, largest cell 0.47 %.

Fresh roofs (igpu-roofline `fast`, this driver, run `roofline/2026-10-04-fast` in the artifacts, clocks not
pinned, every roof confirmed with 3 repeats within 0.6 %): matrix fp16 182.9 TFLOP/s, fp16 -> fp32 92.3 (half
rate), int8 369.1 TOP/s; DRAM read 713, write 643 GB/s. `shared_fp16_read` reads 1344 GB/s again and is not
used for anything.

## Where the time goes (parent)

Warm ETDump, ms per 2048-token prefill (`results/4070ti/sessions/s1-aa/trace/`):

| family | 1B 4w | 1B 8da4w | 3B 4w | 3B 8da4w | 8B 4w | 8B 8da4w |
|---|---:|---:|---:|---:|---:|---:|
| linear GEMM | 34.2 | 26.7 | 98.1 | 75.6 | 242.6 | 185.0 |
| attention (QK^T + softmax + attn*V, all stock) | 51.6 | 51.0 | 100.2 | 98.8 | 152.8 | 150.1 |
| copy / view | 7.8 | 4.6 | 17.7 | 11.2 | 29.7 | 18.1 |
| elementwise | 6.9 | 6.9 | 12.6 | 12.9 | 25.6 | 26.4 |
| 8-bit quantize | - | 3.4 | - | 8.9 | - | 23.7 |
| total | 102.1 | 94.1 | 232.8 | 211.5 | 457.2 | 409.4 |

Attention is 51 % of the 1B prefill and 33 % of the 8B one. One fp16 score matrix is 268 MB per layer on 1B
(32 heads x 2048 x 2048): QK^T writes it, the softmax reads and rewrites it, attn*V reads it. The stock softmax
moves 536 MB in 0.85 ms, i.e. it already runs at the DRAM roof. Attention on this card is bound by memory
traffic, not by arithmetic.

Linear kernels in the model (same traces, fresh roofs): 4w 116.6 to 117.8 TFLOP/s = **63.7 to 64.4 %** of the
fp16 matrix roof; 8da4w 149.3 to 154.5 TOP/s = **40.4 to 41.9 %** of the int8 matrix roof.

Phase timing of the shipped linear kernels (shader clock, share of one wave, `results/4070ti/phases/`):

| kernel | barrier | fetch | MMA | LDS store | prologue + epilog | drain + write |
|---|---:|---:|---:|---:|---:|---:|
| 4w `t256x128k16g42s32ga` (N > 512) | 22 % | 12.5 % | 50 % | 13 % | 0.3 % | 2 % |
| 4w `t128x128k16g24s32ga` (N <= 512) | 32 % | 12 % | 34 % | 19 % | 0.4 % | 1.5 % |
| 8da4w zpgtr `t128x128k64g44s32mk32ra` | 14 % | 34 to 43 % | 30 to 35 % | 7 % | 6 to 8 % | 1 to 5 % |

## Candidate 1: SDPA prefill kernels (`4070ti-refine1`): +42.05 % geomean, REJECTED

Kernel per head_dim, chosen at kernel level (`results/4070ti/screens/sdpa-screen{1,2}.csv`, 61 profiles x 3):

| op | head_dim 64 (1B) | head_dim 128 (3B, 8B) |
|---|---|---|
| QK^T | `df_t64x64k32g11s32nf` (direct feed, one subgroup per workgroup) | `pk_t64x128k32g42s32nf` (packed staging) |
| attn*V | `ml_t32x64k32g42s32` | `ml_t64x128k32g42s32` |

All four accumulate in fp32 like the release SDPA kernels. Their output is bit-identical to the plain port of
the release kernels (`results/4070ti/sdpa-error/diff-port-vs-refine1.csv`: 0 of 12.6 million elements differ).

Measured end to end (`s2-c1b`, pristine parent against build `topic4` with
`ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=4070ti-refine1`; tok/s, median of 5 valid runs per arm):

| cell | parent | candidate 1 | gain | ETDump dispatch total, ms |
|---|---:|---:|---:|---|
| 1B 4w | 19692.3 | 28444.4 | +44.4 % | 102.6 -> 70.9 |
| 1B 8da4w | 21113.4 | 31030.3 | +47.0 % | 96.0 -> 63.9 |
| 3B 4w | 8678.0 | 12487.8 | +43.9 % | 233.6 -> 162.7 |
| 3B 8da4w | 9660.4 | 14524.8 | +50.4 % | 211.3 -> 141.9 |
| 8B 4w | 4481.4 | 5885.1 | +31.3 % | 455.6 -> 346.6 |
| 8B 8da4w | 4995.1 | 6804.0 | +36.2 % | 409.0 -> 300.7 |

Geomean **+42.05 %**, every cell far outside the +-2 % band, repeat spread at most 2.1 %. Against the original
`dev/1.5` numbers of `cells.csv` (19692 / 8752 / 4491 and 20898 / 9660 / 5032) the gains are the same within
1 %, because the parent is `dev/1.5`'s NVIDIA rows.

Where the gain comes from (warm ETDump, ms, parent -> candidate, `sessions/s2-c1b/trace/families.csv`):

| family | 1B 4w | 3B 4w | 8B 4w | 8B 8da4w |
|---|---|---|---|---|
| QK^T | 16.4 -> 4.2 | 42.8 -> 8.5 | 65.2 -> 13.0 | 63.1 -> 12.7 |
| attn*V | 21.6 -> 5.6 | 39.7 -> 8.2 | 60.2 -> 11.6 | 59.6 -> 11.6 |
| softmax (release kernel in both arms; truncated SARC variant in the candidate) | 13.6 -> 10.5 | 17.9 -> 13.9 | 27.3 -> 21.2 | 27.3 -> 21.2 |
| linear GEMM | 34.5 -> 34.3 | 98.6 -> 97.7 | 241.1 -> 239.2 | 184.8 -> 181.5 |
| everything else | 16.5 -> 16.3 | 34.6 -> 34.4 | 61.8 -> 61.6 | 74.2 -> 73.7 |

- QK^T: the masked half is neither computed nor written (`NO_MASK_FILL`, valid with the truncated SARC softmax
  only; the test's pairing check enforces it), and the rest is a cooperative-matrix product instead of scalar
  dot products. Per layer at kernel level 1012 -> 266 us (1B), 1508 -> 311 (3B), 2000 -> 410 (8B).
- attn*V: reads each row only up to its causal limit, cooperative-matrix product. 1347 -> 348, 1434 -> 294,
  1912 -> 359 us per layer.
- The direct-feed QK^T kernel of this campaign wins over the 780M-style staged kernel for head_dim 64 only
  (266 against 289 us); direct-feed attn*V loses to the staged one (500 to 700 against 350 to 430 us).
  fp16 accumulation was screened (`dfg`, `dfh`): at most 3 % on QK^T, slower on attn*V; not pursued.

Why it is rejected:

1. `verify.sh` (unmodified): `1b 8da4w unaligned: default vs tiled output DIFFER`; the parent control says
   SAME. Everything else in that gate passed (24 of 24 SDPA correctness passes with 0 mismatches and
   `pairing=ok`, correctness rc=0, 12 of 12 production-diff, the other three default-vs-tiled items SAME,
   decode 31 tokens, 22 of 22 runner calls rc 0).
2. Owner decision 2026-10-04, applied with thresholds fixed before the data existed (per cell and metric,
   candidate-default vs parent-default <= 2 x parent-tiled vs parent-default; 41 real-text prompts; evidence
   `results/4070ti/probe/`): **outside in 4 of 6 cells**.

| cell | top-1 differences (floor / cand) | mean KL, nats | max KL | max abs logit diff | abs ln ppl ratio | within 2x |
|---|---|---|---|---|---|---|
| 1B 4w | 0 / 0 | 0.000077 / 0.00082 | 0.0011 / 0.0102 | 0.46 / 0.66 | 0.0025 / 0.0170 | NO |
| 1B 8da4w | 2 / 6 | 0.0618 / 0.0678 | 0.606 / 0.560 | 4.42 / 3.90 | 0.105 / 0.063 | NO |
| 3B 4w | 0 / 0 | 0.00057 / 0.00024 | 0.0222 / 0.0028 | 0.45 / 0.63 | 0.0068 / 0.0099 | yes |
| 3B 8da4w | 1 / 1 | 0.0106 / 0.0246 | 0.152 / 0.226 | 3.73 / 3.74 | 0.052 / 0.109 | NO |
| 8B 4w | 0 / 0 | 0.000028 / 0.00029 | 0.00022 / 0.0027 | 0.27 / 0.36 | 0.00027 / 0.0019 | NO |
| 8B 8da4w | 1 / 1 | 0.0278 / 0.0227 | 0.447 / 0.275 | 2.72 / 4.17 | 0.028 / 0.0052 | yes |

At the differing position itself (1B 8da4w, 1792-token prompt, `probe/refine1/position/summary.csv`): the
parent has three tokens within 0.5 logit (top-2 margin +0.53 tiled, +0.33 default); with the candidate the
margin is -0.016 in the tiled arm and +0.20 in the default arm. The arm that flips is the reference arm of the
check; the candidate's own path gives the parent's token.

What this does and does not show:

- Measured: the top-1 token never differs in the 4w cells; the 4w differences are below 0.001 nat mean KL and
  1.7 % perplexity, and still ten times the 4w floor, because the parent's two 4w linear kernels are
  numerically almost identical while the candidate changes the attention arithmetic. In the 8da4w cells the
  candidate is at the floor in KL and logit difference; 6 against 2 of 41 on 1B is not a resolved difference.
- Measured: against the fp32 CPU reference of the SDPA test the candidate is the more accurate arm. The stock
  kernels accumulate in fp16 (`T` is the fp16 component type throughout `sdpa_compute_attn_weights_tiled` and
  `sdpa_compute_out_tiled`); rms error at S = 2048 is 8.5e-5 for stock and 3.5e-5 for the candidate
  (`results/4070ti/sdpa-error/summary.csv`).
- Inferred, not measured: that the distance from the parent is mostly the parent's own rounding. There is no
  fp32 reference for whole-model logits here.
- The rule is the owner's and it is applied as written: **candidate 1 is rejected**. Whether SDPA changes on
  this device should be judged against a reference rather than against the parent is an open question to the
  owner (`STATUS.md`).

Not complete in the evidence of `s2-c1b`: the session's own check says REJECT, because the first timed parent
run of 8B 8da4w aborted at exit (`corrupted double-linked list`, rc 134) after printing its token, which leaves
that next-token row INVALID. The other 23 rows are SAME, and the aborted run printed the same token. One traced
candidate run (3B 4w) also aborted at exit after writing its ETDump.

## Linear kernels: swept, nothing found

Kernel level, `test_llama_microbench --linear --regime=prefill`, texture3d (the model path), three models,
3 repeats, tokens interleaved; per-layer-weighted time against the shipped rows
(`results/4070ti/screens/screen1-4w.csv`, `screen1-8da4w.csv`, `screen2-8da4w.csv`):

- 4w, 14 `ga` tiles (128 or 256 on each side, k 16 or 32, four grids, drain in Ash, ColumnMajor B): best
  `t256x128k16g42s32gac` 0.995x; the 256 x 256 tile 0.63x; every k 32 tile 0.70 to 0.86x.
- 8da4w, 13 zpgtr tiles: best 0.84x.
- 8da4w half-texel weight staging (`bh`: each staging slot keeps two words of its packed-weight texel instead of
  one, half as many weight fetches): 1.011x with a 1.2 % repeat spread; production-diff 6 of 6 ALL PASSED. The
  fetch phase is 34 to 43 % of a wave on this kernel, but halving the number of fetches does not shorten it,
  so it is not the fetch count that costs. What does was not established.

Static pruning before any run: shared memory at most 49152 bytes (the double-buffered 4w staging excludes k 32
on a 256-row tile and every k 64 tile), staging thread maps that divide the workgroup, the texture3d drain band
fitting in Ash. The space that survives is small enough to enumerate; no sampling was needed.

Candidates 2 and 3 (`4070ti-refine2`, `4070ti-refine3`) put the two near-noise variants through the full gate
to close this part with end-to-end numbers: see "Gated linear candidates" below.

## Two more SDPA kernels, measured, both behind hooks

**Softmax without the full zero tail** (`sarc_sdpa_attn_weights_softmax_buffer_half_4070ti_nz`). With
candidate 1 the release softmax is half of the attention time and runs at the DRAM write roof; half of what it
writes is zeros above the diagonal that the SARC attn*V kernels never read beyond their own row tile. Kernel
level (`sdpa-screen3.csv`): softmax 659 -> 462 us per layer (1B, 8B), 494 -> 345 (3B). SDPA output
bit-identical to candidate 1; extended and full tiers PASSED. Ungated quick look end to end: +4.3 % (1B 4w),
+4.8 % (1B 8da4w), +2.5 % (3B 4w) over candidate 1. It is only valid together with attn*V kernels that
truncate; the variant checks the fit condition of this device's table row itself and otherwise behaves as the
release shader. Not gated: it inherits candidate 1's rejection (same numerics) and needs hook 2.

**Fused attention** (`sarc_sdpa_fused_coopmat_4070ti_c<tile>d<head_dim>`). One workgroup owns 16 rows of one
head and walks the visible context twice (row maximum; then exp, row sum and the attn*V accumulation), so the
score matrix never reaches DRAM. Correct on its first run (0 mismatches in the 12 extended and full cases) and
the most accurate of all (rms 2.0e-5 against the fp32 reference), but **slower** than candidate 1 as written:
1B 4w 26.6k against 28.1k tok/s, 3B 4w 10.8k against 12.5k (ungated quick look, column tile 32 or 64). The
kernel computes QK^T twice with a half-rate fp32-accumulate MMA and does its exp and reductions with 32 threads
per workgroup through shared memory with five barriers per tile; which of these dominates was not measured.
Recorded as explored, not as a candidate. The SDPA test prints FAILED for it although every element matches,
because it looks for separate QK^T and attn*V kernel names; the test was not changed.

## Gated linear candidates

TO BE FILLED (candidates 2 and 3).

## What limits further progress

- **Attention is memory traffic.** After candidate 1 the three attention kernels still move 134 MB (QK^T write)
  + 134 MB read and 268 MB write (softmax) + 134 MB (attn*V read) per layer on 1B. The softmax is at the DRAM
  roof; QK^T and attn*V are at about 440 and 380 GB/s against 643 and 713. The softmax variant removes 134 MB.
  Removing the rest needs the fused node (hook 3) and a faster fused kernel than the one here.
- **None of the SDPA work is reachable or acceptable today**: it needs table rows for this device (hook 1), and
  on this device it changes logits more than the parent's own linear arms differ.
- **Linear kernels** sit at 64 % (4w) and 41 % (8da4w) of their matrix roofs and are the largest block of the 8B
  prefill (240 of 347 ms with candidate 1). Tile sweeps and one staging change found nothing. On 4w a wave
  spends 22 % at barriers and 25 % staging against 50 % in the MMA; on 8da4w it spends more time fetching than
  multiplying. The shared-memory limit (49152 bytes) is what excludes the larger 4w tiles; a single-buffered
  staging loop would lift it. Not tried.
- `copy/view` and elementwise operators are 14 % of the 1B prefill with candidate 1 and are upstream operators
  outside both zones.
- The parent aborts or hangs at exit now and then on this card (five times in several hundred runner calls,
  in the parent and in the candidate arm, always after the results were written). It costs a session a finding
  whenever it hits a run the gate needs.

## Limits of this study

- One device, one driver, 2048-token prompts, clocks as found (unpinned; sampled 2565 to 2790 MHz).
- The broad logits comparison uses 41 prompts; counts of top-1 differences on that sample are small numbers.
- Roofs are `fast`-plan, short-run values. The watcher flagged the roofline run after it had finished, because
  the tool starts its runners with a cleaned environment; the run itself completed with rc 0 and its roofs
  agree with the earlier evidence within 0.5 %.
- Raw logs, ETDumps, clock samples, logits and binaries are in
  `gpu-dev-4004:~/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/`, not in the tree.

## Checks on the branch

- `sarc/tools/check.sh --no-build`: PASS (zone rule, twin wrappers, `test_sarc_select` with and without the dev
  zone).
- `sarc/tools/spirv_golden.py` on every build: the 53 shipped variants are unchanged.
- `tools/test_gate_check.py`: 36 tests pass.
