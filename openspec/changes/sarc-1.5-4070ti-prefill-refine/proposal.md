# sarc-1.5-4070ti-prefill-refine

Dev zone only (2026-10-04). Branch `topic/4070ti-prefill-refine`, forked from `topic/780m-prefill-refine` at
`6a7cc8cc6`. That commit with no profile (the shipped NVIDIA rows) is the parent of every comparison. Nothing is
promoted; no release-zone or upstream file is edited. Day-to-day state and incidents are in `STATUS.md`.

## Why

Push the 2048-token Llama prefill on the RTX 4070 Ti SUPER (Vulkan backend, driver 615.71.09) as far as it
goes, measured against the parent in the same session.

## Outcome in one paragraph

SDPA prefill kernels for this card plus an fp32 softmax (**candidate 4**: profile `4070ti-refine1` with softmax
`4070ti_nzf`) are worth **+46.74 % geomean** over the parent (+33.9 % to +55.6 % per cell) and pass the whole
gate, with no next-token item differing and a lower error against the fp32 reference than the parent's own
attention kernels in every test case. All of the gain is attention time (51.7 -> 17.1 ms on 1B, 152.9 -> 39.7 ms
on 8B). It is not reachable from the dev zone alone: it needs two small release-zone hooks (SDPA rows for this
device, a dev name for the softmax), measured here through a local patch that is not committed. The first SDPA
candidate (same QK^T and attn*V kernels with the release fp16 softmax, +42 %) was rejected on precision grounds
and stays rejected. The linear kernels were swept and two variants were put through the gate on top of
candidate 4, twice each: +0.11 % / +0.28 % and +0.54 % / +0.47 % geomean, inside the noise band, which ends the
campaign by its stop rule (with the reservation that none of those four gates completed every step: see
"Gated linear candidates"). What limits the prefill now is the
linear kernels (50 % of the 1B prefill and 70 % of the 8B one at 64 % and about 41 % of their matrix roofs).

**Recommended configuration: candidate 4.** The linear profiles (`4070ti-refine2` to `-refine5`) are recorded
as measured and are not recommended.

## What is in the tree

All under `backends/vulkan/runtime/graph/ops/{glsl,impl}/sarc_dev/`, `backends/vulkan/test/sarc_dev/` and this
directory. Every shader file of this campaign carries `4070ti` in its name; `impl/sarc_dev/Overrides.cpp` is
changed only inside `// >>> 4070ti <id>` .. `// <<< 4070ti <id>` blocks. The generators in `tools/gen_4070ti_*.py`
read the release sources and never write them.

| family (glsl/sarc_dev) | what | generator |
|---|---|---|
| `sarc_sdpa_qk_coopmat_4070ti`, `_pk`, `sarc_sdpa_av_coopmat_4070ti`, `_ml` | the 780M campaign's SDPA dev twins at subgroup 32, 26 tiles | `gen_4070ti_sdpa.py` |
| `sarc_sdpa_qk_coopmat_4070ti_df`, `sarc_sdpa_av_coopmat_4070ti_df` | **new**: direct-feed QK^T and attn*V (no shared-memory staging), fp32 / grouped fp16 / fp16 accumulation, 33 variants | `gen_4070ti_df.py` |
| `sarc_sdpa_attn_weights_softmax_4070ti` | **new**: LLM softmax variants: `nz` (zeroes the masked tail only as far as attn*V reads it), `f32` (row reduction in fp32, one rounding on the store), `nzf` (both) | `gen_4070ti_softmax.py` |
| `sarc_sdpa_fused_coopmat_4070ti` | **new**: fused prefill attention, the score matrix is never written | `gen_4070ti_fused.py` |
| `sarc_linear_q4gsw_coopmat_4070ti`, `sarc_linear_dq8ca_coopmat_zpgtr_4070ti` | sweep tiles of the two linear families (release bodies), 14 + 13 | `gen_4070ti_lin.py` |
| `sarc_linear_dq8ca_coopmat_zpgtr_4070ti_bh` | **new**: zpgtr with half-texel weight staging (twin body) | `gen_4070ti_bh.py` |
| `sarc_dev_prof_4070ti_q4gsw`, `sarc_dev_prof_4070ti_dq8ca_zpgtr` | phase-timing twins, measurement only | `gen_4070ti_prof.py` |

Profiles (`ET_VK_SARC_DEV_PROFILE`): `4070ti-refine1` (SDPA), `4070ti-refine2` (8da4w `bh`), `4070ti-refine3`
(`refine2` + 4w drain in Ash), `4070ti-refine4` (`refine1` + `bh`), `4070ti-refine5` (`refine4` + 4w drain in
Ash), and one screening profile per SDPA tile (`4070ti-qk-*`, `4070ti-av-*`). The softmax variant is chosen by
`ET_VK_SARC_SOFTMAX_VARIANT` in a build that carries hook 2.

`test_llama_microbench --sdpa-correctness-only` gained a reported `[sdpa-error]` line (max and rms error against
its fp32 reference) and `ET_VK_SDPA_DUMP_DIR`; neither changes a verdict.

## What cannot be reached from the dev zone (hooks that would be needed)

These are measured through local patches in `tools/` that are applied to an archived source tree at build time,
recorded in the build provenance, and **not** applied to the branch.

1. **SDPA rows for this device** (`tools/local-hook-nvidia-sdpa.patch`, 13 lines). The dev profile only replaces
   an existing table choice, and the SDPA hooks in `impl/sarc/SdpaCoopmat.cpp` (spec constants, truncated
   softmax) are enabled by a table row. Smallest hook: two `kUnverified` rows in `impl/sarc/table_nvidia.cpp`,
   or a way for a dev profile to mark a device as having SDPA rows. Every SDPA result here depends on it.
   Candidate 4 needs this hook and hook 2, nothing else.
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
  (`tools/build-both.sh`), never from the working copy. All twelve builds (parent, `topic1` to `topic11`):
  `spirv_golden.py` PASS, 53 shipped variants unchanged.
- End to end: `tools/e2e5.sh` (the kit `e2e.sh` protocol: fresh `llama_main` per run, `--warmup`,
  `prompt_2048.txt`, one new token, arms interleaved, cool between runs), 5 valid runs per arm and cell, clock,
  busy, power and temperature sampled every 20 ms with `nvidia-smi`. A run counts only with rc 0, 2048 prompt
  tokens, 0 generated tokens, no GPU process of another owner and a median clock of at least 2502 MHz
  (`results/4070ti/clkmin.json`, from the A/A session). One GPU job at a time under the gpu-lab lock. No build
  ran during a timed session.
- Gate: `tools/gate.sh` / `tools/gate_sdpa.sh`, judged on the content of the result files by
  `tools/gate_check.py` (40 unit tests in `tools/test_gate_check.py`). The owner decisions of 2026-10-04 are
  implemented as `--near-tie <NEAR_TIE.json>` and `--reference-error <REFERENCE_ERROR.json>`
  (`tools/ref_error_rule.py`): the evidence file is only written when the rule is met, and an acceptance that
  used it is labelled as such.
- 1B timer quantisation: a 1B prefill is 62 to 104 ms and the runner's timer is 1 ms, so 1B cells move in steps
  of 1.0 to 1.6 %. The warm ETDump dispatch totals are given alongside.

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
- The rule is the owner's and it is applied as written: **candidate 1 is rejected**.
- The owner then decided (second decision of 2026-10-04) that arithmetic changes are judged against a
  reference. Under that rule candidate 1 is **still not accepted**
  (`results/4070ti/probe/refine1/reference-error-rule.txt`): its rms error against the fp32 CPU reference is
  2.5 times lower than the parent's in every case, but on the 3B head configuration at S = 2048 its maximum
  error is 1.570e-3 against the parent's 1.408e-3, and the criterion is rms and maximum on every head
  configuration.

Not complete in the evidence of `s2-c1b`: the session's own check says REJECT, because the first timed parent
run of 8B 8da4w aborted at exit (`corrupted double-linked list`, rc 134) after printing its token, which leaves
that next-token row INVALID. The other 23 rows are SAME, and the aborted run printed the same token. One traced
candidate run (3B 4w) also aborted at exit after writing its ETDump.

## Candidate 4: the same kernels with an fp32 softmax (`4070ti-refine1` + `4070ti_nzf`): +46.74 %, ACCEPTED

What is different from candidate 1: only the softmax. Candidate 1 missed the reference criterion on one maximum
error, and the one part of its attention block still computing in fp16 was the softmax (row maximum, exp, sum
and division are fp16 in the release shader). `4070ti_nzf` reduces each row in fp32 and rounds once on the
store, and zeroes the masked tail only as far as attn*V reads it. It was measured once against the thresholds
and inputs fixed beforehand; no other variant was tried on this point. Environment
`ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=4070ti-refine1 ET_VK_SARC_SOFTMAX_VARIANT=4070ti_nzf`, build
`topic10` (`267edc6a5` + `tools/local-hook-nvidia-sdpa-softmax.patch`, hooks 1 and 2).

End to end (`s5-c4`, pristine parent against candidate 4; tok/s, median of 5 valid interleaved runs per arm;
`results/4070ti/sessions/s5-c4/`):

| cell | parent | candidate 4 | gain | `dev/1.5` (`cells.csv`) | gain over `dev/1.5` | ETDump dispatch total, ms |
|---|---:|---:|---:|---:|---:|---|
| 1B 4w | 19692.3 | 29681.2 | +50.7 % | 19692.3 | +50.7 % | 103.4 -> 66.5 |
| 1B 8da4w | 20898.0 | 32507.9 | +55.6 % | 20898.0 | +55.6 % | 97.3 -> 61.3 |
| 3B 4w | 8678.0 | 12800.0 | +47.5 % | 8752.1 | +46.3 % | 233.6 -> 159.4 |
| 3B 8da4w | 9660.4 | 14948.9 | +54.7 % | 9660.4 | +54.7 % | 211.7 -> 138.4 |
| 8B 4w | 4471.6 | 5988.3 | +33.9 % | 4491.2 | +33.3 % | 456.3 -> 342.8 |
| 8B 8da4w | 4983.0 | 6942.4 | +39.3 % | 5031.9 | +38.0 % | 410.6 -> 295.9 |

Geomean **+46.74 %** over the parent (+46.2 % over the `cells.csv` numbers), every cell far outside the +-2 %
band, repeat spread at most 2.1 %, no run with a non-zero exit status, one run invalid (`clock_low`) and
replaced. 1B readings are whole milliseconds (104 -> 69, 98 -> 63); the ETDump totals are the finer measure and
say the same (-35.7 % and -37.0 % dispatch time).

Where the gain comes from (warm ETDump, ms per prefill, parent -> candidate 4, `sessions/s5-c4/trace/families.csv`):

| cell | QK^T | attn*V | softmax | linear GEMM | everything else |
|---|---|---|---|---|---|
| 1B 4w | 16.5 -> 4.0 | 21.6 -> 5.6 | 13.6 -> 7.5 | 34.6 -> 33.5 | 17.0 -> 15.8 |
| 1B 8da4w | 16.5 -> 4.2 | 21.6 -> 5.4 | 13.7 -> 7.5 | 27.9 -> 27.7 | 17.7 -> 16.5 |
| 3B 4w | 42.9 -> 8.6 | 39.7 -> 8.2 | 17.9 -> 9.9 | 98.5 -> 98.2 | 34.6 -> 34.5 |
| 3B 8da4w | 41.8 -> 8.5 | 39.2 -> 8.1 | 17.9 -> 9.8 | 75.6 -> 75.0 | 37.2 -> 37.0 |
| 8B 4w | 65.4 -> 13.1 | 60.2 -> 11.7 | 27.3 -> 14.9 | 241.6 -> 241.5 | 61.9 -> 61.7 |
| 8B 8da4w | 63.3 -> 12.7 | 60.0 -> 11.7 | 27.3 -> 14.9 | 185.5 -> 183.0 | 74.6 -> 73.6 |

QK^T and attn*V gain as in candidate 1 (same kernels). The softmax goes from 850 / 636 / 851 us per layer
(1B / 3B / 8B, stock) to 463 / 347 / 463 (`sessions/s5-c4/sdpa-correctness/perf-*.log`): the truncated variant
does not write the zero half, and computing the row reduction in fp32 costs nothing measurable because the
kernel is bound by memory traffic. Over candidate 1 that is another +1.8 % (8B 4w) to +4.8 % (1B 8da4w) end to end (two sessions
compared, not one interleaved measurement).

Gate (`gate.done`: `GATE_ACCEPTED 2026-10-05T02:13:20Z all steps passed`):

- SDPA correctness: 24 of 24 passes (12 extended, 12 full), 0 mismatches, `pairing=ok` on every case.
- Unmodified `verify.sh`, same status as the parent control: correctness rc 0, linear rc 1 / 1 with the same
  24 confirmed + 24 `unexpected_coopmat` cases as the parent (the known texture3d report of this device),
  12 of 12 production-diff ALL PASSED, default vs tiled SAME on all four items, decode 31 tokens, 22 of 22
  runner calls rc 0.
- Next token parent vs candidate: SAME in all six cells on four prompts (24 of 24 rows).
- **No next-token item differs, so the gate recorded a plain pass and did not use the reference-error rule.**
  The candidate changes accumulation and precision, so its measured error is reported regardless, and it is
  the evidence the second owner decision asks for (`results/4070ti/probe/refine1-nzf/`):

| production case (S = 2048) | rms error vs fp32 CPU reference, parent / candidate 4 | maximum error, parent / candidate 4 |
|---|---|---|
| 1B head configuration | 8.55e-5 / 2.10e-5 | 1.713e-3 / 0.914e-3 |
| 3B head configuration | 8.69e-5 / 2.07e-5 | 1.408e-3 / 0.783e-3 |
| 8B head configuration | 8.70e-5 / 2.06e-5 | 1.587e-3 / 0.891e-3 |

Not larger in any of the 12 test cases, for rms and for maximum. On 41 real-text prompts (`compare.csv`; the
parent's tiled arm against its default arm beside it for scale):

| cell | top-1 differences of 41 (parent's two arms / candidate vs parent) | mean KL, nats | max KL | max abs logit diff |
|---|---|---|---|---|
| 1B 4w | 0 / 0 | 0.000077 / 0.00091 | 0.0011 / 0.0080 | 0.46 / 0.58 |
| 1B 8da4w | 2 / 5 | 0.0618 / 0.0756 | 0.606 / 0.825 | 4.42 / 4.34 |
| 3B 4w | 0 / 0 | 0.00057 / 0.00025 | 0.0222 / 0.0019 | 0.45 / 0.73 |
| 3B 8da4w | 1 / 1 | 0.0106 / 0.0238 | 0.152 / 0.408 | 3.73 / 4.00 |
| 8B 4w | 0 / 0 | 0.000028 / 0.00024 | 0.00022 / 0.0017 | 0.27 / 0.36 |
| 8B 8da4w | 1 / 4 | 0.0278 / 0.0251 | 0.447 / 0.299 | 2.72 / 4.56 |

Largest mean KL 0.076 nat (the owner's gross-divergence limit is 0.5), top-1 differs on at most 5 of 41 prompts
(limit: one third). Under the FIRST decision's test (twice the spread of the parent's two linear arms) this
candidate would still be outside in several cells, as candidate 1 was; the second decision replaced that test
for arithmetic changes and is the one applied. To be read plainly: on the 8da4w models the top-1 token changes
against the parent on 5, 1 and 4 of 41 prompts, where the parent's own two arms differ on 2, 1 and 1.

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

The two near-noise variants were then put through the full gate on top of candidate 4, to close this part with
end-to-end numbers: see "Gated linear candidates" below.

## Two more SDPA kernels, measured, both behind hooks

**Softmax without the full zero tail** (`sarc_sdpa_attn_weights_softmax_buffer_half_4070ti_nz`, and its fp32
form `_nzf`, which is part of candidate 4). With candidate 1 the release softmax is half of the attention time
and runs at the DRAM write roof; half of what it writes is zeros above the diagonal that the SARC attn*V kernels
never read beyond their own row tile. Kernel level (`sdpa-screen3.csv`): softmax 659 -> 462 us per layer
(1B, 8B), 494 -> 345 (3B). The fp16 form `nz` gives SDPA output bit-identical to candidate 1 and inherits its
rejection; the fp32 form is what candidate 4 runs. The variant is only valid together with attn*V kernels that
truncate; it checks the fit condition of this device's table row itself and otherwise behaves as the release
shader. Needs hook 2.

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

Both linear variants were gated on top of candidate 4 (`tools/gate.sh`; build `topic11` = `42e001462` +
hooks 1 and 2; both arms run the SDPA kernels and the `4070ti_nzf` softmax; the parent arm is candidate 4,
profile `4070ti-refine1`). tok/s, median of 5 valid interleaved runs per arm, gain over the parent arm of the
same session:

| cell | candidate 5 `4070ti-refine4`, `s6-c5` | again, `s6-c5b` | candidate 6 `4070ti-refine5`, `s7-c6` | again, `s7-c6b` |
|---|---:|---:|---:|---:|
| 1B 4w | 29681.2 (+0.00 %) | 29681.2 (+0.00 %) | 29681.2 (+0.00 %) | 29681.2 (+0.00 %) |
| 1B 8da4w | 32507.9 (+0.00 %) | 32507.9 (+0.00 %) | 33032.3 (+1.61 %) | 33032.3 (+1.61 %) |
| 3B 4w | 12800.0 (+0.00 %) | 12880.5 (+0.63 %) | 12880.5 (+0.63 %) | 12880.5 (+0.63 %) |
| 3B 8da4w | 14948.9 (+0.00 %) | 14948.9 (+0.00 %) | 14948.9 (+0.00 %) | 14840.6 (-0.72 %) |
| 8B 4w | 5988.3 (+0.00 %) | 6005.9 (+0.00 %) | 6005.9 (+0.29 %) | 5988.3 (+0.29 %) |
| 8B 8da4w | 6989.8 (+0.68 %) | 7013.7 (+1.03 %) | 7013.7 (+0.68 %) | 7013.7 (+1.03 %) |
| **geomean** | **+0.11 %** | **+0.28 %** | **+0.54 %** | **+0.47 %** |

- Candidate 5 = candidate 4 + 8da4w zpgtr half-texel weight staging (`bh`).
- Candidate 6 = candidate 5 + the shipped 4w wide tile with the texture3d drain staged in Ash (`gac`, N > 512),
  compared with candidate 4 because candidate 5 had not been accepted.
- No cell is outside the +-2 % band in any of the four sessions. The 1B and 3B figures are single timer steps
  (1 ms in 62 to 69 ms is 1.5 %; 1 ms in 160 ms is 0.6 %). The only reading that repeats is 8B 8da4w, +0.7 % to
  +1.0 % in all four sessions, the size the kernel-level screen gave for `bh` (1.011x). It is below the band.
- In all four gates `verify-check` is ACCEPT with the same status as the parent control (the `bh` and `gac`
  kernels dispatched where the profile asks for them, 12 of 12 production-diff ALL PASSED, default vs tiled
  SAME), and every next-token row that could be compared is SAME (94 of 96; the other two are the INVALID
  rows of `s6-c5` and `s7-c6b`, see below).
- **None of the four gates reached `GATE_ACCEPTED`.** Each lost one step to the runner failing after it had
  produced its output:

| session | `gate.done` | what failed |
|---|---|---|
| `s6-c5` | REJECTED at session-check | first timed parent run of 3B 4w aborted (`corrupted double-linked list`, rc 134): next-token row of the timed prompt INVALID |
| `s7-c6` | REJECTED at the trace step (verify-check and session-check ACCEPT) | traced parent run of 3B 4w aborted (rc 134) |
| `s6-c5b` | REJECTED at the trace step (verify-check and session-check ACCEPT) | traced candidate run of 3B 4w hung after writing its ETDump (one thread at 100 % CPU, GPU idle); ended with SIGTERM after 8.5 minutes |
| `s7-c6b` | REJECTED at session-check | first timed candidate run of 8B 8da4w aborted (rc 134): next-token row of the timed prompt INVALID |

  The failure is in whichever arm it happens to hit (twice the arm that plays the parent, twice the candidate)
  and is the one the pristine parent shows too (below). It still fails a gate step, so the sessions are recorded
  as rejected. They were not repeated a third time: with the failure rate measured below a gate of about 135
  runner calls completes cleanly well under half of the time, and neither variant has a gain to accept.

**Stop rule.** Two consecutive candidates, each gated twice, gain less than 2 % geomean over their parent
(+0.11 % / +0.28 % and +0.54 % / +0.47 %). The campaign stops here. Reservation, stated plainly: the rule says
"gated candidates", and the gates of these two did not complete; what is established is that their measured
gain is inside the noise band in four complete timed sessions, and that everything the gates did check passed.

### The runner fails after its output now and then, in every build

The symptom is always the same: the runner echoes the prompt and the generated token, then either aborts in
the allocator (`corrupted double-linked list`, `free(): chunks in smallbin corrupted`, rc 134) or spins one
thread at 100 % CPU with the GPU idle, before the stats line. `nvidia-smi` answers throughout and the kernel
log shows no Xid. `TECHNICAL-REPORT.md` of the e2e benchmark already describes a rare exit-time crash on this
card.

Counted over every runner call of the staged sessions (timed and next-token runs in `runs.csv`, the 22 calls of
each `verify.sh`, the traces):

| arm | runner calls | failed after output |
|---|---:|---:|
| pristine parent, or a build without the SDPA kernels in use | 299 | 2 (0.7 %) |
| SDPA kernels in use (candidates 1, 4, 5, 6) | 690 | 5 (0.7 %) |

One more in the pristine parent outside the sessions (the 41-prompt logits dump of 3B 8da4w). Four of the five
SDPA-arm failures were on 3B 4w, so that cell was run 150 times per arm, pristine parent and candidate 4
interleaved, a fresh process per run (`tools/exit_probe.sh`, `results/4070ti/exit-probe/3b-4w.csv`): **0 failures
in 150 for either arm.** Read together: the rate is the same with and without this campaign's kernels (about
0.5 % per call), the 3B 4w cluster did not reproduce, and the cause is not known. The symptom points at heap
corruption in the runner process, which no test of the gate looks for; nothing here locates it. At 0.7 % per call
a gate of about 135 runner calls has about a 40 % chance of completing with no failed call, which is what was
seen (one gate of five on candidate 4's builds completed).

## What limits further progress

Shares below are of candidate 4's warm dispatch time (`sessions/s5-c4/trace/`): 66.5 ms on 1B 4w, 342.8 ms on
8B 4w.

- **The linear kernels are now the prefill**: 50 % of it on 1B 4w, 62 % on 3B 4w, 70 % on 8B 4w (45 %, 54 %,
  62 % for 8da4w). In the model they run at 118 TFLOP/s for 4w, **64 to 65 % of the fresh fp16 matrix roof**
  (182.9 TFLOP/s), and at about 144 to 156 TOP/s for 8da4w, **39 to 42 % of the fresh int8 roof** (369.1 TOP/s).
  The tile sweeps and the two staging changes found nothing above the noise band. What the phase timing says
  is left: on 4w a wave spends 22 to 32 % at barriers and 25 to 31 % staging against 34 to 50 % in the MMA, and
  the shared-memory limit (49152 bytes) excludes the larger double-buffered tiles, so a single-buffered staging
  loop is the untried change; on 8da4w a wave spends more time fetching (34 to 43 %) than multiplying (30 to
  35 %), halving the number of fetches did not shorten it, and what the fetch phase is waiting for was not
  established. That is the first thing to find out before writing another 8da4w kernel.
- **fp32 accumulation is half rate on this card** (92.3 against 182.9 TFLOP/s, re-measured), which is why the 4w
  kernels group their fp32 accumulation and why 64 % of the fp16 roof is not simply inefficiency.
- **Attention is memory traffic, and less of it is left**: 26 % of the 1B prefill (17.1 ms), 17 % of 3B, 12 % of
  8B. Per layer on 1B the three kernels still write 134 MB (QK^T, 266 us), read and write about 134 MB each
  (softmax, 463 us) and read 134 MB (attn*V, 336 us): roughly 500, 580 and 400 GB/s against fresh DRAM roofs
  of 643 (write), 646 (copy) and 713 (read). The softmax is within about 10 % of the copy roof; QK^T and attn*V
  have some room. Removing the traffic itself needs the fused node (hook 3) and a faster fused kernel than the
  one written here, which was slower than the three-kernel path.
- **Reachability**: candidate 4 is not usable from the dev zone. It needs hook 1 (SDPA rows for this device)
  and hook 2 (softmax name), both in `impl/sarc/`.
- `copy/view` and elementwise operators are 22 % of the 1B prefill and 16 % of the 8B one with candidate 4.
  They are upstream operators outside both zones.
- The runner's failure after output (above) limits how much can be gated on this card: at the measured rate a
  full gate is more likely to lose a step than to complete.

## Limits of this study

- One device, one driver, 2048-token prompts, clocks as found (unpinned; sampled 2565 to 2790 MHz). The clock
  rule (median `clocks.gr` of a run at least 2502 MHz) is this campaign's own; runs that fail it right after an
  idle period show the same rate as the others, so the sampled value lags the real clock there.
- The broad logits comparison uses 41 prompts; counts of top-1 differences on that sample are small numbers.
  There is no fp32 reference for whole-model logits; the reference criterion is measured on the SDPA block.
- Candidate 4 against candidate 1 is a comparison of two sessions, not one interleaved measurement.
- The four linear gates did not complete (above). Their timing sessions did.
- Roofs are `fast`-plan, short-run values. The watcher flagged the roofline run after it had finished, because
  the tool starts its runners with a cleaned environment; the run itself completed with rc 0 and its roofs
  agree with the earlier evidence within 0.5 %. `shared_fp16_read` (1344 GB/s) is still not understood and is
  not used.
- The control session of the campaign was lost once (about 00:50 UTC on 2026-10-05) while the detached queue
  kept running; the session that took over worked from the files. One `sudo -n dmesg` was issued by it (reading
  the kernel log for an Xid), which is outside what the campaign permits `sudo` for; nothing was changed.
- Raw logs, ETDumps, clock samples, logits and binaries are in
  `gpu-dev-4004:~/hmz-sarc-4070ti/.artifacts/4070ti-prefill-refine/`, not in the tree.

## Checks on the branch

- `sarc/tools/check.sh --no-build`: PASS (zone rule, twin wrappers, `test_sarc_select` with and without the dev
  zone).
- `sarc/tools/spirv_golden.py` on every build: the 53 shipped variants are unchanged.
- `tools/test_gate_check.py`: 40 tests pass.
- `git diff --name-status 6a7cc8cc6 HEAD`: only `glsl/sarc_dev/` (new `4070ti` files), `impl/sarc_dev/Overrides.cpp`
  (marked `4070ti` blocks), `backends/vulkan/test/sarc_dev/test_llama_microbench.cpp` (a reported error line
  and an output dump, no verdict or tolerance touched) and this directory.
