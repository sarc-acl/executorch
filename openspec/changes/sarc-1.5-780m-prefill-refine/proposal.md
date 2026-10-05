# sarc-1.5-780m-prefill-refine

Dev zone only (2026-10-03). Nothing here is promoted and no release-zone or upstream file is touched. Branch
`topic/780m-prefill-refine`, forked from `dev/1.5` at `ef079ac41`; not pushed, no PR.

## Why

`dev/1.5` already runs the Llama prefill on the Radeon 780M with the SARC WMMA kernels (4w, 8da4w) and the SARC
SDPA kernels. This change asks what is still left on that device, measured against `dev/1.5` itself (not against
stock ExecuTorch), using sweep variants only.

## What

All under `backends/vulkan/runtime/graph/ops/{glsl,impl}/sarc_dev/`, `backends/vulkan/test/sarc_dev/` and this
directory:

- `ET_VK_SARC_DEV_PROFILE=<name>` (`impl/sarc_dev/Overrides.cpp`): a named list of preferred sweep tiles with
  optional shape predicates. A shape that a preferred tile does not cover keeps the table's choice, so nothing
  falls back to the tiled kernels and devices without rows are unaffected. `ET_VK_FORCE_TILED_LINEAR` and
  `ET_VK_DISABLE_COOPMAT` still win.
- Sweep tiles for the two linear families (batches 1 to 3 in the sweep yamls).
- `sarc_dev_linear_dq8ca_coopmat_zpg_bt`: the 8da4w zpg kernel with texel-wise weight staging.
- `sarc_sdpa_qk_coopmat_sweep` (adds `NO_MASK_FILL`), `sarc_sdpa_qk_coopmat_pk` (packed staging),
  `sarc_sdpa_av_coopmat_sweep`, `sarc_sdpa_av_coopmat_ml` (multi-pass staging).
- `sarc_dev_linear_q4gsw_coopmat_bx`: 4w with texel-wise weight staging (screened out, kept as a sweep variant).
- Measurement only: `sarc_dev_prof_*` shader-clock twins and `ET_VK_DUMP_OUTPUT_DIR` in the dev test utils.

The generated shader files come from `tools/gen_*.py`, which read the release bodies and never write them.

**Winner: `ET_VK_SARC_DEV_PROFILE=780m-refine5`** (candidates 1, 2, 3 and 5):

| op | kernel | shapes |
|---|---|---|
| 8da4w linear | `sarc_dev_linear_dq8ca_coopmat_zpg_bt_t128x64k32g22s32` | all shapes the row fits; others keep the table's zpg kernel |
| 4w linear | `sarc_linear_q4gsw_coopmat_sweep_t128x256k32g42s32f32c` | texture3d, N >= 1024 and N % 256 == 0; others keep `t128x128k32g42s32f32c` |
| SDPA QK^T | `sarc_sdpa_qk_coopmat_pk_t128x64k32g42s32nf` | tile-aligned prefill |
| SDPA attn*V | `sarc_sdpa_av_coopmat_sweep_t64x64k32g42s32` | tile-aligned prefill |

Profiles `780m-refine1`, `-refine2`, `-refine3` are the earlier steps; `780m-refine6` is candidate 6 (gate
passed, no measurable gain, not part of the winner).

## Method

- Host `rocky-ryzen`, Radeon 780M (RADV PHOENIX, Mesa 25.2.7). Working copy `~/hmz-sarc/executorch`; builds
  with `sarc/tools/build.sh` in `localhost/et-vk-build:rocky10`, rebuilt on that host from the same
  Containerfile (shaderc v2023.8, GCC 14.3.1). The parent build's 53 shipped variants match
  `sarc/golden/spirv.json`.
- Baseline (parent) = `dev/1.5` at `ef079ac41`, measured again inside every session.
- End to end: `tools/e2e5.sh`, the kit `e2e.sh` protocol (fresh `llama_main` per run, `--warmup`,
  `prompt_2048.txt`, one new token, cool to idle + 5 C, arms interleaved) with 5 repeats, the GPU clock,
  busy %, power and temperature sampled every 0.1 s during each run, and a run counted only if rc = 0,
  2048 prompt tokens, 0 generated tokens, no other GPU process and a median clock of at least 2700 MHz inside
  the measured window. Median of the first 5 valid runs per arm. 420 timed runs in 7 sessions, none rejected; the
  median clock in the measured windows was 2736 to 2800 MHz, start temperature 41 to 51 C, peak 94 C.
- Gate per candidate (`tools/gate.sh`, `tools/gate_sdpa.sh`): unmodified `sarc/tools/verify.sh --models
  1b,3b,8b --schemes 4w,8da4w --pdiff` with the candidate's environment, then the e2e session with the
  next-token comparison parent vs candidate on `prompt_2048.txt` and on the unaligned `prompt_check.txt` for
  all six cells, then warm ETDump traces of both arms. SDPA candidates add 12 passes of
  `test_llama_microbench --sdpa-correctness-only`.
- One GPU job at a time, all under the gpu-lab lock of the 780M; no build runs during a measurement.
- A gain inside +-2 % is reported as inside the noise band, even though the A/A session (pristine parent build
  against the topic build without overrides) agreed within 0.05 % with repeat spreads of at most 0.4 %.
- Stop rule: two consecutive gated candidates under 2 % geomean over their parent. Candidates 5 and 6 met it.

## Results

Parent `dev/1.5` (session `s1-aa`, tok/s, median of 5): 1B 2698.3 (4w) / 2544.1 (8da4w), 3B 1146.7 / 1056.2,
8B 521.8 / 489.4.

Each candidate against its parent (the previous accepted candidate), same session:

| candidate (profile) | kind | 1B 4w / 8da4w | 3B 4w / 8da4w | 8B 4w / 8da4w | geomean | gate |
|---|---|---|---|---|---:|---|
| 1 `780m-refine1`: 8da4w `t128x64k32g22s32`; 4w `t128x256k32g42s32f32c` for N >= 1024 | A | +0.80 / **+3.33** % | +1.36 / **+4.08** % | **+2.50** / **+5.73** % | +2.95 % | pass (`s2-c1`) |
| 2 `780m-refine2`: 8da4w texel-wise weight staging | A | 0.00 / **+3.44** % | -0.06 / **+3.94** % | 0.00 / **+4.58** % | +1.97 % | pass (`s3-c2`), 2nd attempt |
| 3 `780m-refine3`: QK^T `NO_MASK_FILL`, attn*V subgroup 32 | B | **+4.29** / **+4.28** % | **+2.44** / **+2.74** % | +1.67 / +1.80 % | +2.86 % | pass (`s4-c3`) |
| 5 `780m-refine5`: QK^T packed staging | B | +0.28 / +0.14 % | +1.00 / +1.55 % | +0.67 / +0.70 % | +0.72 % | pass (`s5-c5`) |
| 6 `780m-refine6`: QK^T 64 x 64 tile, attn*V 128-column tile | B | 0.00 / +0.14 % | +0.71 / -0.11 % | +0.53 / +0.46 % | +0.29 % | pass (`s6-c6`), not adopted |

Bold = outside the +-2 % band. Candidate 4 was screened out before the gate (below).
In every gated session: `verify.sh` correctness rc = 0, 12 of 12 production-diff cases ALL PASSED, next token of
default vs tiled SAME on the real-text and the unaligned prompt, decode 31 tokens; next token parent vs
candidate SAME in all six cells on both prompts; SDPA correctness 4 of 4 cases with 0 mismatches in all 12
passes (candidates 3, 5, 6). `verify.sh` prints `linear <scheme> rc=1` in every run including the parent
control: its microbench report marks a texture3d dispatch of a coopmat kernel as `unexpected_coopmat`, which is
what `dev/1.5` does by design on this device (the same line is in `sarc-1.5-bootstrap/results/780m`).

### Winner against `dev/1.5`, measured directly (session `s7-final`)

Pristine `dev/1.5` build, no environment, against the topic build with `ET_VK_SARC_DEV_PROFILE=780m-refine5`;
tok/s, median of 5 valid runs per arm, arms interleaved:

| cell | `dev/1.5` | winner | gain |
|---|---:|---:|---:|
| 1B 4w | 2698.29 | 2832.64 | +4.98 % |
| 1B 8da4w | 2540.94 | 2828.73 | +11.33 % |
| 3B 4w | 1142.86 | 1201.88 | +5.16 % |
| 3B 8da4w | 1054.04 | 1172.97 | +11.28 % |
| 8B 4w | 519.27 | 545.26 | +5.01 % |
| 8B 8da4w | 488.08 | 552.32 | +13.16 % |

Geomean +8.43 %, every cell outside the +-2 % band; repeat spread at most 0.98 %; next token SAME in all six
cells on both prompts. The parent control (`s0-parent-verify`) has the same `verify.sh` status as the candidates:
correctness rc = 0, 12 of 12 production-diff cases ALL PASSED, `linear <scheme> rc=1`.

Files: `results/780m/sessions/<session>/{STAGE.md,env.txt,runs.csv,summary.csv,nexttoken.csv,verify.out,verify/,trace/,sdpa-correctness/}`.

## Where the gain comes from

Measured before any tile sweep, phase timing with a shader clock (cycles of one wave over the whole kernel,
1B shapes, K = 2048; `results/780m/phases/`):

| kernel | barrier | fetch | MMA | LDS store | prologue + epilog | drain | write |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4w `t128x128k32g42s32f32c` (parent) | 16 % | 7 % | 51 % | 22 % | 1 % | 1.3 % | 2.1 % |
| 4w `t128x256k32g42s32f32c` (candidate 1) | 14 % | 4 % | 53 % | 24 % | 1 % | 1.3 % | 2.8 % |
| 8da4w zpg `t128x64k32g42s32` (parent) | 13 % | 13 % | 38 % | 28 % | 5 % | 1.1 % | 1.9 % |
| 8da4w zpg `t128x64k32g22s32` (candidate 1) | 7 % | 9 % | 39 % | 36 % | 6 % | 1.0 % | 2.9 % |
| 8da4w zpg `bt_t128x64k32g22s32` (candidate 2) | 7 % | 10 % | 43 % | 27 % | 7 % | 1.1 % | 3.5 % |

Staging = fetch + LDS store. Drain and write together are 3 to 5 % of a wave on both parent kernels, so neither
was a target.

- 8da4w (measured): a wave spent more time in staging (40 %) than in the MMA (38 %). Candidate 1 halves the
  waves per workgroup, so each wave owns a 64 x 32 tile and one A load feeds two MMAs; per wave the cycles per
  MMA drop from about 167 to 100. Candidate 2 fetches each packed-weight texel once instead of 8 times and
  drops the per-slot component and parity selects; the LDS-store share falls from 36 % to 27 %.
- 4w (measured): the wide tile stages A once per 256 output columns; its gain grows with the dispatch size
  (kernel level +1 % on 1B, +2 % on 3B, +4 % on 8B) and it loses on N = 512, hence the N >= 1024 predicate.
  Why a 4-wave workgroup helps 8da4w but not 4w was not established.
- SDPA (measured in the 1B 4w trace): QK^T 75.1 -> 47.0 ms with `NO_MASK_FILL`, attn*V 43.8 -> 42.2 ms;
  softmax unchanged at 115 ms. `sarc_sdpa_attn_weights_softmax` reads row s only up to c = s + input_pos, so
  the -inf that QK^T writes above the causal diagonal is never read: fully masked tiles (240 of 512 at 2048)
  are left unwritten and diagonal tiles take the direct store. Packed staging (candidate 5) takes QK^T from
  127.2 to 109.7 ms on 3B and from 191.5 to 161.5 ms on 8B, but only from 46.4 to 45.3 ms on 1B.

Linear kernel rate in the model (warm ETDump, time-weighted, `tools/roof_util.py`; roofs from
`sarc-1.5-e2e-benchmark/evidence/roofline.md`: fast plan, short run, clocks not pinned):

| cell | parent (`s2-c1`) | candidate 1 (`s2-c1`) | candidate 2 (`s3-c2`) | `s7-final`: parent -> winner |
|---|---:|---:|---:|---:|
| 4w 1B / 3B / 8B, % of fp16->fp32 matrix roof (14.772 TFLOP/s) | 72.2 / 72.6 / 69.3 | 73.3 / 73.6 / 70.6 | unchanged | 71.9 / 71.9 / 68.2 -> 72.4 / 72.5 / 69.5 |
| 8da4w 1B / 3B / 8B, % of int8 matrix roof (14.393 TOP/s) | 69.1 / 69.1 / 66.8 | 73.7 / 73.5 / 72.3 | 79.0 / 78.1 / 77.2 | 69.1 / 68.4 / 66.3 -> 78.9 / 76.9 / 76.6 |

Each figure comes from one traced run per arm, so differences of about one point between sessions (for example
the 4w parent at 72.2 and 71.9) are not resolved. The 4w kernel moved by 0.5 to 1.3 points; most of the 4w
cells' end-to-end gain comes from the SDPA candidates.

## Failed and rejected candidates

- 4w `t128x128k64g42s32f32c`, `t128x256k64g42s32f32c`, `t128x128k64g22s32f32c` (batch 1): Ash + Bsh need
  71.7 KiB or more, over the 64 KiB shared-memory limit, and every run (6 in all) ended in a lost Vulkan
  context, i.e. a GPU reset by the kernel driver. The GPU recovered and later runs were normal. The variants
  were removed from the yaml and from `Overrides.cpp`; later batches assert the shared-memory total before
  writing a variant. Raw logs: artifacts `raw/screen2-4w/`.
- Candidate 2, first attempt (`s3-c2-attempt1-gatefail`): the kernel family was named
  `sarc_dev_linear_dq8ca_zpg_bt`, and the correctness gate recognises a cooperative-matrix dispatch by "coopmat"
  in the shader name, so it reported "a rank-3 case did not dispatch coopmat" (the 4 numeric checks passed).
  The gate was not changed; the family was renamed and gated again.
- Subgroup-64 tiles for both linear families (batch 2): 0.59 to 1.00x at kernel level.
- Other tile shapes (batches 1 and 3) and zpgtr on this device (0.80x): `results/780m/screens/`.
- Candidate 4, 4w texel-wise weight staging (`bx`): +0.4 % at kernel level on the wide tile, 0.92x on the
  128 x 128 tile (`screens/screen7-4w.csv`); not gated.
- Candidate 6: gate passed, +0.29 % geomean with one cell negative, all cells inside the band; not adopted.

## What is left (prefill, 1B 4w with the winner, about 720 ms)

From the candidate-arm traces (`sessions/s5-c5/trace/families.csv`): linear GEMM about 51 %, softmax 16 %,
`view_copy` 7 %, QK^T 6 %, attn*V 6 %, elementwise `mul` 6 %, `sigmoid` 2 %, `add` 1.4 %; for 8da4w the
activation quantize is 6 %.

- Softmax (115 ms on 1B, 150 on 3B, 229 on 8B) is now the largest non-GEMM kernel. Its shader name is fixed by
  `impl/sarc/SdpaCoopmat.cpp` (release zone), so no dev variant can replace it. It makes three passes over the
  causal triangle and writes zeros over the masked half, which attn*V never reads beyond the row tile's last
  chunk. This is an inference from the sources, not a measurement of a faster softmax.
- `view_copy`, `mul`, `sigmoid`, `add`, `rms_norm` and the 8da4w quantize shaders are upstream operators outside
  both zones.
- 4w linear sits at 71 to 74 % of its roof and no sweep variant moved it by more than 2.5 % at kernel level.

## Limits

- One device, one driver version, 2048-token prompts, clocks as found. Not a statement about other devices or
  prompt lengths.
- Next-token equality and the sampled production diff are the correctness evidence; full output tensors were
  not compared bit for bit. That the staging changes are bit-identical is an argument from the code (same
  values, same LDS layout, same MMA order), not a measurement.
- `NO_MASK_FILL` is only valid with the truncated SARC softmax, which every device with an SDPA row uses. It
  leaves never-read elements of the attention-weight buffer unwritten.
- The roofs are short-run `fast`-plan values with unpinned clocks; the utilisation figures inherit that.
- Prompts that are not tile-aligned still fall back to the upstream kernels, as on `dev/1.5` (1972 tokens:
  730 tok/s for 1B 4w against 2700 at 2048). Out of scope here.
- The topic branch lives only in `rocky-ryzen:~/hmz-sarc/executorch`; raw logs, ETDumps, clock samples and
  binaries are in `rocky-ryzen:~/hmz-sarc/.artifacts/780m-prefill-refine-2026-10-03/`.

## Checks on the branch

- `sarc/tools/check.sh --no-build`: PASS (zone rule against `release/1.5`, twin wrappers, `test_sarc_select` with
  and without the dev zone: 31 rows, 109 candidates).
- `sarc/tools/spirv_golden.py` on the topic build: the 53 shipped variants are unchanged.
- Not run: `check.sh` steps 4 and 5 on a release export (the release zone is untouched).

## Review follow-up (2026-10-04)

An independent review of this change (numbers recomputed from every `runs.csv`, the staging and `NO_MASK_FILL`
code read against the release bodies and the softmax) found the results reproducible and the change confined to
the dev zone, and asked for three things before any promotion. All three are done; the recommended profile
changes from `780m-refine5` to **`780m-refine3`** (candidates 1, 2 and 3).

1. **Wider SDPA correctness coverage.** `test_llama_microbench --sdpa-correctness-only` gains two opt-in tiers;
   `--sdpa-tier=all` still runs exactly the four original cases.
   - `extended` (8 cases, about 10 s a pass): `input_pos` 64, 128 and 256 with host-supplied cache history, the
     3B and 8B head configurations (head_dim 128), and S = 2048 with two heads. 1075 QK^T tiles: 488 masked,
     503 visible, 84 diagonal (the original tiers: 20 tiles, 4 masked).
   - `full` (4 cases, about 3 min a pass): the 1B, 3B and 8B head configurations at S = 2048 and the 8B
     configuration at S = 1024 with `input_pos` = 1024.
   - Results, build `topic-r1` (`52d015ca4` + this test change), profile `780m-refine3`
     (`results/780m/sessions/r1-sdpa-ext/`): `extended` 12 passes, 8 of 8 PASSED with 0 mismatches each;
     `full` 12 passes, 4 of 4 PASSED with 0 mismatches each (up to 8,388,608 elements compared per case).
     Control with the table kernels (no profile): `extended` 8 of 8 and `full` 4 of 4, 0 mismatches, 1 pass each.
2. **Kernel pairing check.** Each SDPA correctness case now prints the dispatched QK^T, softmax and attn*V
   kernel names (`[sdpa-kernels]`) and FAILS, whatever the numbers, if a `NO_MASK_FILL` QK^T kernel (tile token
   ending in `nf`) is dispatched with any softmax other than `sarc_sdpa_attn_weights_softmax*`. In every pass
   above the candidate ran `sarc_sdpa_qk_coopmat_sweep_t128x64k32g22s64nf` with the SARC softmax. This is a
   test-level guard, not a runtime assertion; no mispaired configuration exists today to show it firing.
3. **Candidate 5 dropped; full gate on the final build.** Candidate 5 (+0.72 %) was inside the +-2 % band, so
   it is not counted. Session `s8-r3final`: pristine `dev/1.5` build against build `topic-r1` with
   `ET_VK_SARC_DEV_PROFILE=780m-refine3`, the same gate as every candidate (`tools/gate_sdpa.sh`).

| cell | `dev/1.5` | `780m-refine3` | gain |
|---|---:|---:|---:|
| 1B 4w | 2694.74 | 2828.73 | +4.97 % |
| 1B 8da4w | 2540.94 | 2824.83 | +11.17 % |
| 3B 4w | 1140.31 | 1187.94 | +4.18 % |
| 3B 8da4w | 1058.40 | 1170.95 | +10.63 % |
| 8B 4w | 514.96 | 536.83 | +4.25 % |
| 8B 8da4w | 486.81 | 548.33 | +12.64 % |

Geomean **+7.91 %**, every cell outside the +-2 % band, repeat spread at most 0.50 %; next token SAME in all six
cells on both prompts. `verify.sh`: correctness rc = 0, 12 of 12 production-diff cases ALL PASSED, default vs
tiled SAME on both prompts, decode 31 tokens, `linear <scheme> rc=1` as on the parent. SDPA correctness
(tier `all`): 12 passes, 4 of 4 with 0 mismatches.

Conditions of `s8-r3final`, which differ from `s7-final`: it started right after the SDPA passes (start
temperature 48 to 53 C, peak 95 C); one timed run was rejected (`8b 4w parent r3`, `clock_low`) and replaced
by the next valid one; the 8B 4w cell ran at a median clock of 2698 to 2735 MHz, the other cells at 2760 to
2800 MHz. The 8B 4w gain (+4.25 % here, +5.01 % in `s7-final` with candidate 5) should be re-measured from a
cool start before it is quoted.

Other review notes, not acted on here:
- The profile matches kernels by name suffix; `t128x64k32g22s32` matches both the sweep and the `bt` 8da4w
  kernels and the first row in the candidate list wins. The dispatched names in every `verify.out` are the
  intended ones.
- The texel-wise staging (`bt`) was checked by reading the index arithmetic against the release body: same
  texel, same widened values, same LDS word for each of the four components. Full output tensors were still not
  compared bit for bit.
- Of the 84 unaligned-prompt check runs in sessions `s1` to `s7`, 22 carry `clock_low`; they are next-token
  checks only and are not timed.

## Round 2 (2026-10-04 to 2026-10-05): the parameter space, and beyond it

Parent: profile `780m-refine3` (+7.91 % over `dev/1.5`, above). Same host, driver and protocol; new raw data in
`rocky-ryzen:~/hmz-sarc/.artifacts/780m-prefill-refine-2026-10-04/`. `STATUS.md` is the running log with every
table; this section is the summary. Dev zone, plus the two release-zone entry points of the owner decision of 2026-10-05 (below).

### Candidates

Each against its parent in the same session, median of 5 valid runs per arm, arms interleaved, cool start:

| candidate | what | 1B 4w / 8da4w | 3B 4w / 8da4w | 8B 4w / 8da4w | geomean | gate |
|---|---|---|---|---|---:|---|
| 7 | softmax r3: the row prefix loaded once, exp once, zero fill bounded to what attn*V reads (`ET_VK_SARC_780M_SOFTMAX=r3`; `hooks/softmax-name-hook.patch`) | **+6.78** / **+6.62** % | **+3.43** / **+3.06** % | **+2.44** / **+2.39** % | +4.10 % | pass; SDPA output byte-identical to the parent on 21 cases |
| 8 | fused SDPA kernel `sarc_dev_780m_sdpa_fused3` (`ET_VK_SARC_780M_SDPA_FUSED`; `hooks/sdpa-fused-hook.patch`) | **+20.00** / **+20.14** % | **+11.69** / **+11.88** % | **+8.10** / **+7.84** % | +13.16 % | ACCEPTED (reference-error rule, owner decision 2026-10-04); one next-token item differs (8B 8da4w, `prompt_2048.txt`) |
| 9 | candidate 8 + one-pass form of the fused kernel + the 4w kernel per shape (`ET_VK_SARC_780M_PROFILE=refine9`) | **+4.83** / +1.80 % | **+4.50** / +1.06 % | **+5.45** / +1.59 % | +3.19 % | pass; no next-token item differs; 4w output byte-identical; the 8da4w cells are inside the band |

Bold = outside the +-2 % band. A/A floor of this round: geomean +0.01 %, cells within +-0.23 % (`t3-aa`).
Chained over the sessions, candidate 9 is about +31 % geomean over `dev/1.5` and +21.6 % over `780m-refine3`;
the direct measurement is in "Final configuration" below once it is run.

### Where the gains come from

- **Softmax (candidate 7).** Traces: 115.1 -> 70.5 ms (1B), 150.2 -> 93.3 ms (3B), 229.1 -> 142.3 ms (8B). The
  kernel is bound by memory traffic: removing exp entirely changes nothing (`m1`), bounding the zero fill does.
- **Fused SDPA (candidate 8).** QK^T, softmax and attn*V each move the S x S attention matrix through memory
  (about 550 MB a layer on 1B). The fused kernel never writes it: per 16 or 32 query rows of a head it walks the
  context in blocks, computes the scores, exponentiates them in shared memory private to one subgroup and
  accumulates e V in fp32. Traces: QK^T + softmax + attn*V 158.9 / 296.2 / 449.4 ms are replaced by the copy pass
  + fused kernel at 45.3 / 120.7 / 184.2 ms. Kernel time per layer at a steady clock: 9.96 -> 2.82 ms (1B),
  10.24 -> 4.32 ms (3B), 13.51 -> 5.54 ms (8B). What made it fast, in the order found (roofline fed-MMA table and
  the disassembly, `STATUS.md`): no shared staging and no barrier (one subgroup per workgroup); Q tiles kept in
  registers; K and V read from tile-packed copies in which a 16 x 16 operand tile is 512 contiguous bytes
  (head_dim 128: 9.95 -> 4.96 ms from that alone). It runs at about 71 % of the fp16 -> fp32 matrix roof.
- **One-pass fused kernel (candidate 9).** A running row maximum with rescaled accumulators instead of a first
  pass over the scores: 2.82 -> 2.26 ms (1B), 4.31 -> 3.22 ms (3B), 5.54 -> 4.05 ms (8B) per layer; -10 / -31 /
  -48 ms in the traces. Alone it is +1.1 to +1.8 % end to end, inside the band.
- **4w kernel per shape (candidate 9).** Traces: linear GEMM 367.5 -> 350.9 ms (1B, -4.5 %), 1049 -> 1018 ms (3B,
  -2.9 %), 2659 -> 2551 ms (8B, -4.1 %). The 256-row tile halves how often the weights are staged and wins once K
  is large, but only with B staged column-major (`B_COLMAJOR`: -18 % on that tile, neutral elsewhere).

### Part 1: the existing parameter space

Static count (`tools/enum_space.py`): 75,497,472 4w combinations, 125,712 survive the device, flag, geometry and
shared-memory rules; 8da4w 165,888 -> 2,238; QK^T 9,216 -> 1,724; attn*V 4,608 -> 470. Thirty timed samples
projected 26.8 days for the 4w survivors, so by owner decision 4w was searched by a seeded uniform sample of
2,500 (2,444 timed), a refinement around the best 22 (991), and a confirmation with the full measurement.

**4w: within the existing kernel bodies, the best configuration per shape on this device and driver is**
(`results/780m/space/confirm-4w/summary.csv`; 5 repeats, spread 0.1 to 0.4 %):

| shapes | kernel | against the `780m-refine3` choice |
|---|---|---|
| N = 512 (1B wk_wv) | the shipped `t128x128k32g42s32f32c` (ten configurations within 0.5 %) | 0 |
| N = 1024 (3B, 8B wk_wv); K = 2048 with N = 2048 (1B wq_wo) | `t128x256k32g42s32f32cbt` | -1.7 to -2.9 % |
| K >= 4096 with N >= 2048 (8B wq_wo, w1_w3, w2; 1B and 3B w2) | `t256x128k32g18s32f32bbt` | -3.4 to -7.0 % |
| N = 8192 with K = 2048 (1B w1_w3) | `t256x128k32g28s32f32bbt` | -4.4 % |
| K = 3072 with N >= 2048 (3B wq_wo, w1_w3) | `t256x128k32g24s32f32bbt`; `..cbt` and `..g18..` are within 2 % | -3.2 to -3.7 % (`..cbt`: -1.8 to -2.3 %) |

Per layer that is 4.0 to 4.8 % less linear time than `780m-refine3`; profile `refine9` (which keeps `..cbt` on
the tied K = 3072 shapes) gets 3.1 to 4.8 %. Response surface: `STATUS.md`, "The random sample" (importance over
the whole space: tile M and N, the grid and the accumulator explain it, through the number of MMA tiles a
subgroup owns; no pair of boolean options interacts) and "Part 1, 4w" (near the optimum: `IMG_A`, `IMG_W` and the
drain mode do not matter; `SH_F16V4` always costs 3 %; `B_COLMAJOR`, tile M / N and the grid are decided by the
shape). 8da4w, QK^T and attn*V: see `STATUS.md` until their enumerations are in.

### Release-zone hook (owner decision 2026-10-05)

Until 2026-10-05 candidates 7 to 10 were measured through two local patches in a scratch tree. The owner decision
of 2026-10-05 ("allow all") permits committing a small, inert entry point for each; they are two commits on this
branch, the local patches and `tools/build_hook.sh` are removed, and `hooks/README.md` describes the dev-zone side.
Nothing else in the release zone changes. `sarc/tools/check.sh` and the reproduction from the committed branch:
"Committed build" below.

**1. Softmax variant name (`b969e8f1c`)**: the Orin campaign's `softmax-variant-hook.patch`, applied unchanged with
`git am` (same diff; one paragraph naming this campaign added to the commit message).

```diff
diff --git a/backends/vulkan/runtime/graph/ops/impl/sarc/SdpaCoopmat.cpp b/backends/vulkan/runtime/graph/ops/impl/sarc/SdpaCoopmat.cpp
index d45f720bc..5a264e27e 100644
--- a/backends/vulkan/runtime/graph/ops/impl/sarc/SdpaCoopmat.cpp
+++ b/backends/vulkan/runtime/graph/ops/impl/sarc/SdpaCoopmat.cpp
@@ -202,7 +202,11 @@ std::string sdpa_softmax_shader_name(
   if (!device_has_active_rows(device_info(&graph), Op::kSdpaQk)) {
     return upstream_name;
   }
-  return "sarc_" + upstream_name;
+  const char* variant = get_override().softmax_variant;
+  if (variant == nullptr) {
+    return "sarc_" + upstream_name;
+  }
+  return "sarc_" + upstream_name + "_" + variant;
 }
 
 } // namespace sarc
diff --git a/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h b/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h
index 93a9b8e96..166eba3e0 100644
--- a/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h
+++ b/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h
@@ -141,6 +141,9 @@ struct Override {
       const DeviceInfo& device,
       const ShapeInfo& shape,
       const std::optional<Choice>& table_choice) = nullptr;
+  // Suffix of a development softmax variant ("sarc_<softmax>_<suffix>"), or
+  // null: the release softmax.
+  const char* softmax_variant = nullptr;
 };
 void set_override(const Override& o);
 const Override& get_override();
```

**2. Fused attention node (`1c8861aa7`)**: 49 added lines. Larger than a switch: **subject to the owner's review
before any promotion.** The node, its kernels and the decision which calls it serves are in the dev zone
(`impl/sarc_dev/780m/Sdpa780mFused.cpp`).

```diff
diff --git a/backends/vulkan/runtime/graph/ops/impl/SDPA.cpp b/backends/vulkan/runtime/graph/ops/impl/SDPA.cpp
index c623a7235..c76a9c52f 100644
--- a/backends/vulkan/runtime/graph/ops/impl/SDPA.cpp
+++ b/backends/vulkan/runtime/graph/ops/impl/SDPA.cpp
@@ -317,6 +317,9 @@ GlobalWorkGrid pick_sdpa_softmax_gwg(
     const std::vector<ArgGroup>& args,
     const std::vector<ValueRef>& resize_args) {
   (void)shader;
+  if (auto skip = sarc::sdpa_fused_skip(graph, resize_args)) { // SARC
+    return *skip;
+  }
   const SDPAMode mode = mode_of(resize_args);
   const ValueRef q = resize_args.at(0);
   // LLM reads H from axis -2, fused from axis -3 (handled by
@@ -815,6 +818,10 @@ void sdpa_impl(ComputeGraph& graph, const std::vector<ValueRef>& args) {
       input_pos_symint,
       out,
       SDPAMode::LLM);
+
+  // SARC development hook: fused attention node (no-op without an override).
+  sarc::add_sdpa_fused(
+      graph, {q_projected, k_cache, v_cache, input_pos_symint, out});
 }
 
 void sdpa_with_kv_cache_impl(
diff --git a/backends/vulkan/runtime/graph/ops/impl/sarc/SdpaCoopmat.cpp b/backends/vulkan/runtime/graph/ops/impl/sarc/SdpaCoopmat.cpp
index 5a264e27e..c6f9e9d3d 100644
--- a/backends/vulkan/runtime/graph/ops/impl/sarc/SdpaCoopmat.cpp
+++ b/backends/vulkan/runtime/graph/ops/impl/sarc/SdpaCoopmat.cpp
@@ -122,6 +122,9 @@ std::optional<GlobalWorkGrid> sdpa_gwg(
     ComputeGraph* graph,
     const vkapi::ShaderInfo& shader,
     const std::vector<ValueRef>& resize_args) {
+  if (auto skip = sdpa_fused_skip(graph, resize_args)) {
+    return skip;
+  }
   const std::optional<TileDims> dims = dims_for_kernel(shader.kernel_name);
   if (!dims.has_value()) {
     return std::nullopt;
@@ -143,6 +146,24 @@ std::optional<GlobalWorkGrid> sdpa_gwg(
       LocalWorkGroup(wg, 1u, 1u));
 }
 
+std::optional<GlobalWorkGrid> sdpa_fused_skip(
+    ComputeGraph* graph,
+    const std::vector<ValueRef>& resize_args) {
+  const Override& o = get_override();
+  if (o.sdpa_fused_serves == nullptr ||
+      !o.sdpa_fused_serves(graph, resize_args)) {
+    return std::nullopt;
+  }
+  return GlobalWorkGrid(
+      {0u, 0u, 0u}, kTiledWorkGrid, LocalWorkGroup(64u, 1u, 1u));
+}
+
+void add_sdpa_fused(ComputeGraph& graph, const std::vector<ValueRef>& refs) {
+  if (get_override().sdpa_fused_add != nullptr) {
+    get_override().sdpa_fused_add(graph, refs);
+  }
+}
+
 vkapi::SpecVarList sdpa_qk_spec_vars(
     ComputeGraph& graph,
     const ValueRef q,
diff --git a/backends/vulkan/runtime/graph/ops/impl/sarc/SdpaCoopmat.h b/backends/vulkan/runtime/graph/ops/impl/sarc/SdpaCoopmat.h
index ab9d546fc..bc8a18fe0 100644
--- a/backends/vulkan/runtime/graph/ops/impl/sarc/SdpaCoopmat.h
+++ b/backends/vulkan/runtime/graph/ops/impl/sarc/SdpaCoopmat.h
@@ -38,6 +38,14 @@ std::optional<GlobalWorkGrid> sdpa_gwg(
     const vkapi::ShaderInfo& shader,
     const std::vector<ValueRef>& resize_args);
 
+// Development hook (Override::sdpa_fused_*, null in a release). The empty
+// launch geometry for a QK^T / softmax / attn*V node whose call the fused
+// attention node serves, else std::nullopt; and the fused node itself.
+std::optional<GlobalWorkGrid> sdpa_fused_skip(
+    ComputeGraph* graph,
+    const std::vector<ValueRef>& resize_args);
+void add_sdpa_fused(ComputeGraph& graph, const std::vector<ValueRef>& refs);
+
 // Spec constants for the QK / AV nodes. `upstream` is release 1.5's list; it
 // is returned unchanged unless the device has an SDPA row, in which case the
 // SARC kernels' constants are appended after it.
diff --git a/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h b/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h
index 166eba3e0..f1278114e 100644
--- a/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h
+++ b/backends/vulkan/runtime/graph/ops/impl/sarc/Select.h
@@ -24,6 +24,9 @@
 #include <vector>
 
 namespace vkcompute {
+
+class ComputeGraph;
+
 namespace sarc {
 
 enum class Op {
@@ -144,6 +147,16 @@ struct Override {
   // Suffix of a development softmax variant ("sarc_<softmax>_<suffix>"), or
   // null: the release softmax.
   const char* softmax_variant = nullptr;
+  // Fused attention node (development; subject to the owner's review before
+  // any promotion). `sdpa_fused_add` may append nodes after the three SDPA
+  // nodes of an LLM-mode op, given the ValueRefs {q, k, v, input_pos, out}.
+  // While `sdpa_fused_serves` is true for a node's resize args, those nodes
+  // write the output and the three SDPA kernels dispatch nothing.
+  void (*sdpa_fused_add)(ComputeGraph& graph, const std::vector<int32_t>& refs) =
+      nullptr;
+  bool (*sdpa_fused_serves)(
+      ComputeGraph* graph,
+      const std::vector<int32_t>& resize_args) = nullptr;
 };
 void set_override(const Override& o);
 const Override& get_override();
```

With no `ET_VK_SARC_780M_PROFILE` and no `ET_VK_SARC_780M_SDPA_FUSED` both entry points are null. The candidates
are now selected by one variable on top of `ET_VK_SARC_DEV_PROFILE=780m-refine3`:
`ET_VK_SARC_780M_PROFILE=c7 | c8 | c9 | c10` (`ET_VK_SARC_780M_SOFTMAX` no longer exists).

### Committed build: candidate 10 reproduced from the branch alone (owner decision 2026-10-05, release-zone hooks)

Build `head1` = branch head `8518659ef`, clean tree, no local patch (`<artifacts 10-05>/build/head1`). Evidence in
`results/780m/sessions/{h0-control,h1-c10}/` and `results/780m/hooks/`.

**Nothing selected: every dispatch as before.**

- `test_sarc_select`: PASS, 1240 checks / 31 rows (release tables) and 1432 checks / 121 candidates (dev zone),
  the counts of the parent. `spirv_golden.py` on `head1`: PASS (53 shipped variants).
- All 1,325 shaders of the pristine `dev/1.5` build (`build/parent`) are byte-identical in `head1`.
- `h0-control`: `head1` with no environment, unmodified `verify.sh`, against the pristine `dev/1.5` control
  (`s0-parent-verify`): the 34 lines are identical once the measured tok/s figures are set aside (kernel names,
  return codes, 12 production-diff cases ALL PASSED, default vs tiled SAME on both prompts, decode 31 tokens).
  `--sdpa-correctness-only` three times: 4 of 4, release QK^T / softmax / attn*V names, `fused=-`.

**Candidate 10 dispatches what was measured.** Selected by `ET_VK_SARC_780M_PROFILE=c10` on top of
`ET_VK_SARC_DEV_PROFILE=780m-refine3`. All 1,468 SPIR-V binaries of the measured build (`fused9`) are
byte-identical in `head1` (the six softmax variants under their new names). Kernel names in `verify.sh`
(`linear 4w` / `linear 8da4w` lines) and in the SDPA tiers (`fused3_d64_t32x32g11s32rko`,
`fused3_d128_t16x64g11s32rko`) are those of the candidate-10 gate; `verify.out` of `h1-c10` equals that gate's
line for line apart from the tok/s figures.

**Gate, once, on the committed build (`h1-c10`, 12:03 to 13:22 PDT):** SDPA tiers `all`, `extended`, `full`, 12
passes each: 48 / 96 / 48 cases passed, 0 failed, 0 mismatches, pairing ok in every line; `verify.sh` rc = 0 as
above. Next token against the parent: SAME in eleven of twelve items; 8B 8da4w on `prompt_2048.txt` DIFFERS,
the item already recorded for candidate 8 (**ACCEPTED (reference-error rule, owner decision 2026-10-04)**,
evidence `results/780m/probe/`, `results/780m/sdpa-error/`; the kernels are byte-identical, so that evidence
stands).

**Timed session against the parent (`780m-refine3`), same binary in both arms, started at 44 C, 5 valid runs per
arm.** This is also the first direct measurement of candidate 10 against the round's parent; until now the
figure was chained over four sessions (1.0410 x 1.1316 x 1.0319 x 1.0047 = +22.13 %).

| cell | `780m-refine3` | candidate 10, committed build | gain | candidate 10 through the patches (`c10-q4-refine10`) | spread parent / candidate |
|---|---:|---:|---:|---:|---|
| 1B 4w | 2828.73 | 3835.21 | **+35.58 %** | 3842.40 | 0.28 / 0.19 % |
| 1B 8da4w | 2824.83 | 3690.09 | **+30.63 %** | 3690.09 | 0.28 / 0.18 % |
| 3B 4w | 1190.01 | 1459.73 | **+22.67 %** | 1459.73 | 0.06 / 0.07 % |
| 3B 8da4w | 1166.29 | 1379.12 | **+18.25 %** | 1360.80 | 0.57 / 1.46 % |
| 8B 4w | 542.09 | 641.40 | **+18.32 %** | 642.21 | 0.16 / 0.28 % |
| 8B 8da4w | 549.50 | 615.20 | **+11.96 %** | 615.20 | 0.43 / 0.42 % |

Geomean **+22.64 %** over `780m-refine3`. Committed build against patch build: -0.19 to +1.35 %, geomean +0.17 %
(different sessions; the A/A floor is +-0.2 %, and 3B 8da4w is the cell with the 1.4 % repeat spread in both).
Traces (warm, one run per arm): 718.2 -> 536.2 ms (1B 4w), 719.2 -> 552.0, 1719.6 -> 1417.5, 1742.7 -> 1484.7,
3809.2 -> 3250.7, 3741.0 -> 3364.6 ms (8B 8da4w); in 1B 4w the three attention kernels (46.5 + 115.3 + 42.1 ms)
are gone, the copy family grows by 35.6 ms (copy pass + fused kernel are counted there), linear GEMM 366.6 ->
352.9 ms.

`sarc/tools/check.sh --no-build`: PASS. Its zone rule compares against `release/1.5`, where the release zone is
allowed and `SDPA.cpp` is already listed in `sarc/HOOKS`, so it does not flag the two entry points; they are the
two commits named above, permitted by the owner decision of 2026-10-05 and by nothing else.

### Roofs and what limits further progress

Roofs re-measured with igpu-roofline (`fast` plan, run `2026-10-04-fast-prefill-refine2`, Mesa 25.2.7, clocks
not pinned): matrix fp16 -> fp32 14.766 TFLOP/s, matrix int8 14.379 TOP/s.

- The matrix unit is not what limits these kernels; feeding it is. The roofline's fed-MMA rows give 2.4 / 4.8 /
  9.7 / 14.7 TFLOP/s when a tile pair is loaded from shared memory for every 1 / 2 / 4 / 8 multiply-adds. The 4w
  kernel loads one tile per two multiply-adds at best (a subgroup owns 4 x 4 accumulator tiles, 128 of its
  registers); larger per-subgroup tiles do not fit the register file with fp32 accumulators.
- A slab layout of the 4w operands (both tiles contiguous, no padding) is 8 to 25 % slower; subgroup size 64,
  tile K 16 or 64 and fp16 accumulation are 14 % or more behind (refinement round 1).
- After candidate 9 the upstream operators are the largest part that is not GEMM: elementwise `mul` / `sigmoid` /
  `add` 68 / 130 / 244 ms, `view_copy` and other copies 61 / 135 / 229 ms (4w) or 36 / 86 / 137 ms (8da4w), the
  8da4w activation quantize 43 / 101 / 167 ms, RMSNorm + RoPE 14 / 40 / 60 ms (1B / 3B / 8B); together 27 %, 21 %
  and 17 % of the 4w prefill (533 / 1420 / 3228 ms). They are outside both zones and were not changed.
- The fused kernel's copy pass costs 0.35 to 0.7 ms a layer, three times its bytes at the copy roof; packing K
  and V where the cache is written would remove it (a change outside the dev zone).

### Failed and rejected in this round

- Softmax `r2` (in-wave tree reductions instead of barriers): same time as `r1`, dropped.
- Fused SDPA, first two structures (`fused`: K and V staged in shared memory with 4 to 6 barriers a block;
  `fused2`: double-buffered staging, one barrier a block): 3.8 / 11.7 ms and no better, against 2.8 / 5.6 ms for
  the third. Removing the row padding from `fused2`'s shared tiles: 2.6 times slower. `fused3` reloading Q from
  the buffer for every block: 2.2 to 3 times slower. All kept as variants, none selected.
- 4w slab layout: above. A 16 x 4 workgroup for the copy pass: 4 % slower than 8 x 8.
- A first start of the candidate-9 gate measured nothing (the sweep had been stopped while holding the gpu-lab
  lock); recorded in `<artifacts>/superseded/`.
- Measurement pitfalls found: the microbench's SDPA suite (3 + 5 runs) and the linear screening mode (1 + 2 runs)
  are taken on a rising GPU clock (`STATUS.md`, "What the clock does to the microbench"); a process whose command
  line contains a runner's name is counted as another GPU process by `e2e5.sh`.
