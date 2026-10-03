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
