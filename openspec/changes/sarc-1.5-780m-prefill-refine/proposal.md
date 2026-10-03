# sarc-1.5-780m-prefill-refine

Status: in progress (2026-10-03). Dev zone only; nothing here is promoted, and no release-zone or upstream file
is touched. Branch `topic/780m-prefill-refine`, forked from `dev/1.5` at `ef079ac41`.

## Why

`dev/1.5` already runs the Llama prefill on the Radeon 780M with SARC WMMA kernels (4w, 8da4w) and the SARC
SDPA kernels. This change asks what is still left on that device, measured against `dev/1.5` itself (not
against stock ExecuTorch), with sweep variants only.

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
- `sarc_dev_linear_q4gsw_coopmat_bx`: 4w with texel-wise weight staging (screened out, kept as a candidate).
- Measurement only: `sarc_dev_prof_*` shader-clock twins and `ET_VK_DUMP_OUTPUT_DIR` in the dev test utils.

The generated shader files come from `tools/gen_*.py`, which read the release bodies and never write them.

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
  the measured window. Median of the first 5 valid runs per arm. No run was rejected in any session so far.
- Gate per candidate (`tools/gate.sh`, `tools/gate_sdpa.sh`): unmodified `sarc/tools/verify.sh --models
  1b,3b,8b --schemes 4w,8da4w --pdiff` with the candidate's environment, then the e2e session with the
  next-token comparison parent vs candidate on `prompt_2048.txt` and on the unaligned `prompt_check.txt` for
  all six cells, then warm ETDump traces of both arms. SDPA candidates add 12 passes of
  `test_llama_microbench --sdpa-correctness-only`.
- One GPU job at a time, all under the gpu-lab lock of the 780M; no build runs during a measurement.
- A gain inside +-2 % is reported as inside the noise band even though the A/A session (pristine parent build
  against the topic build without overrides) agreed within 0.05 % with repeat spreads of at most 0.4 %.

## Results so far

Parent `dev/1.5` (session `s1-aa`, tok/s, median of 5): 1B 2698.3 (4w) / 2544.1 (8da4w), 3B 1146.7 / 1056.2,
8B 521.8 / 489.4.

| candidate (profile) | kind | vs its parent: 1B 4w / 8da4w, 3B 4w / 8da4w, 8B 4w / 8da4w | geomean | gate |
|---|---|---|---:|---|
| 1 `780m-refine1`: 8da4w `t128x64k32g22s32`; 4w `t128x256k32g42s32f32c` for N >= 1024 | A | +0.80 / +3.33, +1.36 / +4.08, +2.50 / +5.73 % | +2.95 % | pass (`s2-c1`) |
| 2 `780m-refine2`: 8da4w texel-wise weight staging `bt_t128x64k32g22s32` | A | +0.00 / +3.44, -0.06 / +3.94, +0.00 / +4.58 % | +1.97 % | pass (`s3-c2`); first attempt failed, see below |
| 3 `780m-refine3`: QK^T `NO_MASK_FILL`, attn*V subgroup 32 | B | +4.29 / +4.28, +2.44 / +2.74, +1.67 / +1.80 % | +2.86 % | pass (`s4-c3`) |

Files: `results/780m/sessions/<session>/{STAGE.md,env.txt,runs.csv,summary.csv,nexttoken.csv,verify.out,trace/}`.

## Where the gain comes from

Phase timing (shader clock, cycles of one wave over the whole kernel, 1B shapes; `results/780m/phases/`):

| kernel | barrier | fetch | MMA | LDS store | prologue + epilog | drain | write |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4w `t128x128k32g42s32f32c` (parent) | 16 % | 7 % | 51 % | 22 % | 1 % | 1.3 % | 2.1 % |
| 8da4w zpg `t128x64k32g42s32` (parent) | 13 % | 13 % | 38 % | 28 % | 5 % | 1.1 % | 1.9 % |
| 8da4w zpg `t128x64k32g22s32` (candidate 1) | 7 % | 9 % | 39 % | 36 % | 6 % | 1.0 % | 2.9 % |
| 8da4w zpg `bt_t128x64k32g22s32` (candidate 2) | 7 % | 10 % | 43 % | 27 % | 7 % | 1.1 % | 3.5 % |

Drain and write are small on both parent kernels, so neither was a target. The 8da4w kernel spent more of a
wave in staging than in the MMA: candidate 1 halves the waves per workgroup (each wave does a 64 x 32 tile, so
one A load feeds two MMAs), candidate 2 fetches each packed-weight texel once instead of 8 times and drops the
per-slot component and parity selects.

Linear kernel rate in the model (warm ETDump, time-weighted, `tools/roof_util.py`; roofs from
`sarc-1.5-e2e-benchmark/evidence/roofline.md`, fast plan, clocks not pinned):

| cell | parent | candidate 1 | candidate 2 |
|---|---:|---:|---:|
| 4w 1B / 3B / 8B, % of fp16->fp32 matrix roof (14.772 TFLOP/s) | 72.2 / 72.6 / 69.3 | 73.3 / 73.6 / 70.6 | unchanged |
| 8da4w 1B / 3B / 8B, % of int8 matrix roof (14.393 TOP/s) | 69.1 / 69.1 / 66.8 | 73.7 / 73.5 / 72.3 | 79.0 / 78.1 / 77.2 |

Candidate 3: `sarc_sdpa_attn_weights_softmax` reads row s of the attention weights only up to
c = s + input_pos, so the -inf that QK^T writes above the causal diagonal is never read. `NO_MASK_FILL` leaves
fully masked tiles unwritten (240 of 512 tiles at 2048) and gives diagonal tiles the direct store. In the 1B 4w
trace QK^T goes from 75.1 to 47.0 ms and attn*V from 43.8 to 42.2 ms; softmax is unchanged (115 ms).

## Failed and rejected candidates

- 4w `t128x128k64g42s32f32c`, `t128x256k64g42s32f32c`, `t128x128k64g22s32f32c` (batch 1): Ash + Bsh need
  71.7 KiB or more, over the 64 KiB shared-memory limit, and every run ended in a lost Vulkan context (GPU
  reset by the kernel driver; the GPU recovered and later runs were normal). Removed from the yaml and from
  `Overrides.cpp`; later batches assert the shared-memory total before writing a variant
  (`results/780m/screens/`, raw logs in the artifacts `raw/screen2-4w/`).
- Candidate 2, first attempt (`s3-c2-attempt1-gatefail`): the kernel family was named
  `sarc_dev_linear_dq8ca_zpg_bt`, and the correctness gate recognises a cooperative-matrix dispatch by
  "coopmat" in the shader name, so it reported "a rank-3 case did not dispatch coopmat" (the 4 numeric checks
  passed). The gate was not changed; the family was renamed and gated again.
- Subgroup-64 tiles for both linear families (batch 2): 0.59 to 1.00x at kernel level.
- Other tile shapes (batches 1 and 3), zpgtr on this device (0.80x): `results/780m/screens/`.
- Candidate 4, 4w texel-wise weight staging (`bx`): +0.4 % at kernel level on the wide tile, 0.92x on the
  128 x 128 tile; not gated.

## Limits

- One device, one driver version, 2048-token prompts, clocks as found (2.77 to 2.80 GHz in every measured
  window). Not a statement about other devices or prompt lengths.
- Next-token equality and the sampled production diff are the correctness evidence; full output tensors were
  not compared bit for bit. The statement that the staging changes are bit-identical is an argument from the
  code (same values, same LDS layout, same MMA order), not a measurement.
- `NO_MASK_FILL` is only valid with the truncated SARC softmax, which every device with an SDPA row uses.
- Prompts that are not tile-aligned still fall back to the upstream kernels, as on `dev/1.5` (1972 tokens:
  730 tok/s for 1B 4w against 2700 at 2048). Out of scope here.
