# Adreno 840 (Galaxy S26 Ultra) — sarc-1.5-e2e-benchmark contribution (2026-09-28)

Labels: [M] measured in this campaign, [R] roofline, [S] source, [I] inference, [O] open.

## Device, driver, clocks, thermal
- Samsung SM-S948U1 (SoC SM8850, Adreno 840), Android 16
  (`samsung/m3quew/m3q:16/BP4A.251205.006/S948U1UEU1AZAB_OYM1AZAB`), kernel 6.12.30-android16; adb serial
  `<s26-serial>` on USB of host-ws1; Vulkan driver version 2150932499 (roofline capabilities) [M].
- No root: GPU clocks float and are not readable (`clocks` = n/a) [M].
- Pacing (phone rule, host-driven over adb): before each run wait for SKIN < 38.5 C (first throttle level 38 C)
  AND GPU (max of `gpuss-*` thermal zones) <= 45 C, at most 300 s. Idle GPU 33 C; temp_pre median 41 C (33-44);
  mean cool wait 28 s; 1 of 89 waits hit the cap [M]. SKIN-only pacing was tried first and let runs start with a
  48-51 C GPU: those 4 runs are in `superseded/pacing-skin-only/` [M].
- One measuring process at a time; no other GPU users found on the device (`others` empty) [M]. Phone rebooted
  (allowed per device notes) before the 8B attempts and again at the end [M].
- gpu-lab/AGENTS.md was not available on this machine; device rules from instruction-for-ai/other-devices.md.

## Builds
- stock: `release/1.5` @ 985c1ceccc8bb8b6a32294f71aeb5d2f299562f7 + `kit/patches/stock-backport-03f41d2031.patch`.
- sarc: `origin/dev/1.5` @ d98227f60 (the Adreno row is already there). Row: "adreno" 4w
  `sarc_linear_q4gsw_coopmat_t64x64k32g21s64m64x32x16` (fp16 MMA 64x32x16, fp16 accumulate), kUnverified; no Adreno
  8da4w row (the 1.4 int8 kernel was wrong and then DEVICE_LOST; not ported) and no Adreno SDPA row [S].
  `sarc/env`: `ET_VK_SARC_UNVERIFIED=1`; stock/env empty [M].
- Toolchain: native `tools/sarc-build-native.sh` android (NDK r29, per-tree uv venv, glslc Vulkan SDK 1.4.350.1),
  not the pinned container; SPIR-V golden not checked [M]. `llama_main` and `logits_probe` built with the same
  route (`--llama`, `--etdump`, `tools/build-probe.sh`).
- sha256: stock/llama_main 7637c605c2b10692b6c30931d0f13a285fdd4d4352dc49f62650763e20f1ec2e,
  sarc/llama_main 3a9d85dea22233b92a55dbfe440c320357d94eaf6fa76f23d8cbe075cb27df03; traced stock e079ddc13fdf…,
  sarc e0d3fd40d8cf…; probe stock c1bfbb65f0e6…, sarc 61d601676e15… (full hashes: campaign env/logs) [M].
- Models: `*_embq_ctx3072.pte` (sha256 == /sarc-c/gpusw/OCL/issues/Executorch/pte/MANIFEST.json), flat-named:
  1b 4w 1ac83440b93b2cde…, 1b 8da4w fb99c89e141f420b…, 3b 4w 92117564851859bd…, 3b 8da4w e9eba0cf5a0f6ca7…,
  8b 4w 695dd232a500e9b7…, 8b 8da4w 6f172bc5590cdf68… [M]. Same exports as the five-GPU campaign: [O].

## Results — 1B and 3B only (medians of 5, paired 95 % CI)
| prompt | model | 4w stock -> sarc tok/s | 4w speedup | 8da4w stock -> sarc tok/s | 8da4w speedup |
|---|---|---|---|---|---|
| "the"x2048 | 1B | 556 -> 551 | 0.99x [0.95, 1.13] | 770 -> 775 | 1.01x [1.00, 1.03] |
| "the"x2048 | 3B | 191 -> 297 | 1.55x [1.24, 1.65] | 364 -> 364 | 1.00x [0.98, 1.03] |
| real 2048 | 1B | 647 -> 633 | 0.98x [0.88, 1.12] | 869 -> 902 | 1.04x [0.99, 1.05] |
| real 2048 | 3B | 197 -> 296 | 1.50x [1.36, 1.67] | 355 -> 343 | 0.97x [0.94, 1.00] |

- 8da4w: both arms dispatch the same stock kernel (`linear_dq8ca_q4gsw_tiled_*`; no Adreno 8da4w row), so ~1.00x is
  expected and its spread is the phone's noise band [M].
- 4w: the SARC Adreno kernel is dispatched in every prefill GEMM (112 / 196, `trace/dispatch.csv`) [M]. Kernel
  rate (warm trace) 1B: SARC 2.71 vs stock 2.73 TFLOP/s — no gain at 1B; 3B: SARC 2.53 vs stock 1.75 [M]. The stock
  4w arm is very noisy (repeat spread 17-30 %) while SARC is 2-10 %, so the 3B CI is wide [M]. Why stock 3B 4w is
  slow and variable (throttling while the stock tiled GEMM runs?) is [O].
- **The SARC 4w row is not correct**: its production diff fails at K >= 3072 (3-9 mismatched elements of 16384 in
  the correctness matrix; fp16 accumulation), consistent with the 2026-09-27 result [M]. The 4w speedup above is for
  a kernel that must not be promoted as is.
- Correctness of the rest [M]: 8da4w production diff (1B/3B/8B x buffer/texture3d, nonzero zp) all correct on the
  stock tiled kernel (SARC falls back); SDPA 4/4, 0 mismatches (stock kernels).

- M2 [M]: warm traces vs timed medians: 4w -0.9 % .. -10.9 %, 8da4w -10.7 % .. -20.3 % — outside the reference
  band (-6.7 .. +3.8 %); the phone's untraced host/driver overhead per run is larger at 8da4w [I]. SARC shows
  `sarc_linear_q4gsw_coopmat_t64x64k32g21s64m64x32x16` for all 4w GEMMs; stock shows no `sarc_*` kernel.
- M3 [R]: roofs from igpu-roofline newdev-20260927 (commit 463b2ff8e19f, plan fast, confirmed, DVFS not pinned).
  SARC 4w accumulates in fp16 (`glsl/sarc/sarc_linear_q4gsw_coopmat.yaml:155`, defaults `ACC_FP32: false` :28,
  `ACC_GROUP_FP32: false` :30) -> `matrix_fp16` 6.95 TFLOP/s: SARC 4w 38.9 % (1B) / 36.5 % (3B). Stock 4w 35.2 / 22.6 %
  of `alu_fp16` 7.76. 8da4w (both arms the stock kernel) 81-85 % of `dot_int8` 7.05; the sarc rows in efficiency.csv
  compare that stock kernel with `matrix_int8` (46-49 %) only because the kit maps "sarc" to the int8 matrix roof.
  Not confirmed on this device: `global_read`, and there is no fp32-accumulate matrix roof (Adreno has none) [R].
- M4 [M]: top-1 identical stock vs sarc in all 8 cells. **Caveat:** on the aligned real input the default SARC arm
  gives bit-identical top-10 to stock for 4w — in the `logits_probe` process the Adreno 4w row did not take the op
  over (a 3-way test on 1B 4w: default sarc 23.7656 == stock; forced tiled 23.5000; forced variant
  `ET_VK_SARC_Q4GSW_VARIANT=t64x64k32g21s64m64x32x16` 23.7812) [M]; why llama_main engages the row and the probe does
  not is [O]. Supplementary probes with the variant forced (`raw_real/probe-supplementary-forced-variant/`): top-1
  agrees on 1B/3B x real/check; top-1 logit 1B real 23.7812 vs stock 23.7656, 3B real 22.9062 vs 22.9062, 1B check
  18.1250 vs 18.1875, 3B check 16.5625 vs 16.5938 [M].

## Anomalies
- **8B not measured.** Llama 3.1 8B aborts with `vkQueueSubmit ... VK_ERROR_DEVICE_LOST` (rc 134) in both builds:
  4w in every attempt (stock and sarc, also after a fresh reboot); 8da4w after a fresh reboot aborted in 8 of 10 runs
  per prompt and the runs that completed varied 193-765 tok/s; the 8B logits probe returned all-zero logits in both
  builds; the ETDump build fails reading query results (`vkGetQueryPoolResults` VK_NOT_READY). All 8B rows, probes
  and next-token lines are kept in `superseded/8b-device-lost/` (with the unfiltered runs.csv) [M]. Cause (GPU
  timeout/watchdog on long submits?) is [O].
- 1972-token check prompt: SARC 4w needs M % 64, so the check runs fall back to stock kernels (report §B1) [I, S].
- Noisy cells (> 3 %): see the Results; the stock 4w arm dominates [M].

## Not measured / deviations
- M1-M4 for 8B (above). M5: no re-tuning on this device -> N/A.
- Rows not promoted (kUnverified; 4w incorrect). kit trace2.sh/probe.sh equivalents run from the host
  (`tools/e2eb-adb.sh`) apply `<build>/env` per build [M].

## Where the raw data is
- host-ws1 `<owner-workspace>/new-workspace/.artifacts/2026-09-28/e2eb/s26/` (raw/, raw_real/ with logs and
  env.txt, trace/trace2/*.etdp, probe/, superseded/, analysis/report/); device dir `/data/local/tmp/e2eb`.
