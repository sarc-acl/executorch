# contrib/mali: Mali-G1-Ultra MC12 (vivo V2502A), sarc-1.5-e2e-benchmark

Labels: [M] measured this campaign, [R] roofline, [S] source, [I] inference, [O] open.
Campaign directory (raw logs, microbench logs, ETDumps, superseded data):
`<owner-workspace>/new-workspace/.artifacts/2026-09-28/e2eb/mali/` (not in git).

## Status: partial. The GPU latched into the degraded state mid-campaign.

- [M] Sentinel (stock `llama_main`, Llama 3.2 1B 4w, prompt_2048, `--warmup`): 323.0–328.7 tok/s at the start
  of M1 (19:14–19:40 UTC), 258.5 tok/s after M1/M2 (sentinel/), i.e. -20 %. 3B 4w stock: 117.9 → 94.4.
- [I] Trigger: the 3B 4w SARC check run (unaligned 1972-token prompt, see "Unaligned M" below) ended at GPU
  80 C at 19:52 UTC. Everything after it is degraded-state data and is in `superseded/degraded-latch/`
  (README there): M1 cells 3B 8da4w, 8B 4w, 8B 8da4w, all M2 traces, a supplementary tile run.
- Device work stopped at that point. The phone was **not** rebooted (owner action). Not measured because of
  this: M1 `prompt_real_2048.txt` (raw_real), M4 logits probe, valid-state 3B 8da4w / 8B cells, valid-state M2
  timings.

## Device

| | |
|---|---|
| Phone | vivo V2502A (PD2502), adb `<mali-serial>`, USB on host-ws1; Android 16, build `vivo/PD2502/PD2502:16/BP2A.250605.031.A3_V000L1/compiler251021000854:user/release-keys`, kernel 6.12.23-android16-5 |
| SoC / GPU | MediaTek MT6993, Mali-G1-Ultra MC12, Arm driver `v1.r54p1-11eac0` (Vulkan 1.3.305, driverVersion 0xd801000), `/vendor/lib64/hw/vulkan.mali.so` md5 `7b6011e1f7b5eda5d3970181b9055503` |
| RAM | 15.0 GiB (all three models fit and ran, 8B included) |
| Root | no. Clocks float and are unreadable (devfreq/GED/gpufreq nodes permission-denied); `clocks` column = n/a |
| Thermal | no readable sysfs zones; pacing on the thermal HAL `GPU` sensor (`dumpsys thermalservice`, "Current temperatures from HAL"). Idle 29.8 C at the start; `IDLE_C=31` → each run waits until GPU <= 36 C, at most 300 s |
| Services stopped | none (phone; screen state as found) |

### Capabilities [M] (`capability/vkcap.txt`, scratch Vulkan probe `vkcap.c`)
- Subgroup size 16, fixed (subgroup-size control range 16–16); max compute shared memory 32768 B;
  maxComputeWorkGroupInvocations 1024; timestampPeriod 0.999 ns.
- `VK_KHR_cooperative_matrix`, compute stage only, subgroup scope. Shapes (M×N×K, A/B → C):
  fp16→fp32 and fp16→fp16 at **16×32×32** and 4×8×8; fp32→fp32 at 16×16×16 and 4×4×4;
  s8→s32 and u8→u32 at **4×16×16** only. No robust-buffer-access for coopmat.
- int8 dot product (`VK_KHR_shader_integer_dot_product`): 8-bit signed/unsigned and 4×8-bit packed signed
  accelerated; mixed signedness not accelerated. shaderFloat16 / shaderInt8 / int64 yes.
- The ExecuTorch device line: `DEVICE,mali-g1-ultra mc12,timestamp_period_ns=0.999001,subgroup_size=16,coopmat=yes,max_shared_mem_bytes=32768`.

## Builds (all Android arm64, NDK r29, `tools/sarc-build-native.sh`, native glslc — not golden-faithful)

| arm | commit | binary sha256 |
|---|---|---|
| stock | release/1.5 @ `985c1ceccc` + `stock-backport-03f41d2031.patch` (built in `stock-1.5/`, reused read-only) | llama_main `7637c605…1ec2e`, llama_main (ETDump) `e079ddc1…b5bf0`, logits_probe `c1bfbb65…33f83e` |
| SARC | `topic/mali-g1-tune` @ `7b858376b` (on dev/1.5 `d98227f60`; kept as local branch `topic/mali-g1-tune-measured`). Rebased afterwards onto dev/1.5 `49121ada0` → `14d8f5f68`; all 184 SARC SPIR-V files of the measured build are byte-identical in the rebased build | llama_main `b7cef8f9…56a4e5`, llama_main (ETDump) `7f611055…d7d08d`, logits_probe `76387af9…b3ce`, test_llama_microbench `c3013b89…f8771` |

Full hashes: `capability/sarc-binaries.sha256`, `…/mali/superseded/degraded-latch/raw-complete/env.txt`.
Per-build env: stock none; SARC `ET_VK_SARC_UNVERIFIED=1` (the Mali row is `kUnverified`).
Models [M]: the shared model store (llama3_{2_1b,2_3b,1_8b}_{4w,8da4w}_embq_ctx3072.pte),
pushed as `models/llama3_2-1b_vulkan_4w.pte` etc.; on-device sha256 equal to MANIFEST.json for all six
(`capability/device-sha256.txt`). Tokenizer sha256 `82e9d319…1b55`.
Driver: `tools/e2eb-adb-mali.sh` (copy of `e2eb-adb.sh.new` with a `mali` case).

## What SARC means on this GPU

- [S] dev/1.5 had no Mali rows (the 1.4 branch had routed Mali-G1 to the upstream GEMM: its comment in
  `<owner-workspace>/workspace/release14-quant-shaders-qualcomm/executorch/backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.cpp:1568` reports 122 vs 250 tok/s for its best coopmat
  tile on 1B 4w).
- New in `topic/mali-g1-tune` [S]:
  - 40 sweep candidates for MMA 16×32×32, SUBGROUP_SIZE 16 (`sarc_dev` yaml + `kQ4gswCandidates`);
  - a new shader flag `CSH_BAND` (default off; `#ifdef` in the body, `$if` in both wrappers, default in both
    yamls): the texture3d drain goes one SG_GRID_Y band at a time, so a 256-thread 64×128 tile fits 32 KiB;
  - a `kUnverified` row in a new `impl/sarc/table_arm.cpp`: device substring `mali-g1` → 4w
    `sarc_linear_q4gsw_coopmat_t64x128k32g44s16m16x32x32gahb` (MMA 16×32×32, 4×4 subgroups of 16, one MMA per
    subgroup, `ACC_GROUP_FP32`, `SH_F16V4`, `CSH_BAND`) for linears with N·K ≥ 2^24 (8B wq/wo/w1/w3/w2,
    3B and 1B w1/w3/w2); smaller linears stay on the stock path;
  - `test_sarc_select` Mali-G1 fixture; `sarc/tools/check.sh --no-build` PASS before and after the rebase.
  - Not done: golden entries (needs the pinned container glslc); `verify.sh`.
- Accumulator for M3 matching: `ACC_GROUP_FP32` (fp16 MMA within a 128-group, fp32 total) → roof
  `matrix_fp16` (fp16-accumulate MMA) plus conversion overhead.
- 8da4w: no SARC kernel [S][I]. The only int8 shape is 4×16×16, and the zpg/zpgtr kernels keep an fp32
  M×N accumulator coopmat, which Mali does not expose at 4×16. SARC 8da4w = the stock tiled/int8-dot kernels.
- SDPA: stock on Mali in both arms (no coopmat SDPA dispatched; SDPA correctness 0 mismatches) [M].

## Correctness (microbench, before the latch) [M]

- Stock 4w tiled kernel (`q4gsw_linear_gemm__tin`) fails the production-diff tolerance (0.5 abs) on Mali:
  max |err| 0.91 / 1.02 / 1.14 / **3.17** on 8B wq_wo / wk_wv / w1_w3 / w2, 2.05 on 1B w2, 1.80 on 3B w2;
  the 4w correctness matrix fails 3 cases at K=4096. fp16 accumulation.
- Mali coopmat needs `SH_F16V4` (fp16 LDS typed as f16vec4): without it the output is garbage
  (max |err| ~90 on 8B).
- The chosen tile (`…gahb`, ACC_GROUP_FP32): all 24 production-diff shapes (1B/3B/8B × 4 × buffer/texture3d)
  PASS, max |err| 0.088–0.191; 4w correctness matrix 24/24.
- The fastest tile (`t128x64k32g28s16m16x32x32hb`, plain fp16 accumulate) reproduces the stock kernel's
  errors **to the printed digit** (0.906715, 1.020, 1.141, 3.166, …), so it fails the same shapes as stock.
  [I] Its fp16 accumulation order matches the tiled kernel's; its output may be identical to stock (not
  checked element-wise).
- 8da4w stock tiled: production diff passes everywhere (max |err| ≤ 0.95 at 8B w2, buffer; ≤ 0.12 texture3d).
- SARC arm as built (row + N·K predicate): row shapes pass; the smaller shapes carry the stock kernel's error
  (3B wq_wo fails like stock). Evidence: `tuning/production-diff.txt`.

## Tuning [M] (microbench `test_llama_microbench`, kernel time, Llama 3.1 8B prefill shapes, M=2048)

Screen (1 repeat, 31 candidates, `tuning/screen-8b-1rep.txt`): register pressure decides. One 16×32 MMA per
subgroup and 256-thread workgroups win; 2–4 MMAs per subgroup are 2–7× slower than stock; fp32 accumulation
costs 1.3–1.8× over fp16; ACC_GROUP_FP32 sits in between.

Finalists, 3 interleaved repeats (`tuning/rep3-summary.txt`), sum of the four 8B shapes vs stock tiled:

| candidate | correct (production diff) | texture3d | buffer |
|---|---|---|---|
| stock tiled `q4gsw_linear_gemm__tin` | fails (max 3.17) | 726.9 ms (1.00×) | 723.9 ms (1.00×) |
| `t64x128k32g44s16m16x32x32gahb` (ACC_GROUP_FP32) — **the row** | passes (max 0.19) | 651.7 ms (**1.12×**) | 687.6 ms (1.05×) |
| `t128x64k32g28s16m16x32x32hb` (fp16 acc) | fails exactly like stock | 480.4 ms (**1.51×**) | 517.0 ms (1.40×) |

Per shape the row tile loses on wk_wv (0.87×) and wins on w1_w3 / w2 (1.13–1.14×); on 1B/3B (1 repeat,
`tuning/small-models-1rep.txt`) it wins only on the N·K ≥ 2^24 shapes, hence the row predicate.
Other 1-repeat results (8B buffer): t64x128k32g44 fp16 1.37×, t64x128k32g44 ACC_FP32 0.79×,
t64x64k32g24 fp16 0.94×, t64x64k32g22 ACC_FP32 (the 1.4-style tile) 0.44×.

## M1 [M] (prompt_2048.txt, 5 interleaved repeats; valid-state cells only)

| model | scheme | stock tok/s | SARC tok/s | SARC/stock | spread stock / SARC | next token (1972) |
|---|---|---|---|---|---|---|
| 1B | 4w | 324.8 | 324.1 | 1.00× [0.98, 1.23] | 1.8 % / 25.0 % | SAME |
| 3B | 4w | 117.9 | 131.7 | 1.12× [0.99, 1.19] | 0.3 % / 20.0 % | SAME |
| 1B | 8da4w | 682.0 | 693.5 | 1.02× [0.98, 1.14] | 3.1 % / 16.9 % | SAME |

- The SARC 4w runs are bimodal: 3B SARC 117.0 / 126.5 / 131.7 / 136.9 / 140.4 against a flat stock
  117.7–118.0; 1B has one SARC run at 398.6 (others 318.9–324.3). The fast runs end hotter (61 C vs 50–54 C).
  [I] A DVFS state change (clocks unreadable), not the kernel alone; the 1B 8da4w SARC arm (same kernels
  as stock) shows one such run too (779 vs ~690).
- Degraded-state cells (superseded, comparable only among themselves): 3B 8da4w 223.4 vs 220.3 (0.99×),
  8B 4w 39.66 vs 35.56 (**0.90×**), 8B 8da4w 115.4 vs 116.4 (1.01×). [O] whether 8B 4w SARC loses in the valid state.
- `nexttoken.csv`: the driver's original comparison said DIFFER everywhere because the Mali driver prints a
  per-process `[MEMPROF]` line; regenerated with that line filtered (original in
  `superseded/nexttoken-memprof-line/`). The driver copy has been fixed.

### Unaligned M (the known fallback pitfall) — a real regression on Mali [M]
On the SARC path, an op built for SARC falls back at run time to release 1.5's `linear_q4gsw_tiled`
(not the stock path's TIN GEMM) for unaligned M. On Mali that kernel is much slower and runs hot:
check runs (prompt_check.txt, 1972 tokens, no warmup): 1B 4w 180.2 vs 338.8 tok/s (GPU 73 C),
3B 4w 65.8 vs 123.4 (80 C — the latch trigger), 8B 4w (degraded) 16.6 vs 41.8. Any real prompt whose length
is not a multiple of 64 hits this. [O] fix: fall back to the stock TIN GEMM (needs its weight layout) or keep
Mali ops off the SARC path at unaligned M.

## M2 [M, degraded state] (warm ETDump, one prefill per cell)

Dispatch evidence (valid regardless of the latch): SARC 4w dispatches
`sarc_linear_q4gsw_coopmat_t64x128k32g44s16m16x32x32gahb_texture3d_texture2d_half` (8B: 160 per prefill,
the N·K ≥ 2^24 linears; wk/wv stay on `q4gsw_linear_gemm__tin`); stock shows no `sarc_*` kernel; 8da4w is
identical in both arms (`linear_dq8ca_q4gsw_tiled_texture3d_texture2d_half_zpint8`). `trace/dispatch.csv`.
Timings (`trace/{families,gemm,totals}.csv`) come from the degraded state: in them the SARC 8B linears take
48.6 s vs 44.4 s stock, and per dispatch the SARC tile is ~1.5× slower than in the microbench while stock is
~1.2× slower. [O] whether the valid-state e2e matches the microbench.

## M3 [O]
No confirmed roofline for this phone on host-ws1 (the owner's campaign ran from their Mac;
`igpu-roofline-newdev/results/` has only a Pixel 7a Mali-G710 fast run). Not started here (long run, and heavy
load is what latches this GPU). `roofline.json` / `efficiency.csv` not produced.

## M4 [O] not measured (device degraded before it; probes staged at `/data/local/tmp/e2eb-mali/probe-{stock,sarc}`).

## Anomalies
- Degraded-state latch (above). No crashes, no device loss, no rejected prefill runs (rc 0, 2048 tokens everywhere).
- One M1 run (8B 8da4w stock r3, degraded cell) started at 46 C: the HAL temperature read returned
  nothing once and the pacing loop exited.
- Mali prints `Could not open module param file '/sys/module/mali_kbase/parameters/large_page_conf'` and a
  `[MEMPROF]` line on every Vulkan process start (harmless; the latter broke next-token comparison).
- Noisy cells (> 3 % spread): all three valid cells on the SARC side (bimodal, see M1).

## Verdict [M]/[I]
Cooperative matrix works on Mali-G1 (16×32×32 fp16) and a SARC tile can beat the stock 4w kernel at the kernel
level, but only narrowly when it is also accurate: 1.12× (texture3d, 8B shapes) with ACC_GROUP_FP32, versus
1.51× for plain fp16 accumulation that keeps the stock kernel's out-of-tolerance error. End to end the
accurate tile gave 1.00× (1B) and 1.12× (3B, bimodal) in the valid state, and 0.90× on 8B in the degraded
state. The SARC path's unaligned-M fallback halves 4w prefill on Mali. The 8da4w path, 2–3× faster than 4w on
this GPU, has no usable int8 coopmat shape. Recommendation: keep the Mali row `kUnverified`; worth pursuing
only with (a) a stock-TIN fallback for unaligned M and (b) a valid-state 8B e2e after a reboot.
