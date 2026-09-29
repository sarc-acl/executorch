# Radeon RX 7600 — sarc-1.5-e2e-benchmark contribution (2026-09-28)

Labels: [M] measured in this campaign, [R] roofline, [S] source, [I] inference, [O] open.

## Device, driver, clocks
- Host `host-ws1`, Ubuntu 22.04.5, kernel 6.8.0-107-generic, Xeon Gold 5220R. GPU: AMD Radeon RX 7600
  (Navi 33, gfx1102, PCI 1002:7480, 8 GiB VRAM, `/sys/class/drm/card1`), discrete; GPU1 is llvmpipe [M].
  The marketing name comes from the new driver's deviceName "AMD Radeon RX 7600 (RADV NAVI33)"; whether the
  board is an RX 7600 or 7600 XT/other Navi 33 SKU is [O] (the 8 GiB VRAM fits the RX 7600).
- Driver for EVERYTHING below (roofline, microbench, stock arm, SARC arm): a user-space RADV,
  Mesa **26.2.3** (tag `mesa-26.2.3`, commit 31e9a6b2e95e30d84bf3177d1f497d063e59b6b2), built RADV-only with ACO
  (`-Dvulkan-drivers=amd -Dgallium-drivers= -Dplatforms= -Dllvm=disabled`, release) against libdrm 2.4.134
  (commit e984d448b8b1) in `<owner-workspace>/mesa/install`, RUNPATH to that prefix; selected per process with
  `VK_ICD_FILENAMES=<owner-workspace>/mesa/install/share/vulkan/icd.d/radeon_icd.x86_64.json` (never system
  wide). Vulkan 1.4.354, conformance 1.4.5.3 [M]. The system Mesa 23.2.1 exposes no VK_KHR_cooperative_matrix
  on this card (verified by the owner); 26.2.3 does [M].
- VK_KHR_cooperative_matrix (rev 2) properties, all subgroup scope, M=N=K=16: f16xf16+f16->f16,
  f16xf16+f32->f32, and {u8,s8}x{u8,s8}+{u32,s32} (s32 also saturating); no bf16; supported stage compute;
  subgroup size 32..64, default 64 [M] (`driver/driver.txt`, probe `driver/cm.c`).
- Clocks not pinned, left as found: governor `auto` (DVFS), sclk 86..2356 MHz, mclk 96..1124 MHz; post-run sclk
  snapshots 1196-2415 MHz, one sample right after each process exits (`clocks` column) [M]. Idle temperature (kit gtemp = max of edge/junction/mem of the one
  amdgpu hwmon, 0000:04:00.0 = card1) 48 C (raw) / 50 C (raw_real); peak post-run 66 C [M].
- Display: the card drives the COSMIC/Wayland desktop, 2 DisplayPort monitors connected (5 connectors), ~35 MB VRAM
  in use at idle [M]. Not stoppable (no sudo); display-GPU noise is a report pitfall — every cell's repeat spread
  is <= 2.4 % [M].
- Co-tenants: the host also runs other agents (opencode/claude sessions). During the whole campaign `others`
  records another agent's `adb -s <mali-serial> shell ... llama_main` (a Mali phone benchmark driven over adb:
  host CPU only, not this GPU) [M]. No other Vulkan workload on this GPU was observed [M] (not exhaustively
  checked: no per-process GPU accounting on amdgpu here [O]). Load average at phase starts 1.4-4.1
  (`campaign-load.txt`); I ran no builds during timed phases [M].
- Lock: kit `flock` on `$HOME/.cache/gpu-lab/lock-rx7600-sj1` (dir created; `~/.cache` -> /local). gpu-lab has no
  entry for this GPU; M1 -> M2 -> M4 ran strictly in sequence from one driver script (`stage/campaign.sh`) [M].
- 8B fits: peak VRAM 5.47 GiB of 8 GiB for SARC 8B 8da4w at 2048 tokens [M].

## Builds
- stock: `release/1.5` @ 985c1ceccc8bb8b6a32294f71aeb5d2f299562f7 + `kit/patches/stock-backport-03f41d2031.patch`,
  the prebuilt `stock-1.5/executorch/cmake-out-host/{llama,llama-etdump,probe}` (copied, read only). Its
  llama_main/libllama_runner.so hashes equal the 7900 XTX campaign's stock [M]. Note: the stock-1.5 tree now also
  carries an uncommitted `ComputeGraph.cpp` edit (opt-in `ET_VK_EXECUTE_NODE_THRESHOLD`, another agent's) dated
  10:08, after the binaries (09:04); the variable is unset here, so the stock arm is the documented one [M, I].
- sarc: `topic/rx7600-coopmat` @ **5351955ca920a097c8ba962108376012ad8082d2** (local commit, not pushed; on
  `topic/7900xtx-4w-coopmat` ff3f34ef8). Rows for "rx 7600" (kUnverified): 4w
  `sarc_linear_q4gsw_coopmat_t256x128k32g24s32f32cbt` (texture3d, fp32 accumulate), 8da4w
  `sarc_linear_dq8ca_coopmat_zpg_t128x64k32g42s32` (780M kernel), SDPA prefill
  `sarc_sdpa_{qk_coopmat_t128x64k32g22s64,av_coopmat_t64x64k32g22s64}` (780M kernels) [S]. `sarc/env`:
  `ET_VK_SARC_UNVERIFIED=1`; stock/env empty [M]. `check.sh --no-build` passes [M].
- Toolchain: native `tools/sarc-build-native.sh` (tree uv venv + `install_executorch.sh --minimal`, glslc Vulkan SDK
  1.4.350.1), NOT the pinned `et-vk-build:rocky10` container: SPIR-V not checked against `sarc/golden` [M].
- sha256 (stage `.artifacts/2026-09-28/e2eb/rx7600/stage`):
  - stock/llama_main f2a4f255e3a1ef98b7b71d662e75d7596024a8ede94a344ad89535ddcb8d4285,
    stock/libllama_runner.so 4ae495400409e6b5d130e336a264f7ab23cf7a16c749d913a6b27aae30dd989d
  - sarc/llama_main 75be8f83e2532f5a31df24be840cdd39c2a7e654e1687984956f467b7f99dfdb,
    sarc/libllama_runner.so 1a789b62ae9360900f7544efe5a0bb69d9c07b7d220ff385059ba56a9ce8b266
  - traced llama_main: stock 32d7af2434a50e20b9b0023f794aa2292c2611df47a651baca03c7d28dbf47ba,
    sarc f0a3df34bed6acb9d5984b32696b3ebc89e07d9fbf963e25a7f83eacf4313009
  - probe-stock/logits_probe 14da0edba56d1c831fac73dd48d8f76ee0b5e1e12d2ea4be059077ea3f2b27c7,
    probe-sarc/logits_probe ff52b3d68b73f2bebec3c994fdf60392881594459e494bffcde461514984e454
  - libvulkan_radeon.so: see `driver/driver.txt`
- Models (`*_embq_ctx3072.pte`, copied to /local, sha256 == the shared model manifest (MANIFEST.json),
  flat-named): 1b 4w 1ac83440b93b2cde…, 1b 8da4w fb99c89e141f420b…, 3b 4w 92117564851859bd…, 3b 8da4w
  e9eba0cf5a0f6ca7…, 8b 4w 695dd232a500e9b7…, 8b 8da4w 6f172bc5590cdf68…; tokenizer 82e9d31979e92ab9… [M].
- Prompts: kit prompts (same hashes as the 7900 XTX campaign); every timed log (120) and every trace log (12) shows
  `"prompt_tokens":2048` [M].

## Results (medians of 5, paired 95 % CI; kit/analysis/analyze.py)
| prompt | model | 4w stock -> sarc tok/s | 4w speedup | 8da4w stock -> sarc tok/s | 8da4w speedup |
|---|---|---|---|---|---|
| "the"x2048 | 1B | 2705 -> 7787 | 2.88x [2.86, 2.89] | 3793 -> 7340 | 1.94x [1.93, 1.94] |
| "the"x2048 | 3B | 972 -> 3287 | 3.38x [3.36, 3.38] | 1388 -> 3080 | 2.22x [2.21, 2.23] |
| "the"x2048 | 8B | 469 -> 1517 | 3.23x [3.22, 3.27] | 731 -> 1403 | 1.92x [1.91, 1.93] |
| real 2048 | 1B | 2695 -> 7670 | 2.85x [2.83, 2.86] | 3758 -> 7161 | 1.91x [1.88, 1.92] |
| real 2048 | 3B | 973 -> 3235 | 3.32x [3.29, 3.33] | 1375 -> 2999 | 2.18x [2.15, 2.19] |
| real 2048 | 8B | 469 -> 1486 | 3.17x [3.17, 3.18] | 721 -> 1355 | 1.88x [1.87, 1.89] |

Geomean [M]: "the"x2048 4w 3.16x, 8da4w 2.02x, both 2.53x; real text 4w 3.11x, 8da4w 1.98x, both 2.48x.
Real text vs "the" [M]: SARC -1.5 % to -3.4 %, stock -1.4 % to +0.1 %; speedups -0.03x to -0.06x.

- M2 [M]: warm traces (last of two executions) match the timed medians within -5.2 % .. +0.1 % (1B SARC is the
  -5 %: ~260 ms runs, 1 ms timer). SARC dispatches 112/196/224 prefill GEMMs (7 per layer) on the SARC kernels above
  and `sarc_sdpa_{qk,av,softmax}` on every layer; stock shows no `sarc_*` kernel (`trace/dispatch.csv`). Prefill
  GEMM share of GPU time: 4w 53/55/67 % (stock) -> 57/68/77 % (sarc); attention QK^T+AV 1B 266 -> 44 ms,
  8B 1191 -> 128 ms. On 8da4w a large part of the gain is SDPA: stock 8da4w GEMMs are only 37-51 % of its time.
- M3 [R]: igpu-roofline campaign `rx7600-fast-20260928` (commit 463b2ff8e19f, working tree dirty only in `uv.lock`
  (pre-existing, not by this agent), runner c8e8698f86b4f885 built with gcc 15.2 static libstdc++ + Vulkan SDK
  headers; plan fast, 3 confirmed repeats per roof, sentinel 33/33 ok at 7.78 TFLOP/s fp32, 35 confirmed roofs,
  DVFS not pinned; same Mesa 26.2.3 ICD) — `roofline.json`. Key roofs: alu_fp16 20.23 TFLOP/s, dot_int8 29.03
  TOP/s, matrix_fp16 32.31 TFLOP/s, matrix_fp16_fp32 43.42 TFLOP/s, matrix_int8 43.90 TOP/s, global_read
  283.6 GB/s; fed: fp16->fp32 from LDS 7.8/14.9/28.9/38.6 TFLOP/s at 8/16/32/64 ops per loaded byte, int8 from
  LDS 14.3/25.2/31.5/35.8 TOP/s at 16/32/64/128. On RADV the fp16-accumulate WMMA roof is BELOW fp32-accumulate
  (32.3 vs 43.4). The SARC 4w row accumulates in fp32 (`ACC_FP32: true`,
  glsl/sarc/sarc_linear_q4gsw_coopmat.yaml:185) -> matched to `matrix_fp16_fp32`.
  Kernel % of roof (`efficiency.csv`): sarc 4w 63.8-64.3 % of matrix_fp16_fp32 (27.7-27.9 TFLOP/s), sarc 8da4w
  58.6-59.1 % of matrix_int8; stock 4w 48.5-49.9 % of alu_fp16, stock 8da4w 69.4-71.4 % of dot_int8 (stock uses
  the saturating int8 dot, roof kernel non-saturating: report pitfall B6) [S].
  [I]: the 4w tile's 64x64 subgroup tile loads 32 ops per LDS byte, where the fed roof is 28.9 TFLOP/s: the kernel
  runs at ~96 % of the shared-fed roof for its reuse; more reuse needs a bigger subgroup tile, which the 256-VGPR
  budget (4 subgroups/SIMD already) and the 64 KB LDS limit rule out.
- M4 [M]: top-1 identical stock vs sarc on the aligned real 2048-token input for all 6 cells (max |dlogit| over the
  top-10 intersection 0.23-1.00). Check input (1972 tokens): 5/6 identical; 8B 8da4w flips "otherwise"(6062) ->
  "bullying"(45647): stock margin 0.094, sarc margin 0.500 — the known near tie at this position (report §B2; the
  7900 XTX shows the same flip). The e2e check run shows the same flip on both prompts (`raw*/nexttoken.csv`).
- Kernel correctness for the rows (microbench, ET_VK_SARC_UNVERIFIED=1, final binary) [M]: 4w production diff
  1B/3B/8B texture3d pass (SARC kernel dispatched on all 4 shapes); 8da4w production diff 1B/3B/8B x
  buffer/texture3d with nonzero zero points pass; SDPA correctness 4/4, 0 mismatches. The 4w buffer production
  diff and 3 cases of the 4w correctness matrix fail by 0.51-0.63 abs (limit 0.5) in the UPSTREAM fallback kernels
  (`q4gsw_linear_gemm`/`linear_q4gsw_tiled`: buffer IO or M=128 < tile), with the same element indices as the
  7900 XTX logs of 2026-09-27 — not a SARC or device issue [M, I].

## Tuning (8B prefill shapes, texture3d, microbench, 3 interleaved repeats; tables in `.artifacts/2026-09-28/rx7600/tuning-*.md`)
- 4w: 13 tiles + tiled. Winner = the 7900 XTX tile `t256x128k32g24s32f32cbt`, 20.47 ms for the 4 projections
  (27.7 TFLOP/s, 2.89x tiled 59.1 ms); `t256x128k32g24s32f32c` 20.60 ms; 780M tile `t128x128k32g42s32f32c`
  25.24 ms; fp16-accumulate tiles 27.4-35 ms (slower AND the fp16-acc tile failed the 8B diff on the 7900) [M].
  RADV stats: 256 VGPRs, 0 spills, 60 KB LDS, 4 subgroups/SIMD [M]. Deeper K (k64) or 256x256 tiles need > 64 KB LDS [S].
- 8da4w: 780M zpg `t128x64k32g42s32` 22.96 ms kernel (24.7 TOP/s, 1.28x tiled 29.4 ms); xclipse zpgtr tiles 1.05-1.13x
  slower; four new RDNA zpg sweep tiles with bigger subgroup tiles (in the commit's dev zone) all pass the 8B diff but
  are 1.01-1.13x slower: 224-256 VGPRs -> 4 subgroups/SIMD vs 128 VGPRs -> 8 for the 780M tile [M]. A first run of
  two of them without A_MULTI_BLOCK looked 4.6 % faster but was wrong (rows 128-255 garbage): moved to
  `mb/superseded/zpg-a-map-missing-multiblock/` [M].
- SDPA coopmat vs tiled (microbench, S=2048): 3.27x / 5.75x / 6.02x total for 1B/3B/8B [M].

## Anomalies
- No crash, no rejected run, no retry (72 + 72 rows, rc 0 everywhere) [M]. No cell above 3 % spread (max 2.4 %,
  real 1B 8da4w sarc) [M].
- The 1972-token check prompt is not tile-aligned: the 4w row needs M % 256 and the zpg row aligned M, so the linear
  layers fall back to stock kernels there; SDPA coopmat still runs [I from Select.cpp fit rule, S].
- Timer resolution 1 ms: 1B SARC prefill ~263-286 ms, so one step is ~0.4 % [M].
- Roofline campaign ran while this agent's ExecuTorch wheel build loaded the host (load avg up to ~35 during the
  confirm phase); the fp32 sentinel stayed at 7.76-7.88 TFLOP/s throughout (all ok) [M, R].

## System Mesa baseline (separate, informational; NOT part of M1)
- stock arm on the system Mesa 23.2.1 RADV ("AMD Unknown (RADV GFX1102)"), one run per cell, "the"x2048, no
  cool-down pacing: 1B 4w 2441, 1B 8da4w 3683, 3B 4w 877, 3B 8da4w 1355, 8B 4w 412, 8B 8da4w 706 tok/s — i.e. the
  4w stock arm is ~11-14 % faster on Mesa 26.2.3, 8da4w ~2-4 % [M, n=1] (`../../sysmesa-baseline/`).

## Not measured / deviations
- M5 not run: the RX 7600 had no SARC rows before this campaign [S].
- Rows not promoted (kUnverified): golden SPIR-V needs the pinned container; verify.sh evidence is the microbench +
  this campaign [O]. Commit not pushed (local only).
- kit/host/trace2.sh and probe.sh do not read `<build>/env`: both ran with `ET_VK_SARC_UNVERIFIED=1` exported
  (stock 1.5 has no SARC code); `VK_ICD_FILENAMES` was exported for the whole campaign (both arms) [M, S].
- Services not stopped (desktop compositor; no sudo). Sustained roofs: 3 of 35 confirmed roofs have one 120 s batch
  only (fast plan) [R]. No thread trace / RGP (not available here) [O].

## Where the raw data is
- Campaign: host-ws1 `<owner-workspace>/new-workspace/.artifacts/2026-09-28/e2eb/rx7600/` (raw/, raw_real/ with
  all logs and env.txt, raw/trace2/*.etdp, raw_real/probe/, analysis/report/, stage/, driver/, sysmesa-baseline/);
  microbench logs `.artifacts/2026-09-28/rx7600/mb/`, RADV stats `.artifacts/2026-09-28/rx7600/isa/`; roofline
  `<owner-workspace>/workspace/igpu-roofline-newdev/results/rx7600-fast-20260928/host-ws1-gpu-000000000400/`.
