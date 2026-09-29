# RX 7900 XTX — sarc-1.5-e2e-benchmark contribution (2026-09-28)

Labels: [M] measured in this campaign, [R] roofline, [S] source, [I] inference, [O] open.

## Device, driver, clocks
- Host `host-7900xtx`, Ubuntu 25.04, kernel 6.14.0-37-generic; Radeon RX 7900 XTX (gfx1100, Navi31), the only
  Vulkan device (ETVK_DEVICE_INDEX=0); AMDVLK 2025.Q2.1 (LLPC), Vulkan 1.4.313 [M].
- Clocks not pinned, left as found (DVFS; `pp_dpm_sclk` snapshot after each run in `clocks`) [M]. Idle GPU
  temperature 44 C (the card drives the host displays) [M]. Cool-down: e2e.sh rule (idle + 5 C, at most 120 s) [M].
- Co-tenant: `ollama serve` (systemd service, user `ollama`) was running with no model loaded, GPU busy 0 %,
  ~111 MB VRAM (display) [M]. Not stopped (no sudo on this host); recorded in `others` for every run [M].
- Lock: kit `flock` on `$HOME/.cache/gpu-lab/lock-7900xtx-host-7900xtx` with HOME=<owner-home>
  (the NFS `~/.cache` symlink is broken on this host). gpu-lab/AGENTS.md was not available on this machine;
  one measuring process at a time was ensured by running M1 -> M2 -> M4 strictly in sequence [M].

## Builds
- stock: `release/1.5` @ 985c1ceccc8bb8b6a32294f71aeb5d2f299562f7 + `kit/patches/stock-backport-03f41d2031.patch`.
- sarc: `topic/7900xtx-4w-coopmat` @ 8b00c92f1 (on dev/1.5 f4eea5ae3; the 6 newer dev/1.5 commits are docs only).
  Rows for "7900 xtx" (kUnverified): 4w `sarc_linear_q4gsw_coopmat_t256x128k32g24s32f32cbt` (texture3d, fp32
  accumulate), 8da4w `sarc_linear_dq8ca_coopmat_zpg_t128x64k32g42s32` (780M kernel), SDPA prefill
  `sarc_sdpa_{qk_coopmat_t128x64k32g22s64,av_coopmat_t64x64k32g22s64}` (780M kernels) [S].
  Build env `sarc/env`: `ET_VK_SARC_UNVERIFIED=1` (rows are not yet kVerified). stock/env empty [M].
- Toolchain: native `tools/sarc-build-native.sh` (per-tree uv venv + `install_executorch.sh --minimal`, CMake 3.31.10,
  GCC host, glslc from Vulkan SDK 1.4.350.1), NOT the pinned `et-vk-build:rocky10` container: the SPIR-V differs
  from `sarc/golden/spirv.json`, which was not checked [M]. Shader codegen forced every build (ShaderLibrary.cmake
  does not track glsl/sarc*/ edits) [S].
- sha256 (stage `<owner-home>/e2eb`):
  - stock/llama_main f2a4f255e3a1ef98b7b71d662e75d7596024a8ede94a344ad89535ddcb8d4285,
    stock/libllama_runner.so 4ae495400409e6b5d130e336a264f7ab23cf7a16c749d913a6b27aae30dd989d
  - sarc/llama_main 6d4b4698d25d592a6e67cb108b5c2615c57557725222625a3d91407b6ec65419,
    sarc/libllama_runner.so f52d3825a4f4014010deee664da8f703ab9acd774fe90b3ffadb2fa9aa457470
  - traced: stock 3587e4c6e869… / sarc b64509ea322f… (llama_main; ETDump builds, full hashes in the campaign env)
  - probe-stock/logits_probe 14da0edba56d1c831fac73dd48d8f76ee0b5e1e12d2ea4be059077ea3f2b27c7,
    probe-sarc/logits_probe 6374c8d137aaf4ed7e560b688bf78478048ad0a7798906de862be98214e95ff0
- Models (`*_embq_ctx3072.pte`, sha256 == the shared model manifest (MANIFEST.json)), flat-named:
  1b 4w 1ac83440b93b2cde…, 1b 8da4w fb99c89e141f420b…, 3b 4w 92117564851859bd…, 3b 8da4w e9eba0cf5a0f6ca7…,
  8b 4w 695dd232a500e9b7…, 8b 8da4w 6f172bc5590cdf68… [M]. Same export recipe as the five-GPU campaign, exported separately (owner-confirmed, 2026-09-28; files not byte-identical).
- Prompts: kit prompts, sha256 prompt_2048 bfce65eb…, prompt_real_2048 30ec73a2…, prompt_check b5499448…; every timed
  log shows `"prompt_tokens":2048` [M].

## Results (medians of 5, paired 95 % CI; tables from kit/analysis/analyze.py)
| prompt | model | 4w stock -> sarc tok/s | 4w speedup | 8da4w stock -> sarc tok/s | 8da4w speedup |
|---|---|---|---|---|---|
| "the"x2048 | 1B | 6564 -> 20078 | 3.06x [3.03, 3.06] | 10343 -> 22261 | 2.15x [2.13, 2.20] |
| "the"x2048 | 3B | 2516 -> 10089 | 4.01x [3.99, 4.04] | 4008 -> 10396 | 2.59x [2.57, 2.61] |
| "the"x2048 | 8B | 1246 -> 4774 | 3.83x [3.82, 3.84] | 2207 -> 4971 | 2.25x [2.24, 2.27] |
| real 2048 | 1B | 6502 -> 19505 | 3.00x [2.92, 3.02] | 10089 -> 21113 | 2.09x [2.09, 2.11] |
| real 2048 | 3B | 2513 -> 9526 | 3.79x [3.75, 3.83] | 3835 -> 9706 | 2.53x [2.51, 2.55] |
| real 2048 | 8B | 1239 -> 4501 | 3.63x [3.63, 3.65] | 2140 -> 4582 | 2.14x [2.13, 2.15] |

Geomean [M]: "the"x2048 4w 3.61x, 8da4w 2.33x, both 2.90x; real text 4w 3.46x, 8da4w 2.25x, both 2.79x.
Real text vs "the" [M]: tok/s -2.9 % to -7.8 % in the SARC arm, -0.1 % to -4.3 % stock; speedups -0.06x to
-0.22x (largest: 3B 4w 4.01x -> 3.79x). The "the" prompt flatters SARC slightly more than stock here (report §B9).

- M2 [M]: warm traces (last of two executions) match the timed medians within -4.9 % .. +0.6 %. SARC dispatches
  112/196/224 prefill GEMMs (7 per layer) on the SARC kernels above and `sarc_sdpa_*` for QK^T/AV/softmax; stock
  shows no `sarc_*` kernel (`trace/dispatch.csv`). 1B 4w: attention QK^T+AV 107.6 ms (stock) -> 18.6 ms (sarc);
  the prefill GEMM share is 58 % (stock) and 63 % (sarc) of GPU time.
- M3 [R]: roofs from igpu-roofline campaign newdev-20260927 (commit 463b2ff8e19f, plan fast, 3 confirmed repeats,
  sentinel healthy; not pinned) — see `roofline.json`. The SARC 4w row accumulates in fp32 (`ACC_FP32: true`,
  glsl/sarc/sarc_linear_q4gsw_coopmat.yaml:185) -> matched to `matrix_fp16_fp32` 141.9 TFLOP/s.
  Kernel % of roof (`efficiency.csv`): sarc 4w 45.9 / 56.6 / 59.9 %, sarc 8da4w 57.5 / 62.2 / 65.0 % of
  `matrix_int8` 142.6 TOP/s; stock 4w 35.6-37.3 % of `alu_fp16` 63.4, stock 8da4w 83.0-85.6 % of `dot_int8` 69.2.
  Stock uses the saturating int8 dot while the roof kernel is non-saturating (report pitfall B6) [S].
- M4 [M]: top-1 identical stock vs sarc on the aligned real 2048-token input for all 6 cells (max |dlogit| over
  the top-10 intersection 0.30-1.16). Check input (1972 tokens): 5/6 identical; 8B 8da4w flips "otherwise"(6062) ->
  "bullying"(45647): stock margin 0.281, sarc margin 0.055 — the known near tie at this position (report §B2).
  The e2e check run shows the same flip (`raw/nexttoken.csv` 8b,8da4w,DIFFER).
- Correctness evidence for the rows (microbench, ET_VK_SARC_UNVERIFIED=1, 2026-09-28) [M]: 4w production diff
  1B/3B/8B texture3d pass on the release kernel; 8da4w production diff 1B/3B/8B x buffer/texture3d with nonzero
  zero points pass; SDPA correctness 4/4, 0 mismatches.

## Anomalies
- No crash, no rejected run, no retry (72 + 60 rows, rc 0 everywhere) [M]. No cell above 3 % spread on real text;
  on "the"x2048 1B 8da4w sarc 3.3 % [M].
- The 1972-token check prompt is not tile-aligned: the 7900 4w row needs M % 256 and the zpg (4h4w) 8da4w row
  needs aligned M, so SARC falls back to stock kernels there and the check mostly compares stock with stock
  (report §B1) [I from the fit rule in impl/sarc/Select.cpp, S].
- Timer resolution 1 ms: 1B SARC prefill ~92-102 ms, so one step is ~1 % [M].

## Not measured / deviations
- M5 not run: the 7900 XTX had no release-1.5 SARC rows before this campaign; the 2026-09-27 re-tuning (1.4 study,
  1.5 port) is recorded in `openspec/changes/sarc-1.5-7900xtx-4w/` (sweep-1.4, port-1.5, e2e-1.5) [S].
- Rows not promoted (kUnverified): golden SPIR-V needs the pinned container; verify.sh evidence is the microbench +
  this campaign [O].
- kit/host/trace2.sh and probe.sh do not read `<build>/env`: both were run with `ET_VK_SARC_UNVERIFIED=1` exported
  (release 1.5 stock has no SARC code, so the stock arm is unaffected) [M, S].
- Services not stopped (see above). Sustained roofs: 3 of 37 confirmed roofs have one 120 s batch only [R].

## Where the raw data is
- Campaign: host-ws1 `<owner-workspace>/new-workspace/.artifacts/2026-09-28/e2eb/7900xtx/` (raw/, raw_real/
  with all logs and env.txt, raw/trace2/*.etdp, raw_real/probe/, analysis/report/); stage copy on
  `host-7900xtx:<owner-home>/e2eb/`.
