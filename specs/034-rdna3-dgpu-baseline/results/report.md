# RDNA3 Discrete GPU (RX 7900 XTX) — Release/1.3 Baseline

**Date**: 2026-07-22
**Device**: AMD Radeon RX 7900 XTX (Navi 31, `deviceID=0x744c`), reached via `ssh yanwen.xu@xraytracing02`
**Driver**: AMD open-source (Mesa RADV), `driverInfo="2025.Q2.1 (LLPC)"`, Vulkan 1.4.304, conformance 1.4.1.3
**Branch/commit**: `sarc-acl/executorch release/1.3` @ `e2f18eb23` (same commit the M5 EVT1/M41 baseline used)
**Workload**: Llama 1B/3B/8B, 4w & 8da4w, 2048-token prefill (`p2048_exact.txt`, `--num_bos=1`) + 1024-token decode (`--ignore_eos --temperature=0 --warmup=true`)
**Clocks**: Floating only. Pinned clocks were **not attempted** — `power_dpm_force_performance_level` is readable but requires root to write, and `sudo` on `xraytracing02` needs a password (confirmed via `sudo -n true` → "a password is required"); this device joins the S25 Ultra (no root) and RDNA3-iGPU-miniPC (`NR`) precedent rather than getting a pinned column.

## Results

### 4w

| Model | Prefill tok/s (median ± CoV, n=3) | Decode tok/s (median ± CoV, n=3) | Crash notes |
|---|---|---|---|
| 1B | 4841.61 ± 0.71% | 276.786 ± 0.24% | None |
| 3B | 1865.21 ± 0.50% | 131.898 ± 0.08% | None |
| 8B | 867.43 ± 0.27% | 84.756 ± 0.27% | None |

### 8da4w

| Model | Prefill tok/s (median ± CoV, n=3) | Decode tok/s (median ± CoV, n=3) | Crash notes |
|---|---|---|---|
| 1B | 9225.23 ± 0.26% | 252.717 ± 0.11% | None (crash-wise) — **output degenerates into garbled sub-word tokens well before 1024 tokens; see Correctness note below** |
| 3B | 3764.71 ± 0.59% | 120.623 ± 0.11% | None (crash-wise) — same garbled-output caveat |
| 8B | 2021.72 ± 0.21% | 78.469 ± 0.11% | None; output stays coherent (see Correctness note) |

All 18 reps (6 cells × 3 reps) completed with `rc=0`. Zero crashes, zero retries, zero NR cells. Total
sweep wall time ≈ 6 minutes (15:54:08–16:00:10 local).

## Correctness note (not a crash, not a throughput issue)

At `--ignore_eos --temperature=0` over 1024 tokens, **4w stays coherent on all three model sizes** (decode
loops back into repeating the prompt, the expected greedy-decode behavior once the model runs out of new
things to say). **8da4w degenerates into genuinely garbled sub-word/foreign-character text on 1B and 3B**
(e.g. `...ange Hortonaguaails divisême Runner wards underside divid_CURgrades...`) — this is a worse
failure mode than the M41 report's 8da4w caveat (which stayed grammatical English, just repetitive/wrong);
here it's not real language at all. **8da4w/8B stays coherent**, matching 4w's behavior. This looks like it
could be a genuine numerical/dequantization issue specific to the 8da4w path on this GPU+driver for the
smaller models, not a benign decode-loop artifact — worth a follow-up investigation, but out of scope for
this baseline measurement. The throughput numbers above are still valid (the crash/timing harness only
checks process exit code and the presence of a `PyTorchObserver` stats line, not text quality), exactly
per the M41 report's convention for its own gibberish-output cells.

## Cross-device comparison

Sanity bound: the discrete 7900 XTX clearly outperforms the RDNA3 **integrated**-GPU miniPC's floating
numbers from the existing report (e.g. 1B/4w prefill 4841.6 vs 633.3, ~7.6×; 8B/4w decode 84.8 vs 10.1,
~8.4×) — expected for a 24GB discrete card vs an iGPU, and a basic check that these numbers aren't
nonsense before trusting them.

**Does 8da4w beat 4w here too?** Same pattern as every other device in this report: **prefill — yes, on
all three model sizes** (1B 9225 vs 4842, 3B 3765 vs 1865, 8B 2022 vs 867); **decode — no, 4w wins on all
three** (1B 276.8 vs 252.7, 3B 131.9 vs 120.6, 8B 84.8 vs 78.5). This mirrors M41's own split behavior
(8da4w's prefill win was never really universal — M41's decode numbers already favored 4w on every cell
too, once you check its raw table rather than just its prefill-focused Q&A answer).

## Validation

- **Crash attribution**: no crashes occurred; `dmesg` was checked for `amdgpu`/`reset` entries across the
  full sweep window regardless and came back clean, and the driver fingerprint (`vulkaninfo --summary`)
  was re-checked after the sweep and matches the pre-sweep value — no drift.
- **Coherence check**: all 6 `.pte` files passed a 48-token `"The capital of France is"` sanity check
  (coherent "...Paris..." output) before any timed rep was run.
- **DVFS check**: not applicable — no pinning was attempted, so there is no pinned/floating ratio to
  sanity-check; `power_dpm_force_performance_level` read back as `auto` throughout (never written).
- **CoV**: every cell ≤0.71% — no crash/reboot cycles to inflate variance, unlike a couple of the M5 EVT1
  cells in the existing report.

## Reproduce

- **Branch/commit**: `sarc-acl/executorch release/1.3` @ `e2f18eb23`, built natively as Linux x86_64 in
  the `release-1.3/executorch` worktree (this workspace host, not on `xraytracing02` itself — that host has
  no local scratch-disk convention set up, so only the resulting binary was shipped over shared NFS,
  mirroring how this host's own pre-pivot desktop-GPU work already did it).
- **Build**: two-step cmake, `--preset llm`, `-DEXECUTORCH_BUILD_VULKAN=ON`,
  `-DGLSLC_PATH=/local/yanwen.xu/vulkan-sdk/1.4.350.1/x86_64/bin/glslc`, `-DCMAKE_CXX_FLAGS="-include algorithm"`,
  `-DCMAKE_INSTALL_PREFIX=<worktree>/cmake-out-linux-vk`; step 2 builds `examples/models/llama`. Verified
  the resulting `llama_main` links `libvulkan.so.1` / calls `vkCreateInstance`.
- **Staging gotcha**: the runner's RUNPATH falls back to `$ORIGIN` for its `libllama_runner.so`. The shared
  NFS runners directory already had a same-named `.so` left over from an older, unrelated `main`-branch
  desktop-GPU build (2026-06-22) — copying only the new executable there would have silently linked it
  against that incompatible old library. Fixed by staging this build's executable and its own
  `libllama_runner.so` together in a dedicated subdirectory:
  `/sarc-c/gpusw/users/yanwen.xu/android-run/runners/rel1.3_linux_x86/` (verified by `ldd` + `md5sum`
  cross-check that `$ORIGIN` resolves to the matching pair, not the stray old file).
- **Device**: `ssh yanwen.xu@xraytracing02`; GPU `Radeon RX 7900 XTX`; driver `AMD open-source (RADV)
  2025.Q2.1 (LLPC)`, Vulkan 1.4.304.
- **`.pte`**: reused as-is from `/sarc-c/gpusw/users/yanwen.xu/android-run/models/` — the same 6
  release/1.3 `texture`/`ctx3072` files (1B/3B/8B × 4w/8da4w) the M5 EVT1/M41 baseline used. PTE format is
  architecture-agnostic; no fresh export was needed.
- **Command** (per rep):
  ```
  B=/sarc-c/gpusw/users/yanwen.xu/android-run/runners/rel1.3_linux_x86/llama_main
  D=/sarc-c/gpusw/users/yanwen.xu/android-run
  $B --model_path=$D/models/<model>_<quant>_texture_ctx3072.pte \
     --tokenizer_path=$D/assets/tokenizer.model --prompt_file=$D/assets/p2048_exact.txt \
     --num_bos=1 --max_new_tokens=1024 --ignore_eos --temperature=0 --warmup=true
  ```
- **Raw logs**: `/sarc-c/gpusw/users/yanwen.xu/android-run/results_rdna3_dgpu/*_rep{1,2,3}.log` (18 files,
  one per rep) plus the driver script `run_sweep.sh` used to produce them, all on shared NFS.
