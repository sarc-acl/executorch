# Adding your GPU to the SARC 1.5 prefill reports

This is for the agents that own the **RX 7900 XTX**, the **Samsung Xclipse (M51)** device and the **Adreno 840 on
the Galaxy S26**. Five GPUs are already covered: Radeon 780M, Arc B580, Arc Pro B70, RTX 4070 Ti SUPER and
Jetson Orin Nano. You will produce the same measurements for your device, in the same format, so your results
can be merged into the existing reports. **Do not edit the reports themselves.** Deliver data plus notes; the
report owner merges them.

GPU ids to use everywhere, in directory names and in the CSV `gpu` column: `7900xtx`, `m51`, `s26`.

## 1. Read first (in this order)

1. `sarc-acl/CLAUDE.md`: the branch model (release 1.5 only, `dev/1.5` plus `topic/*` branches), clone
   layout and rules.
2. `dev/1.5:sarc/README.md`: the zones, the promotion checklist and `verify.sh`.
3. `dev/1.5:sarc/SWEEP-PARAMETERS.md`: the shader parameters.
4. This change: `openspec/changes/sarc-1.5-e2e-benchmark/`.
   - `REPORT.md` is the public benchmark: stock 1.5 vs SARC.
   - `MANAGER-REPORT.md` has **Part A**, the two-day re-tuning impact measured end to end, and **Part B**, the
     evidence (roofline, kernel traces, correctness, open questions).
   - `kit/` is the reference implementation of every measurement below.
5. `igpu-roofline/CLAUDE.md` and `gpu-lab/AGENTS.md`: device access, locks, and the measurement rules for
   your device class. The Mali/Adreno/Xclipse notes cover thermal behaviour.

## 2. Precondition: your kernels are on `dev/1.5` and verified

The SARC arm of every comparison must run **your device's rows on release 1.5**.
- If your work lives on a release-1.4 branch (a `release14-quant-shaders-<gpu>` branch), port it first:
  1. create a `topic/<gpu>-port` branch from `dev/1.5`;
  2. move your variants into the sweep yaml and then the release yaml;
  3. add or flip the rows in `impl/sarc/table_<vendor>.cpp`;
  4. extend `test_sarc_select`;
  5. run `verify.sh` (the Android variant for phones);
  6. update the SPIR-V golden;
  7. run `check.sh`;
  8. open a PR into `dev/1.5`.

  The procedure is in `sarc/README.md`, "Promotion checklist".
- The M51 and S26 rows already exist in `dev/1.5` as `kUnverified`. The 7900 XTX has none.
- Record the exact commit you measure as "SARC". Use a `sarc/1.5-rN` tag if one contains your rows; otherwise
  use your `topic/*` commit.

## 3. What to measure

Run everything under the gpu-lab lock, one measuring process per GPU:
- stop the device's co-tenant GPU services for the whole campaign, and restore them at the end;
- never delete results; move bad or interrupted runs to `superseded/<reason>/`;
- clocks stay as found; record them.

The Linux host scripts are in `kit/host/`. Phones: see §4.

### M1. Benchmark: stock release 1.5 vs SARC (required)

**Builds**
- **Stock:** `release/1.5` @ `985c1ceccc`, plus the compile-only backport `kit/patches/stock-backport-03f41d2031.patch` (the
  `SharedObject.cpp`/`Squeeze.cpp` `#include <algorithm>` patch from upstream 03f41d2031; it is also in
  `dev/1.5` as commit `158007c35`).
- **SARC:** your commit.
- Build both fresh:
  - x86: `sarc/tools/build.sh --llama --no-tests <tree> <out>`;
  - Android: NDK r26+.
- Make source trees with `kit/host/mktree.sh <commit> <dir>/executorch`. It writes a clean archive plus the
  submodules pinned by that commit.

**Grid**
- 3 models: Llama 3.2 1B, Llama 3.2 3B and Llama 3.1 8B. Use the `.pte` files at
  `/mnt/linux-share/models/<model>/exported/`. They are SHA-256 identical across all hosts, so record the hashes.
  If a model does not fit in memory, say so.
- 2 schemes: 4w and 8da4w.
- 2 builds.
- **5 interleaved repeats**: the two builds run back to back, stock→SARC on odd repeats and SARC→stock on even
  ones.

**Each run**
- A fresh process: `llama_main --prompt_file <prompt> --max_new_tokens 1 --temperature 0 --warmup`.
- Read `prefill_token_per_sec` from the PyTorchObserver line, and check `"prompt_tokens":2048`.

**Prompts** (both are required; they are in `kit/prompts/`)
- `prompt_2048.txt`: "the" × 2048, the original timed prompt.
- `prompt_real_2048.txt`: exactly 2048 real-text tokens (GPL-3.0 preamble), tile-aligned.

**Cool-down.** Before each run, wait until the GPU is within 5 °C of the idle baseline, at most 120 s. On phones,
use the thermal pacing from `igpu-roofline/CLAUDE.md` instead.

**Reference:** `kit/host/e2e.sh --gpu <id> --lock <uuid> [--prompt …] [--out …] [--no-check]`. Builds go in
`<stage>/stock/` and `<stage>/sarc/`. An optional `<build>/env` file holds per-build `KEY=VALUE` environment
lines.

### M2. Warm kernel traces (required)

- Use an ETDump-enabled build of each arm: add `--traced` to `build.sh`.
- Run one prefill per cell with `--warmup` and `--etdump_path` (`kit/host/trace2.sh`), then run
  `kit/analysis/trace_analysis.py`.
- It produces time per kernel family, per GEMM dispatch (with M, N, K) and per linear operator.
- Also list the dispatched kernel names with `kit/analysis/dispatch.py`. SARC must show your
  `sarc_linear_*` kernels, and stock must show none.

### M3. Roofline evidence (required)

- Run a confirmed igpu-roofline campaign on your device (`--plan fast` at minimum), or cite an existing confirmed
  one. Only confirmed roofs count.
- Report these roofs, each with its value, unit, REPORT.md path and line, and date:
  - `alu_fp16` (fp16 FMA);
  - `dot_int8` (int8 dot);
  - `matrix_fp16` and `matrix_fp16_fp32`;
  - `matrix_int8`;
  - `global_read` (DRAM);
  - the fed-matrix roofs, if measured.
- State which accumulator your SARC 4w kernel uses (the `ACC_FP32` or `ACC_GROUP_FP32` flag), so the right roof
  is matched.
- Then run `kit/analysis/efficiency.py` to get each kernel's rate and % of its matched roof.

### M4. Correctness via logits (required)

- Build `kit/logits_probe` (a small C++ tool) against each arm's install prefix. The x86 pattern is in its
  `CMakeLists.txt`; cross-compile it for Android with the NDK.
- Run it for every model × scheme × build on:
  - `real_ids.txt`: 2048 tokens, tile-aligned, so the SARC kernels are engaged;
  - `check_ids.txt`: 1972 tokens, unaligned. On 4w and 4h4w rows, SARC falls back to stock kernels there by
    design.
- Reference: `kit/host/probe.sh`; analysis: `kit/analysis/probe_analysis.py`.
- Report whether SARC and stock agree on the top-1 token, the largest logit difference and the smallest
  top-1 margin.

### M5. Before/after of your recent re-tuning (only if you re-tuned in the last days)

- Rebuild your **previous best** kernel commit and your **re-tuned** commit from the exact hashes (`mktree.sh`).
- Give the previous build the opt-in environment it needed in `stock/env`, e.g. `ET_VK_TEXTURE_COOPMAT=1`.
- Prove the opt-in mattered with `kit/host/optin_check.sh`: run the previous build on 1B with and without its
  env.
- Then run M1's protocol on `prompt_real_2048.txt`, with previous = `stock/` and re-tuned = `sarc/`.
- Also give the kernel-level before/after per shape from your study's microbench runs, with the same 3-repeat
  discipline. The format is `evidence/refinement/refinement.csv`.

## 4. Phones (M51, S26): same protocol, adb transport

- The kit scripts are bash for Linux GPU hosts. On Android, drive the **same protocol from the host**:
  - each run is one `adb shell 'cd <dir> && LD_LIBRARY_PATH=<dir> ./llama_main …'` invocation (a fresh
    process);
  - read the stats line from its output;
  - pace on the device's thermal zones over adb.
- Write exactly the CSV schema in §5.
- `build.sh --android` builds the backend and tests only. `kit/host/build-android.sh --tree <dir> --llama
  --etdump --probe` cross-compiles `llama_main`, the ETDump `llama_main` and the probe with the NDK (natively,
  no container). Record the commits and binary hashes.
- `kit/host/e2e-adb.sh --serial <serial> --gpu <id> --out <dir> [--mode e2e|trace|probe]` runs M1, M2 and M4
  over adb with the same order, CSV schema, retries and per-build `env` files as `e2e.sh`, `trace2.sh` and
  `probe.sh`. It paces on SKIN and the `gpuss-*` zones (`--pacing skin`, Adreno) or on one thermal zone
  relative to idle (`--pacing idle --temp-zone …`, M51), and `--guard-md5` / `--guard-value` stop the campaign
  if the driver or a clock pin changes. The device-dir layout is in its header.
- Heed the device notes in `igpu-roofline/CLAUDE.md`:
  - Mali: degraded-state latch; reboot to recover.
  - Adreno: stepwise throttling after 40–90 s.
  - Xclipse: clock pinning procedure only if the owner asks.
- Never mix device states within one comparison.
- If the 8B model does not fit, or a phone cannot sustain the run, report it as not measured. Do not
  substitute other data.

## 5. Deliverables

Commit on a `topic/<gpu>-e2e-benchmark` branch, and open a fork PR into `dev/1.5`. Put everything under
`openspec/changes/sarc-1.5-e2e-benchmark/contrib/<gpu>/`:

```
contrib/<gpu>/
  NOTES.md          device, driver, host/adb serial, OS, clocks and thermal policy, stock and SARC commits and
                    binary sha256, model .pte sha256, env per build, services stopped, anomalies (crashes,
                    rejected runs), what was NOT measured and why
  raw/runs.csv      M1 with prompt_2048.txt
  raw/nexttoken.csv (optional) 1972-token next-token comparison
  raw_real/runs.csv M1 with prompt_real_2048.txt
  raw_real/probe/   M4 JSONs: <model>-<scheme>-<build>-<real|check>.json
  trace/            M2 outputs of trace_analysis.py (families.csv, gemm.csv, totals.csv) and dispatch.csv
  roofline.json     M3, same structure as evidence/roofline.json → gpus.<gpu>
  efficiency.csv    M3 kernel rate and % of roof
  refine/runs.csv   M5 (if applicable), plus refine/optin.txt and refine/refinement.csv
```

- Keep raw logs and ETDumps in your own `.artifacts/` campaign directory, not in git, and give that path in
  `NOTES.md`.
- **`runs.csv` schema** (header exactly):
  `gpu,host,model,scheme,build,rep,slot,tok_s,rc,temp_pre,temp_post,cool_s,clocks,others,utc,log`
  - `model` ∈ {1b, 3b, 8b}; `scheme` ∈ {4w, 8da4w}; `build` ∈ {stock, sarc};
  - `rep`: 1–5 for timed runs, `Nx` for retries;
  - `log` starts with `logs/prefill-` for timed runs;
  - no commas inside fields (replace them with `;`);
  - a run is rejected (and kept) only if rc ≠ 0 or `tok_s` is missing; other GPU processes are recorded in
    `others`.
- Check your data before the PR: copy it into a scratch campaign root with `raw/<gpu>/runs.csv` and run
  `kit/analysis/analyze.py raw` there. It must produce your cells without errors.

## 6. Reporting rules

- Label every claim in `NOTES.md` with its evidence:
  - **[M]** measured this campaign;
  - **[R]** roofline;
  - **[S]** source;
  - **[I]** inference;
  - **[O]** open.
- Report crashes, rejected runs, noisy cells (repeat spread > 3 %) and anything not measured.
- The known pitfalls are already in `MANAGER-REPORT.md` Part B. Check yours:
  - the unaligned-M fallback;
  - timer resolution (1 ms) on fast runs;
  - co-resident GPU processes;
  - display-GPU noise;
  - exit-time crashes (seen on the 4070 Ti).
- Report to the owner in Chinese. Ask before pushing, before `standard` roofline plans on phones, and before
  anything that changes device state.
