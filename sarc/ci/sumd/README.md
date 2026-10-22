# SUMD performance gate (branch `ci/1.5`)

This branch pins the ExecuTorch side of a TeamCity job that guards the SUMD Vulkan driver (Samsung
userspace) against performance regressions. ExecuTorch, its binaries and the models are frozen here.
Each driver commit runs the same prefill benchmark, which is then compared against a stored baseline.

- ExecuTorch (SARC path): this branch, which is `dev/1.5` at `ee3a5ebb75` plus this directory. It
  includes the Xclipse (M51) kernels from 2026-09-28/30 that the release tag `sarc/1.5-r2` does not have.
- ExecuTorch (stock path): upstream `release/1.5` at `985c1ceccc` (`origin/release/1.5` in this fork).
- The job measures both builds. A driver change can slow down the SARC kernels, the stock kernels or both.

Do not move this branch with `dev/1.5`. A new ExecuTorch commit here changes the workload, so it needs
a deliberate rebaseline (see the last section).

## What runs

| | |
|---|---|
| Models | Llama 3.2 1B, Llama 3.2 3B, Llama 3.1 8B |
| Schemes | 4w and 8da4w |
| Workload | `llama_main`, 2048-token prompt (`prompt_2048.txt`), `--warmup`, 1 new token, temperature 0 |
| Runs | 5 reps per cell, stock/sarc alternating, a fresh process per run, thermal pacing before each run |
| Value | median prefill tok/s per (build, model, scheme): 12 cells |
| Correctness | next token on `prompt_check.txt`, sarc vs stock, per model/scheme |
| Env (sarc) | `env.sarc`: `ET_VK_SARC_UNVERIFIED=1`, `ET_VK_SARC_M51_PROFILE=c8` |
| Env (8B) | `env.8b`: `ET_VK_EXECUTE_NODE_THRESHOLD=32` on both builds (owner decision N10) |

`run.sh` drives `openspec/changes/sarc-1.5-e2e-benchmark/kit/host/e2e-adb.sh`, the adb harness of the
e2e benchmark. The protocol and the CSV schema are the same as in that study. During a run, a change of the
driver file's md5 stops the run.

## One-time setup (per device)

1. Build `llama_main` for arm64-v8a twice, once from each tree:
   ```bash
   KIT=openspec/changes/sarc-1.5-e2e-benchmark/kit/host
   $KIT/build-android.sh --tree <this branch>/executorch  --out out-sarc  --llama
   $KIT/build-android.sh --tree <release/1.5>/executorch --out out-stock --llama
   ```
   Keep both binaries as TeamCity artifacts and reuse them. Shaders are compiled into the binary at build
   time, so rebuilding with a different glslc changes the workload.
2. Export or copy the six models as `llama3_2-1b`, `llama3_2-3b` and `llama3_1-8b`, each with the suffixes
   `_vulkan_4w.pte` and `_vulkan_8da4w.pte`. Get the tokenizer as well.
3. Push everything to the device:
   ```bash
   sarc/ci/sumd/push.sh --serial <serial> --stock-bin out-stock/llama/examples/models/llama/llama_main \
     --sarc-bin out-sarc/llama/examples/models/llama/llama_main --models <dir> --tokenizer <tokenizer.model>
   ```
   `device-files.sha256` records what was pushed. Keep it with the baseline.
4. On the current SUMD driver, check correctness once. The M51 rows at this commit are still `kUnverified`
   in the fork. `sarc/tools/verify.sh --pdiff` (1B/3B/8B, 4w/8da4w) is the full check, but it runs on Linux
   GPU hosts, so on the phone it has to be driven over adb (see `sarc/README.md`). `run.sh` does a lighter
   check on every run: the next-token comparison.
5. Calibrate the noise: run `run.sh` twice on the same driver (an A/A run) and compare the two. Set
   `--cell-min` and `--geomean-min` above the noise you see.
6. Write the baseline:
   ```bash
   sarc/ci/sumd/run.sh --serial <serial> --out base-run --write-baseline baseline.json \
     --pacing idle --temp-zone /sys/class/thermal/thermal_zone<N>/temp
   ```

## TeamCity build step (per driver commit)

Flash or install the driver, then run:

```bash
sarc/ci/sumd/run.sh --serial <serial> --out run-$BUILD_NUMBER --baseline baseline.json \
  --pacing idle --temp-zone /sys/class/thermal/thermal_zone<N>/temp
```

- Exit status: 0 = pass, 1 = regression, 2 = bad input (a cell is missing or has fewer than 3 valid runs),
  3 = device or harness failure.
- The script prints TeamCity service messages: `buildStatisticValue` for each cell
  (`sumd.<build>.<model>.<scheme>.tok_s`) and for each build's geomean ratio, plus a `buildProblem` for each
  failed check. Chart the statistics to see trends.
- A regression is any of the following:
  - a cell below `--cell-min` × its baseline (default 0.95);
  - a build's geomean ratio below `--geomean-min` (default 0.97);
  - a next-token check that was `SAME` in the baseline and is not now.
- Artifacts to keep: `run-*/results.json`, `run-*/run-info.txt` (ExecuTorch commit, driver md5, build
  fingerprint), `run-*/{small,big}/run/runs.csv` and the logs.
- One full run is 120 timed runs (6 cells × 5 reps × 2 builds) plus 12 check runs. Each is a fresh process
  with thermal pacing first, and each campaign starts with a 60 s idle settle. Time the first run. If it is
  too long for every commit, use `--models 1b,3b --reps 3` per commit and run the full set nightly.

## Notes

- M51 numbers stay inside the company. Do not publish results or raw logs from this job.
- The board's timing sometimes switches between a fast and a slow state about 21 % apart, independent of
  the build. A single run can fail on that alone. Before treating a failure as a driver regression,
  rerun it and look at the per-run values in `runs.csv`.
- Rebaselining is a deliberate step. Do it after a driver change that was accepted as a speed-up or an
  expected slowdown, or after a change on this branch. Use `--write-baseline`, keep the old baseline file,
  and record why.
