# sarc-1.5-7900xtx-prefill-refine

Second-layer prefill tuning of the Radeon RX 7900 XTX (Navi31, gfx1100, AMDVLK 2025.Q2.1 / LLPC), in the 780M's position
(owner decision N1): port the 780M's and the RX 7600's second-layer results, gate each, stop when two consecutive gated
candidates gain under 2 %. Dev zone only. Branch `topic/7900xtx-prefill-refine`, forked from `origin/topic/780m-prefill-refine`
at `90fe4d013` (the parent). Development, git and builds are on the control workstation (`host-ws1`); the GPU host
(`<gpu-host>`) only runs staged binaries.

## Parent

`90fe4d013dd39a6fcf7d68fd5f8038de5324bca3` with `ET_VK_SARC_UNVERIFIED=1`, the AMDVLK ICD
(`VK_ICD_FILENAMES=/etc/vulkan/icd.d/amd_icd64.json`), no profile. The "7900 xtx" rows are already in
`impl/sarc/table_amd.cpp` (kUnverified): 4w `t256x128k32g24s32f32cbt`, 8da4w zpg `t128x64k32g42s32`, SDPA
`qk_coopmat_t128x64k32g22s64` / `av_coopmat_t64x64k32g22s64`, SARC truncated softmax. The release-zone hooks of D4 (softmax
variant name, fused attention node) are on the starting branch (`git grep softmax_variant` to be confirmed before the first
candidate).

## Thresholds (fixed 2026-10-08 before the first measurement; `tools/thresholds.txt`)

| threshold | value | source |
|---|---|---|
| noise band | a difference inside +-2 % is noise, not a gain | R6 |
| baseline tolerance | each of the six parent cells within 3 % of the published value (`sarc-1.5-e2e-benchmark/contrib/7900xtx/NOTES.md`, 2026-09-28: 4w 20078 / 10089 / 4774, 8da4w 22261 / 10396 / 4971 tok/s), else explained before anything is optimised | R4.4 |
| clock floor | 97 % of the lowest per-run median clock (sclk, prefill window) of the valid runs of the A/A session, one device-wide value; written into `thresholds.txt` once, before the first candidate | R6, L16 |
| clock samples | sampled every 5 ms (`sample_period_s`; measured interval on the GPU host 6.3 ms median, 7.1 ms max), at least 5 samples inside the prefill window (a 1B prefill is about 90 ms) | R6, L19 |
| thermal throttle (amended after the A/A, see Calibration) | `gpu_metrics` v1.3 `indep_throttle_status` bits 32 to 47 in the prefill window. The A/A session decides each temperature bit it shows, per cell: if the median clock of the A/A runs whose window carries the bit is within 1 % of that of the runs without it (or, where every run carries it, within 1 % of the other cells' A/A clocks), the bit is masked (recorded, not rejected); otherwise it rejects. A bit no A/A run shows rejects. Until the calibration is written, every bit rejects (`thermal_mask=0xffff`). Power and current bits are recorded, not rejected | R6, L16 |
| foreign GPU use | the card drives the host's display. (a) any other runner process of anyone (`llama_main`, `test_llama_micr`, `llama-*`, `vulkaninfo`, `igpu-roofline`, an `ollama runner`) or other holder of `/dev/dri/{renderD128,card1}` that `fuser` can see, before, during (0.5 s poll) or after a run, makes it invalid (the idle `ollama serve` is not a runner and is never stopped); (b) `gpu_busy_percent` is sampled 10 x 50 ms before every run (`busy_pre_max` column); the ceiling is the highest value among the valid timed A/A runs rounded up to a multiple of 5, at least 5, and a later run above it is invalid (`foreign_busy`). Other users' processes are invisible without root and no per-process engine accounting exists; (b) bounds them. Every row records the values | R6, L22 |
| host builds | nothing is built on the GPU host by this campaign; a compiler, linker or ninja process of anyone on it before, during or after a run makes the run invalid | R5 |
| repeats | 5 valid runs per arm and cell; 7 if any cell of the A/A session shows a repeat spread ((max - min) / median, first 7 valid runs) above 2 % in either arm | R6, L22 |
| kernel screen | a kernel replaces the incumbent for a shape only if at least 3 % faster in every round; a tie keeps the incumbent | R8 |
| stop rule | two consecutive gated candidates each under 2 % geometric mean over their parent | R11 |
| cooling | to idle + 5 C, or until the temperature (max of edge, junction, memory) has not fallen for 30 s, at most 300 s, before every run | R6, L20 |
| page cache | the GPU host has 123 GB RAM, the six models (14 GB) stay in the page cache; the model file is read once before each cell all the same (D5, both arms alike); the load time of every run is recorded | D5 |
| next-token items | D1 / D3 as written (near-tie evidence; reference-error rule for arithmetic changes) | owner |
| tok/s | `prefill_token_per_sec` of the runner (1 ms timer: one step is about 1 % for the 1B cells); ETDump dispatch time is reported beside it | L19 |

## Calibration and baseline (A/A session `aa`, 2026-10-08 17:36 to 18:16 UTC; `results/7900xtx/sessions/aa/`)

The parent build (`90fe4d013`, `ET_VK_SARC_UNVERIFIED=1`) in both arms, 7 repeats plus up to 3 extra pairs per cell (the session ran before its
own calibration was written: every temperature bit rejected and no clock floor was set, as announced above, so `runs.csv` of the session
carries `thermal_throttle` on every run; `tools/revalidate.py` re-judges the recorded runs under the final thresholds without editing them).
Values written into `tools/thresholds.txt` once, by the rules above (`tools/calibrate.py`, output `raw/calibration.txt`):

| threshold | value | derivation |
|---|---|---|
| clock floor | 2670 MHz | 97 % of the lowest per-run median clock of the 120 timed A/A runs (2752.5 MHz; range 2752.5 to 2901.5) |
| repeats | 7 | repeat spread above 2 % in at least one cell and arm (largest 5.95 %, 3B 4w parent) |
| foreign-busy ceiling | 5 % | highest pre-run `gpu_busy_percent` of the valid timed runs was 0 |
| thermal mask | `0xffef` (bit 36 masked) | **amended, see below** |

**Amendment of the thermal-mask rule (2026-10-08, after the A/A and before any candidate was measured; for the owner to ratify).**
The rule as fixed above, applied by `calibrate.py`, gives `thermal_mask=0xffff`: bit 36 of `indep_throttle_status` is in the window of all 120 timed
runs, there are no runs without it to compare with, and the fallback comparison (cell median clock against the other cells', all of which
carry the bit too) shows +2.19 % and -1.20 % on the two 1B cells, which is the workload dependence of the clock and not an effect of
the bit. Taken literally every later run would be invalid and no candidate could be measured. The sample-level comparison inside the
same runs shows what the bit is: in the prefill windows 4858 of 4937 samples (98.4 %) carry bit 36 and run at a median 2824 MHz (the
highest clock the card reaches); the 79 samples without it are window-edge ramp samples at a median 2126 MHz. Bit 36 is set whenever the
card is loaded (it also shows in about 31 % of all samples including idle ones, and at 44 C in the smoke test) and does not limit the clock. It
is therefore masked (recorded, not rejected); every other temperature bit (32 to 35, 37 to 47) still rejects, and real throttling is caught by the
clock floor. Nothing is lost: the `throttle` field of every run keeps the full word, so any session can be re-judged under the literal rule.
Every result of this campaign stands under this amendment until the owner decides (question in `STATUS.md`).

Baseline (parent arm of the A/A, median of the first 7 valid runs; published values from `contrib/7900xtx/NOTES.md`, 2026-09-28) and the A/A itself
(same binary in both arms):

| cell | published | measured | vs published | other arm | A/A ratio | repeat spread parent / cand |
|---|---:|---:|---:|---:|---:|---|
| 1B 4w | 20078 | 20277.20 | +0.99 % | 20277.20 | +0.00 % | 1.98 / 1.98 % |
| 1B 8da4w | 22261 | 22021.50 | -1.08 % | 22021.50 | +0.00 % | 3.26 / 3.26 % |
| 3B 4w | 10089 | 10138.60 | +0.49 % | 10039.20 | -0.98 % | 5.95 / 5.05 % |
| 3B 8da4w | 10396 | 10449.00 | +0.51 % | 10449.00 | +0.00 % | 5.16 / 3.03 % |
| 8B 4w | 4774 | 4785.05 | +0.23 % | 4773.89 | -0.23 % | 1.63 / 1.87 % |
| 8B 8da4w | 4971 | 4995.12 | +0.49 % | 4982.97 | -0.24 % | 1.69 / 0.73 % |

All six cells are within 3 % of the published numbers (largest +0.99 % / -1.08 %); the A/A geometric mean is -0.24 % (inside the +-2 % band; the
1B cells' identical medians come from the runner's 1 ms timer: 91 to 94 ms per prefill, one step is 1.1 %). Next token on the timed, the
real-text and the unaligned prompt is SAME in all six cells. The 24 untimed next-token runs (real, check) carry `clock_low` under the final floor: they
run cold without `--warmup` and their window is shorter; they are compared as text only. The prompt is the kit's `prompt_2048.txt`
(sha256 bfce65eb...), as in the published notes.

## Host and environment

- Control workstation (`host-ws1`): Ubuntu 22.04, native builds (`tools/build-native.sh`): host gcc, the toolchain Python 3.12.9
  (torch, yaml), Vulkan SDK 1.4.350.1 glslc; podman works on neither machine, so the shipped-SPIR-V golden is **pending**: the
  DIFF set of every build is compared with the DIFF set of the parent build (R5 and the lesson of the RX 7600 campaign).
- GPU host (`<gpu-host>`): Ubuntu 25.04, kernel 6.14, 123 GB RAM, RX 7900 XTX at `0000:03:00.0` = `card1`, AMDVLK 2025.Q2.1
  (LLPC) the default ICD. Governor `auto` (`power_dpm_force_performance_level`), as found, not changed; power cap 327 W
  (read 2026-10-08). Another user may log in to the host; `ollama serve` runs idle. Work only under `<gpu-root>`.
- Models: the six `*_embq_ctx3072.pte` of the GPU host's model directory (read-only, sha256 equal to its `sha256.txt` and
  the shared manifest), linked as the flat layout `verify.sh --flat-models` expects (`tools/sync-tools.sh`).

## Tools (copied from `sarc-1.5-rx7600-prefill-refine/tools/`, itself from the 780M's; originals untouched)

| tool | change against the RX 7600 copy |
|---|---|
| `env.sh` | runs on both machines (marker `ON_GPU_HOST`): artifact directory, sensors found by name on the GPU host (hwmon index changes with boots), the AMDVLK ICD, `ETVK_DEVICE_INDEX=0`, the lock name from `env.gpu`; host settings of the workstation from the uncommitted `.artifacts/env.local` |
| `rlib.sh`, `rjob.sh`, `sync-tools.sh`, `push-stage.sh` / `pull-stage.sh` (new) | the two-machine workflow: ssh through `bash -s`, detached jobs with a status file on the GPU host (`setsid nohup`, stdin/stdout detached), rsync of tools / stage directories, every command appended to the device command log |
| `sampler.py` | hwmon of the amdgpu found by name; 5 ms period |
| `others.sh` | `ollama serve` is not a foreign user (an `ollama runner` is); `igpu-roofline` is |
| `e2e5.sh` | `gpu_busy_percent` before each run (`busy_pre_max`, `foreign_busy`); sample period from `thresholds.txt`; host column is a placeholder |
| `hold.sh` | HOLD looked up in the artifact directory of the machine it runs on (workstation: `.artifacts/HOLD`; GPU host: `<gpu-root>/HOLD`) |
| `verify_stage.sh` | the unmodified `sarc/tools/verify.sh` copied by `sync-tools.sh` (sha256 logged); `HOME` set so that its lock path is the GPU host's lock file |
| `build-native.sh`, `build-both.sh`, `build_probe.sh`, `export_commit.sh` | toolchain from `env.local`; `nice -n 10`, 16 jobs (other work runs on the workstation); no `.building` guard on the GPU host |
| `calibrate.py` | also writes `busy_pre_max` |
| `thresholds.txt` | the table above |
| `chain0..7.sh` | not carried over (they hard-code the RX 7600 commits); this campaign writes its own queue scripts |

The other copied tools (search, screen and plot scripts) are unchanged and used only where named.

## Plan (`CAMPAIGN.md`, section 6)

1. Snapshot `s0-parent-verify`: unmodified `verify.sh` on the parent build, stored once.
2. Baseline and A/A: parent against the same binary, six cells; calibrate the clock floor, the thermal mask, the foreign-busy
   ceiling and the repeat count.
3. Locate: per-operator ETDump breakdown for the six cells, phase timing of the linear kernels.
4. Port list (each a gated candidate): softmax `r3` (D4.1 hook); fused attention kernel with the M2a fix (D4.3 hook); linear
   kernel per layer shape; whole-texel 8da4w weight staging only if the 8da4w linear kernels are bound on weight loads.
5. Further candidates only where step 3 points. No tile sweep first, no sampled search (N1).
6. Stop rule, final verification on the committed branch head, final session against the pristine parent (R11).

## Results

(none yet; A/A and baseline above)
