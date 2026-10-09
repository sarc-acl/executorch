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
| cool start | the wait before a gate / timed session (`gate.sh`: at most 48 C, at most 1800 s) and the waits before screen, tier and trace jobs (55 / 60 C, at most 300 s) use the **core temperatures only: the maximum of edge and junction, not memory** (`gtemp_core` in `env.sh`; owner decision 2026-10-08 22:13 UTC, repeated 2026-10-09 00:56 UTC). The memory sensor reads 46 to 50 C at idle on this card and made every 48 C wait run to its cap. Committed on its own (`9a61374ed`), synced to the GPU host 2026-10-09 before the next gate. It changes only when a session starts, not which runs are valid; sessions measured before it (candidates 1 to 6, the final session) stand. The per-run cooling above is unchanged | owner |
| single-scheme candidates | a candidate that by construction changes only one scheme (only 4w or only 8da4w kernels) is judged on the cells it changes: adopted when each of those cells gains at least 2 % (outside the band), with next token and the gate as usual; when its six-cell geomean is under 2 % it still counts as one candidate under 2 % for the stop rule. Owner decision 2026-10-09 12:57 UTC, written here before it is used (2026-10-09, this commit); it does not reopen decisions already taken (candidate 6 changes both schemes and stays decided by its own rule) | owner |
| foreign GPU user, monitors | the rule above stays (owner decision 2026-10-09 00:56 UTC): no run is taken while another process holds the card; the queue waits and logs the wait (`foreign_wait_s`); finished runs are not invalidated by a monitor that appears later | owner |
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

**Amendment of the thermal-mask rule (2026-10-08, after the A/A and before any candidate was measured; **ratified by the owner 2026-10-08 19:30 UTC and again 2026-10-09 00:56 UTC**).**
The rule as fixed above, applied by `calibrate.py`, gives `thermal_mask=0xffff`: bit 36 of `indep_throttle_status` is in the window of all 120 timed
runs, there are no runs without it to compare with, and the fallback comparison (cell median clock against the other cells', all of which
carry the bit too) shows +2.19 % and -1.20 % on the two 1B cells, which is the workload dependence of the clock and not an effect of
the bit. Taken literally every later run would be invalid and no candidate could be measured. The sample-level comparison inside the
same runs shows what the bit is: in the prefill windows 4858 of 4937 samples (98.4 %) carry bit 36 and run at a median 2824 MHz (the
highest clock the card reaches); the 79 samples without it are window-edge ramp samples at a median 2126 MHz. Bit 36 is set whenever the
card is loaded (it also shows in about 31 % of all samples including idle ones, and at 44 C in the smoke test) and does not limit the clock. It
is therefore masked (recorded, not rejected); every other temperature bit (32 to 35, 37 to 47) still rejects, and real throttling is caught by the
clock floor. Nothing is lost: the `throttle` field of every run keeps the full word, so any session can be re-judged under the literal rule.
Owner decision 2026-10-08 19:30 UTC: bit 36 is masked (recorded per run, not rejecting); every other temperature bit still rejects; the clock floor (2670 MHz) stays; the raw status words stay in the raw files; the measurements made under the amended rule are valid.

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

## Adoption rule for candidate 6 (written 2026-10-09 04:00 UTC, before the data; the thresholds table above is unchanged)

Candidate 6 (`7900xtx-refine5`: candidate 5 plus attn*V `sweep_t32x32k32g22s32` for head dimension 64 only, i.e. the 1B cells) comes from the second attention screen, which found it 1.16x faster
than the incumbent on that op in every round. At the 1B model's attention share this is worth about 1.5 ms of 93 ms, below what a 7-repeat session resolves (repeat spreads of 2 to 8 % on the
1B cells), so a session cannot confirm a gain; it can only show harm. It enters the final stack if (a) its gate passes (SDPA tiers all / extended / full, 12 passes each, 0 mismatches,
`pairing=ok`; `verify.sh` as the snapshot except kernel-name lines; the SDPA output byte-identical to candidate 5's, else D3.1: error against the reference not larger), and (b) its geometric-mean change
over candidate 5 is not worse than -2 % and neither 1B cell is worse than -2 %. Otherwise candidate 5 is the final attention stack. A gain is not claimed for it unless it is outside +-2 %.
It counts toward the stop rule as a gated candidate under 2 % if its geomean gain is under 2 %.

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
| `chain0..7.sh` | not carried over (they hard-code the RX 7600 commits); this campaign writes its own queue scripts (`q-*.sh`: A/A + snapshot, candidate gates, screens, phase timing, roofs, final verification) |
| `e2e5.sh`, `gl.sh` (later change) | wait for a foreign GPU user to leave instead of invalidating runs / refusing jobs (an `amdgpu_top` of another session of this account and an `nvtop` held the card during the campaign); the wait is recorded (`foreign_wait_s`) |
| `revalidate.py`, `collect-results.sh`, `trace-analyze.sh`, `phases.sh` (two-machine), `spirv_same.sh` (toolchain python) | A/A re-judging under the final thresholds; scrubbed evidence copies; ETDump analysis on the workstation; phase timing run on the GPU host and decoded locally |
| `sdpa_screen.sh`, `sdpa_screen2.sh`, `sdpa_pick.py`, `q-screens.sh`, `q-screens2.sh`, `q-sdpa*.sh` (new) | kernel-level screens (fused variants, linear kernels, unfused attention kernels, exact-name attention variants) with the R8 rule applied by `fused_pick.py` / `screen_pick.py` / `sdpa_pick.py` |

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

Stop rule counter: candidate 1 (+1.56 %) and candidate 2 (-13.79 %) were two consecutive candidates under 2 %; candidate 3 (+2.64 %) is above, so the count starts again.

| # | candidate | parent | geomean | state | evidence |
|---|---|---|---:|---|---|
| 1 | softmax `r3` (the 780M's, `ET_VK_SARC_780M_PROFILE=c7`, D4.1 hook, on the parent binary) | pristine parent | +1.56 % | gated: `verify.sh` identical to `s0-parent-verify` (32 of 32 lines), SDPA tiers all / extended / full 12 passes each, 0 failed, 0 mismatches, `pairing=ok`, bit-identical to the release softmax in 21 of 21 SDPA cases, error against the reference not larger; next token SAME on the timed, real-text and unaligned prompts in all six cells | `results/7900xtx/sessions/c1-softmax/` |

| 2 | fused attention kernel `fused3sb` (`rk` variants picked by the fused-variant screen; D4.3 hook; build `c2`) | 1 | **-13.79 %** | **rejected on performance**; gate not completed (see below); next token SAME everywhere; the SDPA tiers that ran (all x 12, extended x 6) passed with 0 mismatches | `sessions/c2-fused`, `fused/` |

| 3 | linear kernel per layer shape (`7900xtx-refine2`, build `c3` = commit 773e306d8; the screen's picks by the 3 % rule, texel-wise family held out) | 1 | **+2.64 %** | gated: `verify.sh` as the snapshot except the two dispatched-kernel-name lines (30 of 32 identical), outputs byte-identical in 24 of 24 prefill linear shapes (no arithmetic change), next token SAME on the timed, real-text and unaligned prompts in all six cells, golden DIFF set equal to the parent build's | `sessions/c3-linear/`, `screens/` |

| 4 | whole-texel 8da4w staging (the texel-wise `zpg_bt` family on the 8da4w shapes where the screen has it best; `7900xtx-refine3`, same build `c3`) | 3 | -0.42 % | gated (verify.sh as the snapshot except the two kernel-name lines, outputs byte-identical to candidate 3 in 24 of 24 shapes, next token SAME), **not adopted** | `sessions/c4-texel/` |

| 5 | unfused attention kernels of the screen (`7900xtx-refine4`, build `c5` = commit 958bc07e7: QK^T `pk_t128x128k32g42s32nf`, attn*V `sweep_t64x64k32g42s32`, on every prefill call that fits) | 3 | **+4.09 %** | gated: SDPA tiers all / extended / full 12 passes each + 1 control pass each, 0 failed, 0 mismatches, `pairing=ok`; `verify.sh` as the snapshot except the two kernel-name lines; SDPA output **bit-identical** to candidate 3 in 21 of 21 cases and error against the reference not larger (D3.1); real-text probe 32 prompts x 6 cells: 0 top-1 differences, KL 0; next token SAME on all three prompts | `sessions/c5-sdpa/`, `sdpa-screen/` |

| 6 | attn*V `sweep_t32x32k32g22s32` for head dimension 64 (`7900xtx-refine5`, build `final` = commit a8dd09570) | 5 | +0.67 % | gated: tiers 39 passes 0 mismatches `pairing=ok`, `verify.sh` as the snapshot except the two kernel-name lines, SDPA output bit-identical to candidate 5 in 21 of 21 cases; **adopted** under the rule written before its session (geomean not worse than -2 %, neither 1B cell worse than -2 %; the gain is under the +-2 % band and is not claimed) | `sessions/c6-attn/`, `sdpa-screen2/` |

| 7 | the sweep 128 x 64 tile `t128x64k32g22s32` on 8B w2 instead of the release row (`7900xtx-refine6`, build `c10` = commit 5950764fa; the one pick of the first second-set screen, 1.034 worst round) | 6 | +1.16 % | gated (no SDPA tiers: no attention change): `verify.sh` as the snapshot except the two kernel-name lines, linear outputs byte-identical in 24 of 24 shapes, next token SAME; **not adopted** (single-scheme rule, owner decision 2026-10-09 12:57 UTC: its one changed cell gains +0.79 % < 2 %) | `sessions/c7-w2/`, `set2/` |

Candidate 1, per cell (median of 7 valid runs, `sessions/c1-softmax/raw/summary.csv`): 1B 4w 20078.40 -> 20686.90 (+3.03 %), 1B 8da4w 22260.90 -> 22755.60 (+2.22 %),
3B 4w 10138.60 -> 10240.00 (+1.00 %), 3B 8da4w 10449.00 -> 10502.60 (+0.51 %), 8B 4w 4762.79 -> 4841.61 (+1.65 %), 8B 8da4w 4982.97 -> 5031.94 (+0.98 %); geomean +1.56 %.
Where it came from (warm ETDump, ms per 2048-token prefill, parent -> candidate 1, `sessions/c1-softmax/trace/families.csv`): softmax 10.2 -> 8.2 (1B 4w),
11.8 -> 9.3 (3B 4w), 18.7 -> 15.2 (8B 4w); the other families unchanged within 1 ms.

Candidate 2, per cell against candidate 1 (median of 7 valid runs, 0 invalid runs, `sessions/c2-fused/raw/summary.csv`): 1B 4w 20480.00 -> 20480.00 (+0.00 %), 1B 8da4w 22755.60 -> 22755.60 (+0.00 %),
3B 4w 10138.60 -> 7907.34 (-22.01 %), 3B 8da4w 10449.00 -> 7937.98 (-24.03 %), 8B 4w 4785.05 -> 4055.45 (-15.25 %), 8B 8da4w 5019.61 -> 4104.21 (-18.24 %); geomean -13.79 %.
Why (warm ETDump, `sessions/c2-fused/trace/`, ms per 2048-token prefill, attention of candidate 1 = QK^T + softmax + AV against the fused node with its K / V copy pass):
1B 26.3 -> 28.3 (4w), 3B 41.3 -> 109.5, 8B 55.2 -> 152.2. The existing three-kernel coopmat attention costs 1.5 to 1.7 ms per layer on this card; the best fused variant needs 1.7 ms (d64) and 3.7 to 4.5 ms
(d128) per layer. The fused-variant screen (`fused/`) compared the fused variants with each other only and could not show this; on the RX 7600, where the unfused path is slow, the same kernel gained +18 %.
The rest of the gate (SDPA tiers beyond the passes above, `verify.sh`, the reference-error evidence and the real-text probe) was not run for a candidate that loses 13.8 %: it would only
re-check the correctness of a rejected kernel. Nothing of candidate 2 enters the final stack. The attention share left to gain after candidate 1 is small anyway (QK^T + AV about 20 % of the 1B prefill, 9 % of 8B).

Candidate 3, per cell against candidate 1 (median of 7 valid runs, 0 invalid runs, `sessions/c3-linear/raw/summary.csv`): 1B 4w 20277.20 -> 21113.40 (+4.12 %), 1B 8da4w 23011.20 -> 23814.00 (+3.49 %),
3B 4w 10088.70 -> 10138.60 (+0.49 %), 3B 8da4w 10556.70 -> 10951.90 (+3.74 %), 8B 4w 4818.82 -> 4830.19 (+0.24 %), 8B 8da4w 5044.33 -> 5237.85 (+3.84 %); geomean +2.64 %.
Where it came from (warm ETDump, GEMM family, ms per prefill, candidate 1 -> candidate 3, `sessions/c3-linear/trace/families.csv`): 1B 4w 61.6 -> 58.5, 1B 8da4w 48.9 -> 47.9, 3B 8da4w 129.3 -> 124.2,
8B 8da4w 307.1 -> 296.0; 3B and 8B 4w unchanged (every 4w shape there keeps the table kernel: no screened 4w kernel is 3 % faster in every round).
Dispatched kernels per shape: `sessions/c3-linear/linear-bitwise/kernels.txt` (8da4w: the 256 x 64 sweep tile on 6 shapes, the 128 x 64 sweep tile on the 8B wk_wv / wq_wo, the 780M's `afmb1` on 1B w2, the 64 x 64 tile on 1B wk_wv;
8B w2 and 3B wk_wv keep the table kernel; 4w: the 128 x 128 `cbt` sweep tile on the three 1B shapes).

Candidate 4, per cell against candidate 3 (median of 7 valid runs, 0 invalid, `sessions/c4-texel/raw/summary.csv`): 1B 4w 21787.20 -> 21333.30 (-2.08 %), 1B 8da4w 23540.20 -> 23540.20 (+0.00 %), 3B 4w 10088.70 -> 10088.70 (+0.00 %),
3B 8da4w 11070.30 -> 11070.30 (+0.00 %), 8B 4w 4818.82 -> 4785.05 (-0.70 %), 8B 8da4w 5278.35 -> 5291.99 (+0.26 %); geomean -0.42 %. The 4w picks of `refine2` and `refine3` are the same kernels, so the 4w cells are the same configuration in both arms:
their -2.08 % / -0.70 % / +0.00 % are the run-to-run noise of this session (repeat spread up to 7.4 % in the 1B 4w parent arm). The kernel-level edge of the texel-wise family over the other picks (0.1 to 3.5 % in the worst round, 1B
wk_wv the other way) does not show end to end: whole-texel staging is not adopted, as the phase timing (weight fetch 11 % of the wave) predicted.

Candidate 5, per cell against candidate 3 (median of 7 valid runs, 0 invalid, `sessions/c5-sdpa/raw/summary.csv`): 1B 4w 21787.20 -> 23011.20 (+5.62 %), 1B 8da4w 23814.00 -> 24674.70 (+3.61 %),
3B 4w 10138.60 -> 10556.70 (+4.12 %), 3B 8da4w 10893.60 -> 11570.60 (+6.21 %), 8B 4w 4807.51 -> 4923.08 (+2.40 %), 8B 8da4w 5224.49 -> 5361.26 (+2.62 %); geomean +4.09 %.
Where it came from (warm ETDump, ms per prefill, candidate 3 -> candidate 5, `sessions/c5-sdpa/trace/families.csv`): QK^T 6.4 -> 4.8 (1B), 17.7 -> 9.8 (3B), 21.0 -> 14.0 (8B); attn*V 11.8 -> 10.7, 14.4 -> 13.0, 19.4 -> 17.2; softmax and GEMM unchanged.
The screen (`sdpa-screen/`): QK^T without the mask fill and with packed K staging, in the 128 x 128 tile: 1.30x (head dim 64) and 1.45x (head dim 128) in the worst round; attn*V `sweep t64x64k32g42s32` 1.08x / 1.10x.
The "recommended" QK^T / attn*V pair is the largest QK^T tile that the dev zone had; the second attention screen below extends the tile family around it.

Candidate 6, per cell against candidate 5 (median of 7 valid runs, 0 invalid, `sessions/c6-attn/raw/summary.csv`): 1B 4w 21787.20 -> 22260.90 (+2.17 %), 1B 8da4w 24094.10 -> 24674.70 (+2.41 %),
3B 4w 10611.40 -> 10556.70 (-0.52 %), 3B 8da4w 11505.60 -> 11505.60 (+0.00 %), 8B 4w 4911.27 -> 4923.08 (+0.24 %), 8B 8da4w 5347.26 -> 5333.33 (-0.26 %); geomean +0.67 %. The kernel is selected only for head dimension 64,
so the four other cells run the same kernels in both arms: their -0.52 to +0.24 % are the noise of the session. Attn*V in the traces of the two 1B cells: 10.6 -> 8.8 ms (4w), 10.6 -> 9.0 ms (8da4w).

Candidate 7, per cell against candidate 6 (median of 7 valid runs, 0 invalid, `sessions/c7-w2/raw/summary.csv`): 1B 4w 23540.20 -> 24094.10 (+2.35 %), 1B 8da4w 25284.00 -> 25600.00 (+1.25 %), 3B 4w 10836.00 -> 10778.90 (-0.53 %),
3B 8da4w 11636.40 -> 11976.60 (+2.92 %), 8B 4w 4911.27 -> 4923.08 (+0.24 %), 8B 8da4w 5333.33 -> 5375.33 (+0.79 %); geomean +1.16 %. Only the 8B 8da4w cell runs a different kernel (8B w2 of 8da4w: the table row -> `sweep_t128x64k32g22s32`; the other 8da4w layers of 8B are the same in both arms);
the other five cells are the same configuration in both arms, so their -0.53 to +2.92 % (the parent arm of 1B 4w spreads 15 %) are the noise of this session, which exceeds the +-2 % band on two cells. The affected cell gains +0.79 % (trace, GEMM family of 8B 8da4w: 292.2 -> 290.3 ms), inside the band.
The pick passed the 3 % rule in one of three screens of the same pair (1.034; 1.026 and 1.024 in the others). Candidate 7 changes only 8da4w kernels (refine6 differs from refine5 only for N = 4096, K = 14336, i.e. 8B w2), so under the owner's adoption rule for single-scheme candidates (Thresholds, 2026-10-09 12:57 UTC)
it is judged on the cell it changes, 8B 8da4w: +0.79 % < 2 %, so it is **not adopted** (single-scheme rule, owner decision 2026-10-09 12:57 UTC); with a six-cell geomean of +1.16 % it counts as one candidate under 2 % for the stop rule. The final stack stays `7900xtx-refine5`.

**Phase timing (work-order step 3), 8da4w table kernel `t128x64k32g42s32`** (PROF twin `sarc_dev_prof_dq8ca_zpg_t128x64k32g42s32p`, shader clock, twelve prefill shapes, `results/7900xtx/phases/parent-8da4w.csv`),
share of the wave's cycles, median over the shapes (range over the shapes): barrier wait 31.4 % (18.0 to 33.7), LDS store 25.6 % (24.8 to 37.3), MMA 21.8 % (21.0 to 22.7), global weight fetch 10.9 % (10.4 to 14.5),
prologue 3.5 %, group epilogue 4.2 %, drain 0.9 %, write 0.6 %. The MMA is a fifth of the wave's time; staging into LDS and the barriers around it are over half; the weight fetch itself is 11 %: the kernel is not bound
on weight loads (the precondition the work order sets for whole-texel staging, candidate 4). No twin of the 4w table kernel (`t256x128k32g24s32f32cbt`) exists in the dev zone; a new one would be shader work and was not made.

**Stop rule, reading.** Candidates 1 (+1.56 %) and 2 (-13.79 %) are two consecutive candidates under 2 %, so the rule of R11 is met by its letter. The port list of the work order still has two items that nothing has tried
(the linear kernel per layer shape, and the whole-texel 8da4w staging), and the complete linear screens already show kernel-level gains of 4 to 40 % on single shapes. This campaign therefore continues with
item 3 (candidate 3) as a further gated candidate and reports the reading to the owner (`STATUS.md`); if candidate 3 gains under 2 % as well, the campaign stops there (it gained +2.64 %; the count restarted, candidate 4 is the first of a new pair).
Candidate 4 (whole-texel 8da4w staging, the texel-wise `zpg_bt` family): its precondition (weight-load bound) is not met by the phase timing above, but the 8da4w screen has the family at 1.05 to 1.07 against the table kernel
in the worst round on 11 of 12 shapes, so it is measured once as `7900xtx-refine3` against candidate 3 (`7900xtx-refine2`, the same picks with the texel-wise family held out) and adopted only by the rule of the thresholds table.

Locate (same traces, parent, share of the dispatch time): linear GEMMs 63 % / 71 % / 78 % (1B / 3B / 8B, 4w) and 56 % / 66 % / 75 % (8da4w); attention (QK^T + softmax + AV)
30.1 % / 22.2 % / 13.9 % (4w); everything else under 12 %. The fused attention kernel can therefore pay most on 1B and least on 8B, where the GEMMs decide.

## Second set, items 2 and 3: 8da4w and 4w linear kernels, per-K-step synchronisation (owner decision 2026-10-08 23:38 UTC, coordinator 2026-10-09 04:26 UTC; `results/7900xtx/set2/`)

Ordered before the final verification. The RX 7600's round-2 recipe (A staging row pitch of 24 bytes for 8da4w, 72-byte A / 88-byte B rows for 4w) and the owner's own levers (a branch-free chunk loop, stores interleaved with the MMAs, fewer waves, B staging layout)
were built as dev-zone variants under the `7900xtx` prefix, default off: copies of the release 8da4w zpg body and 4w body with options (`sarc_dev_7900xtx_{dq8ca_zpg,q4}_body.glslh`, reproducible from the release bodies by
`tools/gen_7900xtx_{dq,q4}_body.py`; variant tables `tools/gen_7900xtx_{dq,q4}.py`; 43 + 21 variants, measurement-only twins included). Read for races (pitch changes addresses only, every stage store keeps a unique address; the drain tile
in the A buffer is first written after a `memoryBarrierShared()` + `barrier()` that ends the last MMA reads and every drain iteration starts with that barrier pair). Nothing is selected unless named; the head build with nothing selected reproduces the parent
snapshot (32 of 32 `verify.sh` lines, `sessions/inert/`).

| step | result |
|---|---|
| phase timing first (shader clock, share of a wave's cycles, median over the twelve shapes) | 8da4w t256x64k32g24: barrier 31.5 %, LDS stores 17.4 %, MMA 34.0 %; with the 24-byte pitch barrier 31.6 %, LDS stores 31.4 % (barrier + LDS 48.9 -> 63.0 %: **up, not down**). 4w table kernel: barrier 51.8 %, LDS stores 15.9 %, MMA 28.6 %, fetch 1.9 % (same pattern, so the treatment applies) |
| screen, 8da4w (25 kernels, 3 rounds, incumbents = the profile's own kernel per shape, 3 % in every round) | **0 of 12 shapes for the pitch variants**: pitch 20 / 24 / 32 bytes 0.55 to 0.9x of the incumbent, controls (pitch 16) 0.98 to 1.00. The screen has one pick, which is not a pitch variant: the existing 128 x 64 tile `sweep_t128x64k32g22s32` on 8B w2, worst-round ratio **1.034, passing the 3 % rule** (1.026 and 1.024 in two other screens of the same pair). It is gated as candidate 7 below |
| screen, 8da4w synchronisation variants (branch-free loop, interleaved stores; 22 kernels) | **0 of 12 shapes** (medians 0.90 to 0.97) |
| screen, 4w (20 kernels: 72 / 88-byte rows, 16 waves, the RX 7600 `ap4bp12`) | **0 of 12 shapes** (0.42 to 1.00; control 0.986) |
| ceilings from ablation twins (measurement only: removed work, ratio incumbent / twin, median over the shapes) | 8da4w: no barrier 1.008, no stores of the next chunk 1.069, no staging 1.069, staging + barrier removed 1.101; 4w: no barrier 1.069 (up to 1.143 on one shape), no next-chunk stores 1.025, both 1.090 |

So the barrier of the chunk loop is worth under 1 % of the 8da4w kernel and the whole staging at most 7 to 10 %; the 31 % (8da4w) and 52 % (4w) barrier shares of the phase timing are time a wave waits while the other waves of the workgroup work, which removing the barrier does not recover.
A tile of K step 64 for 4w would halve the barriers but was not tried: the release table kernel already declares 61,440 bytes of workgroup memory in its SPIR-V (see "Known risk" below) where the device query reports 32,768, and a K64 tile needs more.
**No variant passes the pre-registered screen rule, so there is nothing to gate: items 2 and 3 end as negative results with measured ceilings.** The stop rule cannot count candidates that do not exist; it applies to this set as "no gated candidate" (see the stop-rule paragraph below).
The RX 7600's +5 % from the 24-byte pitch does not carry over to this card (cause not investigated; UNVERIFIED).

## Final stack against the pristine parent, on the build of the committed head (2026-10-09, `sessions/final3/`; supersedes `sessions/final2/` on build c9 and `sessions/final/`)

Build `c10` = commit `5950764fa`, exported from the git object stores with pinned submodules, built natively on the scratch disk, no local patch. It is the code of the committed head: `git diff --name-only 5950764fa HEAD -- . ':!openspec'` is empty (the later commits are evidence, documents and tools).
The head contains the second-set variants and the `refine6` profile, all inert unless named: with nothing selected the build reproduces the parent snapshot (`sessions/inert3/`, 32 of 32 `verify.sh` lines). Native build, shipped-SPIR-V golden pending as in the sibling campaigns: 14 of 53 variants differ
from `sarc/golden/spirv.json` (owners 780M / Arc), the same 14 as in the parent build, 0 differ from the parent build's own (`results/7900xtx/golden-diff-parent.txt`, `sessions/final3/golden-c10.txt`); the dispatched kernel set of the final stack on c10 equals that of the earlier session on c9
(139 (model, scheme, kernel, count) rows in `kernels.csv`, identical). Pristine parent = build `parent` (commit `90fe4d013`), `ET_VK_SARC_UNVERIFIED=1` only. One timed session, both arms interleaved, 7 valid runs per arm and cell (84 of 84 timed runs valid; the 24 untimed next-token rows of the file hold 23 invalid runs, all `clock_low`:
cold runs without `--warmup`, compared as text only, kept in `runs.csv`; every observer value in the 84 timed logs equal to `runs.csv`, prompt 2048 tokens, 0 generated; no foreign GPU user, `foreign_wait_s` at most 1). The cool-start wait used the core temperature (`gtemp_core`).

| cell | pristine parent tok/s | final stack tok/s | gain | published 2026-09-28 | parent vs published |
|---|---:|---:|---:|---:|---:|
| 1B 4w | 20480.00 | 23011.20 | +12.36 % | 20078 | +2.00 % |
| 1B 8da4w | 22021.50 | 24381.00 | +10.71 % | 22261 | -1.08 % |
| 3B 4w | 10138.60 | 10666.70 | +5.21 % | 10089 | +0.49 % |
| 3B 8da4w | 10502.60 | 11770.10 | +12.07 % | 10396 | +1.03 % |
| 8B 4w | 4762.79 | 4899.52 | +2.87 % | 4774 | -0.23 % |
| 8B 8da4w | 5019.61 | 5375.33 | +7.09 % | 4971 | +0.98 % |

Geometric mean **+8.33 %** (min +2.87 %, max +12.36 %; every cell outside the +-2 % band). Caveat: per-cell gains of a few percent, e.g. 8B 4w +2.87 %, are of the size seen as noise between identical kernels in this setup: in candidate 7's session (`sessions/c7-w2/`) the 1B 4w and 3B 8da4w cells ran the same kernels in both arms and read +2.35 % and +2.92 %. The thresholds stay as fixed; the large gains are not in doubt, the small ones are indicative only. Earlier sessions of the same stack: +8.24 % on build `c9` (`sessions/final2/`) and +7.91 % on build `final` (`sessions/final/`, before the second set): same shipped paths, the differences are session noise
(repeat spreads of 1 to 6 % in the parent arms; candidate 7's session shows two cells with identical kernels in both arms moving +2.4 and +2.9 %). The N1 expectation was +20 to +30 %; this card lands below it (what limits it: below). The stages measured one by one multiply to +9.2 % (candidates 1, 3, 5, 6, each in its own session); the single final session is the figure of record.

**Recommended configuration** (all committed code): `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_7900XTX_PROFILE=7900xtx-refine5` with the AMDVLK ICD (`VK_ICD_FILENAMES=/etc/vulkan/icd.d/amd_icd64.json`). The profile selects the softmax variant `780m_r3` (candidate 1), the linear kernel per layer
shape of the screens (candidate 3), QK^T `pk_t128x128k32g42s32nf` and attn*V `sweep_t64x64k32g42s32` (candidate 5) and attn*V `sweep_t32x32k32g22s32` for head dimension 64 (candidate 6). Candidate 7's `7900xtx-refine6` is not part of it.
No release-zone file changed (the D4 hooks were already on the starting branch): `git diff --name-status 90fe4d013 HEAD` outside the change directory lists only dev-zone files (`glsl/sarc_dev/` and `impl/sarc_dev/`: the `fused3sb` pair, the 7900xtx linear bodies, wrappers, yaml and row files,
appended blocks in `sarc_sdpa_av_coopmat_sweep.yaml`, `sarc_sdpa_qk_coopmat_pk.yaml` and `sarc_dev_prof_q4gsw.yaml`, and the `7900xtx` block of `Overrides.cpp`; `sessions/final3/files-outside-change-dir.txt`); nothing under `sarc/tools`, `sarc/golden`, no tolerance, prompt or
threshold file changed (`tools/thresholds.txt` only by the calibration commit).

**Final verification of everything together** (all on the build `c10`, `sessions/final3/`, `sessions/inert3/`):
- unmodified `verify.sh --models 1b,3b,8b --schemes 4w,8da4w --pdiff` on the timed binaries and environment against `s0-parent-verify`: 30 of 32 lines identical; the two that differ are the dispatched-kernel-name lines of the `linear 4w` and `linear 8da4w` microbench (candidate 3's picks); every correctness, production-diff,
  default-vs-tiled and decode line is identical, including the parent's own non-pass lines (`correctness rc=1`, 4w buffer production-diff FAILED). With the parent's environment only (nothing selected): 32 of 32 identical;
- SDPA tiers all / extended / full: 12 passes each with the final environment and 1 control pass each: 39 runs, 0 failed cases, 0 mismatches, `pairing=ok` everywhere;
- SDPA output of the final stack against the parent configuration (no profile) on the same c10 binary and the same inputs (`tools/sdpa_evidence.sh` runs the stage's one `test_llama_microbench` for both arms; the arms differ only by environment): byte-identical in 21 of 21 cases and rms / maximum error against the fp64 reference not larger (`sdpa-error/error.csv`), so the stack does not change the arithmetic (D3.1 holds with equality). The parent configuration (`ET_VK_SARC_UNVERIFIED=1`, no profile) on the same c10 binary stands for the pristine parent because (i) `inert3` (`verify.sh` on c10 with nothing selected) is identical to `s0-parent-verify` in 32 of 32 lines and (ii) all 53 shipped SPIR-V variants are byte-identical between build c10 and build `parent` (the reviewer checked it with `cmp`). Nothing more is claimed: these two comparisons were not run on the pristine parent's own binaries;
- every prefill linear output byte-identical to the parent's in 24 of 24 shapes (candidate 3's check; candidates 5 and 6 do not touch linear kernels);
- real-text probe, 32 prompts x 6 cells (`tools/probe_run.sh` uses the stage's one `lp` binary, built from c10, for both arms; the arms differ only by environment): final-default against parent-configuration-default: 0 top-1 differences, mean and maximum KL 0, maximum logit difference 0;
- next token SAME parent against final on the timed, the real-text and the unaligned prompt in all six cells;
- `sarc/tools/check.sh --no-build` on the head: PASS (`sessions/final3/check-no-build.txt`); the golden step is the pending native-glslc comparison above.

**Where each gain came from** (warm ETDump, ms per 2048-token prefill, pristine parent -> final stack; GEMM = prefill linear layers; the GEMM rate is the linear-layer FLOPs of the prefill, 2 x 2048 x sum(N x K) = 3.99 / 11.54 / 28.59 TFLOP for 1B / 3B / 8B, over the traced GEMM time; percent of the freshly measured roofs, `results/7900xtx/roofline/`:
matrix fp16 with fp32 accumulate 140.83 TFLOP/s for the 4w kernels, matrix int8 141.60 TOP/s for the 8da4w kernels; in brackets the parent's):

| cell | QK^T | softmax | attn*V | GEMM | everything else | dispatch total | GEMM rate, final | % of roof (parent) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1B 4w | 6.2 -> 4.7 | 10.3 -> 7.6 | 12.4 -> 8.8 | 59.9 -> 57.0 | 6.4 -> 6.5 | 95.3 -> 84.7 | 69.9 TFLOP/s | 50 % (47 %) |
| 1B 8da4w | 6.5 -> 4.9 | 10.1 -> 7.6 | 12.6 -> 9.0 | 48.2 -> 47.4 | 9.4 -> 9.5 | 86.8 -> 78.5 | 84.0 TOP/s | 59 % (58 %) |
| 3B 4w | 17.2 -> 9.6 | 11.8 -> 9.3 | 14.9 -> 12.9 | 137.9 -> 139.5 | 14.5 -> 14.6 | 196.4 -> 186.0 | 82.7 TFLOP/s | 59 % (59 %) |
| 3B 8da4w | 17.6 -> 9.6 | 11.8 -> 9.3 | 15.0 -> 13.0 | 125.5 -> 120.5 | 20.6 -> 20.5 | 190.5 -> 172.9 | 95.8 TOP/s | 68 % (65 %) |
| 8B 4w | 20.7 -> 13.9 | 18.8 -> 15.1 | 19.8 -> 17.3 | 332.7 -> 332.8 | 32.4 -> 32.4 | 424.4 -> 411.5 | 85.9 TFLOP/s | 61 % (61 %) |
| 8B 8da4w | 21.0 -> 13.8 | 18.9 -> 14.9 | 19.9 -> 17.4 | 306.0 -> 293.3 | 41.3 -> 41.1 | 407.2 -> 380.6 | 97.5 TOP/s | 69 % (66 %) |

Softmax `r3` (candidate 1) takes 2.5 to 4.0 ms off; QK^T without the never-read mask fill and with packed staging (candidate 5) 1.5 to 8.0 ms; attn*V 2.0 to 3.6 ms (candidate 5 on all cells, candidate 6 on 1B); the linear kernels per shape (candidate 3) 0.8 to 12.7 ms on the 1B 4w cell and the three 8da4w cells, and nothing on 3B and 8B 4w
(their GEMM family differs by +1.6 and +0.1 ms, within the +-3 ms by which the family time varies between traces of the same configuration). The ETDump sums are single warm executions; the tok/s medians of the timed session are the figures of record.

**Negative results** (all with numbers above): the fused attention kernel `fused3sb` (-13.79 %; the existing three-kernel coopmat attention is 2.6x faster than the best fused variant on 3B / 8B; the kernel itself is correct on AMDVLK: 12 + 6 tier passes, 0 mismatches before its gate was stopped); whole-texel 8da4w staging (-0.42 %);
every 4w kernel of the screens on 3B and 8B (none 3 % faster in every round); the wider QK^T grids and the large attn*V tiles of the second attention screen (0.62 to 1.02x); the online-softmax fused variants (1.3 to 2.6x slower than the two-pass ones); and the whole second set: the RX 7600's padded staging rows, the branch-free chunk loop and the interleaved stores
(0 of 12 shapes in each screen; ceilings in the second-set section). Process notes: the first fused-kernel guard stopped its queue on a wrong grep pattern (a tool bug, corrected); jobs were lost to foreign GPU users and re-run; one attention variant (`t32x32k32g41s32`) did not compile and was removed; the first final verification ran before the second set was done
(an ordering mistake, corrected by this redo).

**What limits further progress.** The linear GEMMs are 60 to 81 % of the final prefill and run at 50 to 69 % of the matrix roofs. On 4w nothing in the dev zone is 3 % faster than the table kernel on 3B and 8B. For 8da4w the ablation twins say the kernel is MMA-loop bound, not synchronisation bound: with the
barrier removed it gains 0.8 %, with all staging removed 7 to 10 %; the MMA loop is about 80 % of the kernel (without it the incumbent tile is 5.3x faster) and the kernel as a whole runs at 59 to 69 % of the int8 roof (the RX 7600's round 2 put the ceiling of this loop structure near 78 % of its roof with staging and barrier removed: cited from its README, not measured here). A faster GEMM would need a different MMA-loop structure (fragment reuse across K slabs,
register-resident B fragments), i.e. a new kernel family with its own gates; the owner decisions leave kernel redesign out of a port campaign unless ordered. After the attention picks, attention is 11 to 25 % of the prefill and every named kernel of the dev zone has been screened. The elementwise and copy kernels are about 7 % of the 8B prefill and sit outside the dev zone's kernels.

**Known risk: workgroup memory above the device limit (reviewer's finding, verified from the SPIR-V of build `c9`).** AMDVLK reports `maxComputeSharedMemorySize` = 32,768 bytes (the microbench's `max_shared_mem_bytes`), but the shaders declare more workgroup memory than that:
QK^T `sarc_sdpa_qk_coopmat_pk_t128x128k32g42s32nf` (every QK^T of the final stack) 53,248 bytes, the 780M's `afmb1` 8da4w kernel (1B w2 in the final stack) 60,160, the 4w `sweep_t128x128k32g42s32f32cbt` (1B 4w picks) 40,960, and the parent's own release 4w row
`t256x128k32g24s32f32cbt` 61,440. The overrun is pre-existing (the parent's table row is the largest) and not a regression of this campaign; the driver accepts these pipelines and every correctness gate passes, but it breaks a Vulkan valid-usage rule and `Select.cpp` checks shared memory
only for texture3d linear rows, not for attention rows. No kernel was changed for this. Whether to accept it on AMDVLK or to require in-limit kernels before any promotion is the owner's decision (`STATUS.md`).

**Stop rule.** Candidates 1 and 2 were two consecutive ones under 2 % (the second not fully gated: rejected on performance after its timed session); candidate 3 (+2.64 %) restarted the count; candidates 4 (-0.42 %) and 5 (+4.09 %) are not both under 2 %;
**candidate 6 (+0.67 %, inside the +-2 % noise band: adopted by the rule written before its session, not claimed as a gain) and candidate 7 (+1.16 % geomean; +0.79 % on its one changed cell; gated, not adopted under the single-scheme rule) are two consecutive gated candidates under 2 %: the stop rule of R11 / section 9 is met.**
The second set (items 2 and 3) produced no candidate of its own: every variant failed the screen rule that precedes a gate (the one pick of its screens became candidate 7).
