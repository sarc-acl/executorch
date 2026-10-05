# STATUS: sarc-1.5-orin-prefill-refine

**2026-10-05 05:10 UTC. RUNNING: SDPA screen on the device. Baseline, A/A, parent control, roofs and traces done. No candidate yet.**

(The first version of this file carried a wrong clock time; all times here are UTC from `date -u`.)

## Running now

- Device (`duck-naughty`), detached, one GPU job at a time under the gpu-lab lock; status in
  `~/hmz-sarc-orin/jobs/<job>.status` / `.out`, from the workstation `tools/dstat.sh`. These survive a reboot of
  the workstation:
  - `chain2b` (`tools/chain2.sh`): SDPA screen 1 (59 `orin-qk-*` / `orin-av-*` profiles + stock, 1 round, about
    1.5 min per profile), then the 8da4w phase timing. Its earlier steps (session `s1-aa`, traces) are done.
  - `chain3`, waiting: kernel screen of the 4070 Ti campaign's 8da4w dev tiles (13 sweep tiles, 4 half-texel
    `bh` tiles) on the Orin, then the stock arm of the SDPA reference error.
  - `chain4`, waiting: build `topic2`: production-diff and a 2-round screen of the Orin whole-texel 8da4w
    twins (`orin_bf*`, see "8da4w" below).
- Workstation: nothing.

## Next step

Read SDPA screen 1, pick the kernel per head_dim (`tools/orin_refine.py` -> profile `orin-refine1`), build,
measure the reference error of both arms, run `gate_sdpa.sh`.

## 8da4w: what the kernel does with its weight fetches (hypothesis, being measured)

The shipped `t128x128k64g44s32mk32ra` stages the weights with 2 `texelFetch` per thread and chunk and keeps one
32-bit word of each fetched 4-word texel: every packed-weight texel is fetched four times per chunk, by four
threads, through the texture2d path, the slowest read path of this device (roofs above). For the 1B `w1` shape
that is 33.5 million texel fetches per dispatch. `tools/gen_orin_bf.py` generates a twin of the release body in
which a slot is a whole texel (one fetch, eight shared-memory words; same values, same shared-memory layout,
same MMA order): `orin_bf_*` with two staging slices as shipped, `orin_bf1_*` with one slice and K = 128 per
chunk (one texel per thread for 512 threads). 9 tiles, build `topic2` (`dcd7cbe88`), compiled, not measured yet.

## Baseline and A/A (session `s1-aa`, pristine parent against build `topic1` with no environment)

tok/s, median of 5 valid runs per arm, arms interleaved, record-only clock (`results/orin/sessions/s1-aa/`):

| cell | parent | topic1, no env | ratio | `cells.csv` (dev/1.5) | parent vs `cells.csv` |
|---|---:|---:|---:|---:|---:|
| 1B 4w | 890.05 | 890.82 | 1.0009 | 890.82 | -0.09 % |
| 1B 8da4w | 823.48 | 823.48 | 1.0000 | 822.82 | +0.08 % |
| 3B 4w | 360.44 | 360.37 | 0.9998 | 360.37 | +0.02 % |
| 3B 8da4w | 320.30 | 320.25 | 0.9998 | 320.30 | 0.00 % |
| 8B 4w | 189.68 | 189.72 | 1.0002 | 189.74 | -0.03 % |
| 8B 8da4w | 170.47 | 170.41 | 0.9997 | 170.43 | +0.02 % |

The baseline agrees with `cells.csv` within 0.1 % in every cell (threshold 3 %). A/A geomean +0.01 %, largest
cell difference 0.09 %, repeat spread at most 0.36 %: the noise on this device is far inside the +-2 % band.
60 timed runs, all valid; next token parent vs topic SAME in all six cells on the four prompts (24 of 24).
`gate_check.py session --calibration --require-logs`: ACCEPT, 0 findings.

Clock: the devfreq clock reads 612 MHz (the 15 W cap) as the median of every timed run.
`results/orin/clkmin.json`: device-wide threshold floor(0.97 x 612) = **593 MHz**; no run below it.
Temperature 60 to 65 C at run start (idle 60 C; the fan keeps it there). Memory: 5.6 to 6.6 GB available
before every run; 13 of the 60 timed runs paged something out while they ran (at most 202 pages = 0.8 MB,
3.9 MB in total), with no visible effect on the rates.

## Parent control (`s0-parent-verify`)

Unmodified `sarc/tools/verify.sh --models 1b,3b,8b --schemes 4w,8da4w --pdiff --flat-models`, no environment,
22 of 22 runner calls rc 0, default vs tiled SAME on the check and the unaligned prompt for 1B 4w and 8da4w,
decode 31 tokens. What the pristine parent itself shows on this device that a naive checker calls a failure
(all listed as `PARENT-STATUS` in `results/orin/sessions/s0-parent-verify/verify-check.txt`; the same lines are
in the Orin evidence of `sarc-1.5-4w-port` and `sarc-1.5-8da4w-port`):

| line | parent | why |
|---|---|---|
| `correctness rc=1` | 3 upstream cases FAILED (`linear_q4gsw_M128_K4096_N128`, texture3d / buffer / rank-3 buffer), 41 PASSED | the Orin rows cover only the measured M = 2048 projections; M = 128 runs the upstream fp16 kernel, which fails at K = 4096 |
| `linear 4w rc=1`, `linear 8da4w rc=1` | 24 `unexpected_coopmat` + 24 `fallback_tiled` | texture3d coopmat dispatch reported as unexpected (as everywhere); buffer IO has no Orin row |
| `pdiff <model> <scheme> buffer rc=1`, 6 of 6 | FAILED | buffer IO has no Orin row: "NOT coopmat -- fallback, cannot validate the shader under test"; the upstream 4w buffer kernel also fails numerically at K >= 4096 (5 shapes) |
| `pdiff <model> <scheme> texture3d`, 6 of 6 | ALL PASSED | the model path |

`gate_check.py verify` first rejected the control because it required every correctness and production-diff
case to pass. It now records the status of every correctness case and of every production-diff shape (coopmat
or fallback, PASSED / FAILED / threw, rc, final line) and requires a candidate to EQUAL the control per case;
where the parent passes, the candidate must pass. Re-run on the same files: ACCEPT, 0 findings; the first
verdict is kept as `gate.done.first-check`. 41 unit tests.
SDPA correctness on the parent (recorded only): 0 mismatches in 8 + 4 cases, upstream kernels (`qk_coopmat=NO`).

## Where the time goes (parent, warm ETDump, ms per 2048-token prefill; `results/orin/sessions/s1-aa/trace/`)

| family | 1B 4w | 1B 8da4w | 3B 4w | 3B 8da4w | 8B 4w | 8B 8da4w |
|---|---:|---:|---:|---:|---:|---:|
| linear GEMM | 667 (29 %) | 871 (35 %) | 1895 (33 %) | 2577 (40 %) | 4863 (45 %) | 6054 (50 %) |
| QK^T (stock) | 581 | 581 | 1488 | 1488 | 2268 | 2268 |
| attn*V (stock) | 458 | 457 | 1194 | 1194 | 1817 | 1816 |
| softmax (stock) | 177 | 177 | 231 | 231 | 351 | 351 |
| attention total | 1215 (53 %) | 1215 (49 %) | 2913 (51 %) | 2912 (46 %) | 4436 (41 %) | 4435 (37 %) |
| copy / view / other | 267 | 154 | 581 | 351 | 985 | 574 |
| elementwise | 101 | 101 | 190 | 190 | 363 | 377 |
| 8-bit quantize | - | 101 | - | 254 | - | 425 |
| total dispatch | 2285 | 2477 | 5666 | 6373 | 10773 | 11992 |

Attention, all stock kernels, is half of the 1B and 3B prefill and 37 to 41 % of the 8B one.
Linear kernels in the model against the fresh roofs: 4w `t256x128k16g22s32` 5.74 to 6.21 TFLOP/s = 59 to 64 %
of the fp16 matrix roof (9.716); 4w `t128x128k32g42s32f32` (8B, K = 14336) 5.23 = 54 % of the fp16 -> fp32 roof
(9.722); 8da4w zpgtr 4.39 to 4.85 TOP/s = **22.5 to 24.9 %** of the int8 matrix roof (19.482). 8da4w linear
is 30 % slower than 4w linear in every model, on a device whose int8 matrix rate is twice its fp16 rate.

## Done so far

- Builds (cross image `localhost/et-jetson-cross:jp7.2.1`, GCC 13.3, shaderc v2026.1, 8 jobs, under the desktop
  build lock): `parent` = pristine `6a7cc8cc6`; `topic1` = `1d827a139` (dev zone only: `OrinSdpa.cpp`, the
  `orin-*` profiles). Provenance `.artifacts/orin-prefill-refine/build/<tag>.src.txt`.
- Shipped SPIR-V (`tools/shipped.py`, `build/topic1.shipped.txt`): all 53 shipped variants byte-identical
  between `parent` and `topic1`. Against the golden, 14 variants differ in BOTH builds (the cross image's glslc
  is not the one the goldens were made with; the e2e report documents the same for 11 of 48); none belongs to
  the Orin rows, whose variants all match the golden.
- Device capabilities (`results/orin/vk-caps.txt`): subgroup size 32 only (min = max = 32), shared memory
  49152 bytes, cooperative matrix fp16 16x16x16 / 16x8x16 / 16x8x8 with fp16 or fp32 result, int8 16x16x32 /
  16x8x32. The same shapes the 4070 Ti SDPA kernels use (16x16x16 fp16 -> fp32, subgroup 32).
- Fresh roofs, igpu-roofline `fast`, driver 595.78, run `raw/roof-2026-10-05-fast` (2026-10-05 02:52 to 03:33
  UTC, 2474 s, clocks not pinned: 612 MHz under load, 15 W mode; every roof confirmed with 3 repeats within
  1.4 %; report in `results/orin/roofline/2026-10-05-fast/`): matrix fp16 9.716 TFLOP/s, fp16 -> fp32 9.722,
  int8 19.482 TOP/s; fed from shared memory 9.511 / 8.771 / 17.818; DRAM read 62.1, write 58.0, copy 64.1 GB/s;
  texture2d read from DRAM 20.1 GB/s against 40.1 for texture3d and 62 for a buffer.
  The tool is the fleet copy already on the device (`~/.cache/igpu-roofline/fleet-quick-20260925`, runner
  `c7fba81beb1e`), run from a copy under `~/hmz-sarc-orin/roofline` with `tools/roof_fast.py`: that tree's
  controller without its clock pinning (the original pins the GPU to 1020 MHz with sudo, which is not allowed
  here). The report's own clock field says "unavailable"; the clock was sampled every 10 s beside it.
- First numbers of the parent control (still running): 1B 4w 890.4, 1B 8da4w 822.5, 3B 4w 360.4, 3B 8da4w
  320.2 tok/s (`cells.csv`: 890.8 / 822.8 / 360.4 / 320.3).

## Incidents

- I edited `tools/build-orin.sh` while its first invocation was running from the same file. I stopped that
  invocation before it reached the edited lines (its tree step ran on and completed), made the build resumable
  and added `tools/wsrun.sh`, which runs workstation jobs from a private copy of the tools.
- `chain2` was started while the parent control still had 20 minutes to go; its session would have given up
  after the 900 s lock wait. Killed before any run (`jobs/chain2.status`); restarted as `chain2b`, which waits
  for the control. Killing it by a name pattern also killed my own ssh shell twice; `tools/dkill.sh` now ends a
  job by its recorded session id.
- `nvidia-smi pmon -c 1` hangs on this device (my probe, killed). Not used by any tool.
- With the 8B model loaded the device has about 1.3 GB available and swap use rose from 107 to 155 MB during
  the parent control. Every session records memory and swap counters per run (`logs/<run>.mem`).

## Thresholds, fixed before any measurement

- Baseline: each of the six cells within 3 % of the Orin SARC median in
  `sarc-1.5-e2e-benchmark/results/cells.csv` (890.8 / 360.4 / 189.7 tok/s for 4w, 822.8 / 320.3 / 170.4 for
  8da4w). Otherwise stop and find out why.
- Noise: a difference inside +-2 % is not a gain. The A/A session reports the real floor.
- Normal clock: `calibrate_clock.py` on the baseline and A/A runs: one device-wide threshold,
  floor(0.97 x the lowest per-cell median of the per-run median devfreq clock). A timed run below it is invalid.
- Arithmetic changes (SDPA kernels): the owner's reference-error rule of 2026-10-04 as written (rms and maximum
  error against the fp32 CPU reference not larger than the parent's on every S = 2048 head configuration, all
  tiers 0 mismatches; gross divergence: mean KL <= 0.5 nat and top-1 differences <= one third of the prompts in
  every cell).
- Stop rule: two consecutive gated candidates each below +2 % geomean over their parent.

## What differs from the sibling (4070 Ti) tools

The measuring host is not the build host. `tools/common.sh` serves both sides; the device holds a copy of the
tools, the unmodified `sarc/tools/verify.sh` and the kit prompts under `~/hmz-sarc-orin/executorch/` (same
relative paths), the builds under `~/hmz-sarc-orin/build/<tag>/bundle/`.

| tool | change |
|---|---|
| `build-orin.sh`, `jetson-cross/` | replaces `build-both.sh`: `mktree.sh` tree + the cross recipe of the campaign that produced the Orin rows of `cells.csv` (`reference-tools/jetson-cross`, image `localhost/et-jetson-cross:jp7.2.1`), under the desktop build lock. One runner serves timed and traced runs (ETDump is linked, as in that campaign). Added to the recipe: `vk-caps.cpp` (capability query) |
| `deploy.sh`, `drun.sh`, `dstat.sh`, `pull.sh` | new: copy tools and builds to the device, start a tool there detached with a status file, show status, mirror the results to `.artifacts/orin-prefill-refine/device/` |
| `common.sh` | temperature from `/sys/class/thermal` (zone `gpu-thermal`), clock from devfreq `17000000.gpu`, load from the nvgpu node, power from ina3221 `VDD_IN`; `nvidia-smi` is not used (N/A on a Jetson, and `nvidia-smi pmon` hangs there). No per-process GPU client list exists without root: foreign jobs are found by name (known GPU programs, a running Actions job `Runner.Worker`). Cooling waits also end when the temperature has stopped falling |
| `e2e5.sh` | sampler 0.1 s from sysfs; flat model directory; memory and swap counters before and after every run (`logs/<run>.mem`) |
| `trace.sh` / `trace_analyze.sh` | the ETDump runs happen on the device, the analysis on the workstation |
| `gen_orin_sdpa.py`, `devzone.py` | the Orin SDPA base rows (`impl/sarc_dev/OrinSdpa.cpp`, the Xe2 mechanism, no hook) and `orin-*` profiles in marked blocks of `Overrides.cpp` |
| not carried over | `gen_4070ti_*.py`, the local hook patches, `Containerfile`, `podman-shim.sh` (they stay in the sibling's directory) |
