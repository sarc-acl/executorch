# STATUS: sarc-1.5-orin-prefill-refine

**2026-10-05 03:45 UTC. RUNNING: parent control on the device, baseline + A/A queued behind it. No candidate yet.**

(The first version of this file carried a wrong clock time; all times here are UTC from `date -u`.)

## Running now

- Device (`duck-naughty`), detached, one GPU job at a time under the gpu-lab lock:
  - job `s0-parent-verify` (`tools/parent_verify.sh parent`): unmodified `verify.sh` on the pristine parent.
  - job `chain2b` (`tools/chain2.sh`), waiting for it: stage `s1-aa`, baseline + A/A session (`--calibrate`),
    warm traces, SDPA screen 1 (59 `orin-qk-*` / `orin-av-*` profiles + stock), 8da4w phase timing.
  - Status on the device: `~/hmz-sarc-orin/jobs/<job>.status` and `.out`; from the workstation `tools/dstat.sh`.
  These survive a reboot of the workstation.
- Workstation: job `build-extra1` (`logits_dump` for builds `parent` and `topic1`), waiting for the desktop
  build lock, which the B580 campaign holds. It does not survive a workstation reboot (start again).

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
