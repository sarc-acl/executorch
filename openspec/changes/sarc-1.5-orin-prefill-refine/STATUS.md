# STATUS: sarc-1.5-orin-prefill-refine

**2026-10-04 20:20 UTC. SETTING UP. Nothing measured yet. The parent cross-build is running on the workstation.**

## Running now

- Workstation (`fedora`, build host): `tools/build-orin.sh parent 6a7cc8cc6`, detached, under
  `flock ~/.cache/gpu-lab/lock-desktop-build`. Output `.artifacts/orin-prefill-refine/jobs/build-parent.out`,
  end marker `jobs/build-parent.done`, provenance `build/parent.src.txt`. If the workstation reboots this job is
  gone and has to be started again (the tag directory must be removed first: tags are immutable).
- Device (`duck-naughty`): nothing.

## Next step

Deploy the parent build, test the temperature / clock / power probes on the device, parent control
(`parent_verify.sh`), baseline + A/A session (`--calibrate`), then the SDPA screen.

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
