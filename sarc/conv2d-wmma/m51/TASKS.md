# M51 conv2d-pw-coopmat: tasks

Each task lists how it is verified. Raw data and device identifiers stay in the local record; committed files carry ratios
only (`SPEC.md`, "Publication of M51 results").

> **Owner amendment, 2026-11-19 (SPEC "MODIFIED Requirements"). Execution order from now on:**
> 2.4 → 4.0 (early session; it picks the parent) → 7.2 FP16 → 9.2 (k×k gate and timing) → 5.2 (phase timing) → at most two 1×1
> candidates (6), with `KSPEC` and `XP` as candidates only → 9.3/9.4 if time allows → 7.4 → 8.
> Workload (7): all ten models; export and runner build on the host now; FP16 on the board right after 4.0 and
> before 9.2; FP32 and INT8 on idle board time.
> Not required: 3.1–3.4 as a mandatory parent, 4.1/4.2 as a separate parent session (the parent's numbers come from
> 4.0), 5.1, 5.3.

## 1. Environment and capability
- [ ] 1.1 Build `etvk_ops_bench` for Android arm64 (`backends/vulkan/test/ops/benchmark_v2/build.sh --android`).
  Verify: the binary runs `--help` on the board; its checksum is in the local record.
- [ ] 1.2 Qualify the board: SoC model, root, canonical driver checksum, pinned clocks read back, profiler config aside.
  Verify: all five are in the local record for the session.
- [ ] 1.3 Run `device_coopmat_properties.json`; record properties, Vulkan version, `maintenance4`, subgroup range and
  shared-memory limit. Verify: there is a committed summary `CAPABILITY.md` (shapes, types, scope, subgroup sizes only);
  the stop condition of the capability requirement is evaluated and written down.

## 2. Existing variants as they are
- [ ] 2.1 Compute the shared-memory size of every AMD-family variant; list exclusions over 64 KiB.
  Verify: a table in `RESULTS.md`, cross-checked against the descriptor's `shared_bytes()`.
- [ ] 2.2 Record whether pipeline creation of the AMD-family variants (trip count from a uniform buffer) succeeds.
  Verify: an outcome per variant in the local record and a one-line summary in `RESULTS.md`.
- [ ] 2.3 Add the many-workgroup and reduced-grid correctness collections (timed shapes, long-K GEMM-equivalent shapes,
  bias, residual) under `benchmark_v2/.../cases/`. Verify: the collections run on the host, or on a desktop device, and
  pass with `pw_tiled_4x8`.
- [ ] 2.4 Gate the AMD-family variants on the M51 (10 repetitions). Verify: pass/fail per variant, with mismatch
  coordinates for failures.

## 3. M51 family parent
- [ ] 3.1 Add the `KSPEC` option (loop bound and any coopmat offset/stride by specialization constant, default off) to the
  body, the wrapper, the descriptor and `shared_bytes()`. Verify: the SPIR-V of all existing variants is byte-identical
  before and after; a resize test re-creates the pipeline or reports the variant ineligible.
- [ ] 3.2 Add the `XP` option (FP32 to FP16 before staging, one-pass full-tile drain, FP32 epilogue). Verify: the generated
  code shows one store pass; the fp32 cancellation cases are run and recorded; the reference-error rule holds.
- [ ] 3.3 Add the `m51` family: yaml block and descriptor for `conv2d_pw_coopmat_m51_t128x128_f32_kspec_xp_half`, plus
  the family in `Conv2dPW.cpp` and `gen_pw_coopmat_cases.py`. Verify: the registry check is clean; static safety holds;
  the race read is written in the commit body.
- [ ] 3.4 Gate the parent on the M51. Verify: 10/10.

## 4. Baseline timing
- [x] 4.1 A/A of the parent. Verify: within `THRESHOLDS.md`. (Done as part of 4.0: two A/A opens,
  geomeans 1.0015 / 0.9959, pass the 1% band; widened-band cells recorded in `RESULTS.md`.)
- [x] 4.2 Time the shape set with `auto`, `pw_tiled_4x8`, the passing AMD-family variants and the parent.
  Verify: the ratio table in `RESULTS.md` is recomputable from the local `runs.csv`. (Done as part of
  4.0: five rotated runs; `RESULTS.md` 4.0; local `runs.csv` + `tools/timing_analysis.py`.)

## 5. Locate
- [ ] 5.1 On a separate branch `topic/m51-conv2d-wmma-inspect`, port the pipeline-statistics dump. Record the parent's
  registers, shared memory, scratch and instruction mix per K step. Verify: the branch is never merged; `RESULTS.md`
  states the bottleneck without absolute counts.
- [ ] 5.2 Shader-clock phase timing of the parent (fetch, staging, sync, multiply, drain). Verify: phase shares in
  `RESULTS.md`.
- [ ] 5.3 (optional) Vendor counter capture in a non-timed session. Verify: a session note in the local record.

## 6. Candidates
- [ ] 6.n For each candidate chosen from 5.x: one option against a named parent, static safety, race read, gate, then an
  interleaved timed session. Verify: per-technique gain in `RESULTS.md`; the stop rule is checked after each one.

## 7. Workload baseline (parallel with 2–6)
- [x] 7.1 Export the ten CI models in FP16 with `../survey-tools/export_models.py`; build a traced runner with the event
  tracer enabled. Verify: ten `.pte` files and a runner that writes an ETDump on the host.
  (ic4 exported with random weights — HF CDN unreachable from this network; deviation recorded locally.)
- [ ] 7.2 Run on the M51 (12 executions, first dropped) and aggregate. Group the time by shader and shape. Also group it
  by convolution family, using the desktop survey's families: 1×1, im2col GEMM, direct k×k, dilated, 1D, depthwise,
  other. Group it by kind × K as well. Verify: shares and rankings per family and per shader are committed; times stay
  local.
- [ ] 7.3 Add the top ten uncovered 1×1 shapes to the timed set (task 4.2 / 6 sessions).
- [ ] 7.4 End-to-end projection. For resnet18, resnet50, mv2, mv3, ic3, ic4, densenet161 and dl3, take the M51 1×1 ratio
  of each matching layer shape in the ETDump and leave every other layer unchanged. Report the projected GPU-time
  reduction in %, labelled "projection, lower bound". Verify: the table can be recomputed from the local ETDump CSV and
  `runs.csv`.

## 8. Report
- [ ] 8.1 `RESULTS.md`: per-shape verdict, per-technique gains, what the workload says to do next.
  Verify: the publication grep is clean; the final table comes from the committed head's build.
- [ ] 8.2 A sanitized note for the desktop side (ratios and shapes only).
- [ ] 8.3 A stage-2 input note in `RESULTS.md`: the M51 ranking of convolution families by share, and which of im2col
  GEMM, direct k×k and dilated 3×3 to take first. Verify: the note exists and cites the 7.2 shares.

Section 9 runs after sections 1–8 in the same campaign run (owner decision, 2026-11-19). Section 10 waits for the routing
switch.

## 9. k×k convolution (stage 2, PENDING: desktop push)
Starts only when `origin/topic/conv2d-wmma` has moved and contains a cooperative-matrix route with im2col input.
- [ ] 9.1 Rebase `topic/m51-conv2d-wmma` onto the pushed `origin/topic/conv2d-wmma`. Re-run the 1×1 parent gate and
  A/A. Verify: 10/10, and the parent's ratios are within the noise band of the stage-1 results.
- [ ] 9.2 The im2col GEMM stage on the M51.
  - Apply the M51 rules to the im2col-input variants: `KSPEC`, `XP`, the fence, and shared memory ≤ 64 KiB.
  - Check that no im2col scratch storage (texture2d, texture3d, buffer) brings back a multi-pass drain.
  - Gate with a k×k collection: the M51 top k×k shapes from 7.2 and the desktop layer table (3×3 512 ch 7×7, 3×3
    stride 2 512 ch, 3×3 448→384 5×5, 3×3 256 ch 14×14, 1×7 192→224 17×17, 3×3 128 ch 28×28, 3×3 192→48 14×14,
    3×3 64 ch 56×56), full grid and reduced grid, 10/10.
  - Time the whole convolution, gather included, against today's automatic route.
  Verify: the gate result and the ratio table in `RESULTS.md`.
- [ ] 9.3 Direct k×k through im2col+WMMA (large planes, edsr) and dilated 3×3 (dl3), only if the desktop has pushed them.
  Verify: gated and timed like 9.2, or recorded as "not available".
- [ ] 9.4 Locate and candidates for k×k, with the discipline of sections 5–6. `_sk` split-K is the first menu entry.
  Verify: per-technique gains; the stop rule is checked after each one.
- [x] 9.5 The k×k thresholds are committed in `THRESHOLDS.md`. The end-to-end bar is still to be set with the device
  owner before TASKS 10.

## 10. Measured end to end (stage 2, PENDING: desktop routing switch)
- [ ] 10.1 With the desktop routing switch, pick the two or three models with the largest projected gain (7.4).
  Interleave A/B runs with the switch off and on: same binary, pinned clocks, cooled board, median of 5. Verify: the
  gain in % of GPU time and of the execute call, compared with the projection.
