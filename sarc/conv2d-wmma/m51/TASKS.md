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
- [x] 1.1 Build `etvk_ops_bench` for Android arm64 (`backends/vulkan/test/ops/benchmark_v2/build.sh --android`).
  Verify: the binary runs `--help` on the board; its checksum is in the local record.
  (Builds recorded in the local record; the binary digest is verified on the device before every session.)
- [x] 1.2 Qualify the board: SoC model, root, canonical driver checksum, pinned clocks read back, profiler config aside.
  Verify: all five are in the local record for the session.
  (Local record `SESSION.md`; driver verified after every run of every session; pins verified pre-session and read back post-session.)
- [x] 1.3 Run `device_coopmat_properties.json`; record properties, Vulkan version, `maintenance4`, subgroup range and
  shared-memory limit. Verify: there is a committed summary `CAPABILITY.md` (shapes, types, scope, subgroup sizes only);
  the stop condition of the capability requirement is evaluated and written down.
  (`CAPABILITY.md` committed; stop condition PASS, recorded in `PROGRESS.md`.)

## 2. Existing variants as they are
- [x] 2.1 Compute the shared-memory size of every AMD-family variant; list exclusions over 64 KiB.
  Verify: a table in `RESULTS.md`, cross-checked against the descriptor's `shared_bytes()`.
  (`RESULTS.md` 2.1.)
- [x] 2.2 Record whether pipeline creation of the AMD-family variants (trip count from a uniform buffer) succeeds.
  Verify: an outcome per variant in the local record and a one-line summary in `RESULTS.md`.
  (`RESULTS.md` 2.2; per-variant outcomes in the local correctness runs.)
- [x] 2.3 Add the many-workgroup and reduced-grid correctness collections (timed shapes, long-K GEMM-equivalent shapes,
  bias, residual) under `benchmark_v2/.../cases/`. Verify: the collections run on the host, or on a desktop device, and
  pass with `pw_tiled_4x8`.
  (`pw_coopmat_gate_{full,reduced}_grid.json` committed; the baseline arms pass in every gate run on the device - local record `gate/`.)
- [x] 2.4 Gate the AMD-family variants on the M51 (10 repetitions). Verify: pass/fail per variant, with mismatch
  coordinates for failures.
  (`RESULTS.md` 2.4; per-variant verdict table from the local record `gate/` and `tools/gate_verdict.py`.)

## 3. M51 family parent
- [~] 3.1 Add the `KSPEC` option (loop bound and any coopmat offset/stride by specialization constant, default off) to the
  body, the wrapper, the descriptor and `shared_bytes()`. Verify: the SPIR-V of all existing variants is byte-identical
  before and after; a resize test re-creates the pipeline or reports the variant ineligible.
  (Dropped, owner amendment 2026-11-19: `KSPEC` and `XP` are candidates only - see the amendment note above.)
- [~] 3.2 Add the `XP` option (FP32 to FP16 before staging, one-pass full-tile drain, FP32 epilogue). Verify: the generated
  code shows one store pass; the fp32 cancellation cases are run and recorded; the reference-error rule holds.
  (Dropped, owner amendment 2026-11-19: candidates only.)
- [~] 3.3 Add the `m51` family: yaml block and descriptor for `conv2d_pw_coopmat_m51_t128x128_f32_kspec_xp_half`, plus
  the family in `Conv2dPW.cpp` and `gen_pw_coopmat_cases.py`. Verify: the registry check is clean; static safety holds;
  the race read is written in the commit body.
  (Dropped, owner amendment 2026-11-19: the campaign times the existing `amd`/`nvidia` variants.)
- [~] 3.4 Gate the parent on the M51. Verify: 10/10.
  (Dropped with 3.3; the parent that was actually gated and timed is the selected `amd` variant - `RESULTS.md` 2.4, 4.0.)

## 4. Baseline timing
- [x] 4.1 A/A of the parent. Verify: within `THRESHOLDS.md`. (Done as part of 4.0: two A/A opens,
  geomeans 1.0015 / 0.9959, pass the 1% band; widened-band cells recorded in `RESULTS.md`.)
- [x] 4.2 Time the shape set with `auto`, `pw_tiled_4x8`, the passing AMD-family variants and the parent.
  Verify: the ratio table in `RESULTS.md` is recomputable from the local `runs.csv`. (Done as part of
  4.0: five rotated runs; `RESULTS.md` 4.0; local `runs.csv` + `tools/timing_analysis.py`.)

## 5. Locate
- [~] 5.1 On a separate branch `topic/m51-conv2d-wmma-inspect`, port the pipeline-statistics dump. Record the parent's
  registers, shared memory, scratch and instruction mix per K step. Verify: the branch is never merged; `RESULTS.md`
  states the bottleneck without absolute counts.
  (Dropped, owner amendment 2026-11-19: not required.)
- [x] 5.2 Shader-clock phase timing of the parent (fetch, staging, sync, multiply, drain). Verify: phase shares in
  `RESULTS.md`.
  (Done by ablation, disclosed in `RESULTS.md` 5.2: shader-clock support was not checked and ablation was used for the
  time budget; phase shares located, the in-data-path split marked UNVERIFIED because the ablations overlap.)
- [~] 5.3 (optional) Vendor counter capture in a non-timed session. Verify: a session note in the local record.
  (Dropped, owner amendment 2026-11-19: not required.)

## 6. Candidates
- [~] 6.n For each candidate chosen from 5.x: one option against a named parent, static safety, race read, gate, then an
  interleaved timed session. Verify: per-technique gain in `RESULTS.md`; the stop rule is checked after each one.
  (Skipped: 5.2's in-data-path split is UNVERIFIED, so no evidence-based 1x1 candidate exists; the kk side's next arms
  were already measured in 9.2 and are statistically tied with the parent. Reason recorded in `PROGRESS.md`; candidates
  are optional per the owner amendment and notes 32/48.)

## 7. Workload baseline (parallel with 2–6)
- [x] 7.1 Export the ten CI models in FP16 with `../survey-tools/export_models.py`; build a traced runner with the event
  tracer enabled. Verify: ten `.pte` files and a runner that writes an ETDump on the host.
  (ic4 exported with random weights — HF CDN unreachable from this network; deviation recorded locally.)
- [x] 7.2 Run on the M51 (12 executions, first dropped) and aggregate. Group the time by shader and shape. Also group it
  by convolution family, using the desktop survey's families: 1×1, im2col GEMM, direct k×k, dilated, 1D, depthwise,
  other. Group it by kind × K as well. Verify: shares and rankings per family and per shader are committed; times stay
  local.
- [x] 7.3 Add the top ten uncovered 1×1 shapes to the timed set (task 4.2 / 6 sessions).
  (The ten heaviest uncovered 1x1 shapes of the 7.2 survey, gated 10/10 on both grids and timed with the 8.1 protocol;
  `RESULTS.md` 7.3; local record `task7-3/`.)
- [x] 7.4 End-to-end projection. For resnet18, resnet50, mv2, mv3, ic3, ic4, densenet161 and dl3, take the M51 1×1 ratio
  of each matching layer shape in the ETDump and leave every other layer unchanged. Report the projected GPU-time
  reduction in %, labelled "projection, lower bound". Verify: the table can be recomputed from the local ETDump CSV and
  `runs.csv`.
  (`RESULTS.md` 7.4; the 1x1 ratio is vs `auto` - the route the ETDump layers ran - with the
  `pw_tiled_4x8`-based values as a labelled secondary column, plus kxk and vs-best-stock;
  recomputable with the local `tools/e2e_projection.py` over the local ETDump CSV and the run
  records; extended with the 7.3 matches.)

## 8. Report
- [x] 8.1 `RESULTS.md`: per-shape verdict, per-technique gains, what the workload says to do next.
  Verify: the publication grep is clean; the final table comes from the committed head's build.
  (Final tables in `RESULTS.md` 8.1 from the one interleaved session; publication grep clean on the whole branch;
  binary provenance stated in 8.1.)
- [x] 8.2 A sanitized note for the desktop side (ratios and shapes only).
  (`DESKTOP-NOTE.md`.)
- [x] 8.3 A stage-2 input note in `RESULTS.md`: the M51 ranking of convolution families by share, and which of im2col
  GEMM, direct k×k and dilated 3×3 to take first. Verify: the note exists and cites the 7.2 shares.
  (`RESULTS.md` 8.3.)

Section 9 runs after sections 1–8 in the same campaign run (owner decision, 2026-11-19). Section 10 waits for the routing
switch.

## 9. k×k convolution (stage 2, PENDING: desktop push)
Starts only when `origin/topic/conv2d-wmma` has moved and contains a cooperative-matrix route with im2col input.
- [x] 9.1 Rebase `topic/m51-conv2d-wmma` onto the pushed `origin/topic/conv2d-wmma`. Re-run the 1×1 parent gate and
  A/A. Verify: 10/10, and the parent's ratios are within the noise band of the stage-1 results.
  (The rebase is a no-op: the origin tip is already the merge-base. The 2.4 gate and the 4.0 session
  ran on this tree, so the parent gate and ratios stand as recorded.)
- [x] 9.2 The im2col GEMM stage on the M51.
  - Apply the M51 rules to the im2col-input variants: `KSPEC`, `XP`, the fence, and shared memory ≤ 64 KiB.
  - Check that no im2col scratch storage (texture2d, texture3d, buffer) brings back a multi-pass drain.
  - Gate with a k×k collection: the M51 top k×k shapes from 7.2 and the desktop layer table (3×3 512 ch 7×7, 3×3
    stride 2 512 ch, 3×3 448→384 5×5, 3×3 256 ch 14×14, 1×7 192→224 17×17, 3×3 128 ch 28×28, 3×3 192→48 14×14,
    3×3 64 ch 56×56), full grid and reduced grid, 10/10.
  - Time the whole convolution, gather included, against today's automatic route.
  Verify: the gate result and the ratio table in `RESULTS.md`. (All 32 `_i2c` variants 10/10 on both
  grids; every variant creates its pipeline on the device. Timed session: best arm
  `amd_t128x128_f16_alc_sk4_i2c_half`, 3.69x over `auto`, 5.66x over `im2col`, 14/14 vs the
  tolerance-holding incumbent. The `direct` route fails tolerance on the K=576-class shapes —
  recorded in `RESULTS.md` 9.2.)
- [x] 9.3 Direct k×k through im2col+WMMA (large planes, edsr) and dilated 3×3 (dl3), only if the desktop has pushed them.
  Verify: gated and timed like 9.2, or recorded as "not available".
  (Not available: the pushed base has neither route. The k×k coopmat multiply exists only behind
  the im2col gather, which refuses dilation ("dilation is not 1", `Conv2dGemm.cpp`; the base's
  own generator lists a `3x3 dilation 2` case as an expected SKIP) - so dl3's dilated layers
  cannot reach it; no direct in-kernel-gather k×k route exists (CAMPAIGN-PLAN step 3 is future
  desktop work) and the base's im2col collections cover only the eight 9.2 shapes, with no edsr
  large-plane cases. Evidence in `RESULTS.md` 9.3.)
- [~] 9.4 Locate and candidates for k×k, with the discipline of sections 5–6. `_sk` split-K is the first menu entry.
  Verify: per-technique gains; the stop rule is checked after each one.
  (Skipped, as 6.n: the menu's next arms were already measured in the 9.2 session - 3.38-3.50x
  over `auto` against the parent arm's 3.69x - so a candidate session would re-measure existing
  variants rather than test a new technique; `_sk` split-K is already in the parent's name.
  Candidates are optional per notes 32/48; reason recorded in `PROGRESS.md`.)
- [x] 9.5 The k×k thresholds are committed in `THRESHOLDS.md`. The end-to-end bar is still to be set with the device
  owner before TASKS 10.

## 10. Measured end to end (stage 2, PENDING: desktop routing switch)
- [ ] 10.1 With the desktop routing switch, pick the two or three models with the largest projected gain (7.4).
  Interleave A/B runs with the switch off and on: same binary, pinned clocks, cooled board, median of 5. Verify: the
  gain in % of GPU time and of the execute call, compared with the projection.

## 11. Stage 3: kernel efficiency, no shape slower than stock (owner, 2026-11-20)
Goal: every conv2d coopmat kernel beats the best stock route on every timed shape, with efficiency measured against
the roofline (`THRESHOLDS.md`, stage 3). The input facts are in the local `ROOFLINE.md` and the argus analysis.
- [x] 11.1 Re-baseline on the new board. Run a short interleaved session of the 1×1 and k×k parents against all stock
  routes on the 45 timed shapes. Verify: a per-shape table of ratio and % of roofline bound; the losers are listed.
  (RESULTS 11.1: 45-case session, zero invalid rows, A/A inside the band, geomeans reproduce 8.1 within 0.6%;
  losers confirmed - three 784-plane 1x1 cells beyond band plus one at parity, and the four kxk cells.)
- [x] 11.2 B-operand LDS loads. First probe the ISA of the existing `BCM` variants (argus pipeline dump, untimed).
  Then build one change that turns the B gather (`ds_load_u16_d16`) into dword loads: BCM fixed, or A/B role swap
  (Cᵀ = Bᵀ·Aᵀ, weights as the A operand). Verify: the ISA shows the change; LDS ops per WMMA and LDS Busy fall
  (argus); gated; timed.
  (RESULTS 11.2: probe answered - ColumnMajor lowers to dword pairs; `_bcm2` built and gated 1240/1240; the ISA
  shows the designed lowering and LDS ops/WMMA fall 10.6 -> 4.58, but LDS cycles per instruction rise 2.11 -> 6.62
  (bank conflicts), LDS Busy rises 87 -> 94%, WMMA rate falls to 0.80x, and the candidate loses 1.011-1.318x on
  all 31 shapes - REJECTED with evidence; the note-6 odd-stride iteration `_bcm3` (gated 1240/1240) fixes the load
  walk (zero u16, LDS ops/WMMA 5.13) but the store walk still conflicts (4 x 17 = 68 dwords = 4 mod 32): cycles
  per instruction 4.98, LDS Busy 91.3%, WMMA 0.92x - hypothesis not supported, family closed per note 6, timing
  skipped; the role swap is contraindicated by the same measurement.)
- [x] 11.3 More fp32 accumulators per wave, or a larger fp32 tile, for the large-plane 1×1 losers (784/3136 pixels).
  Verify as in 11.2.
  (RESULTS 11.3: t32x32 gated 1240/1240; blanket parent rejected at 1.202x geomean; adopted per-regime - the
  1x1 "after" state is the predicate "M = 784 and N·K ≥ 2^18" (5 of 31 shapes, 0.929x geomean vs parent on them,
  no losers selected); argus shows an LDS-capacity occupancy limit (LDS-limited launch 68.8%, VGPR- and
  wave-slot-limited 0%); stop counter stays at one - the candidate gained 8.6% on its targets.)
- [x] 11.4 im2col gather cost: no runtime div/mod, no fp16 rounding emulation if not needed, or the gather fused into
  the A load. Also split-K 1 for large M. Target: the Cin=64 large-plane k×k losers. Verify as in 11.2.
  (RESULTS 11.4: fast-gather candidate gated 1120/1120 with the rounding emulation dropped, as note 4 allows;
  0.930x geomean vs the kk parent (0.815-1.008), faster on 13 of 14 shapes; gather VALU Busy 106.7% -> 41.9%,
  SALU Busy 30.1% -> 9.3%, div/mod and rounding constants gone from the ISA; ADOPTED as the stage-3 kk parent,
  runtime plumbing default-off and unrouted as before.)
- [ ] 11.5 Small-M, DRAM-bound shapes: move toward the DRAM roof (deeper split-K, more loads in flight).
  Verify as in 11.2.
- [x] 11.6 Final interleaved session from the committed HEAD. Report per-shape ratio and % of roofline before and
  after stage 3, and per-candidate gains. Re-run from the final HEAD after 11.8, with every adopted arm gated on
  the exact binary timed (the 11.6a arms were gated on earlier binaries). Add a per-cell noise-band column to the
  final table and mark which gains exceed it. Cool the board fully first (the 11.6a opening A/A had a +24% cell).
  (11.6a provisional checkpoint, HEAD e4087bd2ef, RESULTS 11.6a: 840 rows, zero invalid, close A/A 1.007, parent
  drift vs 11.1 0.999; after/before geomean 0.970 over all 45 (1x1 0.988 via the 11.3 predicate route, kxk 0.930
  via fast-gather); after/best-stock 0.578; geomean % of bound 27.6% -> 28.5%; goal met beyond the noise band on
  37 of 45 - 784,256,1024 crosses to 0.987 within the parity band, 7 cells remain behind (three 784 1x1, four
  kxk), down from 8; oracle 0.986 vs predicate 0.988.)
  (FINAL s119 - the note-17 re-run, HEAD 163f5bfef1, RESULTS 11.6 final: 11 min full idle then 120 s cooling idle then
  905 rows zero invalid on the same binary as every gate; thermal state read back in the loop before and after
  (local log);
  A/A open 1.0000 (5 cells outside ±2%) / close 0.9987 (3 outside) - both geomeans inside 1%, so the note-17 validity
  condition is met and s119 stands as final; the 4.0/7.3 precedent widening, per-cell bands in the table; parent
  drift 1.001. After state = the note-18 route directly: 1x1 M >= 784 and N >= 256 (the three N=128 cells run on the
  parent only; their s118 both-arm ratios are the record of the cost: 3136,128,256 0.798 / 784,128,512 0.930 /
  784,128,1152 1.050; boundary-cell requests closed); the nv alternative arm ran on the 8 predicate cells -
  same-session amd/nv predicate geomeans 0.721 / 0.683, nv faster on all 8 with no loser, matching s117; amd stays
  adopted, nv recorded as the owner's alternative. After/before geomean 0.903 (1x1 0.919, kxk 0.868);
  after/best-stock 0.541; geomean % of bound 27.6% -> 30.6%. Gains beyond band on 21 of 22 changed cells, one inside
  its widened band. Goal met on 43 of 45: the three 784 losers land at 0.755/0.748/0.938 vs best stock, 64-256
  224x224 at 0.975, 1024-256 28x28 at 0.703; the two remaining cells are the 11.5 DRAM-roof cells (1.810 / 1.492).
  Superseded first run s118: open 1.0033 (5 outside) / close 0.9896 (7 outside) fired the re-run trigger; its
  note-18-recomputed geomeans (0.904 / 0.539 / 27.8% -> 30.7%) agree with s119 to a rounding step.)
- [ ] 11.7 (superseded by 11.8) 784-cell gap, re-attributed: the 11.3 occupancy reading is corrected by the M51
  linear-GEMM evidence (note 13) - a kernel at one workgroup per CU with 16 independent accumulator chains per
  wave reaches about two times our efficiency, so the limiter is **too few independent accumulator chains per
  wave** (2 in the parent, 4 in t32x32, against 16), not occupancy. The measured t32x32 launch limiter stands
  (LDS-limited launch 68.8%, VGPR- and wave-slot-limited 0%); it caps resident waves but is not the binding limit
  on throughput. Next lever -> 11.8.
- [x] 11.8 (note 13, before 11.5) Fragment-reuse amortization with the existing large tiles - no new shader code:
  `amd_t128x128_f32_alc` and `nvidia_t128x128_f32_alc` (stage-1 best on the 3136 cells and `784,512,128`), never
  yet timed on the 784/3136 losers. Gate 10 reps on the pw collections (include the many-workgroup lesson of the
  `amd_t64x64_f32` failure); time on all 31 1x1 shapes with the stock arms and the parent; argus before/after on
  the 784 losers (LDS ops per WMMA target ~3-4, WMMA ops per cycle clearly above the parent's, occupancy, and the
  drain-pass count and time share); judge per shape with a predicate of at most two terms (notes 9 and 11). If the
  drain dominates, a one-pass drain (`f32xp` form) only if it is a yaml combination gateable by 16:45, within the
  64 KiB cap. If time is left: the kxk `amd_t128x128_f32_alc_sk2/sk4_i2c` arm with the fast gather on the 4 kxk
  losers.
  (RESULTS 11.8/11.8k: pw gate 4 arms x 1240/1240 and kk gate 560/560 each + fast-gather re-gate + kk t128+fg gate
  (6 arms: parent +-fg, sk2 +-fg, sk4 +-fg), all on the final binary; s117 775 rows zero invalid, s117k 180 rows
  zero invalid; argus pw: LDS ops/WMMA ~3.1, WMMA rate 1.56-1.59x parent, occupancy 28-32% vs 23.6%, LDS Busy
  90%->51%, drain 2 passes unmeasured share; argus kk: RESULTS 11.8k. The three 784 losers cross stock parity on
  the adopted amd route (0.755/0.748/0.895; nvidia alternative 0.703/0.699/0.833); predicates adopted: 1x1
  M >= 784 and N >= 256 (note 18; supersedes t32x32 and the first N >= 128 reading; the three dropped N=128 cells
  listed with ratios in RESULTS 11.6/11.8; boundary-cell requests closed), kk M >= 784 and N >= 256 ->
  t128_sk2+fg (plain adoption; supersedes the 11.4 flat cell; 64-256 224x224 crosses to 0.98). kk cells behind
  stock fall to two (the 11.5 cells). No f32xp drain: the 784 losers already cross parity with the two-pass form,
  and no new candidate after 16:45.)
