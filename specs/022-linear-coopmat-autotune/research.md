# Phase 0 Research: Smart Autotuning for q4gsw CoopMat Tile Configuration

All items below are **decisions**, not open questions — this feature builds
directly on work already done earlier in this session (the 642-config
enumeration, 10 real on-device measurements, and a review of how production
autotuners like AutoTVM/Ansor/Triton/Tensile avoid brute force), so there are
no unresolved `NEEDS CLARIFICATION` items from the plan's Technical Context.

## Decision 1: Where this executes

**Decision**: Shader-variant edits, the Android build, and all on-device
measurement happen in the existing isolated experiment worktree
`.artifacts/tsweep-256x256-smoketest/executorch` (branch
`exp/tsweep-256x256-4x4-smoketest`), created earlier this session. Spec-kit
documentation and the analysis/orchestration scripts live in this repo
(`quant-perf-optimization/executorch`, under `specs/022-linear-coopmat-autotune/`).

**Rationale**: The `linear_q4gsw_coopmat_tsweep.{glsl,yaml}` template and its
`QuantizedLinear.cpp` dispatch hook only exist as uncommitted work in
`.tmp-origcm` (per workspace `CLAUDE.md`); this session already forked that
into the `.artifacts/tsweep-256x256-smoketest` worktree, ported the exact
same uncommitted diff, and has a warm, already-built Android toolchain there
(`cmake-out-android-vk/` with `libvulkan_backend.a` and the bench binaries
already built). This `quant-perf-optimization` worktree has none of that
infrastructure — it hosts a different, spec-kit-governed sweep
(`dq8ca_q4gsw_coopmat_sweep`) for a different quantization scheme (spec
008). Reusing the warm worktree avoids a second full submodule-init +
Android cross-build (previously measured at ~15-20 minutes even with
ccache) and matches this workspace's own precedent of keeping spec-kit
documentation separate from the worktree that physically holds a given
uncommitted shader family.

**Alternatives considered**:
- *Fork a fresh worktree scoped to this feature*: rejected — no benefit
  over reusing the existing one, and doubles submodule-init/build cost for
  identical shader source.
- *Move the tsweep family into this `quant-perf-optimization` worktree
  permanently*: rejected as out of scope — that would mean committing
  `.tmp-origcm`'s uncommitted work into a different branch/workstream
  entirely, a decision for the workstream owner, not this feature.

## Decision 2: Analytical cost model for zero-device-time pruning

**Decision**: Score every one of the 642 valid buffer-storage candidates
with two hardware-derived proxies computed purely from each candidate's
already-known derived properties (no simulation, no ML model):

1. **Occupancy proxy**: `min(64KB / LDS_bytes(candidate), 1024 / WG_SIZE(candidate))` —
   a rough upper bound on concurrently-resident workgroups per compute unit,
   using the confirmed HW limits (`maxComputeSharedMemorySize=64KB`,
   `maxComputeWorkGroupInvocations=1024`) as the two binding resources.
2. **Register-pressure proxy**: accumulator count per subgroup
   (`MMAS_PER_SG_M × MMAS_PER_SG_N`) — penalize candidates far outside the
   `[2, 16]` range actually observed across this session's 10 real data
   points (the winner uses 8; the worst performer, 256×256/4×4, also uses
   16 but at ~2.4x the winner's LDS footprint — so accumulator count alone
   doesn't explain 256×256's loss, LDS/occupancy does; extremes in either
   direction are untested and riskier).

Combine as `score = occupancy_proxy / (1 + max(0, accumulators - 8) * 0.15)` —
a simple, auditable penalty that derates candidates with unusually high
accumulator counts without needing a fitted/learned model. This mirrors
exactly what the research pass on real autotuners found: CUTLASS/Tensile
prune with hardware-legality + resource-budget rules, not learned models, at
this scale of search space.

**Minimum-parallelism floor (added after a dry run of this scoring
formula over the full 642-candidate universe, before any shortlist was
acted on)**: candidates with `WG_SIZE < 128` (i.e. `SG_GRID_X × SG_GRID_Y < 4`
at `SUBGROUP_SIZE=32`, or `< 2` at `SUBGROUP_SIZE=64`) are excluded from
`top-rank` shortlisting entirely — they still appear in the full ranking
(auditable per FR-008) but are never selected regardless of score. The
raw occupancy proxy has no upper reward bound on shrinking `LDS_bytes`/
`WG_SIZE`, so unconstrained top-N selection returned 28 single-subgroup
tiles (16×16 through 64×64 at `SG_GRID=1×1`, `WG_SIZE` as low as 32) as
the "best" candidates — a workgroup with only one subgroup cannot use this
shader family's double-buffered prefetch/compute overlap the way it's
designed (that overlap happens via barriers across all subgroups sharing
a workgroup's `Ash`/`Bsh` tile), and every one of the 10 real
configurations ever measured on this hardware uses `WG_SIZE >= 128`. This
is a floor grounded in the shader's own design and the full extent of
real evidence in hand, not a fit to the 10-point sample's *values* — it
excludes a whole class of candidates the shader architecture makes
implausible, rather than tuning constants to match observed numbers.

**Alternatives considered**:
- *Let the staged search's Round 1 cheap gate eliminate degenerate tiles
  instead of filtering them out here*: rejected — Round 1 still costs a
  real device measurement per candidate (spec FR-006's whole point is
  minimizing that), and a class of candidates this predictably
  unpromising doesn't need device time to rule out.
- *Add a reward term for larger tile area instead of a hard floor*:
  rejected as needing calibration against the same 10-point sample
  Decision 2's main formula already avoids fitting to.

**Rationale**: This session's own 10 real measurements already show the
qualitative trend an analytical model needs to capture: **smaller,
LDS-lighter tiles at moderate accumulator counts win; large tiles lose even
when legal** (256×256/4×4: 16 accumulators, 40.5KB LDS, ~96% of dbuf1's
throughput; 128×64/K64/4×4: only 2 accumulators but 54KB LDS from the larger
K-step, ~88% of dbuf1's throughput; 128×64/K16/2×2, the winner: 8
accumulators, but only 16.5KB LDS, 125% of dbuf1's throughput — LDS/occupancy
is the dominant signal here, not accumulator count alone). An occupancy-first,
register-pressure-penalized score is directionally consistent with all 10
points without needing a trained cost model — appropriate given the sample
size (10 points is far too small to fit a reliable ML cost model, but
sufficient to sanity-check a simple analytical one).

**Alternatives considered**:
- *Learned cost model (XGBoost, as AutoTVM does)*: rejected for this
  feature's scale — AutoTVM's model is trained on hundreds to thousands of
  measurements; this session has 10. A learned model here would be
  overfit noise dressed up as rigor.
- *Random/uniform sampling of the 642 space*: rejected — would ignore the
  strong, already-observed "smaller tile wins" trend and risk wasting
  device-time budget on large-tile candidates already known to be weak.
- *Pure roofline/FLOP-per-byte model*: considered but rejected as the sole
  signal — this GEMM is not clearly memory- or compute-bound in isolation on
  this hardware; the occupancy/register proxy is more directly tied to the
  actual observed variance across the 10 points.

## Decision 3: Shortlist size and anchor inclusion

**Decision (revised during implementation — see calibration finding below)**:
Take the top ~24-32 candidates by analytical score, then force-include
**all 9 previously-measured, compiling known configurations** from
`known-measurements.json` (not just the 2 originally planned: the
dbuf1-equivalent geometry 128×128/K16/4×2/s32, and the sweep winner
128×64/K16/2×2/s32), regardless of their analytical rank. The one known
configuration that failed to compile (128×64/K16/4×4/s32) is explicitly
excluded with `shortlist_reason: "known_compile_failure"` — re-attempting
it on-device would waste budget on an already-known outcome.

**Calibration finding that drove this revision** (task T009): scoring all
10 known points with the Decision 2 formula showed the model correctly
separates the *worst* known performers (256×256/4×4 and 128×64/K64/4×4
rank 8th/9th of 9 compiling points, matching their real bottom-2 ranking)
but does **not** reliably rank the single *best* performer — the true
winner (128×64/K16/2×2, real 1.25×) scored 3rd of 9, behind
64×64/K16/2×2 (real 1.15×, scored highest) and 64×128/K16/2×2 (real
1.18×, scored 2nd). More strikingly, 128×256/K16/4×2 — a real, solidly
mid-pack performer (1.14×, 4th of 9 by real speedup) — scored 8th of 9
analytically, because its 16 accumulators and 28.5KB LDS trip the
occupancy/register penalty harder than its real throughput would justify.
The root cause: the occupancy proxy rewards small tiles for higher
per-CU residency, but doesn't model that a too-small tile does less useful
work per dispatch — a real GEMM tuning tradeoff this simple, unfitted
heuristic doesn't capture. Retuning the formula's constants to fix this
on 10 points was rejected as exactly the overfitting risk Decision 2
already flagged (10 points is too few to fit reliably) — broadening the
force-include set instead fixes the actual problem (real data being
overridden by an imperfect heuristic) without touching the formula.

**Rationale**: 24-32 (plus up to 9 anchors, most of which likely already
overlap the top-32 analytically-ranked set) keeps the shortlist
comfortably under the SC-001 budget of ≤96 configs measured on real
hardware, even before any candidates are dropped in later rounds. Forcing
in every already-measured config costs nothing (we're not spending new
device time reproducing known results — Round 1 can skip straight past
already-known configs when producing the report, using their existing
`known-measurements.json` result) but guarantees the search can never
silently regress below any real result already in hand.

**Alternatives considered**:
- *A fixed top-N with only 2 forced anchors* (original plan): rejected
  after the calibration finding above showed the model would have
  actually dropped 128×256/K16/4×2 — a real, non-trivial performer — from
  a naive top-28 cutoff.
- *Retune the formula's constants against the 10 points*: rejected as
  overfitting a heuristic to a sample too small to generalize (Decision 2).
- *A fixed top-N with no forced anchors at all*: rejected — the original
  concern from spec.md User Story 1 Acceptance Scenario 2, now confirmed
  empirically rather than just hypothesized.

## Decision 4: Staged, successive-halving-style search

**Decision**: Three measurement rounds over the shortlist, each stricter and
more expensive than the last:

- **Round 1 — cheap gate** (every shortlisted candidate): compile, run
  `COOPMAT_BENCH_CORRECTNESS_ONLY=1` (correctness gate, Principle I) plus a
  single production shape at default harness rigor. Drop anything that
  fails to compile (as 128×64/K16/4×4 did) or fails correctness immediately —
  zero further budget spent on it.
- **Round 2 — full shape coverage** (top third of Round 1 survivors, by
  Round 1's single-shape GFLOP/s): run the full 12-13 production-shape,
  FLOP-weighted pass (`COOPMAT_BENCH_M=2048`) at the harness's default
  rigor (`warmup=3, runs=5` internally).
- **Round 3 — statistically rigorous confirmation** (top 3-5 of Round 2):
  repeat the Round 2 measurement multiple independent times (separate
  process invocations, not just the harness's internal repeat count) to
  report run-to-run mean and stddev, satisfying Constitution Principle IV's
  "a number is only reportable with its iteration count and stddev
  alongside it."

**Rationale**: This directly mirrors the successive-halving pattern
identified in the earlier research pass (AMD Tensile's coarse-then-fine
benchmark pipeline; Ansor's evolutionary elimination) — spend the least
device time on the candidates most likely to be weak, and reserve full
statistical rigor for only a handful of finalists. Round 1's per-candidate
cost is small (compile + one shape); Round 3's is the most expensive but
only applies to ≤5 candidates.

**Alternatives considered**:
- *Measure every shortlisted candidate at full rigor immediately*: rejected —
  this is exactly the "try all of them" pattern the feature exists to avoid,
  just scoped to the shortlist instead of all 642.
- *Two rounds instead of three*: considered, but a dedicated final
  confirmation round is needed to satisfy Principle IV's stddev requirement
  without inflating Round 2's per-candidate cost across the whole
  shortlist-survivor set.

## Decision 5: Budget accounting against SC-001/SC-002

**Decision**: Track "configurations measured on real hardware" as the
distinct candidates that receive a Round 1 measurement (the first point any
candidate ever touches the device). With a shortlist of ~24-32, this alone
satisfies SC-001 (≤96, i.e. ≤15% of 642). Total device time across all three
rounds is estimated and reported against an exhaustive-equivalent estimate
(642 × Round-2-equivalent cost) to demonstrate the ≥5x reduction required by
SC-002.

**Rationale**: SC-001 is about *which* configs ever get real measurement,
not total invocation count — a shortlist of 24-32 already clears the ≤96
bar even before Round 1 eliminates anything. SC-002's 5x device-time claim
needs an explicit, reported estimate (not just an implicit assumption) so
the final report can substantiate it rather than asserting it.

## Decision 6: Tie-breaking rule

**Decision**: If two or more Round 3 finalists are statistically
indistinguishable (overlapping mean ± stddev), prefer the candidate with the
smaller shared-memory footprint, then the smaller accumulator count, as the
final recommendation.

**Rationale**: Per spec User Story 2 Acceptance Scenario 2, a documented
rule is required rather than an arbitrary pick. Smaller LDS/register
footprint is preferred because it leaves more headroom for whatever else
shares the GPU in a real inference pipeline (concurrent shaders, other
model ops) — a tile that ties on raw throughput but uses fewer resources is
the safer production choice.

## Decision 7: Device/driver safety cadence

**Decision**: Re-verify driver hash (`md5sum` of `vulkan.samsung.so`) and
device availability (no other `llama`/`coopmat` process running) at the
start of every round (not just once at the start of the whole search), and
re-pin clocks at the start of every round. If either check fails or drifts
mid-round, halt that round and report the partial results collected so far
rather than continuing under unknown conditions.

**Rationale**: Directly implements Constitution Principles VII/VIII and
spec FR-007. This session already established the exact commands for both
checks (`md5sum /vendor/lib64/hw/vulkan.samsung.so`,
`ps -A | grep -iE "llama|coopmat"`, `pin_freqs.sh`) — reused as-is, not
re-derived.

## Decision 8: Correctness gate and reporting format reuse

**Decision**: Reuse `COOPMAT_BENCH_CORRECTNESS_ONLY=1` (fp32-reference,
tile-aligned small shapes, `abs=0.5`/`rel=0.05` tolerance) exactly as-is for
every candidate's correctness gate. Reuse the exact comparison-table format
already established in `jira-tile-sweep.md` (speedup vs T-tiled, vs dbuf1,
FLOP-weighted GFLOP/s per shape) for the final report, so the new result is
directly comparable to existing published numbers without re-deriving a
methodology.

**Rationale**: Constitution Principle IV requires comparability across
features and over time; inventing a new correctness or reporting format
would violate that without adding any value.

## Decision 9: `WEIGHT_STORAGE` clarification (discovered during T011)

**Finding**: The yaml's `WEIGHT_STORAGE` parameter (`texture2d`/`buffer`)
generates two shader variants per tile geometry, but `QuantizedLinear.cpp`'s
weight-prepack code (`add_q4_linear_weight_pack_node`, around the
`utils::StorageType storage_type = utils::kTexture2D` line) does not let a
caller choose between them — it defaults to `kTexture2D` unconditionally and
only falls back to `kBuffer` when the packed weight matrix exceeds the
device's `max_texture2d_dim()`. None of this workstream's production Llama
shapes come close to that limit, so the `WEIGHT_STORAGE=buffer` shader
variant is **structurally unreachable** for any realistic measurement —
consistent with every real measurement taken this session (and in the
original `jira-tile-sweep.md` sweep) using the `_texture2d_half` suffix
exclusively.

**Decision**: Build and measure only the `WEIGHT_STORAGE=texture2d` variant
for every new shortlisted candidate. This isn't a scope restriction chosen
for this feature — it's the only variant any candidate can actually be
exercised through given how weight storage is currently selected. (This
also clarifies the earlier "buffer storage only" framing from this
session's conversation: the constitution's Buffer-vs-Texture3D distinction
— B-coopmat vs T-tiled — is about the op's activation/output tensor
storage, which is already fixed to Buffer for every coopmat dispatch
regardless of `WEIGHT_STORAGE`; it was never actually in question here.)

**Alternatives considered**: Building both variants regardless was rejected
as pure wasted build/device time for a code path that cannot fire.

## Decision 10: `COOPMAT_BENCH_QUICK` mode (added mid-Round-1, per user feedback)

**Decision**: Added a `COOPMAT_BENCH_QUICK=1` env var to
`test_coopmat_linear_bench.cpp` that restricts the harness to 3
representative shapes (largest FFN shape, a down_proj shape, and the
smallest KV-proj shape) and to just the `linear_q4gsw` op, instead of the
default 13 shapes x 4 ops. Rounds 1 and 2 use this mode; Round 3's final
confirmation uses the unrestricted default (full 13-shape, FLOP-weighted
methodology matching `jira-tile-sweep.md`, per Decision 8).

**Why this was needed**: Round 1 (25 candidates, full-shape/full-op mode)
took ~2.7 minutes per candidate in practice (~67 minutes total) — far
slower than estimated, because the harness always runs its complete case
list (13 shapes x 4 ops x 2 storage = 104 perf cases, each with
warmup=3+runs=5 GPU submissions) regardless of which coopmat variant env
var is set; only ~26 of those cases are even relevant to this feature (the
other 3 ops' shaders are unaffected by our variant selection and were being
uselessly re-measured every single invocation). Validated: quick mode
reproduces the known winner's real GFLOP/s numbers (2655/2655/2215 vs the
already-known 2653/2667/2359 at the same shapes) while cutting
per-invocation wall time from ~162s to ~24s (~7x).

**Rationale**: This is exactly the successive-halving principle from
Decision 4 (spend less per-candidate cost on rounds with more candidates),
just applied one level lower than originally planned — since the harness
can't cheaply skip *shapes* via `COOPMAT_BENCH_M` alone (that only scales
M, not which shapes run), the actual lever was restricting the shape *list*
and *op list* directly. Round 1's 25-candidate pass already in flight when
this was added was left to finish on the slower full-mode binary rather
than restarting it (see Decision 7 -- don't discard in-progress real
measurements over a process improvement); this mode is used starting with
Round 2.

**Alternatives considered**: Reducing `COOPMAT_BENCH_M` for early rounds
was considered first but rejected — it would reduce per-case GPU compute
time but not the number of cases (the actual dominant cost, given each
case's fixed Vulkan dispatch/sync overhead and the 8 GPU submissions per
case from warmup+runs), so it wouldn't have addressed the real bottleneck.
