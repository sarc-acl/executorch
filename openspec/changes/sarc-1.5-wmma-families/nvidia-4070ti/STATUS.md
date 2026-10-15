# NVIDIA WMMA matrix status

Updated: 2026-10-15 05:16 UTC

Stage: REPLACEMENT SNAPSHOTS AND A/A COMPLETE — INDEPENDENT REVIEW CHECKPOINT.
Running: none. feature-control-aa-r1 completed rc=0 at 05:12:32 UTC. No build,
runner, poller, sampler or campaign lock holder remains. No owner question is
pending. No candidate gate or session has started. The finite matrix remains
INCOMPLETE; this checkpoint is feature enablement, control behavior and calibration.

## New measured evidence

All four probes (parent/matrix × llama/backend exact compiled libraries) observed
the actual vkCreateDevice chain: the memory-model extension enabled,
vulkanMemoryModel=1, DeviceScope=0, availability/visibility chains=0, driver rc=0.
They create/destroy the compiled Runtime with no model, shader or dispatch.
Source/library/helper/command hashes are retained. Probe instrumentation is absent
from the measured runners. Evidence: results/replacement-snapshots.json and
artifact checks/enabled-features/, stage/enabled-features-mm-r1/.

Replacement s0-parent-mm-verify and s0-topic-mm-control each have all 37 command
observations: 22 runners rc0, correctness rc0, twelve production-diff commands
rc0/ALL PASSED, four default/tiled SAME lines, two decode commands rc0. Both
linear benchmarks remain rc1: 24 cases per scheme, ok=true, twelve confirmed
and twelve unexpected_coopmat labels; benchmark correctness is SKIPPED because
verify.sh requests --skip-correctness there. This is preserved control behavior,
not a plain gate PASS. No foreign/thermal sample, slow-load possibility or runner
failure occurs across the 74 observations. Raw outputs are identical line by
line with rates removed, also matching the old parent observations. Empty diffs
and all logs stay in the replacement stages. Actor audit:
checks/snapshot-mm-audit.json; public summary results/replacement-snapshots.json.
Independent review must redo these checks from raw evidence.

## Calibration and remeasured historical control

A/A s1-aa-mm-r1 has 84/84 valid timed rows: seven per arm for all six cells,
zero invalid/replaced rows, 2048 prompt / 0 generated tokens, rc0, >=9 samples
inside each prefill window, no thermal/foreign observations and fully warmed
model pages. Actual odd/even arm order, binary/source/environment provenance
and generated-text equality match in every pair. The first three post-run child
checks find zero remaining descendants; no sampler process remains on recovery.

The pre-registered rule extended every cell from five to seven because the first
five include a >1% arm ratio and >3% spread. The lowest valid per-run median
clock is 2610 MHz, so the one device-wide floor is floor(0.97*2610)=2531 MHz.
No threshold changed. A/A geomean is 0.995829116934, within the 1%
limit; every cell is within 2%. All historical-baseline differences are within
3%, largest +2.44%, so none requires an out-of-band explanation. The table uses
valid-run medians and paired-bootstrap intervals, seed 407014 and 20,000 draws.

| Cell | Parent tok/s | Topic tok/s | Topic / parent | Paired 95% interval | Parent vs historical |
|---|---:|---:|---:|---:|---:|
| 1b-4w | 35310.30 | 35929.80 | 1.017544 | 1.000000–1.017544 | -1.72% |
| 1b-8da4w | 40156.90 | 39384.60 | 0.980768 | 0.961538–1.040000 | +1.96% |
| 3b-4w | 14222.20 | 14124.10 | 0.993102 | 0.993102–1.000000 | +1.39% |
| 3b-8da4w | 16650.40 | 16384.00 | 0.984000 | 0.976193–1.000000 | +2.44% |
| 8b-4w | 6420.06 | 6420.06 | 1.000000 | 0.996865–1.003145 | +0.31% |
| 8b-8da4w | 7501.83 | 7501.83 | 1.000000 | 1.000000–1.003676 | +0.00% |

The 1 ms runner timer step is about 1.7–2.0% in the 1B cells. Their median arm
differences are one timer step; 1B 8da4w's interval extends from about -3.85% to
+4.00%. This is noise/uncertainty, not a candidate gain or permission to widen
the limits. Seven repeats and the 2531 MHz floor apply to future candidate runs.

Actor recomputation reads every saved CSV row, runner JSON stats, sensor log,
record, binary, paired order and output hash, then recomputes the extension rule,
medians, spread, bootstrap intervals, floor, geomean and historical comparisons.
It matches the queue's results (geomean differs only by final floating-point
rounding). Source tool: tools/recompute_calibration.py. Raw audit:
checks/calibration-audit.json; public summary results/calibration.json. CSV:
stage/s1-aa-mm-r1/runs.csv; original calibration.json remains immutable there.
All raw observer records retain UNASSESSED for timing; per-row CSV decisions and
this raw-data audit supply the calibration validity, without relabeling records.

## Exact builds and approved source scope

parent-mm-r1: 5f02679b9a20d0b43ed44ea42cb1d779e1f269ac, original base plus
only the approved three-file feature fix. matrix-mm-r1:
dd57ce0e4fda4d7a5c72f9ce3c13b075428eec9e, the same runtime source bytes plus
eleven existing dev-only selectors and run-local evidence/tools. Both use exact
git-object exports of 30 pinned submodules and the pinned image/compiler.
All four unmodified golden checks PASS (53 shipped variants), and every one
of 2,091 SPIR-V modules per component/build matches the retained old parent.
All eleven shared-array audits match source. Exact binary hashes rechecked.
results/feature-builds.json; raw manifests, hashes and golden outputs remain
in build/parent-mm-r1/ and build/matrix-mm-r1/.

New parent CPU metadata execution passes all nine constant methods on all six
models, context/sequence length 2560. It links core runtime/mmap loader only;
ldd has no Vulkan dependency. Group-128 linear and packed 4-bit/group-32 embedding
facts remain from serialized-graph inspection. Evidence:
results/runtime-metadata-memory-model.json and metadata/parent-mm-r1/.

The 04:08 owner decision covers exactly Adapter.cpp, Device.cpp and Device.h
from Orin cbb7df9b6a7a605d28192fbc2a8b62c6f3611d9c. Supplied patch SHA256
5434b8e5c94ba8b18fb11e502577054bc3dea3af4e94263e6470113e316938a9.
Feature commit 8ebe66c2b and results/memory-model-fix.json record it. No release
shader, sarc/tools, golden, HOOKS, prompt or tolerance changed. Unmodified
check.sh --no-build still reports the two owner-authorized zone failures for
Device.cpp/Device.h; selector tests pass 1240 release / 1921 dev checks. That
check is NOT a plain PASS. Raw output: checks/check-memory-model-no-build.txt.

The old builds and snapshots remain preserved as historical observations.
Only superseded CMakeFiles were retired after configuration/link provenance and
removed-file hashes were archived; all exact binaries/libraries/SPIR-V/exports
and failed job records remain. About 30 GiB is free; no new build is needed.

## Reviewed inventory, controls and remaining stage

Inventory f393c3ff7ff4fc41e4651a5109a8509097c501f6; archive SHA256
65d1e3ca0eb7e9dffa8e9fbbb1a100bf2cf609868720b9053de67ff8bf5247d8;
NVIDIA manifest SHA256
ef5d13471421898c8515e9a8356caff64a20aa97432175cbcab7bf3a5cd748b1.
All 18 input hashes confirmed. Queried properties, compiled modules and the
whole-model matching-resource audit remain available; no aggregate peak-memory
or allocation-limit exception is inferred. The manifest has seven native
texture3d and four alternate-buffer candidates. All candidate entries remain
PENDING for correctness; fixed-model fallback must never count as candidate
coverage, and alternate buffer entries need honest separate correctness evidence.

Both arms retain 4070ti-fused1: k4070tiRefine1 fallback, 4070ti_nzf softmax,
fused3sb_d64_t32x32g11s32rko / fused3sb_d128_t16x64g11s32rko and existing K/V
copy. The full gate still checks decode; no attention/decode/fp-linear tuning.

Next: independent review of these NEW feature/snapshot/calibration observations,
then complete the finite candidate gates, native six-cell sessions with same-
session controls, all-six-cell next-token checks on timed/real/unaligned prompts,
alternate-storage correctness, dispatch evidence, uncertainty and margins. A
D1/D3 difference needs its stated evidence; no candidate has been accepted.
This single member cannot establish family coverage, minimality or a split.

Reviewer: no builds or GPU jobs. Recompute from the raw stage logs/CSV, check the
exact measured-build goldens and exported sources, and verify the three approved
release edits. No new/changed shader exists for a synchronization diff audit.
No source-only review is requested and no existing owner blocker is re-reviewed.
HOLD remains untouched; all future jobs use the inherited lock and yield at
boundaries. No pushes, PRs, services, policy changes, new bodies or search.
