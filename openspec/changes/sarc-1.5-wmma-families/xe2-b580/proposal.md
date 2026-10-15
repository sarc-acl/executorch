# Xe2 B580 bounded linear matrix

Source base `617897184529ff180af491ee0d5d7de778cbb6c6`. Inventory pinned at
`f393c3ff7ff4fc41e4651a5109a8509097c501f6`; all input and 53 source hashes checked.
Only existing 4w and 8da4w kernels are reachable through additional dev profiles.
The finite manifest contains 12 storage-specific candidates, seven kernel bases.
`matrix.json` retains each exact dispatch name, fingerprint and origin rule.

## Pre-registration

Five valid fresh-process runs per arm/cell, alternating control first on odd repeats,
candidate first on even repeats; up to three replacement pairs. All six cells in each
session. Rates are valid-run medians, with spreads and bootstrap uncertainty
reported (10,000 unpaired bootstrap resamples, seed 58020261014, 95% interval).
Use the first five valid runs per arm in CSV order; preserve every extra valid/invalid
row. Noise band ±2%; baseline discrepancy above 3% per cell requires explanation
before candidates. Kernel-screen margin 3% in every round; no screening or search queued.
Family bounds remain geomean throughput ratio >=0.97 and each cell >=0.92 against the
same-session remeasured tuned control. No family-wide acceptance from this member alone.

A/A uses the pristine base and selector build with the same final B580 profile in both
arms. Clock floor is 97% of the lowest valid per-run median clock in A/A, device-wide.
Minimum five clock samples within prefill; thermal reasons reject. Power/current limits
are recorded. Foreign compute activity before/during/after rejects. The desktop-idle
exemption of the historical run does not apply; wait for the existing idle indication.
Any positive observed foreign DRM engine delta invalidates a prefill, including A/A.
Unreadable counters invalidate. The foreign ceiling is zero; no historical 5% allowance.
No clock, power, session or service policy is changed.

Context is 2560 in all six serialized constant methods. Serialized linear arguments
confirm group 128; q4 embedding op and group 32 are present. Prefill projection IO is
texture3d, width packed. The buffer output projection is M=1, outside this matrix.
Buffer candidate variants remain in production-diff coverage, never represented as
measured prefill candidates. Parent SPIR-V declarations have now been checked;
whole-model allocation legality is BLOCKED and runtime dispatch remains unobserved.

## Profiles and attention control

All arms have `ET_VK_SARC_UNVERIFIED=1`, no unrelated override/variant/force variables.
Control is `b580-fused1-order-p13`. New names use `b580-fused1-order-wmma-` so the existing
B580 fused-pair lookup is preserved; QK/AV preference entries are copied exactly from
the control, and softmax remains the control's default. No shader or attention logic edits.

`q4-narrow` substitutes B70's flw only at N<=512; `q4-band` substitutes its flib where
it fits. Both keep the B580 p13/i 8da4w control. `dq-table` tests the inventoried table
base, `dq-i` the ident base, `dq-n26` B70's n26 at N*K>=33554432 with i otherwise.
Those three keep the B580 4w table. `b70-linear` reproduces the B70's complete final
linear rule set with B580 attention. All use existing fit and parent-service checks.
`q4-table` and `dq-n13` are identity aliases of control, covered by baseline/A/A rather
than repeated candidate sessions. The n13 rule remains N*K>=24000000; no threshold tuning.

The full unmodified verify gate runs on each measured build/environment, including
production-diff in both storage modes, decode and three-prompt next-token comparisons.
Compare statuses with the parent snapshot; preserve failures and invalid rows.
A changed linear accumulation order is judged by D3 where necessary; no precision claim
is inferred from matching tokens. Attention tiers for a new attention candidate are
inapplicable because attention remains unchanged.

## Adapter changes

Selected adapters copied from `sarc-1.5-b580-fused-port/tools/` in this clone.

| File | Adaptation |
|---|---|
| host.sh | Extra change-directory nesting, approved base/profile/root, pinned image, eight build jobs, own artifacts/TMPDIR/HOLD; restore idle wait and reject device-loss marker |
| build-both.sh | Preserve artifact TMPDIR; exact-commit recursive export and unchanged sarc build/golden tools |
| hold.sh | Explicit own artifact directory from host adapter rather than old depth calculation |
| session.sh | Drop historical extra-12 desktop exception; standard replacement limit |
| e2e5.sh | Zero raw foreign-workload ceiling in A/A and candidates; add row-level provenance and propagate device loss, retaining invalid rows |
| sampler.py | Rescan foreign clients at each 50ms poll; include initial cycles of newly discovered clients; no child poller processes |
| stage.sh, launch_runner.py, clean_exec.py | Stage recorded binaries plus identical cache/environment launcher, prime model before every process, clear unrelated ET/Vulkan overrides |
| gate_linear.sh | Wrap unmodified verify with explicit controls, sampler/guard, loss propagation, preserved outputs and strict foreign-activity check |
| summarize.py, gate_check.py | Removed unused historical adjudication references; no historical acceptance code is reused |

Raw paths, infrastructure facts, manifests and hashes stay under this run's artifact
root. Stage records include commit, binary, profile, session/candidate, device/driver,
cell, validity/reason and sensor/log paths; no local patch in measured builds.

No measurements exist yet. Runtime projection must be updated from A/A and snapshot
wall times before candidate queue launch; a projection over 48 hours requires an owner
report. No automatic promotion, upstream work, push, PR, roofline or parameter search.

## Preflight validation

`check.sh --no-build` PASS after recursive submodule initialization completed:
release selector 1240 checks; dev selector 1921 checks. Initial run during preparation
reported incomplete gitlinks and is retained in artifacts. `git diff --check` passes.
CPU-only launcher tests show unknown ET/Vulkan settings removed, expected controls
retained and model cache read recorded. Named HOLD dry-run checks wait, acknowledgement
and shared unit lock on the NAS; a write-only descriptor initially failed shared flock
on NFS and was corrected to a read/write descriptor before any campaign unit.
Synthetic sensor-window tests reject a positive foreign delta below the old rounding
precision and retain exit 77 as invalid. No benchmark result is inferred from these tests.
Fresh Codex preflight source review passes pinned inputs/profile attention/manifest matching
and identifies orchestration fixes. Exact compiled module, formats and measured evidence
remain prerequisites; the review does not close the matrix.

The first nohup launcher was reaped with its tool session before a source/build tag
was created. Its empty log is retained. A setsid subprocess dry-run is checked before
using that detachment mechanism. HOLD is checked atomically by the outer unit wrapper;
inner scripts do not wait for HOLD while holding the busy lock.

Static follow-up found the table 8da4w token collides with an existing dev xe2bt
candidate in candidate-before-row suffix lookup. `dq-table` now has attention
preferences only, so its linear choice is the original table choice. No new exact-name
selection API or kernel body is introduced. A CPU dispatch-name audit will verify every
profile against the pinned manifest before GPU candidates.

## Recovery allocation and CPU evidence, 2026-10-15

The preserved parent build was reused. Its five recorded binary/library hashes and
53 shipped SPIR-V variants still match. Twelve candidate modules have declarations
equal to their inventory totals and within the queried shared limit; these are
module declaration bounds, not a driver allocation measurement.

The actual image creation path confirms the old 8B width conflict and also exposes
a height conflict in the RoPE prepack outputs of every model. Pure extent functions
linked from the existing parent archive reproduce the violating extents. All six
whole-model cells are therefore BLOCKED; no partial legal model subset exists with
these inputs. BLOCKER.md gives exact source references, query matching, counts and
the conforming-resolution dependency. No model GPU work or parent rebuild occurred.

The CPU constant-method helper executed nine methods per model, with all 60 outputs
matching static metadata. Context 2560 is runtime-verified; group/embedding metadata
is validated from serialized graph arguments because no such constant methods exist.

The CPU selector audit contains 432 linear rows and 36 fallback attention rows across
control, six candidates and two identities. It reaches every one of the 12 pinned
inventory shader names in the intended storage. Identity profiles equal control;
single-scheme candidates leave the other scheme unchanged. Every arm has default
softmax and fused variants `d64_t16x64s16m8g4roj,d128_t16x128s16m8g8oj`.
`results/selector-dispatch.csv` records source selection, not observed GPU dispatch.
`results/cell-coverage.csv` records zero executed runs for every blocked cell.

Small CPU helpers use the host compiler and existing parent archives; command arrays,
source/archive/binary hashes, compiler identity and outputs are private artifacts.
They neither rebuild kernels nor construct model resources. Failed helper compilation
diagnostics are retained. There are no timing rows, calibration, gates, margins,
accepted candidates or family acceptance. Existing GPU orchestration still requires
exact-run validation if a separately authorized conforming resolution enables launch.
