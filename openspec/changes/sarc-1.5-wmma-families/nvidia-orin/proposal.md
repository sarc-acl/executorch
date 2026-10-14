# Orin bounded WMMA matrix

Actor: Codex. Source base `617897184529ff180af491ee0d5d7de778cbb6c6`.
Input inventory `f393c3ff7ff4fc41e4651a5109a8509097c501f6`; NVIDIA manifest
SHA256 `ef5d13471421898c8515e9a8356caff64a20aa97432175cbcab7bf3a5cd748b1`.
The archive and every supplied file/source hash were checked. No mutable inventory
or historical measurement is substituted for this reviewed snapshot.

## Stage boundaries

Complete the legal finite matrix, subject to prerequisites and the approved cap;
otherwise report an explicit blocker. No promotion, release selection, new bodies,
parameter search, push, PR, service changes or hardware policy changes.
All attention controls remain the original `orin-fused1` environment.
No measured profile or correctness acceptance is claimed by preflight.

## Threshold registration before measurements

Five valid runs per arm/cell; odd pairs control first, even pairs candidate first.
At most three additional pairs replace invalid rows; retain every invalid row.
A valid timed row has rc 0, finite positive rate, 2048 prompt tokens, zero generated
tokens, at least five clock samples inside prefill, no foreign workload before,
during or after, median clock at least the floor and no thermal throttle.
Clock floor: `0.97 * lowest valid per-run median clock in A/A`, device-wide.
Keep fractional MHz precision; do not round the threshold downward.
A/A validity before floor calibration uses all other criteria; apply the resulting
floor back to A/A. Never use lowest per-cell median as a substitute.
Noise band: +/-2%; baseline discrepancy beyond 3% per cell must be explained before
candidates. Kernel screen margin 3% every round if any screen is needed; no search.
Family limits are unchanged: for each cell define elapsed-time ratio as control
median tok/s divided by candidate median tok/s. Require its geomean <=1.03 across
all six cells and every cell <=1.08. Also report throughput ratios from the same
medians. This definition is conservative and does not widen either limit. Report
uncertainty.
Memory and swap counters are recorded for every 8B process; one process at a time.
Models enter page cache before a cell and whenever the model changes, both arms.
The unmodified full correctness gate still checks decode. Exact-build shipped
SPIR-V must pass; no historical compiler exemption carries over.
Reference-error and near-tie decisions apply only under their full criteria.

## Current preflight

The reviewed NVIDIA manifest has 11 source-distinct storage-specific entries.
Current queried MMA types, subgroup requirement, declared shared-memory sizes and
workgroup invocation counts match all 11. Exact compiled allocation and complete
format/module capability checks remain pending; this is not final legality.
Seven texture3d entries match projection IO in all six models. Four buffer entries
are retained for correctness coverage, not silently measured through texture IO.

Direct read-only flatbuffer inspection confirms 2560 context, group size 128 for
all delegated linears, q4 embedding op/group 32 and texture3d width-packed projection
IO. This is stronger evidence than the kit's historical size-based inference;
constant-method execution on this run's exported binary remains pending.

The current cross image is present and its compiler reports shaderc v2026.1.
No exact parent binary or parent golden result exists yet; no exemption is claimed.
Private device/host/model paths and raw query outputs stay outside the source tree.

## Adapter review

Copied adapters have not been changed or launched. The following existing behavior
cannot be reconciled by changing host constants alone:

| Source | Existing behavior | Required behavior |
|---|---|---|
| `sarc-1.5-orin-fused-port/tools/calibrate_clock.py:18,25,30` | minimum of per-cell medians; accepts two samples and does not filter all validity criteria | lowest valid per-run median; at least five samples; full A/A validity |
| `tools/common.sh:79,106-107` | sampler/watch stop kills only the outer shell; no child cleanup trap | samplers stop their own children and process counts are checked early |
| `tools/common.sh:31`; `tools/deploy.sh:16,25,27,29,33` | ssh executes command strings directly | every ssh script uses `bash -s` |
| `tools/e2e5.sh:85,102` | rows identify parent/cand tags but omit per-row commit, binary, profile, candidate and driver | every row carries the task's complete provenance |

Semantic corrections to copied routines exceed the host-constant-only rule.
The independent reviewer distinguishes supplemental orchestration: separate A/A
analysis, a sampler with explicit child cleanup, bash-s deployment commands and a
deterministic provenance join can implement task requirements without modifying
copied routines. The task does not expressly forbid that route. Do not describe
such new logic as constant adaptation. The existing owner question is pending;
launch remains held while that question is outstanding. Original historical tools,
gates, prompts, goldens and fixed thresholds remain untouched.
