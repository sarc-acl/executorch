# Independent preflight review

Fresh Codex reviewer, 2026-10-14. Reviewed commit
`3aa690c580ff7543c9ef48c41a3287a9170a66ce` against
`617897184529ff180af491ee0d5d7de778cbb6c6`, plus the pre-measurement
bootstrap clarification in `proposal.md` (seed 407014, 20,000 draws,
percentile 95% interval, ratio of arm medians), and the focused adapter/status
corrections made during this review.

**PASS for the bounded preflight.** The recorded source, numeric and static-model
conclusions reproduce, and the findings below were corrected and checked in
this round. This review does not authorize launch or complete the device matrix.
The existing owner question remains untouched.

## Findings resolved in this round

- `tools/common.sh` mixes private and source status evidence: `gpu_gone()`
  includes the hostname and absolute marker path in a message appended to
  tracked `STATUS.md`. The correction keeps details in artifacts and appends
  only a generic loss notice and artifact-relative marker reference to source
  status. Inspected the focused diff; no private value is written there.
- The inherited parent comment incorrectly describes a closing arm with no
  environment. The correction describes the invariant final attention controls;
  the executable parent constants were already correct.
- Removed the extra blank line at the end of `STATUS.md`. The final
  `git diff --check` against the base and `bash -n tools/common.sh` pass.

## Independently checked

- Read all 978 campaign lines, including the launch decision, then `INPUT.md`,
  provenance and reviewed snapshot documentation. Inspected processes, locks,
  queues and artifacts: no build, measurement or sampler process was present;
  source preparation recorded rc=0. Verified all 30 recursive submodule HEADs
  and pinned objects in this clone.
- Recomputed the archive hash, all 18 input-file hashes, all 53 source hashes
  against pinned git objects and the checkout, and all recorded preflight
  artifact hashes. `results/input-hashes.json` matches reviewed provenance.
  Checked 101 NVIDIA source-reference occurrences, exact row/YAML names and
  all 11 source fingerprints.
- Recomputed candidate geometry, declared shared-array totals and tile fit for
  all 12 production NK pairs. All 11 CSV rows match the manifest and
  `preflight/candidate-numeric-checks.json`. Checked the complete 15-record
  matrix signature and required types/scope against local Vulkan headers,
  and subgroup/control, compute-stage, shared/workgroup limits and basic
  features against `preflight/vulkaninfo.txt`. These are partial prerequisites.
- Compared the static inspector with program/Vulkan schemas and header
  definitions, then reran its CPU mmap inspection on all six models with
  identical parsed results. Independently reconstructed `model-summary.json`:
  kit sizes, constant values, group-128 linear calls, group-32 packed embeddings,
  112/196/224 prefill calls per scheme, and 12 distinct NK pairs agree.
  Native prefill IO is width-packed texture3d: seven manifest entries have
  native coverage; four buffer entries remain alternate correctness coverage.
- Recomputed polling intervals from all saved monotonic rows: requested 5 ms
  median 5.133012 ms, range 5.103417–5.230897 ms; requested 20 ms median
  20.143821 ms. These idle readout intervals do not establish loaded sensor
  freshness or sufficient samples in a future prefill window.
- Verified copied export/shim/warm/prompt files are byte-identical to this
  clone's historical adapters; inspected the complete constants diff,
  attention-control source and new tools. Shell/Python parse checks and the
  current source privacy scan pass. Read `preflight/hold-dry-run.txt` and
  the HOLD implementation; the recorded test acknowledges, releases the lock,
  resumes and cleans its markers. The reviewer did not repeat that mutation.

## Pending evidence and limits

No candidate selector/profile, exported build, measured binary, gate or timing
exists. Runtime-method validation, descriptor/format support, actual runtime
graph layout, exact-module allocation, baseline/A/A calibration, launch-driver
guard/sampler behavior and finite-matrix projection remain pending. Raw-run
medians, validity, uncertainty, gains, margins and member coverage cannot be
recomputed yet; no performance or family acceptance result is asserted.

Golden and exact-build/gate comparisons require future builds and are pending
launch evidence. A synchronization audit of changed shaders is inapplicable:
no shader changed. All 11 committed additions and the reviewed working-tree
clarification/corrections are inside this run directory; there is no release-zone,
`sarc/tools`, golden, tolerance or upstream-prompt edit. The copied unaligned
prompt is unchanged. `check.sh --no-build` was not run because it compiles.

No reviewer GPU query/job, build, inference, push, PR, external message,
service or host-policy change was performed. This is the single independent
review of this newly prepared preflight; subsequent work waits as recorded
in `STATUS.md`.
