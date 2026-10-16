# Independent source and preflight review

2026-10-14. Reviewer: fresh Codex context, independent of the actor.
Reviewed preregistration at base `617897184529ff180af491ee0d5d7de778cbb6c6`
with the appended `b70-wmma` selector and current run documents.

**Source/manifest review PASS; measured-stage completion and GPU launch readiness
remain pending.** No selector or inventory-coverage defect was found.

Read the complete 978-line task, including the final owner decision, before other
checks. Read INPUT, all five required inventory documents and provenance; inspected
STATUS, processes, artifacts and locks before review. No owner question was pending.
Independently confirmed the archive and all 18 per-file hashes. All 53 source-input
hashes match pinned git objects; the live source differs only by the appended selector.

Profiles and candidate coverage exactly match all seven Xe2 kernel bases and twelve
storage variants, including fingerprints and declared shared bytes. Checked registered
storage masks, candidate-before-row lookup, prior service, fit/layout checks and origin
predicates: n13 requires N*K>=24,000,000, n26 requires N*K>=33,554,432 and k32 requires
N<=512. The selector preserves prior choices for absent/unmatched profiles, unrelated
devices/ops, decode and failed fits. It copies the existing override, preserving fused
attention and softmax fields; the final base profile and attention preferences remain
identical across arms. No shader was added or changed.

Queried matrix types/shapes, subgroup controls and declared geometry satisfy the
manifest's source requirements. The kernel-required subgroup differs from the device
default, as already documented historically; supported compute size control covers it.
Serialized model groups, embedding op/group and context constants, and current file
sizes agree with the model specification. Runtime constant-method execution and exact
compiled allocation/capability/format checks remain required before timing. Private
unit, sensor, occupancy and model details stay in artifact-local reviewer evidence.

The preregistered seven profiles cover six cells each: 42 comparisons, 420 timed
processes at five repeats or 588 at seven, excluding gates and next-token checks.
Family limits remain 3% geometric-mean and 8% per-cell slowdown. Historical reference
numbers resolve to the cited records and are explicitly separate from future controls.
Release code, sarc/tools, goldens, prompts and tolerances are unchanged; new deliverables
are confined to this run. `git diff --check` passed. The actor's existing check log
reports selector checks passing; the reviewer did not compile or rerun them.

## Launch-adapter finding

The copied `tools/trace.sh` changes model cells without filling the page cache before
each change, whereas D5 explicitly requires that for traces. `tools/gate.sh` invokes
this trace path unconditionally. Do not launch that path as-is: either keep these
historical wrappers inactive for this bounded stage or establish compliant cache
preparation before using it. The unmodified correctness gate remains required.
The copied control wrapper also requires historical SDPA tier passes; fresh attention
candidate tier testing is inapplicable because this stage changes no attention kernel.

## Checks not performed

No builds or GPU jobs were started by the reviewer. Timing/validity recomputation,
golden checks on measured binaries, exact gate/binary/environment matching and measured
margins have no evidence yet and remain pending for the device stage. A/A calibration,
runtime metadata, dry HOLD check and focused selector execution are actor prerequisites.
A change-specific shader synchronization audit is inapplicable because shaders are
unchanged. This review does not accept a candidate, a family set or stage completion.
