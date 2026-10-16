# Preflight and blocker review

Fresh Codex reviewer, 2026-10-14. One review of the unstarted bounded matrix.
The complete assembled campaign, including its final owner decisions, was read
before other inspection; the supplied input record and this run's status followed.

Verdict: the saved preflight results and monitoring blocker are supported. The actor
reports the status consistency corrections below completed. This is not launch approval, a completed
matrix, a full candidate-legality verdict, or member/family acceptance. No candidate
profile, measured build, gate or per-cell result exists.

## Findings and corrections

The inspected STATUS.md said all candidates were pending queried properties, then
reported the completed partial property checks. It also carried an update time
predating the saved hold test. The actor reports correcting the stale sentence to
the remaining prerequisites (exact compiled allocation, runtime storage/layout and
origin/service binding) and refreshing the time before the consistent-state commit.
These documentation corrections were not separately re-reviewed; no repeated review
is required while the owner question remains pending.

No incorrect reported preflight count, provenance mismatch, prohibited source edit
or privacy leak was found. The proposal explicitly leaves the downstream throttle
bit interpretation unresolved before timing; this review does not resolve it.

## Independently checked

The branch is the assigned topic branch and HEAD is the approved full base
`617897184529ff180af491ee0d5d7de778cbb6c6`. Process and lock inspection found no
campaign build, measurement, sweep, queue, watcher or sampler, and no device-lock
holder. Existing display processes remain. Read-only attempts to list both display
processes' DRM fdinfo directories independently returned PermissionError, confirming
the monitoring coverage problem. No process or service was changed.

Archive SHA-256 reproduces
`65d1e3ca0eb7e9dffa8e9fbbb1a100bf2cf609868720b9053de67ff8bf5247d8`.
All 18 snapshot-file hashes reproduce the supplied provenance record, and the archive
contains exactly those 18 file payloads with the same hashes. The RDNA3 manifest
reproduces `edcfa39ea7d8fac9c907a6a901754b96aa2d2894302b9c8ad623b90c2780d923`.
All 53 source-hash rows match both the pinned base's git objects and the checkout.
The RDNA3 manifest's 58 unique source references resolve within 28 pinned source
files, with the full approved base in each reference. Supplied inventory review,
validation, profile closure and legality documentation were read. The inventory
commit itself is absent from this clone's object store: its attribution rests on
the supplied coordinator provenance, authenticated here by the archive/file hashes,
rather than an independent comparison to that commit's tree.

The matrix query was inspected without executing it. Its application/instance and
matrix structure fields match the recorded Vulkan header. On this ABI their sizes
are 48, 64 and 56 bytes with 8-byte alignment; matrix field offsets are sType=0,
pNext=8, M/N/K=16/20/24, A/B/C/result=28/32/36/40, saturation=44 and scope=48.
The KHR property sType 1000506001, subgroup scope 3 and component-type values
fp16=0, fp32=1, signed-int8=3, signed-int32=5 match the header. The function-pointer
signature and initialized output records also agree. The saved enumeration has
14 records, including both required non-saturating subgroup-scope 16x16x16 tuples.
The saved device profile and Vulkan output agree on RADV, the shared-memory and
workgroup limits. Required-size control covers compute and subgroup size 32 even
though the recorded default subgroup is 64.

Independent arithmetic against the saved profile, matrices, models and manifest
reproduces every CSV candidate ID, fingerprint, exact shader, declared shared bytes,
workgroup size, origin-route set and fit count:

| Reported partial check | Recomputed result |
|---|---|
| Declared MMA/subgroup/workgroup/shared requirements | 19 of 19 pass the implemented checks |
| Predicted production storage | 15 candidates, each with 12 aligned production fits |
| Alternate storage only | rdna3-08, -10, -11 and -15, each with zero predicted production fits |
| Production projection NK union | 12 distinct pairs; final vocabulary projections have M=1 and are excluded |
| Enumerated cooperative-matrix records | 14 |
| Serialized model files / sizes | 6, each equal to the kit's byte count |

The model inspector's hard-coded table fields and union/type/storage/layout values
were compared with the repository schemas. Read-only reinspection of the six model
files reproduces models.json byte for byte; the repeat output is private raw evidence
as `preflight/reviewer-models.json`. Serialized context and sequence constants are
2560 in all six, prefill linear group arguments are 128, embedding operators are
q4 group 32, and prefill projection IO is fp16, texture3d, width-packed. Default
weight storage remains an inference. Its source prediction agrees with the actual
packed-weight extent test in QuantizedLinear.cpp for these shapes. These checks do
not establish runtime constant-method execution or dispatch.

Copied export and hold adapters differ from the historical originals only as
documented: checkout/root discovery and removal of historical-root comments.
Recursive object export and hold logic are unchanged. Shell syntax passes. The
saved isolated hold-test log supports the reported block/release behavior; no real
HOLD was altered by the reviewer. Measurement chains are not installed or running.

Registered limits retain the six-cell 0.97 geomean and 0.92 individual throughput
ratios, +/-2% noise, 3% published-baseline explanation threshold and 3% screen
margin. The A/A repeat/floor rules and bootstrap seed/count are specified before
data exists. The listed thermal and unknown bits reproduce mask `0xFFFFE670`.
Twenty saved sensor-read bundles took 219.710 to 1039.909 microseconds; availability
and read latency do not establish timed-window coverage or an A/A clock floor.
Attention controls are explicitly invariant and source descriptions are correctly
distinguished from future dispatch evidence.

HEAD versus base has no changes. The inspected staged additions are confined to
this run's directory; no release file, shader, sarc/tools, golden, tolerance, prompt
or another device's content changed. Privacy scanning of the deliverables found no
private home/mount path, host/user identity, UUID or network address. Raw inspection
remains outside the checkout. Staged whitespace checks pass.

## Not checked, and why

No GPU query, job or build was started by the reviewer. GPU properties were checked
against saved raw records and header ABI, without requerying the device. Whole-model
SHA-256 values in the private actor log were not independently rehashed; serialized
metadata and current byte sizes were checked read-only. Runtime graph initialization,
packed extents/layout, origin predicates, preceding service availability, exact
compiled allocation and actual linear/attention dispatch remain future prerequisites.
The downstream kernel's throttle source is not established by the saved header
record. Current hardware-policy reinspection was not performed by this reviewer.
After the inspection, the actor reported adding durable private host/policy/permission
and compiler hash records and updating the proposal's evidence index. Those late
additions were not independently rechecked in this single review.

There are no runs.csv measurements or timed binaries, so medians, gains, uncertainty,
validity counts, parent snapshot comparison, same-binary gate matching, reference
error and shipped-SPIR-V golden checks are inapplicable to this preflight record.
No new or changed shader exists for a change-specific shared-write audit. Existing
candidate shader synchronization has not been independently gated or certified here.
check.sh --no-build was not run because it still compiles selector binaries.
Builds, gates, goldens and timings are not conditions for validating this blocker
record. External member evidence, legal matrix completion, margins and family
coverage remain unstarted or unavailable and cannot be inferred from this member.

The single monitoring question stays in STATUS.md. Await the owner ruling after
recording this review; do not launch a build/measurement queue or repeat review and
recomputation while waiting. No commits, pushes, PRs, other-checkout access, system
package installation, service changes or host-policy changes were performed.
