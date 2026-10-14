# rdna3-780m status

Updated 2026-10-14 19:41 UTC. Actor: Codex.
Stage: bounded RDNA3 RADV matrix; launch blocked on the owner monitoring decision. Nothing built, measuring or queued.
Full assembled campaign including final owner decision read before other inspection.
Initial branch clean on topic/wmma-rdna3-780m; no existing campaign job or device lock holder found.

Reviewed inventory commit: f393c3ff7ff4fc41e4651a5109a8509097c501f6.
Source base: 617897184529ff180af491ee0d5d7de778cbb6c6.
Archive SHA256: 65d1e3ca0eb7e9dffa8e9fbbb1a100bf2cf609868720b9053de67ff8bf5247d8.
RDNA3 manifest SHA256: edcfa39ea7d8fac9c907a6a901754b96aa2d2894302b9c8ad623b90c2780d923.
Archive and all 18 provenance file hashes confirmed against the supplied record.
Inputs remain read-only; all 19 candidates pass the partial queried declared requirements.
Runtime model storage, exact module allocation and origin binding remain pending.

No new per-cell results exist. Planned attention control is 780m-final unchanged across arms; actual dispatch evidence is pending.
Preflight source/model/property checks are recorded in proposal.md and
results/preflight-candidates.csv. All 53 pinned source hashes also match.
All 19 candidates pass the queried declared requirements; 15 have predicted production
storage and four are retained for alternate-storage correctness. None is launch-ready:
compiled allocation, runtime storage and origin binding remain future prerequisites.
The complete property signature and private host/compiler/sensor/model records remain
under the artifact root. Six model sizes and serialized context/group/embedding metadata
match the kit; runtime constant-method execution is pending. No per-cell measurements,
profiles, builds or gate outputs exist. Limits are registered in proposal.md.

Independent preflight/blocker review completed; see REVIEW.md.
Next: wait for the owner monitoring ruling. This matrix remains unstarted.
No repeated recomputation/review while waiting; check for an answer at most every 20 minutes.
No pushes, PRs, services, host policy changes or work outside this stage.

## Decision needed from the owner

Current foreign-workload monitoring cannot read the display users' DRM fdinfo
(PermissionError for the active greeter and Xwayland). The historical campaign accepted
own-user engine sampling plus device-wide busy percent for other users, but this task
explicitly says historical exceptions do not become new exceptions. R6 requires absence
of foreign GPU work before/during/after each run. May this run use that limited monitoring,
with the coverage limitation attached to every result, or must an owner-provided fully
observable idle environment be available first? No sudo, service or policy changes will
be performed by this campaign. No build or measurement queue will launch while this
question is pending. Independent source/property/model-readiness work and the single review are complete.
The preflight record is supported; measured-stage completion is not claimed.

## Coordinator hold

HOLD and HELD are under the assigned artifact root; the copied adapter derives that root
from this checkout. A separately named dry-run hold passed. The real HOLD was untouched.
No queue/watcher/sampler exists; future units are a build, gate or interleaved session,
with duration pending calibration.
