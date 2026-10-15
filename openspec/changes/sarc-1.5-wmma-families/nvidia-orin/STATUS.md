# NVIDIA Orin WMMA matrix status

Updated: 2026-10-15T00:38:29.077965+00:00

Stage: recovery preparation complete; detached exact-parent cross-build queued.
The legal matrix is incomplete. No model GPU job, gate or timed session has run.
Base: `617897184529ff180af491ee0d5d7de778cbb6c6`.
Own branch: `topic/wmma-nvidia-orin`.
Actor: Codex. Independent review is provided by the controller's fresh reviewer.

## Running and next step

Build queue: `.artifacts/jobs/parent-20261015-pinned1/state.json`, relative to the
private campaign root. Build: `.artifacts/build/parent-20261015-pinned1/`.
The queue honors HOLD and the shared desktop build lock, uses at most eight jobs,
and records terminal success/failure in JSON. It performs no deployment or GPU
execution. Export and build are conservatively expected within three hours;
there is no >48-hour projection or unbounded matrix queue at this checkpoint.
No build tag/export existed on recovery; prepared own-clone submodule objects
and verified model/property records are reused. No known target measurement job
or device lock holder was observed. Target deployment root is absent; the current
governor is unchanged. No services, host policies or other workloads were changed.

After successful build: check parent golden/compiler evidence, execute CPU-only
constant methods on six models, resolve compiled-module legality, complete the
run-local measurement orchestration and committed selectors before GPU launch.
A failing golden result remains a blocker; no old exemption carries over.

## Reviewed input

Input record: external `INPUT.md` and `inputs/REVIEWED-INPUT.json`.
Inventory: `f393c3ff7ff4fc41e4651a5109a8509097c501f6`.
Archive SHA256: `65d1e3ca0eb7e9dffa8e9fbbb1a100bf2cf609868720b9053de67ff8bf5247d8`.
NVIDIA manifest SHA256: `ef5d13471421898c8515e9a8356caff64a20aa97432175cbcab7bf3a5cd748b1`.
Recovery hash checks pass: archive, all 18 per-file hashes and 53 source hashes
against pinned base git objects. Only this reviewed snapshot is used.
Seven texture3d and four buffer entries remain; alternate storage is correctness
coverage, not a silent substitution for the models' texture3d projections.
Numeric property checks and static model metadata were reviewed previously.
Exact compiled allocations/features and runtime constant execution remain pending.

## Owner question resolved

The question recorded once on 2026-10-14 about adapter scope was answered by the
owner's recovery clarification of 2026-10-15 00:32 UTC in CAMPAIGN.md. Necessary
run-local corrections are authorized; no pending scope question remains.
Historical sources, sarc/tools, goldens, prompts, tolerances and registered
thresholds remain unchanged. The earlier REVIEW.md describes the prior preflight
only; no repeat preflight review has been requested.

## Evidence limits

No golden, gate, A/A calibration, baseline or candidate timing has passed yet.
All six static contexts are 2560; group 128 and q4/group-32 embeddings match the
serialized delegate nodes. Runtime constant-method validation remains mandatory
before any model GPU job. Both future arms preserve `orin-fused1` attention.
No family-wide acceptance or matrix-completion claim is made.
