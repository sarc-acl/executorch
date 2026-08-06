<!--
Sync Impact Report
==================
Version change: 2.4.0 → 2.5.0

Context: this workspace stopped using the speckit plan/tasks/implement loop
(the speckit *format* is still used for standing docs like this one, just
not the interactive ceremony). Separately, the user found during a doc
audit that this constitution's "Default Scope for Every Benchmark" .pte_out
rule (added v2.3.0 below) had gone stale: 2026-08-04 established
`/sarc-c/gpusw/users/yanwen.xu/android-run/models` (NFS, manifest-tracked)
as the permanent .pte store, demoting `.pte_out` to transient export
scratch — the opposite of what v2.3.0 mandated. This amendment reverses
that one bullet to match current practice; no other principle changes.

Modified sections:
  - Default Scope for Every Benchmark, ".pte_out" bullet -- reversed:
    .pte_out is scratch only now; the NFS models/ dir + MANIFEST.json is
    the permanent store. Full history of the flip (buffer/fp16 generations
    also archived along the way) lives in
    `.shared-context/instruction-for-ai/setup/README.md`, not restated here.
  - Historical Sync Impact Reports below (pre-dating this one) are left
    as-is per this file's own convention.

Added content: none.
Removed sections: none.

Templates requiring updates:
  - .specify/templates/plan-template.md ......... n/a (loop retired)
  - .specify/templates/spec-template.md ......... n/a (loop retired)
  - .specify/templates/tasks-template.md ........ n/a (loop retired)
  - .specify/templates/commands/*.md ............. n/a (not present)

Follow-up TODOs: none new.
-->

<!--
Sync Impact Report (previous amendment, retained for history)
==================
Version change: 2.3.0 → 2.4.0

Context: `specs/017-workstream-agent-housekeeping` closed a gap the user
identified directly: a fresh agent session in this folder had no path to
this workstream's real operating knowledge until *after* a `/speckit-*`
command loaded this constitution — and even then, ten expensive,
already-root-caused operational gotchas from prior sessions stayed
scattered across `specs/014-016`'s individual `research.md` files with
nothing pointing a new session toward them. This amendment is
documentation-only: no principle is redefined, no target/scope changes.

Added content:
  - New `.specify/memory/gotchas.md` -- a living, append-as-you-go
    consolidation of ten operational gotchas (each with symptom, root
    cause, fix/workaround, and a citation), with a header instructing
    future sessions how to append new entries. Not itself part of this
    constitution, but cross-referenced from it (see Modified sections).
  - This folder's root `CLAUDE.md` gained a pointer block naming this
    constitution, the M5 EVT1 target, and `.shared-context/` -- closing
    the gap where a fresh agent's very first file read pointed at none of
    this workstream's real operating knowledge.

Modified sections:
  - Principle VI (Verify With Tools, Never Assume) -- added a bullet
    citing the ETDump per-event `kernel_name`-attribution finding as a
    concrete, already-observed instance of this principle's failure mode,
    pointing to `.specify/memory/gotchas.md` G6 and to the doc as a whole
    for the full list of similar findings.
  - Principle X (Consult `instruction-for-ai` Before Acting) -- added a
    fourth numbered step warning that a `.shared-context/instruction-for-ai/`
    doc's literal mechanism/command can itself be actively wrong for this
    repo (citing the `ET_VK_FORCE_BUFFER` example, `.specify/memory/gotchas.md`
    G2), not just that the doc should be read first.
  - Development Workflow, "Issue & Open-Question Tracking" -- added a
    "Gotchas Reference" paragraph introducing `gotchas.md`, its append
    convention, and its parallel (not replacement) relationship to
    `open-questions.md`.

Removed sections: none.

Templates requiring updates:
  - .specify/templates/plan-template.md ......... ✅ no change needed
  - .specify/templates/spec-template.md ......... ✅ no change needed
  - .specify/templates/tasks-template.md ........ ✅ no change needed
  - .specify/templates/commands/*.md ............. n/a (not present)

Follow-up TODOs: none new. `.specify/memory/gotchas.md` G6's underlying
ETDump-attribution root cause remains open (see `open-questions.md` Q11);
this amendment only ensures the finding itself is easy to find, not that
the underlying mechanism is fixed.
-->

<!--
Sync Impact Report (previous amendment, retained for history)
==================
Version change: 2.2.0 → 2.3.0

Context: during `specs/015-m5-e2e-wmma-validation` implementation, an 8B
`8da4w` export was redirected to ad hoc scratch locations (first `/tmp`,
which filled up; then a job-specific NFS tmp dir) purely to work around
disk space, without moving the result into this workspace's one canonical
`.pte` location. The user caught this and required exports to always land
in `/local/yanwen.xu/workspace/.pte_out` — never a scratch/job-tmp dir —
made an explicit, standing rule rather than something re-derived per
session.

Added content:
  - Default Scope for Every Benchmark -- new bullet: all exported `.pte`
    files MUST land in `/local/yanwen.xu/workspace/.pte_out` (workspace
    root, sibling of every branch worktree, shared across them). Since
    `export.output_dir` isn't honored by `export_llm` (output lands in
    CWD), this means `cd`-ing into `.pte_out` before running the export
    command, not exporting elsewhere and copying/moving the result after.

Modified sections: none.

Removed sections: none.

Templates requiring updates:
  - .specify/templates/plan-template.md ......... ✅ no change needed
  - .specify/templates/spec-template.md ......... ✅ no change needed
  - .specify/templates/tasks-template.md ........ ✅ no change needed
  - .specify/templates/commands/*.md ............. n/a (not present)

Follow-up TODOs: none new.
-->

<!--
Sync Impact Report (previous amendment, retained for history)
==================
Version change: 2.1.0 → 2.2.0

Context: a 2026-07-05 session on the real M5 EVT1 target made two avoidable
mistakes that a pre-existing workspace doc would have prevented: (1) it ran
`adb devices` on the wrong host and wrongly concluded the M5 EVT1 was
unreachable (the correct access path was already documented in
`.shared-context/instruction-for-ai/devices-and-access.md`); (2) it spent
effort diagnosing a stale-library Android link failure that
`instruction-for-ai/build.md`'s own documented two-step build sequence
would have prevented. The user explicitly asked that this constitution be
amended to make consulting that directory first, not after getting stuck,
an explicit rule -- not just a set of scattered citations.

Added content:
  - New Principle X, "Consult `.shared-context/instruction-for-ai` Before
    Acting, Not After" -- requires starting at that directory's README.md
    task-to-doc router before any Android build/device/export/profiling/
    driver action, rather than inferring the procedure or concluding
    something is broken/unreachable without checking there first.

Modified sections:
  - Governance -- "nine principles" -> "ten principles" (Principle X added).

Removed sections: none.

Templates requiring updates:
  - .specify/templates/plan-template.md ......... ✅ no change needed
  - .specify/templates/spec-template.md ......... ✅ no change needed
  - .specify/templates/tasks-template.md ........ ✅ no change needed
  - .specify/templates/commands/*.md ............. n/a (not present)

Follow-up TODOs: none new.
-->

<!--
Sync Impact Report (previous amendment, retained for history)
==================
Version change: 2.0.0 → 2.1.0

Context: user supplied a historical JIRA comment thread (2026-02-27 through
2026-06-18, the actual source material `.shared-context/report-for-human/`
was compiled from on 2026-06-17) and asked for a gap analysis against this
constitution. Most of the JIRA data traced cleanly to already-documented
facts (4w headline numbers, driver history, M4-vs-M5 parity); this
amendment fixes two factual errors introduced in the v2.0.0 amendment and
folds in three findings from that thread not yet reflected here.

Modified sections (corrections to v2.0.0's own content):
  - Principle IV (Two-Tier, Statistically Sound Benchmarking) -- corrected:
    v2.0.0 said every tier-2 run "MUST capture an ETDump trace alongside
    the tok/s number," which would have you profile the exact run you're
    reporting. `.shared-context/report-for-human/e2e-spec.md` is explicit
    that profiler-on numbers are NOT valid for reporting on this hardware;
    the actual methodology is a separate, small etdump-enabled run purely
    for dispatch confirmation. Fixed to require two runs, never one.
  - Quantization Scheme Matrix -- corrected: v2.0.0's PARKED reason for
    both `8w` and `8da8w` said "exceeds M5 EVT1 RAM budget," conflating
    two different, independently-confirmed blockers. `8w` has never
    reached the RAM question -- it has no export pattern producing
    `et_vk.linear_q8csw` at all. Only `8da8w` is actually RAM-blocked
    (9.6GB weights > ~8.8GB available). Split the two reasons.

Added content:
  - Principle VII (Clock Discipline) -- added the observed
    throttle-differential under floating clocks: tiled-shader configs
    throttle -19% to -27% run-over-run under sustained load, coopmat/dbuf
    configs stay flat (<4% variation). A blended floating mean without
    accounting for this misstates a tiled-vs-coopmat comparison in
    coopmat's favor.
  - Metrics Philosophy -- added the mandatory pre-benchmark coherence
    check (short low-token prompt; garbage output means diagnose via
    Principle VIII before benchmarking, never report through it).
  - Development Workflow -- added a caution to check `quant-dev`'s
    existing dbuf1-4 variant harness and matmul tile-sweep harness before
    building new tuning infrastructure for `specs/007-012`-style work, to
    avoid duplicating tooling that already exists in a sibling worktree.

Removed sections: none.

Templates requiring updates:
  - .specify/templates/plan-template.md ......... ✅ no change needed
  - .specify/templates/spec-template.md ......... ✅ no change needed
  - .specify/templates/tasks-template.md ........ ✅ no change needed
  - .specify/templates/commands/*.md ............. n/a (not present)

Follow-up TODOs: none new.
-->

<!--
Sync Impact Report (previous amendment, retained for history)
==================
Version change: 1.5.0 → 2.0.0

Modified principles:
  - II. Renamed "Samsung RDNA3 iGPU Is the Target, Not a Fallback" ->
    "Samsung M5 EVT1 Is the Only Active Target" -- BACKWARD INCOMPATIBLE:
    removed the `rocky-ryzen` MiniPC dual-validation requirement entirely.
    Samsung M5 EVT1 is now this workstream's sole active validation
    platform; MiniPC and any non-Samsung/Adreno-based device are retired
    to archived/historical reference only. Explicit user instruction:
    "From now on, we only work on Samsung's devices, no more MiniPC or
    other phones... keep as archive when we need to refer to them for
    data." Consequently `8w`/`8da8w` (whose only implementation/validation
    path was MiniPC) move from "MiniPC/RDNA3 iGPU only" to PARKED in the
    Quantization Scheme Matrix -- no active platform, not a scope this
    workstream revisits ad hoc.
  - IV. Two-Tier, Statistically Sound Benchmarking -- tier-2 wording
    tightened: e2e **prefill tok/s** is now named explicitly as this
    workstream's sole headline metric (previously a non-committal "e.g.,
    tokens/sec, ms/token"), paired with a required ETDump capture per e2e
    claim (cross-referencing Principle VI). Dropped the now-stale
    "8-bit schemes are minipc/iGPU-only" scoping parenthetical. Explicit
    user instruction: "Use ETDump to verify e2e results. we care about
    tok/s for e2e."
  - VII. Clock Discipline -- dropped the `rocky-ryzen`-specific DVFS
    cross-check exception now that MiniPC is not an active target.

Modified sections:
  - Quantization Scheme Matrix: `8w`/`8da8w` On-device scope -> PARKED.
  - Reference Hardware Inventory: `rocky-ryzen` MiniPC block relabeled
    ARCHIVED / historical-only (kept for citing specs 001-013's data, not
    as a live platform).
  - Reference Build Recipe (Development Workflow): relabeled Archived --
    no longer the required pre-Android-build step.
  - Default Scope for Every Benchmark: added the explicit `_ctx3072.pte`
    export requirement for the 2048-prefill/1024-decode workload (was
    previously implied, not stated). Explicit user instruction: "2k
    prefill and 1k decode (2048 and 1024, thus pte use the 3072 version)."
  - Samsung/Xclipse Build, Export, Deploy (Development Workflow): added
    the driver-flashing doc and the named clock-pin script; generalized
    the floating-clock allowance to "whenever explicitly requested," not
    only DVFS/thermal analysis. Explicit user instruction: "we often run
    with frequency pinned. but also sometimes i ask for floating results."

Added sections:
  - Core Principles: IX. "Never Disclose Samsung-Internal Specifics
    Upstream" (NON-NEGOTIABLE) -- per explicit user instruction ("Must
    never put samsung specific internal knowledge to Upstream!!! most
    critical"), the single most important scope boundary this
    constitution enforces: upstream-bound (`pytorch/executorch`)
    contributions must never carry internal board/codenames (e.g.
    "ERD9975", "M5 EVT1"), device serials, internal hostnames/NFS paths,
    driver build hashes/filenames, or JIRA ticket references -- describe
    hardware behavior only via runtime-queryable Vulkan capabilities.
  - Performance & Portability Standards: "Metrics Philosophy" -- imports
    `.shared-context/report-for-human/RESEARCH-GOALS.md`'s "e2e is the
    deliverable, microbench is for analysis" framing directly into the
    constitution, per user instruction to integrate
    `.shared-context/instruction-for-ai` knowledge not yet reflected here.

Removed sections: none (superseded MiniPC-era content retained verbatim in
older Sync Impact Reports below, for history).

Templates requiring updates:
  - .specify/templates/plan-template.md ......... ✅ no change needed
  - .specify/templates/spec-template.md ......... ✅ no change needed
  - .specify/templates/tasks-template.md ........ ✅ no change needed
  - .specify/templates/commands/*.md ............. n/a (not present)

Follow-up TODOs: none new.
-->

<!--
Sync Impact Report (previous amendment, retained for history)
==================
Version change: 1.4.0 → 1.5.0

Modified sections:
  - Reference Hardware Inventory -- corrected. This workstream's MiniPC
    phase (specs 001-013) prototyped without access to this workspace's
    pre-existing Samsung/Xclipse validation history, so v1.0.0-1.4.0 carried
    an unconfirmed device table (Samsung `SM-S926B` "believed not to expose
    cooperative-matrix support", a second WMMA-capable device "not yet
    connected"). That workspace history exists and predates this workstream
    (real target validated on-device since 2026-06-08, with working
    build/export/deploy/profile tooling and e2e coopmat results already in
    hand). Replaced the unconfirmed table with the real target (M5 EVT1 /
    ERD9975 / Xclipse 970, WMMA confirmed) and pointers to the live docs
    that hold its volatile specifics, instead of copying values that will
    drift.

Added sections:
  - Core Principles: VII. "Clock Discipline: Pinned by Default, Verified
    Bound" -- codifies this workstream's now-required Samsung/Android
    measurement practice (pin clocks; verify the pin actually bound via a
    GFLOP/s cross-check) after the MiniPC phase's own handoff report named
    on-device validation as the explicit next step with no equivalent
    practice yet defined.
  - Core Principles: VIII. "Verify the Driver Before Every Coopmat
    Measurement" -- codifies re-checking the on-device Vulkan driver
    identity before trusting a coopmat measurement, and points at this
    workspace's existing catalog of known Xclipse driver defects, for the
    same reason as VII.
  - Performance & Portability Standards: "Shader/Storage Configuration
    Taxonomy" -- imports the T-tiled/B-tiled/B-coopmat/dbuf1-4 naming
    already established and used consistently in this workspace's prior
    Samsung work, so the next phase doesn't reinvent ad hoc names for the
    same storage x shader comparisons.
  - Development Workflow: "Samsung/Xclipse Build, Export, Deploy (M5 EVT1)"
    -- points at this workspace's existing, working Android build/export/
    run/profile pipeline, which predates and already answers most of what
    `specs/013-minipc-handoff-report`'s Runbook flagged as "needs
    adaptation" or "newly established."
  - Development Workflow: "Issue & Open-Question Tracking" -- adopts this
    workspace's existing open-questions.md -> root-causes.md -> JIRA
    pipeline for this workstream's own anomalies going forward.

Removed sections: none (the superseded MiniPC-phase device table is kept
verbatim in the v1.0.0->1.1.0 Sync Impact Report below, for history)

Templates requiring updates:
  - .specify/templates/plan-template.md ......... ✅ no change needed
  - .specify/templates/spec-template.md ......... ✅ no change needed
  - .specify/templates/tasks-template.md ........ ✅ no change needed
  - .specify/templates/commands/*.md ............. n/a (not present)

Follow-up TODOs: the prior TODO(HW_INVENTORY) is resolved by this amendment
(the real Samsung target is now documented); none new.
-->

<!--
Sync Impact Report (previous amendment, retained for history)
==================
Version change: 1.3.0 → 1.4.0

Modified principles: none

Added sections:
  - Core Principles: VI. "Verify With Tools, Never Assume" -- mandates
    ETDump-based dispatch confirmation for model-level (tier 2) WMMA
    studies, compiled-SPIR-V inspection for any WMMA shader change (to
    confirm real cooperative-matrix instructions and correct behavior),
    and profiler-driven optimization using available tooling (ETDump, GPU
    timestamps, RGA/ISA tooling, Vulkan validation layers) instead of
    source-level guessing. Added on explicit user request while planning
    007-wmma-improvement-microbench, generalized beyond that one feature
    to apply to all future WMMA work in this workstream.

Removed sections: none

Templates requiring updates:
  - .specify/templates/plan-template.md ......... ✅ no change needed
  - .specify/templates/spec-template.md ......... ✅ no change needed
  - .specify/templates/tasks-template.md ........ ✅ no change needed
  - .specify/templates/commands/*.md ............. n/a (not present)

Follow-up TODOs: none new (see the pre-existing TODO(HW_INVENTORY) below,
carried over unchanged)
-->

<!--
Sync Impact Report (previous amendment, retained for history)
==================
Version change: 1.2.0 → 1.3.0

Modified principles: none

Added sections:
  - Repository & Distribution Scope: documents that `sarc-acl/executorch`
    (this workstream's fork/remote) is a safe, standing home for
    everything this workstream produces -- specs/, .specify/, speckit
    tooling, raw benchmark artifacts -- with no curation needed to land
    there. Upstream `pytorch/executorch` PRs are a narrower, separate
    surface requiring explicit per-PR instruction on what to include.
    Added on explicit user request after the first full commit pass
    landing specs/001-006 to this fork.

Removed sections: none

Templates requiring updates:
  - .specify/templates/plan-template.md ......... ✅ no change needed
  - .specify/templates/spec-template.md ......... ✅ no change needed
  - .specify/templates/tasks-template.md ........ ✅ no change needed
  - .specify/templates/commands/*.md ............. n/a (not present)

Follow-up TODOs: none new (see the pre-existing TODO(HW_INVENTORY) below,
carried over unchanged)
-->

<!--
Sync Impact Report (previous amendment, retained for history)
==================
Version change: 1.1.0 → 1.2.0

Modified principles: none

Added sections:
  - Performance & Portability Standards: "Default Scope for Every
    Benchmark" — codifies as a standing default what every feature (001-005)
    already did ad hoc: all three Target Models x both int4 schemes (six
    configurations), and a fixed 2048-token-prefill/1024-token-decode
    workload for every tier-2 (model-level) measurement. Added on explicit
    request while specifying 006-e2e-storage-comparison, to stop re-deriving
    this scope in each new feature's Assumptions section.

Removed sections: none

Templates requiring updates:
  - .specify/templates/plan-template.md ......... ✅ no change needed
  - .specify/templates/spec-template.md ......... ✅ no change needed
  - .specify/templates/tasks-template.md ........ ✅ no change needed
  - .specify/templates/commands/*.md ............. n/a (not present)

Follow-up TODOs: none new (see the pre-existing TODO(HW_INVENTORY) below,
carried over unchanged)
-->

<!--
Sync Impact Report (previous amendment, retained for history)
==================
Version change: 1.0.0 → 1.1.0

Modified principles:
  - II. "Samsung RDNA3 iGPU Is the Target, Not a Fallback" — corrected and
    sharpened using code-level evidence: the quantized-linear coopmat path
    (`can_use_q4gsw_coopmat` in QuantizedLinear.cpp) already targets wave64
    iGPUs correctly; the `!is_integrated_gpu()` exclusion lives specifically
    in the generic unquantized fp16 GEMM/matmul path (`is_coopmat_eligible`
    in GemmCoopmat.h), which matters once this workstream reaches SDPA.
    Also narrows on-device scope to the int4 quant schemes only (4w,
    8da4w) per user's memory-budget constraint for Samsung/Android targets.
  - IV. Renamed "Benchmark Against the Established Baseline, at Real Shapes"
    → "Two-Tier, Statistically Sound Benchmarking" — expanded into an
    explicit microbenchmark tier (shader-level, using the existing
    BenchmarkResult harness) and model-level tier (.pte end-to-end vs.
    stock ExecuTorch baseline), across the named quant schemes and models.

Added sections:
  - Performance & Portability Standards: Quantization Scheme Matrix, Target
    Models, Reference Hardware Inventory (dated, probe-don't-trust caveat).
  - Development Workflow: Environment & Build Bootstrap (uv + fish venv,
    worktree init sequence), Reference Build Recipe (MiniPC/linux preset).

Removed sections: none

Templates requiring updates:
  - .specify/templates/plan-template.md ......... ✅ no change needed
  - .specify/templates/spec-template.md ......... ✅ no change needed
  - .specify/templates/tasks-template.md ........ ✅ no change needed
  - .specify/templates/commands/*.md ............. n/a (not present)

Follow-up TODOs:
  - TODO(HW_INVENTORY): the attached Samsung device (SM_S926B / e2s) is
    believed not to expose cooperative-matrix support; a second,
    WMMA-capable Samsung device exists but was not yet connected as of
    2026-07-03. Re-verify via `test_coopmat_probe.cpp` once available and
    update the Reference Hardware Inventory table below.
-->

# Vulkan Cooperative-Matrix (WMMA) GEMM Constitution

This constitution governs one contributor's workstream inside ExecuTorch:
bringing cooperative-matrix (WMMA/coopmat) acceleration to the Vulkan
backend's matrix-multiplication shaders — linear/GEMM today, SDPA/attention
next — with the Samsung M5 EVT1 board (Exynos Xclipse) as the sole active
performance target (Principle II), validated end-to-end on real LLaMA
models.
It supplements, and never overrides, the repository-wide guidance in
`CLAUDE.md`; it applies to
`backends/vulkan/runtime/graph/ops/{impl,glsl}/*coopmat*` and
`*linear*coopmat*`, `backends/vulkan/runtime/vk_api/Adapter.*` /
`Device.h` capability plumbing, and `backends/vulkan/test/custom_ops/test_coopmat_*`.

Several principles below cite `.shared-context/...` paths. That directory
is a **sibling of this git worktree**, not part of this repository — it
lives at the workspace root (alongside the `quant-perf-optimization/`
checkout this constitution ships in, not inside it) and is local-only,
never committed or cloned with this repo. It is the canonical home for
this workspace's pre-existing Samsung/Xclipse device knowledge, build/run
tooling, and results; see the workspace-root `CLAUDE.md` for the full
layout. A future clone of this repo onto a different machine (as
`specs/013-minipc-handoff-report` itself anticipated) will not carry
`.shared-context/` along — re-establish equivalent device/driver
references on that machine before relying on the citations below.

## Core Principles

### I. Correctness Before Performance (NON-NEGOTIABLE)
No coopmat shader change is "done" until it passes the existing correctness
tests (`backends/vulkan/test/op_tests`, and the per-op `test_*_linear`
correctness checks at small, tile-aligned shapes) against the CPU/tiled
reference, for whichever quantization scheme it touches (fp16, 4w, 8da4w,
8da8w, or 8w — see the Quantization Scheme Matrix below). A perf-only
benchmark with no CPU reference (as in `test_coopmat_linear_bench.cpp`) is
valid only for shapes already covered by a correctness test elsewhere — it
must not be the first or only signal that a new dispatch path is correct.
Performance numbers are never reported as a substitute for a passing
correctness check.
Rationale: coopmat introduces subgroup-shape and driver-specific numerics
(mixed-precision accumulation, tile padding, component-type packing) that
are easy to get silently wrong, and mobile drivers have already shown
correctness regressions invisible on desktop (see commit `10ef1eaa9`,
"Fix coopmat quantized-linear correctness on Xclipse").

### II. Samsung M5 EVT1 Is the Only Active Target
Every coopmat kernel or dispatch path added under this workstream is
developed and validated — correctness and performance — exclusively on
the Samsung M5 EVT1 board (see Reference Hardware Inventory) before it
counts as complete. No other device is an active validation platform:
- The `rocky-ryzen` MiniPC RDNA3 iGPU that this workstream's `specs
  001`-`013` were built and validated on is **retired from active use**.
  Its results remain valid, citable historical/baseline data — consult
  them for comparison — but this workstream does not re-run, extend, or
  add new dependencies on MiniPC-only tooling going forward.
- Any Adreno-based or other non-Samsung-Xclipse phone (e.g. the device
  gated off in the separate `adreno-fix` branch) is likewise out of active
  scope. Discrete-GPU or non-target-mobile results alone never satisfy
  this mission.

The two coopmat dispatch paths in this codebase currently differ on
mobile-readiness, and that difference is itself the workstream's roadmap:
- The **quantized-linear coopmat path** (`can_use_q4gsw_coopmat` in
  `QuantizedLinear.cpp`) already gates correctly for mobile: it checks
  `supports_cooperative_matrix()` and `subgroup_size() == 64` (wave64,
  covering Xclipse), not GPU class. This is the path to keep extending.
- The **generic unquantized fp16 GEMM/matmul path**
  (`is_coopmat_eligible()` in `GemmCoopmat.h`, shared by `add_linear_coopmat_node`
  and `add_matmul_coopmat_node` — the latter is the entry point for future
  SDPA work) still hardcodes `!adapter->is_integrated_gpu()`, even though
  Xclipse-specific tuning already exists in its shader (`coopmat_mm.glsl`,
  commit `e0e9130c6`). Treat that exclusion as a known gap this workstream
  exists to close once mobile correctness/perf on that path is validated —
  not as a design constraint to preserve by default.

On the Samsung M5 EVT1 target, only the **int4-weight** schemes (4w,
8da4w) are in active scope: 8-bit-weight schemes (8w, 8da8w) do not fit
the memory budget of the target phones for the 8B/3B models and, with
MiniPC retired, currently have **no validation platform at all** — they
are PARKED, not a scope this workstream revisits until a Samsung device
with sufficient RAM exists (see `.shared-context/report-for-human/RESEARCH-GOALS.md`
for the RAM-budget reasoning).
Rationale: the mission is Samsung mobile-iGPU performance specifically,
under a mobile memory budget; splitting active validation effort across a
proxy device and the real target dilutes confidence in the one that
actually matters once the real target is available and accessible, and a
quant scheme that can't fit on the phone does not advance the mission
regardless of what a proxy device shows.

### III. Explicit Eligibility Gating, Safe Fallback Always
Coopmat shaders may impose hard shape, alignment, or subgroup requirements
(M/N/K tile alignment, `subgroup_size() == 64`, 2D-output only, and similar)
instead of handling every case in-shader — but every such requirement MUST
be encoded in an explicit, testable eligibility check (as
`can_use_q4gsw_coopmat` / `is_coopmat_eligible` already do) and MUST fall
back to an already-validated path (the tiled or double-buffered reference
shader) whenever it doesn't hold. New device-capability requirements
(component types, shared-memory budget, subgroup size, cooperative-matrix
configuration) are probed at runtime via `Adapter`/`Device` and
`test_coopmat_probe.cpp`, never assumed from the Vulkan spec text or from a
config table like the one below. Silent miscompute on an unchecked
assumption is not an acceptable trade for speed.

### IV. Two-Tier, Statistically Sound Benchmarking
Every performance claim under this workstream is made at two levels, and a
claim at one level never substitutes for the other:

1. **Shader microbenchmark** — isolates one op/shape/scheme using the
   existing `BenchmarkResult` harness in `backends/vulkan/test/custom_ops`
   (adaptive probe-then-scale iteration count, explicit warmup runs, and
   `get_avg_time_us()` / `get_std_dev_us()` reporting). A number is only
   reportable with its iteration count and stddev alongside it — a single
   untimed run is not evidence. Compare against the existing tiled and
   double-buffered reference shaders (the fp16 GEMM bench and per-op
   coopmat linear bench already in tree). Use shapes drawn from real LLaMA
   prefill/decode, not synthetic square shapes chosen for convenience.
2. **Model-level benchmark** — exports a real model to a `.pte` (see
   `/export`) and measures end-to-end **prefill throughput in tokens/sec**
   via the standard ExecuTorch LLaMA runner on the Vulkan backend — this
   workstream's sole e2e headline metric (see Metrics Philosophy below;
   decode tok/s is secondary and reported alongside it, not in its place).
   The baseline for this tier is **the default behavior of ExecuTorch
   running that model** — i.e., the same `.pte` executed without this
   workstream's coopmat dispatch path enabled — not another research
   prototype. Every model-level (tier-2) tok/s claim MUST be paired with a
   **separate** ETDump-confirmation run (same model/config, an
   etdump-enabled runner variant, a small `--max_new_tokens` to keep the
   capture light) confirming the intended kernel dispatched (Principle
   VI) — **never the same run used for the reported number**: profiler
   overhead measurably distorts timing on this hardware, and a
   profiler-on or thermally-degraded number is not valid for reporting
   (`.shared-context/report-for-human/e2e-spec.md`).

Both tiers are run across the target model set and quantization schemes
(see tables below), scoped per Principle II — only the int4 schemes (4w,
8da4w) currently have an active validation platform; 8-bit schemes are
PARKED. A change counts as a win only when it beats the relevant baseline
at both tiers it applies to; regressions on any previously-passing shape
or model are called out explicitly, never dropped silently.

### V. Document Every Driver Workaround at the Point of Use
Mobile Vulkan drivers (Xclipse in particular) have shown crashes and
correctness bugs that require spec-constant workarounds, loop-bound
restructuring, or shader splitting (see commits `e0e9130c6`, `f8f3313a1`,
`10ef1eaa9`). Any such workaround MUST carry an inline comment naming the
specific device/driver, the observed symptom, and enough detail that a
future contributor does not "clean it up" by reverting it. This is the one
category of comment this workstream requires beyond the repository's
default minimal-comment style, because the failure mode — a driver crash on
an unrelated-looking device — is invisible from the code alone.

### VI. Verify With Tools, Never Assume
Every WMMA/coopmat performance or correctness claim under this workstream is
backed by tool-driven verification, never by inference from eligibility-gate
logic or source reading alone:
- **Model-level (tier 2) WMMA studies** MUST capture an ETDump trace and
  confirm, from the actual per-op kernel names it records, that the WMMA/
  coopmat kernel dispatched for the operation(s) under study. An eligibility
  check (`can_use_q4gsw_coopmat` or similar) passing in code is not
  sufficient evidence that the intended kernel actually ran end to end.
- **Any change to a WMMA/coopmat shader** MUST have its compiled SPIR-V
  inspected (e.g. via `spirv-dis`/`spirv-cross` or equivalent disassembly)
  to confirm the expected cooperative-matrix instructions are actually
  present in the generated binary, and MUST re-confirm the shader's overall
  behavior is correct (Principle I). A shader that "looks right" in GLSL
  source is not evidence of what the driver actually compiled or executed.
- **Optimization work is profiler-driven, never guessed.** Use the tooling
  available — ETDump, GPU timestamp queries (already in `BenchmarkResult`),
  the Radeon GPU Analyzer (RGA) or equivalent ISA/occupancy tooling, Vulkan
  validation layers — to identify where time actually goes and to confirm a
  change had its intended effect, rather than reasoning from source code
  alone about what should be faster.
- **ETDump's own per-event `kernel_name` field is not itself immune to this
  principle.** This workstream directly observed a case where the
  `kernel_name` ETDump recorded for a dispatch diverged from the shader
  that actually ran, in the full ~100+-node LLaMA-graph context (see
  `.specify/memory/gotchas.md` G6) — cross-check an ETDump-based dispatch
  claim with at least one independent method (a wall-clock A/B against a
  forced-fallback path, or an isolated shader microbenchmark with its own
  kernel-name capture) before trusting `kernel_name` alone.

Rationale: kernel-selection logic, shader templates, and driver behavior
have already diverged from expectations more than once in this workstream
(the dead `default_storage` check silently no-opping a compile option,
Xclipse-specific driver crashes invisible from the GLSL source) — trusting
code-level reasoning without tool-level confirmation is exactly the failure
mode Principles I and V already guard against elsewhere. This principle
makes verification-by-tooling the explicit default, not an occasional
afterthought. `.specify/memory/gotchas.md` collects the concrete instances
of this principle being violated in practice, this workstream's own
ETDump-attribution finding among them — consult it for the full list.

### VII. Clock Discipline: Pinned by Default, Verified Bound
Every Samsung/Android performance measurement under this workstream pins
GPU/MIF/INT clocks to the workspace's documented default before measuring
(current values and pin script are in
`.shared-context/instruction-for-ai/README.md` §Conventions — this
constitution does not copy them, since they drift with the board). A
number is reported as "pinned" only after verifying the pin actually
bound: cross-check the in-graph GFLOP/s (or e2e tok/s) against an
equivalently-configured pinned microbenchmark. If they disagree, the
process did not inherit the pin and the number reflects DVFS boost, not
the reported clock config — it MUST NOT be reported as pinned. Floating
(unpinned) runs are permitted whenever explicitly requested — not only
for DVFS/thermal analysis — always clearly labeled as floating, never
presented as the pinned headline number.

Floating runs additionally have a known, shader-dependent thermal
behavior on this target: under sustained back-to-back load with no
cooldown, tiled-shader configs throttle hard run-to-run (observed -19% to
-27% from cold-start peak to steady state on 8B), while coopmat/dbuf
configs stay essentially flat (observed variation <4%). A floating "mean"
across repeated runs is only meaningful if this is accounted for — report
per-rep numbers (or note explicitly that a mean mixes cold-start peak with
throttled steady state) rather than a single blended average, especially
when comparing a tiled baseline to a coopmat config.

Rationale: this exact pin-verification failure already happened once on
this workspace's Samsung target — a previously accepted "pinned" baseline
turned out to be a ~980MHz DVFS-boost artifact rather than the intended
509MHz pin, caught only by this GFLOP/s cross-check, not by the pin
command appearing to succeed (`.shared-context/report-for-human/root-causes.md` Q10).
The throttle-differential above is a separate, already-observed effect on
the same target (JIRA, 2026-06-18) — silently averaging it away would
misstate a tiled-vs-coopmat floating comparison in coopmat's favor.

### VIII. Verify the Driver Before Every Coopmat Measurement
Samsung/Android boards used by this workstream are shared, reference-class
hardware, not exclusively controlled by this workstream — the flashed
Vulkan driver can change between sessions (reflash, reboot, another
experiment). Before any coopmat correctness or performance measurement,
confirm the on-device driver identity (e.g. `adb shell logcat -d | grep
SUMD`, or the workspace's equivalent probe) and record which build was
present; never assume a prior session's driver is still there. Current
driver state lives in `.shared-context/ACTIVE-STATUS.md` (volatile,
maintained separately from this file); known Xclipse Vulkan compiler
defects that shape which shader workarounds are currently load-bearing are
catalogued in the workspace-root `TODO.md` P0 section — consult it before
attributing an anomaly to this workstream's own shader code.
Rationale: this is not a hypothetical risk. A specific driver build
previously miscompiled the coopmat path silently — no crash, no error,
plausible-looking decode output — and was caught only by a small-shape
correctness bench, not by code inspection or a passing eligibility check
(`.shared-context/report-for-human/root-causes.md` Q9). This is the same
failure mode Principle VI already guards against for shader/kernel
selection; this principle extends it to the driver binary itself.

### IX. Never Disclose Samsung-Internal Specifics Upstream (NON-NEGOTIABLE)
This is the single most important scope boundary this constitution
enforces. Any change destined for the public `pytorch/executorch`
repository (the narrower surface defined in Repository & Distribution
Scope below) MUST NOT contain Samsung-internal identifiers or
infrastructure details, in code, comments, commit messages, or PR
descriptions — including but not limited to:
- Internal board/codenames (e.g. "ERD9975", "M5 EVT1", "M41") or
  pre-release chip-stage designations (e.g. "EVT1").
- Device serials, internal hostnames (`*.samsung.com`, `sj1-*`, etc.), or
  internal network/NFS paths.
- Driver build hashes, filenames, or version strings tied to unreleased
  driver builds.
- JIRA ticket numbers or content, or references to this workspace's
  internal `.shared-context/` docs.

Describe hardware behavior upstream only in terms that are already public
or runtime-queryable: Vulkan capability bits (`subgroup_size()`,
`supports_cooperative_matrix()`, component-type support), GPU architecture
family in general terms (e.g. "RDNA-derived mobile iGPU"), and observed
symptoms — never the specific internal board that exposed them. When a
driver workaround (Principle V) documents a device/driver by name for
this workstream's own internal use, that same comment MUST be reworded or
dropped before the containing change is proposed upstream.
Rationale: this workstream operates on Samsung's internal, often
pre-release validation hardware and infrastructure; the public
`pytorch/executorch` repository is not the place for any detail that
identifies that internal environment. This is a confidentiality boundary,
not a code-quality preference — a violation here is treated as a serious
error, not a style nit, and blocks the change until corrected. This
supplements, and is stricter than, the file-scope guidance already in
Repository & Distribution Scope and the workspace-root `CLAUDE.md`'s
branch discipline: those govern *which files* go upstream, this principle
governs *what strings may appear* even inside a file that otherwise
belongs there.

### X. Consult `.shared-context/instruction-for-ai` Before Acting, Not After
A large share of this workstream's actual, load-bearing operational
knowledge — which host a given phone is physically attached to, the exact
multi-step Android cross-build sequence, driver hash-to-meaning tables,
canonical scripts — lives in `.shared-context/instruction-for-ai/`, not in
this constitution. This constitution only summarizes and points to that
directory (see the Development Workflow section below); it is not a
substitute for reading it. Before attempting any Android build, device
access, export, profiling, or driver operation under this workstream:
1. Start at `.shared-context/instruction-for-ai/README.md` — a task → doc
   router — and read the one doc that owns that task.
2. Do not infer the procedure from source-reading, from habits carried
   over from a different workstream/worktree, or from a partially-built
   local tree.
3. Do not conclude a device or resource is unreachable, or that a build
   step is broken/missing, without first checking whether that folder
   already documents the answer.
4. A mechanism or command a `.shared-context/instruction-for-ai/` doc
   describes can itself be actively wrong for this repo's own source —
   e.g. `ET_VK_FORCE_BUFFER`, which that directory's `export-pte.md`
   documents but which does not exist anywhere in this codebase (see
   `.specify/memory/gotchas.md` G2 for the real mechanism). Reading the
   doc first (steps 1-3 above) does not itself guarantee the doc is
   correct for this repo — check `.specify/memory/gotchas.md` for known
   instances of this before trusting a documented mechanism at face
   value.

Rationale: on 2026-07-05, a session ran `adb devices` on the wrong host
and concluded "no M5 EVT1 device reachable" — the phone was reachable the
whole time via `ssh` to a different host, documented in
`devices-and-access.md`. The same session separately spent effort
diagnosing a stale prebuilt-library link failure that `build.md`'s own
documented two-step Android recipe (rebuild core runtime + `--target
install` before any dependent sub-build, e.g. `test_coopmat_linear_bench`)
would have prevented outright. Both were avoidable by reading the doc
first, not after getting stuck; this principle makes that the required
order of operations, not a best-effort courtesy.

## Performance & Portability Standards

- **Scope boundary**: this constitution governs the coopmat/WMMA GEMM and
  (future) SDPA workstream only. It does not redefine standards for the
  rest of the ExecuTorch Vulkan backend or the wider project — `CLAUDE.md`
  remains authoritative there.
- **Shader reuse**: coopmat shader templates are shared across ops (linear,
  matmul, and future SDPA) wherever the tile/data-flow shape allows, per the
  existing consolidation (commits `801b62d9d`, `2fb740798`). Prefer
  parameterizing one `coopmat_mm`/`linear_qw_coopmat`/`linear_dq8ca_qw_coopmat`
  -family template over hand-writing a new one per op.
- **Precision**: fp16 storage is universal; accumulation precision follows
  the scheme's arithmetic path in the matrix below (fp32 accumulate for
  fp16 WMMA, int32 accumulate for int8 WMMA). Any reduced-precision
  accumulation path must demonstrate it stays within the existing per-op
  correctness test's tolerance before landing.
- **Runtime feature detection**: new Vulkan/GLSL extensions
  (`VK_KHR_cooperative_matrix`, specific component-type support, etc.) are
  feature-detected at runtime via `Adapter`/`Device`; there is no
  compile-time assumption that a given device exposes a given coopmat
  configuration.

### Quantization Scheme Matrix

| Scheme  | Activation            | Weight              | Arithmetic path        | Coopmat (WMMA) status today                              | On-device (Samsung/Android) scope |
|---------|------------------------|----------------------|-------------------------|------------------------------------------------------------|------------------------------------|
| fp16    | fp16                   | fp16 (unquantized)   | fp16 WMMA               | Implemented (PR #19009); `coopmat_mm.glsl` / `GemmCoopmat.*` | Reference/SDPA groundwork only    |
| 4w      | fp16                   | int4, grouped-sym.   | fp16 WMMA (weight dequant to fp16) | Implemented: `linear_qw_coopmat.glsl` (`linear_q4gsw_coopmat_*`) | **In scope** |
| 8da4w   | int8, dynamic per-row  | int4, grouped-sym.   | int8 WMMA (coopmat\<int8\> × coopmat\<int8\> → coopmat\<int32\>) | Implemented: `linear_dq8ca_qw_coopmat.glsl` (`linear_dq8ca_q4gsw_coopmat_*`) | **In scope** |
| 8w      | fp16                   | int8, channel-scaled | int8 arithmetic (tiled only today) | Not yet ported to coopmat; tiled only (`linear_q8csw_tiled.glsl`) | **PARKED** — no export pattern emits `et_vk.linear_q8csw` (never even reaches the RAM question); MiniPC (its only other validation platform) also retired |
| 8da8w   | int8, dynamic per-row  | int8, channel-scaled | int8 arithmetic (tiled only today) | Not yet ported to coopmat; tiled only (`linear_q8ta_q8csw_tiled.glsl`) | **PARKED** — exports fine but RAM-blocked at the default 2048-context workload (9.6GB weights > ~8.8GB available); MiniPC (its only other validation platform) also retired |

### Shader/Storage Configuration Taxonomy

Comparing coopmat on real hardware means comparing storage type × shader
path combinations, not just "coopmat on/off." Use these exact names —
defined once in `.shared-context/report-for-human/RESEARCH-GOALS.md` and
already used consistently across this workspace's prior Samsung work — in
every report and table; do not invent new ad hoc names for the same
comparisons in a new spec.

| Name | Meaning | How it's produced |
|---|---|---|
| **T-tiled** | Stock ExecuTorch default: texture storage, tiled shader — the honest baseline (what a user gets today) | texture `.pte` |
| **B-tiled** | Diagnostic baseline: buffer storage, tiled shader — isolates the kernel effect from the storage effect | buffer `.pte`, coopmat disabled |
| **B-coopmat** | Buffer storage, coopmat (WMMA) shader — this workstream's contribution | buffer `.pte`, coopmat gate fires |
| **dbuf1–4** | Experimental double-buffered loop-structure variants of B-coopmat | buffer `.pte` + variant switch |

Headline speedup = **B-coopmat vs T-tiled** (does coopmat beat what a user
gets today, end to end). Pure-kernel speedup = **B-coopmat vs B-tiled**
(storage held constant). Reusing this vocabulary keeps results comparable
across specs and over time, per Principle IV's benchmarking discipline.

### Metrics Philosophy

Imported from `.shared-context/report-for-human/RESEARCH-GOALS.md`, which
already establishes this for the workspace — this workstream adopts it
as-is rather than deriving its own:

- **E2E prefill tok/s is the deliverable.** The only number this
  workstream reports as "the result" for a tier-2 claim (Principle IV) is
  end-to-end prefill tokens/sec on the M5 EVT1 target, at the Default
  Scope workload below. Decode tok/s is reported alongside it as
  secondary context, never as the headline (decode is a single-token
  `M=1` gemv where the coopmat gate does not engage, so it stays roughly
  constant across configs and is not this workstream's signal).
- **Microbenchmark is analysis, not the deliverable.** Tier-1 shader
  microbenchmarks (Principle IV) exist to (1) gate correctness at small
  aligned shapes before any tier-2 run, and (2) explain a tier-2 tok/s
  result after the fact — microbench GFLOP/s should be consistent with
  the ETDump-measured per-op linear time, and an Amdahl's-law rollup over
  the ETDump breakdown should predict the observed e2e speedup. If those
  three don't agree, something is wrong (unbound clock pin, driver
  miscompile) — treat the disagreement itself as a signal to investigate
  via Principles VII/VIII, not as noise to average away.
- **Every e2e claim is ETDump-verified**, per Principle IV/VI, via a
  separate confirmation run — never the reported run itself (Principle
  IV): a tok/s number with no ETDump trace confirming the intended kernel
  dispatched is not a reportable result under this workstream.
- **A short coherence check precedes every benchmarking session**: run a
  brief, low-token prompt (e.g. "The capital of France is") through the
  config under test before trusting any timing from it. Garbage or
  incoherent output means diagnose first (check the driver identity,
  Principle VIII, before anything else) — never benchmark a config whose
  correctness hasn't been sanity-checked that session.

### Target Models

Primary validation set, in priority order: **LLaMA 3.1 8B**, **LLaMA 3.2
3B**, **LLaMA 3.2 1B**. Every model-level benchmark claim (Principle IV,
tier 2) names which of these three it was run against; claims about "LLaMA
performance" without a named model/size are not acceptable evidence.

### Default Scope for Every Benchmark

Unless a feature explicitly documents a deviation and justifies it, every
benchmark under this workstream runs:

- **All three Target Models above**, at **both** `4w` and `8da4w` (the two
  int4 schemes in scope per the Quantization Scheme Matrix) — six
  configurations total. A result reported for "a model" or "a scheme"
  without covering all six, or without an explicit stated reason for a
  narrower scope, is incomplete evidence under this workstream.
- **A fixed workload for every tier-2 (model-level) measurement**:
  **2048-token prefill, 1024-token decode**. This keeps e2e numbers
  directly comparable across features and over time without re-deriving or
  re-justifying a workload size each time. Tier-1 shader microbenchmarks
  use shapes drawn from this same prefill/decode split (Principle IV).
- **This workload is served by a single context-length export**: `.pte`
  files are exported at `MAX_SEQ=MAX_CTX=3072` (canonical naming
  `*_ctx3072.pte`, per `.shared-context/instruction-for-ai/setup/README.md`),
  which comfortably covers the 2048-prefill/1024-decode split above. Don't
  export a different context length for this default workload without
  updating this section and justifying the change.
- **Every exported `.pte` is moved to the NFS canonical store** at
  `/sarc-c/gpusw/users/yanwen.xu/android-run/models` immediately after
  export, tracked by that directory's `MANIFEST.json` (sha256/size/mtime;
  regenerate via `.shared-context/scripts/pte_manifest.py`). This reverses
  the v2.3.0 rule that `.pte_out` was the permanent destination — as of
  2026-08-04 `.pte_out` is transient scratch only. `export_llm`'s
  `export.output_dir` config key is still not honored (the file lands in
  the process's CWD, so `cd` into `.pte_out` before invoking the export),
  but the result must then be hashed and `mv`'d (not `cp`'d, and not left
  permanently) to the NFS path above — never kept as a duplicate in both
  places. See `.shared-context/instruction-for-ai/setup/README.md`
  §"Where PTEs live now" for the full procedure.

### Reference Hardware Inventory

*Corrected 2026-07-05. Through v1.4.0 this table treated the Samsung
target as unconfirmed — that reflected only this workstream's own
MiniPC-phase device inventory (`specs/001-013`, done without adb access to
the real target). This workspace has an independent, pre-existing
Samsung/Xclipse validation history predating this workstream (on-device
since 2026-06-08, with working build/export/deploy/profile tooling and
real e2e coopmat results already in hand) — see
`.shared-context/report-for-human/RESEARCH-GOALS.md`. Device/driver/clock specifics still
drift session to session (shared board); this table intentionally does not
copy those volatile values — pull them from the docs cited below, always.*

**`rocky-ryzen` MiniPC (AMD Ryzen APU, RDNA3 iGPU) — ARCHIVED, not an
active target (Principle II).** Retired 2026-07-05; kept here only to
interpret specs `001`-`013`'s existing MiniPC results, not as a platform
to build or benchmark on going forward. Exposed 14 cooperative-matrix
configurations, all 16×16×16 at Subgroup scope:

| # | M | N | K | AType | BType | CType | ResultType | Scope |
|---|---|---|---|-------|-------|-------|------------|-------|
| 0 | 16 | 16 | 16 | float16 | float16 | float16 | float16 | Subgroup |
| 1 | 16 | 16 | 16 | float16 | float16 | float32 | float32 | Subgroup |
| 2 | 16 | 16 | 16 | uint8 | uint8 | uint32 | uint32 | Subgroup |
| 3–13 | 16 | 16 | 16 | int8 variants | int8 variants | int32 | int32 | Subgroup |

**M5 EVT1 — PRIMARY SAMSUNG TARGET.** Samsung ERD9975 reference board
(Exynos S5E9975 / "Exynos 2500"), Xclipse 970 GPU (AMD RDNA-derived),
wave64 default, subgroup size 32–64. **Cooperative matrix CONFIRMED**: fp16
and int8 WMMA, 16×16×16, Subgroup scope. This is the on-device validation
target Principle II requires, not the `rocky-ryzen` proxy. Live
serial/host/NFS-path defaults → `.shared-context/instruction-for-ai/README.md`
§Conventions (the paste-block every runnable doc uses); which driver is on
the device *right now* (good vs. known-bad hash) →
`.shared-context/ACTIVE-STATUS.md`. Do not copy those values into this
file — see Principle VIII.

**M41 — secondary quick-experiment Samsung device.** Reachable via a
different host/ADB path (`.shared-context/instruction-for-ai/devices-and-access.md`
§1b). WMMA support not assumed present; use for fast non-target-critical
iteration, not as this workstream's validation target.

The Pixel 7a / `SM-S926B` / `SM-N950U` table previously here was this
workstream's own MiniPC-phase device inventory; it is retained verbatim in
the v1.0.0→1.1.0 Sync Impact Report above for history, and superseded by
the M5 EVT1 / M41 pair above.

## Development Workflow

- Land in small, single-purpose, reviewable commits, following this
  workstream's existing pattern (capability probe → single-dtype prototype
  → benchmark → generalize), each with an `[ET-VK]` subject prefix
  consistent with existing history.
- Before extending coopmat to a new op family (e.g., SDPA), first extend the
  capability probe (`test_coopmat_probe.cpp`) to confirm the target device
  exposes the needed configuration, then prototype correctness at small
  aligned shapes, then benchmark at real shapes — in that order.
- Any change to eligibility gating (`can_use_q4gsw_coopmat`,
  `is_coopmat_eligible`, or their successors) requires re-running both the
  correctness test suite and the relevant benchmark, since widening the
  gate changes which shapes now depend on coopmat correctness.
- Before building new loop-structure variants or a tile-geometry sweep
  harness for this workstream's shaders, check the workspace's `quant-dev`
  worktree first: it already has a dbuf1-4 double-buffer variant harness
  and a matmul tile-sweep harness (see the workspace-root `CLAUDE.md`) —
  both directly relevant to `specs/007-012`'s tuning work. Port/reuse that
  tooling rather than re-deriving it independently on Samsung.

### Environment & Build Bootstrap

All Python tooling in this workstream runs inside the project's `uv`-managed
virtualenv — never system Python or an ad hoc `venv`/`pip` setup:

```fish
source .venv/bin/activate.fish   # bash: source .venv/bin/activate
```

Any agent or contributor working from a **new git worktree** MUST bootstrap
it before attempting a build — a worktree does not inherit the parent
checkout's `.venv`:

```fish
uv venv .venv --seed
source .venv/bin/activate.fish   # or activate for bash
./install_executorch.sh --minimal
```

### Archived Reference Build Recipe (MiniPC / `rocky-ryzen`, Linux preset)

**Historical only (Principle II) — retired 2026-07-05.** This was the
go-to local validation loop before touching an Android build during the
MiniPC phase; retained verbatim to reproduce specs `001`-`013`'s results,
not as a required or recommended step for current work. Use the
Samsung/Xclipse pipeline below instead.

```bash
rm -rf cmake-out-vk
cmake . -Bcmake-out-vk --preset "linux" \
    -DCMAKE_INSTALL_PREFIX=cmake-out-vk -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DEXECUTORCH_PAL_DEFAULT=posix \
    -DEXECUTORCH_BUILD_VULKAN=ON -DEXECUTORCH_BUILD_TESTS=ON \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_FLAGS="-include algorithm"
cmake --build cmake-out-vk -j$(nproc) --target install --config Release

cmake backends/vulkan/test/custom_ops/ \
    -Bcmake-out-vk/backends/vulkan/test/custom_ops \
    -DCMAKE_INSTALL_PREFIX=cmake-out-vk -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DEXECUTORCH_ROOT=$(pwd) \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
cmake --build cmake-out-vk/backends/vulkan/test/custom_ops -j$(nproc)
```

Model-level (tier 2) benchmarks additionally require a `.pte` exported for
the model/scheme under test (see `/export`) and a run through the standard
LLaMA runner against that same build.

### Samsung/Xclipse Build, Export, Deploy (M5 EVT1)

Per Principle X: read the relevant doc below FIRST, before attempting the
task. `specs/013-minipc-handoff-report`'s own Runbook flagged Android
build, export, and deploy as "needs adaptation" or "newly established,"
written without visibility into this workspace's pre-existing pipeline for
exactly that target. That pipeline already exists and is validated — check
it before writing new Android tooling for this workstream:

- **Build** (runtime + `llama_main`/ETDump runner, cross-compiled for
  Android): `.shared-context/instruction-for-ai/build.md`, canonical
  script `build_etdump_android.sh`.
- **Export** a `.pte` (texture vs. buffer storage, per quant scheme):
  `.shared-context/instruction-for-ai/setup/README.md`. Canonical recipe (enforced
  2026-08-04) is a raw `python -m executorch.extension.llm.export.export_llm`
  invocation with inline Hydra overrides, run from vanilla `release-1.3/` — NOT a
  script. The old `export_quant.sh` wrapper is retired (archived to
  `.shared-context/scripts/archive/`); do not use or revive it.
- **Run** an e2e/microbench measurement, including clock pinning
  (Principle VII, script `pin_freqs.sh`): `.shared-context/instruction-for-ai/commands.md`.
  Pinned is the default for every reported number; run floating (unpinned)
  whenever explicitly requested (not only for DVFS/thermal analysis) —
  always label a floating result as such, never as the pinned headline.
- **Profile** via ETDump: `.shared-context/instruction-for-ai/profiling.md`,
  `.shared-context/scripts/analyze_etdump_shaders.py`.
- **Device access / current driver state** (Principle VIII):
  `.shared-context/instruction-for-ai/devices-and-access.md`,
  `.shared-context/ACTIVE-STATUS.md`.
- **Flash / A-B the Vulkan driver**: `.shared-context/instruction-for-ai/flash-sumd-driver.md` —
  use this when Principle VIII's driver-identity check finds an
  unexpected or known-bad hash.

Only the pieces this workstream's MiniPC phase never had cause to build —
e.g. anything specific to the SDPA-coopmat or `8da4w`-tuning work coming
from `specs/007-012` — are genuinely new; the underlying Android
build/export/run/profile mechanics are not.

### Issue & Open-Question Tracking

Anomalies encountered during this workstream's work on real hardware (an
unexplained perf ranking, a correctness mismatch, a driver crash) are
logged, not silently resolved ad hoc and forgotten: open questions go in
`.shared-context/report-for-human/open-questions.md` (numbered `Q`
entries: phenomenon → hypothesis, explicitly marked unverified → next
step → status); once root-caused, they move to
`.shared-context/report-for-human/root-causes.md`; anything that's a
driver or tooling defect (not this workstream's own code) additionally
gets a ticket under `.shared-context/report-for-human/jira-tickets/`.
Reuse this existing pipeline rather than starting a parallel one scoped
just to this workstream.

**Gotchas Reference**: `.specify/memory/gotchas.md` is a separate,
parallel doc to `open-questions.md` above — not a replacement for it.
`open-questions.md` tracks *unresolved* phenomena on real hardware
(perf anomalies, correctness mismatches, driver crashes) working toward a
root cause; `gotchas.md` is the living, append-as-you-go consolidation of
this workstream's already-root-caused *operational* lessons (a build
trap, a documented-but-nonexistent env var, a naming collision between
similarly-named files) that cost real time to rediscover once already.
Consult it before repeating a mistake this workstream has already made;
append a new entry to it, per its own header's convention, whenever a
future session root-causes a new multi-hour or repeat-mistake issue.

## Repository & Distribution Scope

- This workstream's standing home is the `sarc-acl/executorch` fork/remote.
  Everything it produces -- `specs/` (plans, research, tasks, reports, raw
  benchmark logs/JSON), `.specify/`'s speckit scaffolding, and the
  `speckit-*` skills under `.claude/skills/` -- is safe to commit and push
  there without curation. None of it needs to be scrubbed, squashed, or
  hidden to land on that remote.
- Contributions bound for the upstream `pytorch/executorch` repository are
  a different, narrower surface: typically only the production code
  change (e.g. a coopmat correctness fix or perf win under
  `backends/vulkan/`), not this workstream's `specs/`/`.specify`/speckit
  scaffolding or raw benchmark artifacts.
- Do not assume which commits go upstream, and do not prepare or open an
  upstream PR unprompted. Wait for explicit instruction identifying
  exactly which commits/files to include before curating one.

## Governance

This constitution governs the coopmat/WMMA workstream only; it supplements,
and never supersedes, the root ExecuTorch `CLAUDE.md` and repository-wide
review standards. Amend it when the mission's scope changes (e.g., expanding
beyond Vulkan, beyond Samsung/Xclipse as the reference target, or beyond the
three named LLaMA models) or when a principle is proven wrong in practice
(e.g., Principle II's fallback constraint becomes moot once the mobile-iGPU
path is fully validated and the `!is_integrated_gpu()` gate is removed).
Amendments are made directly to this file, versioned per semantic-versioning
rules (MAJOR: principle removed/redefined incompatibly; MINOR: principle or
section added/materially expanded; PATCH: wording/clarification only), and
recorded in a Sync Impact Report prepended to this file. Check each PR under
this workstream's scope against the ten principles above before merge —
Principle IX above all, since it is NON-NEGOTIABLE for anything upstream-
bound; any other deviation must be justified in the PR description, not
merged silently.

**Version**: 2.5.0 | **Ratified**: 2026-07-03 | **Last Amended**: 2026-08-06
