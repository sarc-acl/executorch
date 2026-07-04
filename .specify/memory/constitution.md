<!--
Sync Impact Report
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
next — with Samsung's RDNA3-based mobile integrated GPU (Exynos Xclipse) as
the primary performance target, validated end-to-end on real LLaMA models.
It supplements, and never overrides, the repository-wide guidance in
`CLAUDE.md`; it applies to
`backends/vulkan/runtime/graph/ops/{impl,glsl}/*coopmat*` and
`*linear*coopmat*`, `backends/vulkan/runtime/vk_api/Adapter.*` /
`Device.h` capability plumbing, and `backends/vulkan/test/custom_ops/test_coopmat_*`.

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

### II. Samsung RDNA3 iGPU Is the Target, Not a Fallback
Every coopmat kernel or dispatch path added under this workstream must be
validated — correctness and performance — on real RDNA3 hardware (the
`rocky-ryzen` MiniPC iGPU as a fast local proxy, and Samsung Xclipse mobile
hardware as the actual target) before it counts as complete. Discrete-GPU
results alone do not satisfy this mission.

The two coopmat dispatch paths in this codebase currently differ on this
point, and that difference is itself the workstream's roadmap:
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

On the actual Samsung/Android target, only the **int4-weight** schemes (4w,
8da4w) are in scope: 8-bit-weight schemes (8w, 8da8w) do not fit the memory
budget of the target phones for the 8B/3B models and are validated on the
`rocky-ryzen` RDNA3 iGPU only, not on-device.
Rationale: the mission is mobile iGPU performance under a mobile memory
budget; a coopmat path that only helps desktop dGPUs, or a quant scheme
that can't fit on the phone, does not advance it.

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
   `/export`) and measures end-to-end (e.g., tokens/sec, ms/token) via the
   standard ExecuTorch LLaMA runner on the Vulkan backend. The baseline for
   this tier is **the default behavior of ExecuTorch running that model** —
   i.e., the same `.pte` executed without this workstream's coopmat
   dispatch path enabled — not another research prototype.

Both tiers are run across the target model set and quantization schemes
(see tables below), scoped per Principle II (int4 schemes are the ones that
must run on-device; 8-bit schemes are minipc/iGPU-only). A change counts as
a win only when it beats the relevant baseline at both tiers it applies to;
regressions on any previously-passing shape or model are called out
explicitly, never dropped silently.

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
| 8w      | fp16                   | int8, channel-scaled | int8 arithmetic (tiled only today) | Not yet ported to coopmat; tiled only (`linear_q8csw_tiled.glsl`) | MiniPC/RDNA3 iGPU only — does not fit target phones |
| 8da8w   | int8, dynamic per-row  | int8, channel-scaled | int8 arithmetic (tiled only today) | Not yet ported to coopmat; tiled only (`linear_q8ta_q8csw_tiled.glsl`) | MiniPC/RDNA3 iGPU only — does not fit target phones |

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

### Reference Hardware Inventory

*As of 2026-07-03 — capabilities change with driver/OS updates; always
re-verify with `test_coopmat_probe.cpp` rather than trusting this table.*

**`rocky-ryzen` MiniPC (AMD Ryzen APU, RDNA3 iGPU)** — primary local
dev/test platform, used before every Android build. Exposes 14 cooperative-matrix
configurations, all 16×16×16 at Subgroup scope:

| # | M | N | K | AType | BType | CType | ResultType | Scope |
|---|---|---|---|-------|-------|-------|------------|-------|
| 0 | 16 | 16 | 16 | float16 | float16 | float16 | float16 | Subgroup |
| 1 | 16 | 16 | 16 | float16 | float16 | float32 | float32 | Subgroup |
| 2 | 16 | 16 | 16 | uint8 | uint8 | uint32 | uint32 | Subgroup |
| 3–13 | 16 | 16 | 16 | int8 variants | int8 variants | int32 | int32 | Subgroup |

**Android devices (via `adb devices -l`)**:

| Device | Model / codename | GPU | Role |
|--------|-------------------|-----|------|
| `3A021JEHN02756` | Pixel 7a (`lynx`) | Adreno | Non-target Vulkan regression check only |
| `R5CY21Y3VEV` | Samsung `SM-S926B` (`e2s`) | Exynos/Xclipse (RDNA3-based) | Believed primary target; **currently believed not to expose cooperative-matrix support** — verify via probe before relying on it |
| `ce0717178d7758b00b7e` | Samsung `SM-N950U` (`greatqlte`) | Older Mali/Adreno-era | Not a coopmat target |

A second Samsung device with confirmed WMMA support exists but was not
connected as of this writing — see the `TODO(HW_INVENTORY)` in the Sync
Impact Report above; use it as the on-device validation target once
attached, and update this table.

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

### Reference Build Recipe (MiniPC / `rocky-ryzen`, Linux preset)

The go-to local validation loop before touching an Android build:

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
this workstream's scope against the five principles above before merge; any
deviation must be justified in the PR description, not merged silently.

**Version**: 1.3.0 | **Ratified**: 2026-07-03 | **Last Amended**: 2026-07-04
