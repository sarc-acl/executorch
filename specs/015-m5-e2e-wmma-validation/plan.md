# Implementation Plan: M5 EVT1 End-to-End WMMA Validation (Linear 4w/8da4w + SDPA)

**Branch**: `015-m5-e2e-wmma-validation` | **Date**: 2026-07-05 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/015-m5-e2e-wmma-validation/spec.md`

## Summary

Nine configurations (3 models × {`4w`, `8da4w`} linear, plus 3 models ×
SDPA-coopmat) need a dispatch-confirmed, tool-verified e2e prefill/decode
tok/s number captured on the real M5 EVT1 target, using this repo's own
current build (128x64 tile + `specs/014`'s fp16-accumulate/loop-flattening/
vectorized-dequant changes, measured as-is per spec Clarifications). No new
shader or dispatch code is needed -- this is export + build + deploy +
measure + report, reusing every mechanism this workstream already has.
**Per explicit user instruction during planning**: measure **1B first**
(lowest compute/watchdog risk), report its results as soon as they exist,
then proceed to 3B and 8B rather than batching all nine configurations
before reporting anything.

## Technical Context

**Language/Version**: Python 3.10 (AOT export, this repo's own `.venv`,
already editable-installed) for `.pte` generation; C++17 (existing
`llama_main`/ETDump runner, unmodified) for the on-device measurement; no
new code in either language.

**Primary Dependencies**:
- `.shared-context/scripts/export_quant.sh` -- canonical export script.
  AOT export is graph/quantization-level, not shader-dependent, so it can
  run from this repo's own venv (confirmed importable:
  `executorch.extension.llm.export.export_llm`) rather than requiring the
  `quant-dev` worktree `export-pte.md`'s examples `cd` into -- the three
  `4w` **Buffer**-storage `.pte`s already in the shared `.pte_out/` (one
  per model; matching `Texture3D` exports also exist for the same three
  models but are not used by this feature) were very likely produced this
  same way and are reused (spec FR-001); only the three `8da4w` buffer
  `.pte`s are new exports.
- `cmake-out-android-vk/examples/models/llama/llama_main` -- this repo's
  own already-built Android runner (built during `specs/014`'s session,
  reflecting the current `vulkan_backend` library with the 128x64 tile +
  all three `specs/014` shader changes). **Critical distinction from
  `.shared-context/instruction-for-ai/commands.md`'s example commands**:
  those reference `llama_main_origcm`/`llama_main_etdump_origcm`, prebuilt
  runners staged from a *different* worktree's `.tmp-origcm` build (per
  workspace `CLAUDE.md`) -- those do NOT contain this repo's shader
  changes and MUST NOT be used for this feature's measurements. This
  feature builds and pushes its own runner.
- `build_etdump_android.sh` (already present in this repo, not yet run
  this session) -- builds the ETDump-enabled runner variant needed for
  Principle VI's separate dispatch-confirmation run.
- `pin_freqs.sh` (adb host, per `commands.md` §5) -- clock pinning,
  default for every reported number. **Pinning alone is not sufficient**:
  constitution Principle VII requires verifying the pin actually bound
  (cross-check in-graph GFLOP/s or e2e tok/s against an
  equivalently-configured pinned microbenchmark) before any number is
  reported as "pinned" -- this workstream's own Q10 root-cause (a
  ~980MHz DVFS-boost number mistaken for a 509MHz pin) is the direct
  precedent for why this check is not optional. Not persistent across
  reboots -- re-pin and re-verify if the device reboots mid-feature.
- `.shared-context/scripts/analyze_etdump_shaders.py` -- ETDump per-shader
  breakdown, used to confirm coopmat/WMMA dispatch (FR-002).
- `ET_VK_SDPA_COOPMAT=1` (runtime env, confirmed present in
  `backends/vulkan/runtime/graph/ops/impl/SDPA.cpp`) -- enables SDPA
  coopmat dispatch, combined with a `Buffer`-storage `.pte` (same PTE the
  linear coopmat path uses).
- `ET_VK_EXECUTE_NODE_THRESHOLD=16` (already committed in this repo's
  `ComputeGraph.cpp`, unlike its uncommitted state in the `quant-dev`
  worktree) -- the GPU-watchdog mitigation for 2048-token prefill,
  confirmed present before relying on it.

**Storage**: Flat files -- `.pte_out/` for exports (shared workspace dir),
this feature's own `results/` for capture logs and the final report; NFS
staging (`$NFS` per `README.md` §Conventions) between this host and the
M5 EVT1 adb host, matching every prior on-device feature in this
workstream.

**Testing**: No new test framework. Dispatch confirmation (US1) uses
ETDump + `analyze_etdump_shaders.py`, reused unmodified. E2E timing (US2/
US3) uses the standard `llama_main` runner, **3 repeated runs per
configuration reporting mean + CoV** (not a single-shot capture) --
matching `e2e-spec.md`'s own established "3-run means" methodology,
already found and grounded during `/speckit-analyze` (this feature's own
first draft under-specified this as a single run).

**Target Platform**: Samsung M5 EVT1 (Exynos 2500 / Xclipse 970), reached
via `ssh yanwen.xu@sj1-dmckee-d01` then `adb -s 0000088f8e579c33` (per
Principle X and `devices-and-access.md` -- NOT local `adb`, the mistake
corrected during `specs/014`). Driver identity re-verified before any
measurement (Principle VIII) -- last confirmed as known-good `f14c51b6f8`
at the end of `specs/014`'s session, but boards drift; re-check, don't
assume.

**Project Type**: Real end-to-end hardware measurement + report. No
production source changes; the only new artifacts are `8da4w` `.pte`
exports (data, not code) and this feature's own `results/` documents.

**Performance Goals**: No target set by this feature (that's `005`'s and
the user's own prior MiniPC work's role) -- this feature *measures* what
today's M5 EVT1 build actually delivers, directional against `quant-dev`'s
already-published figures per spec Clarifications, not a pass/fail bar.

**Constraints**:
- **Execution order, per explicit user instruction during planning**:
  measure **1B first** (lowest layer-count/compute, lowest GPU-watchdog
  risk), report its results as soon as captured, then proceed to 3B, then
  8B -- never batch all nine configurations silently before the first
  report.
- Per spec Clarifications: measure today's shader as-is (128x64 tile +
  all three `specs/014` changes together); no isolation/revert step.
- Per spec Clarifications: SDPA-coopmat e2e is in scope, extending the
  existing partial M5 finding (1B fully measured at 2048-prefill; 8B/3B
  previously blocked by the GPU-watchdog issue at 2048).
- Per constitution Principle IV: every e2e tok/s number requires a
  *separate* ETDump-enabled dispatch-confirmation run -- never the same
  run used for the reported number.
- Per constitution Principle VII: pinned clocks (509/2730/663 MHz) by
  default; floating only if explicitly requested, always labeled. **The
  pin's effect MUST be verified** (GFLOP/s or e2e tok/s cross-check
  against an equivalently-pinned microbenchmark) before it is reported as
  "pinned" -- commanding the pin is not the same as confirming it bound
  (found missing from this feature's first draft during `/speckit-analyze`,
  corrected here per the Q10 precedent).
- Per constitution Principle IV / `e2e-spec.md`: every reported e2e number
  is a **3-run mean with CoV**, not a single-shot capture -- matching this
  workstream's established methodology (also found missing from this
  feature's first draft during `/speckit-analyze`).
- Per `commands.md` §10: the on-device PAL GPU-profiler settings file
  (`amdPalSettings.cfg`) may need to be moved aside before benchmarking if
  present and active -- **requires explicit user approval to touch**,
  per that doc's own note; not done unilaterally.
- Per FR-008/Edge Cases: any dispatch failure or recurrence of the
  GPU-watchdog issue at 2048-token prefill is reported explicitly with a
  stated reason -- never silently retried into a different (weaker)
  configuration and reported as if it were the intended one.

**Scale/Scope**: 9 configurations (3 models × `4w`/`8da4w` linear = 6,
plus 3 models × SDPA-coopmat = 3), backed by 6 distinct PTE files (3
`4w` Buffer, already exist, shared by both `linear_4w` and `sdpa_coopmat`
configs; 3 `8da4w` Buffer, new). Each of the 9 configurations needs a
dispatch-confirmation run and a 3-run e2e prefill/decode capture --
sequenced 1B → 3B → 8B per the user's ordering instruction, not grouped by
scheme/op family.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Checked against `.specify/memory/constitution.md` (v2.2.0, current
committed `HEAD`):

- **I. Correctness Before Performance (NON-NEGOTIABLE)**: PASS. This
  feature reports timing only for configurations that already passed
  correctness (the linear shapes were correctness-verified in `specs/014`
  at production K; SDPA's existing correctness tests are unmodified and
  unaffected by this feature). No new shader code means no new correctness
  surface to gate.
- **II. Samsung M5 EVT1 Is the Only Active Target**: PASS, directly
  implements this principle -- the entire point of this feature is
  replacing MiniPC/`quant-dev`-only evidence with real M5 EVT1 measurement
  on this repo's own code.
- **III. Explicit Eligibility Gating, Safe Fallback Always**: PASS, N/A to
  modify. This feature doesn't touch `can_use_q4gsw_coopmat`,
  `is_coopmat_eligible`, or SDPA's eligibility gate -- it exercises them
  as they exist.
- **IV. Two-Tier, Statistically Sound Benchmarking**: PASS by design --
  this feature *is* the tier-2 (model-level) measurement, with the
  required separate ETDump dispatch-confirmation run (FR-002) and pinned
  clocks (Principle VII) by default.
- **V. Document Every Driver Workaround at the Point of Use**: N/A -- no
  new driver workaround introduced.
- **VI. Verify With Tools, Never Assume**: PASS by design -- FR-002
  requires ETDump-confirmed dispatch before any number is trusted, for
  every one of the nine configurations independently (not inferred from
  one configuration's success), matching `009`'s and `011`'s precedent.
- **VII. Clock Discipline**: PASS *by design, corrected during
  `/speckit-analyze`* -- this feature's first plan draft claimed PASS on
  "pinned by default" alone, without a task that verifies the pin bound
  (Principle VII's own GFLOP/s-cross-check requirement) or captures the
  3-run mean/CoV `e2e-spec.md` establishes as this workstream's actual
  methodology. Both gaps are now explicit in Constraints above and carried
  into `tasks.md`'s Foundational and e2e-capture tasks. Any floating run
  would be explicitly requested and labeled, not needed for this feature's
  scope.
- **VIII. Verify the Driver Before Every Coopmat Measurement**: PASS,
  directly implements this principle (Technical Context above) --
  re-verifies rather than trusting `specs/014`'s end-of-session state.
- **IX. Never Disclose Samsung-Internal Specifics Upstream**: PASS, N/A.
  Entirely internal-workstream work; no upstream-bound artifact.
- **X. Consult `.shared-context/instruction-for-ai` Before Acting**: PASS,
  directly implements this principle -- this plan was written after
  reading `build.md`, `export-pte.md`, `commands.md`, `devices-and-access.md`,
  and `flash-sumd-driver.md`, and explicitly calls out one place those
  docs' example commands would mislead this feature if followed literally
  (the `_origcm` runners belong to a different worktree's build).

No violations. Complexity Tracking is not needed.

## Project Structure

### Documentation (this feature)

```text
specs/015-m5-e2e-wmma-validation/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── checklists/
│   └── requirements.md  # Spec quality checklist (already created by /speckit-specify)
└── tasks.md             # Phase 2 output (/speckit-tasks, not this command)
```

No `contracts/` directory: this feature has no external interface of its
own -- it's a measurement + report, matching the no-contracts precedent of
specs `001`/`006`/`009`/`011` (this workstream's other e2e-measurement
features).

### Source Code (repository root)

No new production source files. New data artifacts only:

```text
.pte_out/
├── llama3_2_1b_8da4w_buffer_ctx3072.pte   # new export
├── llama3_2_3b_8da4w_buffer_ctx3072.pte   # new export
└── llama3_1_8b_8da4w_buffer_ctx3072.pte   # new export
                                            # (4w buffer x3 models: already exist, reused; matching
                                            #  texture exports also exist but are unused by this feature)

cmake-out-android-vk-etdump/                # new build dir (ETDump-enabled runner), via build_etdump_android.sh

specs/015-m5-e2e-wmma-validation/
└── results/
    ├── raw/                                # per-configuration capture logs + etdump traces
    ├── 1b-results.md                       # 1B's linear + SDPA results, published as soon as captured
    ├── 3b-results.md
    ├── 8b-results.md
    └── m5-e2e-validation-report.md         # consolidated final report (User Story 4)
```

**Structure Decision**: Same lightweight, no-new-production-code structure
as this workstream's other e2e features. The one deliberate deviation from
those specs' single final-report file is per-model result files
(`1b-results.md`, etc.), published incrementally in the user's requested
1B → 3B → 8B order, with the consolidated report assembled last from
those three once complete.

## Complexity Tracking

*No violations -- table not needed.*
