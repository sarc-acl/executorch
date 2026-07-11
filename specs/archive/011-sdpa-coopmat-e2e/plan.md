# Implementation Plan: SDPA Coopmat E2E Validation

**Branch**: `011-sdpa-coopmat-e2e` | **Date**: 2026-07-05 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/011-sdpa-coopmat-e2e/spec.md`

**Note**: This template is filled in by the `/speckit-plan` command. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

Extend `010`'s proven tier-1 SDPA coopmat win (66.8% average prefill
speedup, all three target models real-effect) to a real end-to-end
measurement: confirm the coopmat shaders actually dispatch through a real
exported model (not `010`'s synthetic harness), then measure e2e
prefill/decode tok/s with the toggle enabled, against `009`'s already-
published baseline. Planning confirmed (empirically, not assumed) that
enabling `ET_VK_SDPA_COOPMAT` requires no new export and no rebuild --
`009`'s existing `Buffer`-storage `.pte` files already dispatch both
coopmat shaders correctly once the toggle is set, since the coopmat gate is
a pure runtime check. This makes the feature unusually lightweight: no new
production code, no new build targets, no new exports -- purely reusing
`009`'s exports and capture methodology with one added env var.

## Technical Context

**Language/Version**: Python 3.10+ (`uv`-managed `.venv`) for the
comparison/report script and ETDump parsing, matching every prior feature
in this workstream. No new C++/build work is anticipated (Constraints).

**Primary Dependencies**:
- `009`'s six existing `Buffer`-storage `.pte` exports, reused verbatim
  (research.md Decision 1 -- confirmed empirically during planning, not
  assumed).
- `009`'s existing `cmake-out-vk` (standard) and `cmake-out-vk-etdump`
  (event-tracer) build trees, reused as-is -- confirmed no production
  runtime code changed since `009`'s last build (`010` only added test
  code).
- `SDPA.cpp`'s existing `ET_VK_SDPA_COOPMAT` opt-in toggle (already
  implemented, unchanged by this feature).
- `009`'s and `010`'s already-published reports, read as read-only inputs
  (baseline e2e numbers; the tier-1 microbenchmark finding to cross-check
  against).

**Storage**: Flat files -- new ETDump captures under `results/etdump/`, new
e2e capture logs under `results/raw/`, and
`specs/011-sdpa-coopmat-e2e/results/sdpa-coopmat-e2e-report.md`.

**Testing**: The ETDump dispatch-confirmation check (FR-002) and the e2e
smoke behavior implicit in a clean capture run are this feature's own
inline verification, matching `009`'s precedent -- no separate automated
test suite, and no new test code (unlike `010`, which added genuinely new
correctness coverage for code that had never been tested at all; here the
SDPA coopmat path was already correctness-proven by `010`).

**Target Platform**: `rocky-ryzen` MiniPC -- real device work (ETDump
capture, e2e capture), matching every prior tier-2 feature.

**Project Type**: A pure measurement-and-report feature -- zero new
production code, zero new build targets, zero new exports. The lightest
-weight feature in this workstream to date.

**Performance Goals**: N/A -- this feature measures performance; it does
not carry its own target.

**Constraints**:
- No e2e number is reported for a configuration until ETDump confirms both
  `sdpa_compute_attn_weights_coopmat` and `sdpa_compute_out_coopmat`
  actually dispatched for it (FR-002, constitution Principle VI).
- No new `.pte` export and no rebuild -- if a configuration's existing
  export or build turns out not to support the toggle correctly (spec.md
  Edge Cases), that is reported as a scope correction, not silently worked
  around with an ad hoc new export.
- Prefill comparisons against `009`'s baseline inherit `006`'s documented
  cross-session variance caveat (research.md Decision 6).
- No concurrent GPU load during any capture (established workstream
  discipline).

**Scale/Scope**: 3 target models x 2 int4 schemes = 6 configurations
(constitution default scope), each needing only an ETDump dispatch check
and an e2e capture -- no export, no build.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Checked against `.specify/memory/constitution.md` (v1.4.0):

- **I. Correctness Before Performance (NON-NEGOTIABLE)**: PASS --
  correctness of the SDPA coopmat shaders themselves was already
  established by `010`'s genuinely-new correctness check; this feature
  only adds dispatch confirmation at the real-model tier (FR-002), which is
  the appropriate tier-2 verification per Principle VI, not a duplicate of
  `010`'s tier-1 correctness work.
- **II. Samsung RDNA3 iGPU Is the Target, Not a Fallback**: PASS with scope
  note -- `rocky-ryzen` MiniPC only, consistent with every prior tier-2
  feature; Samsung/Xclipse validation is a future feature.
- **III. Explicit Eligibility Gating, Safe Fallback Always**: PASS. This
  feature only observes the existing `sdpa_coopmat_device_ok`/
  `sdpa_buf_half`/`sdpa_cm_aligned` gates; it does not modify eligibility
  logic at all.
- **IV. Two-Tier, Statistically Sound Benchmarking**: PASS, tier-2
  (model-level) -- reuses `001`/`006`/`009`'s exact e2e methodology and
  JSON shape; every reported tok/s number carries its run count (FR-003).
  This is precisely the tier-2 confirmation `010`'s tier-1 finding still
  needed per Principle IV's "never substitutes" rule.
- **V. Document Every Driver Workaround at the Point of Use**: N/A -- no
  new driver workaround anticipated; nothing about this feature touches
  shader or dispatch code.
- **VI. Verify With Tools, Never Assume**: PASS, central to this feature --
  FR-002's ETDump-based dispatch confirmation is exactly this principle's
  model-level clause, and this plan itself already applied it once during
  planning (research.md Decision 1) rather than assuming the toggle would
  "just work" end to end.

No violations identified. Complexity Tracking is not needed -- this
feature adds no code at all beyond one comparison script, reusing every
mechanism `009`/`010` already built.

*Post-Phase-1 re-check*: Phase 1's data model keeps
`qk_dispatch_status`/`av_dispatch_status` as separate, explicit fields per
configuration (never folded into a single pass/fail or into the timing
number itself), so Principle VI stays enforced by the data structure, not
just by convention -- same discipline `009`/`010` already established.

## Project Structure

### Documentation (this feature)

```text
specs/011-sdpa-coopmat-e2e/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/
│   └── sdpa-coopmat-e2e-schema.md
└── tasks.md              # Phase 2 output (/speckit-tasks, not this command)
```

### Source Code (repository root)

No production or test code changes -- this feature is purely
measurement-and-report, reusing `009`'s exports and build trees as-is:

```text
specs/011-sdpa-coopmat-e2e/
├── scripts/
│   └── compare_sdpa_e2e.py   # new: reads 009's report (baseline e2e
│                                # numbers) and this feature's new
│                                # SDPA-coopmat-enabled capture, renders
│                                # the report
└── results/
    ├── etdump/    # new ETDump captures, one per configuration (FR-002)
    ├── raw/       # new e2e capture logs/JSON, ET_VK_SDPA_COOPMAT=1
    └── sdpa-coopmat-e2e-report.md

# Read-only references (not modified):
specs/009-e2e-tokrate-report/results/pte/*.pte                       # reused exports
specs/009-e2e-tokrate-report/results/e2e-tokrate-report.md            # baseline numbers
specs/010-sdpa-coopmat-microbench/results/sdpa-coopmat-microbench-report.md  # cross-check finding
```

**Structure Decision**: Lightest-weight structure in this workstream --
one new Python comparison/report script, no new C++/build/export work at
all. Matches `009`/`010`'s established pattern of one script + a `results/`
directory, scaled down to reflect that this feature adds no new mechanism,
only a new measurement using mechanisms `009`/`002` already built.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

No violations -- this section is intentionally empty.
