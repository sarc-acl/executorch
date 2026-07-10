# Feature Specification: M5 EVT1 Floating-Clock Speedup Table

**Feature Branch**: `019-m5-floating-clock-speedup`

**Created**: 2026-07-06

**Status**: Abandoned — closed 2026-07-08, no longer needed (17/29 tasks were done; remaining 8B floating-clock reps not pursued)

**Input**: User description: "3.2 Result: new record e2e speedup — see table. [pinned 6-row T-tiled-baseline vs full-stack-optimal speedup table for 1B/3B/8B x 4w/8da4w, measured at pinned GPU 509 / MIF 2730 / INT 663 MHz]. Now do the same for floating, no pinning."

## Context

`specs/015-m5-e2e-wmma-validation` and `specs/018-m5-8da4w-t-tiled-baseline`
together produced a complete, six-row **pinned-clock** speedup table (all
3 models x both int4 schemes, T-tiled baseline vs full-stack optimal,
2.60x-3.70x): 1B `4w` 312.7->812.6 (2.60x), 1B `8da4w` 222.30->723.0
(3.25x), 3B `4w` 112.5->334.0 (2.97x), 3B `8da4w` 79.83->286.3 (3.59x), 8B
`4w` 51.4->153.3 (2.98x), 8B `8da4w` 35.17->130.1 (3.70x) -- all at pinned
GPU 509 / MIF 2730 / INT 663 MHz. This feature produces the same six-row
table under **floating (unpinned)** clocks.

This is not simply "the same measurement without a pin command."
Constitution Principle VII already documents a known, asymmetric thermal
behavior on this exact hardware under floating clocks: tiled-shader
configs throttle hard run-to-run under sustained load (observed -19% to
-27% from cold-start peak to steady state on 8B), while coopmat/dbuf
configs stay essentially flat (observed variation <4%). Since this
feature's entire purpose is comparing a tiled baseline against a coopmat
("full-stack optimal") config -- exactly the comparison Principle VII
warns a naive blended floating mean would misstate in coopmat's favor --
this feature's methodology must account for that from the start, not
discover it after publishing a misleading ratio.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 1B floating-clock table measured, throttle-transparent (Priority: P1)

As the engineer who just published the pinned speedup table, I need the
same six numbers for 1B (T-tiled baseline and full-stack optimal, both
schemes) measured under floating clocks, reported per-rep rather than as
a single blended mean, so I can see whether and how much the tiled
baseline's throttle behavior changes the apparent speedup versus the
pinned result.

**Why this priority**: 1B is the fastest, cheapest model to prove the
floating-clock methodology (including confirming clocks are genuinely
unpinned, not still capped) before spending device time on the slower
models.

**Independent Test**: for each of 1B's four configs (`4w` baseline,
`4w` optimal, `8da4w` baseline, `8da4w` optimal), 3 per-rep floating-clock
tok/s values are recorded and published individually -- not collapsed
into one mean -- with dispatch confirmed the same way as the pinned
measurements.

**Acceptance Scenarios**:

1. **Given** the device is not currently clock-pinned, **When** a
   floating-clock run is captured, **Then** a sysfs readback of
   `/sys/kernel/gpu/{min,max}_freq` confirms the values are NOT clamped
   to the pinned 509/2730/663 MHz triple (i.e., genuinely floating, not
   an unpin command that silently no-op'd).
2. **Given** 3 floating-clock reps of a tiled-baseline config, **When**
   they are reported, **Then** all 3 per-rep values are shown (not just a
   mean), so a reader can see whether run-to-run throttle occurred.

---

### User Story 2 - 3B floating-clock table measured (Priority: P2)

Same as User Story 1, for LLaMA 3.2 3B.

**Why this priority**: second-cheapest model, same proven methodology.

**Independent Test**: same as User Story 1, for 3B's four configs.

---

### User Story 3 - 8B floating-clock table measured (Priority: P3)

Same as User Stories 1-2, for LLaMA 3.1 8B -- the model Principle VII's
own throttle observation (-19% to -27%) was originally measured on.

**Why this priority**: slowest, highest device-time cost, and the model
most likely to actually exhibit the documented throttle behavior at full
scale -- sequenced last, after the methodology is proven twice.

**Independent Test**: same as User Stories 1-2, for 8B's four configs,
explicitly checking whether the tiled-baseline configs show the
previously-observed -19% to -27% cold-to-steady-state drop.

---

### User Story 4 - Floating speedup table published alongside the pinned one (Priority: P2)

As a reader comparing the pinned and floating results, I need a
consolidated floating-clock speedup table in the same six-row shape as
the existing pinned one, with the throttle-asymmetry caveat stated
explicitly next to it, so I don't misread a larger floating-clock ratio
as a bigger real-world win rather than partly an artifact of tiled
configs throttling more than coopmat ones.

**Why this priority**: this is the actual deliverable the pinned table's
own reader would want next -- the per-model raw numbers (User Stories
1-3) don't answer "how does this compare to the pinned table" by
themselves.

**Independent Test**: a six-row floating-clock table exists, formatted
the same way as the existing pinned table, with each row's speedup ratio
computed from either (a) matched cold-start-vs-cold-start values or (b)
matched steady-state-vs-steady-state values -- never a tiled cold-start
number divided against a coopmat steady-state number or vice versa --
and a one-paragraph caveat states which of (a)/(b) was used and why.

**Acceptance Scenarios**:

1. **Given** all three models' floating-clock data exists, **When** the
   consolidated table is published, **Then** it sits alongside (not
   replacing) the pinned table, and explicitly labeled as floating in
   every row and in the table's own heading.

---

### Edge Cases

- What if a tiled-baseline config's floating reps show significant
  run-to-run throttle (matching Principle VII's -19% to -27% precedent)?
  -- report all per-rep values and note the drop explicitly; do not
  average it away into a single number that hides which end of the range
  the "speedup" ratio is really comparing.
- What if "floating" doesn't actually take effect (some governor or
  leftover pin state keeps clocks capped near the pinned values, echoing
  the Q10 precedent where a ~980MHz DVFS-boost number was once mistaken
  for something else)? -- verify via sysfs readback (User Story 1,
  Acceptance Scenario 1) before trusting any floating number; if clocks
  are not genuinely floating, stop and fix the environment rather than
  publish a number that isn't what it claims to be.
- What if device thermal state carries over between configs (e.g., 8B
  running hot from a prior model's back-to-back reps affects the next
  config's "cold start" value)? -- log wall-clock run order and note this
  as a possible confound in the published table rather than silently
  presenting all "cold start" values as equally cold.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: This feature MUST measure, under floating (unpinned)
  clocks, all 12 configurations already established under pinned clocks
  in `specs/015`/`specs/018`: T-tiled baseline and full-stack optimal,
  for each of 3 models x 2 schemes (`4w`, `8da4w`).
- **FR-002**: This feature MUST NOT report a single blended mean per
  config when run-to-run values vary meaningfully; it MUST report all
  per-rep values, per constitution Principle VII's explicit floating-run
  requirement.
- **FR-003**: Every number this feature produces MUST be labeled as
  floating in the table/report it appears in -- never presented
  alongside or in place of the pinned headline number without that
  label, per Principle VII.
- **FR-004**: This feature MUST verify, via a sysfs readback of
  `/sys/kernel/gpu/{min,max}_freq` (or the equivalent already-used probe),
  that clocks are genuinely floating (not still clamped to
  509/2730/663 MHz) before trusting any capture as a floating
  measurement.
- **FR-005**: This feature MUST re-verify on-device driver identity
  before measuring (Principle VIII), per this workstream's standing
  discipline -- reuse the existing PTEs from `specs/015`/`specs/018`, no
  new export is needed since the workload and PTEs are identical to the
  pinned measurements.
- **FR-006**: This feature MUST confirm each config's dispatch status
  (tiled vs coopmat) the same way the pinned measurements did -- reusing
  the already-confirmed `dispatch_status` from `specs/015`/`specs/018` is
  acceptable (dispatch status is a property of the shader/PTE, not the
  clock state), rather than re-running dispatch confirmation from
  scratch.
- **FR-007**: This feature MUST publish a consolidated floating-clock
  speedup table in the same six-row shape as the existing pinned table,
  with an explicit caveat paragraph describing which values (cold-start
  or steady-state) were used for each ratio and why, per User Story 4.

### Key Entities

- **Floating Clock Measurement**: one per (model, scheme, config_type)
  where `config_type` is `t_tiled_baseline` or `full_stack_optimal` (12
  total: 3 models x 2 schemes x 2 config types). Fields: model, scheme,
  config_type, per-rep tok/s values (array, not just a mean), observed
  throttle (yes/no, and magnitude if yes), dispatch_status (reused from
  the pinned measurement, not re-derived), sysfs-verified floating state
  (boolean confirmation this run was genuinely unpinned).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All 12 configurations (3 models x 2 schemes x 2 config
  types) have real, measured floating-clock tok/s data.
- **SC-002**: Every published floating number is unambiguously labeled
  as floating -- verifiable by reading any results file and confirming
  no floating number appears unlabeled next to a pinned one.
- **SC-003**: Per-rep values (not a single blended mean) are visible for
  every config, so a reader can independently assess throttle behavior.
- **SC-004**: A consolidated six-row floating speedup table exists
  alongside the pinned one, with its methodology caveat (cold-start vs.
  steady-state comparison basis) stated explicitly.

## Assumptions

- This feature reuses the exact PTEs, workload (2048-prefill/1024-decode),
  and dispatch-confirmation results already established in
  `specs/015-m5-e2e-wmma-validation` and `specs/018-m5-8da4w-t-tiled-baseline`
  -- it does not re-export or re-verify dispatch from scratch, since
  neither depends on clock state.
- "Floating" means no explicit clock pin is commanded; it does not mean
  disabling the GPU's own DVFS/thermal governor, which this feature has
  no ability or intent to control.
- Device thermal history (how hot the board is when a given config's
  floating reps run) is a known, disclosed confound (Edge Cases) --
  this feature does not attempt to fully control for it (e.g., via
  mandatory cooldown periods between configs) unless the user requests
  that level of rigor separately.
- M5 EVT1 device access follows the same access/driver-verification
  discipline as prior specs in this workstream (constitution Principles
  VII/VIII; gotcha G10 -- confirm the device is free before assuming so).
