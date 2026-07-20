# Feature Specification: M41 Release/1.3 Baseline Clock & Quant-Mode Study

**Feature Branch**: `030-m41-release13-baseline`

**Created**: 2026-07-14

**Status**: Draft

**Input**: User description: "Do a study on M41 device (the one we just ran), same release 1.3 vanilla baseline. document the driver hash. Document this existing numbers you just collect. Then, run the same on pinned frequency, for all 4w and 8da4w, for both pinned and floating"

## Clarifications

### Session 2026-07-14

- Q: Given Principle VII's documented -19% to -27% cold-start-to-throttled thermal drift for
  floating-clock tiled configs on this hardware, how should floating-clock results be reported? →
  A: Report a per-rep table AND a mean, with the mean explicitly labeled as potentially mixing
  cold-start peak with throttled steady state (Option B).
- Q: Principle VII documents a prior incident where a "pin command appearing to succeed" was
  actually a DVFS-boost artifact, caught only by cross-checking throughput, not by the sysfs write
  succeeding — how should this study verify a pinned run actually ran pinned? → A: Verify via
  sysfs readback AND an in-graph throughput cross-check (pinned prefill tok/s must fall distinctly
  below the already-collected floating numbers) before trusting a run as pinned (Option B).
- Q: Principle II states "Samsung M5 EVT1 Is the Only Active Target" for this workstream and gives
  the retired MiniPC's results a "citable historical/reference, not headline" treatment — how
  should M41's results be framed relative to that? → A: Frame explicitly as a secondary/cross-device
  reference baseline, the same treatment already given the retired MiniPC — labeled supplementary,
  never compared against or substituted for M5 EVT1 headline numbers (Option A).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Preserve the already-collected baseline with device provenance (Priority: P1)

A performance engineer revisiting this device later needs a trustworthy record of the release/1.3
vanilla 4w-texture baseline numbers already collected on M41 this session (1B/3B/8B, 3 reps each,
floating clocks), together with exactly which driver build was on the device when they were
captured. Without this, the numbers are just chat history — easy to lose and impossible to trust
months later when the device's driver may have drifted.

**Why this priority**: The data already exists; the highest-value, lowest-risk action is writing
it down correctly before it's lost. Every other user story depends on this record existing first
so new numbers can be compared against it apples-to-apples.

**Independent Test**: Can be fully tested by reading the resulting document and confirming it
contains the M41 driver `.so` md5 hash, the device identity (serial/host/SoC), and all 9 already-run
(3 models × 3 reps) prefill/decode numbers, matching the raw command log from this session.

**Acceptance Scenarios**:

1. **Given** the driver hash and baseline numbers were captured earlier this session, **When** the
   study document is produced, **Then** it records the exact driver md5, notes that no documented
   known-good reference hash exists for this SoC family, and reproduces the 1B/3B/8B floating 4w
   numbers with their per-rep values and a caveated mean (per the thermal-drift disclosure required
   elsewhere in this spec).
2. **Given** one 8B rep crashed during collection, **When** the document is written, **Then** the
   crash is recorded as part of the dataset (not silently dropped) with its error signature and
   suspected cause.

---

### User Story 2 - Compare pinned vs. floating clocks for the 4w baseline (Priority: P2)

A performance engineer wants to know whether the release/1.3 vanilla 4w baseline can even be
measured at pinned clocks on M41, and if so, how much floating clocks are inflating the numbers
relative to the pinned/reproducible configuration used elsewhere in this workspace.

**Why this priority**: Pinned clocks are this workspace's default measurement config for a reason
(reproducibility); floating numbers are analysis-only until a pinned counterpart exists. This is
the next most valuable gap after preserving what's already known.

**Independent Test**: Can be fully tested by running the pinned-clock 4w-texture baseline for
1B/3B/8B (3 reps each, continuing through any crash) and checking that all 9 rep-cells show either
a number or "CRASHED", with CoV computed for any model with ≥2 valid reps.

**Acceptance Scenarios**:

1. **Given** clocks pinned to the workspace default (509/2730/663 MHz), **When** the 4w-texture
   baseline is run for each of 1B/3B/8B, **Then** each of the 9 rep-cells shows a valid number or
   "CRASHED" (with cause), a per-model CoV is reported where ≥2 reps succeeded, and the sweep never
   pauses or halts because of a crash.
2. **Given** both pinned and floating 4w numbers exist for a model, **When** they are compared,
   **Then** the document states the relative difference and whether pinned was even viable for that
   model size.

---

### User Story 3 - Extend the pinned/floating comparison to 8da4w (Priority: P3)

A performance engineer wants the same pinned-vs-floating picture for the 8da4w quant mode, so the
device has full quant-mode × clock-mode coverage instead of only 4w.

**Why this priority**: Completes the matrix, but 4w is this project's priority quant mode
elsewhere, so this extension is valuable but not blocking.

**Independent Test**: Can be fully tested by running the 8da4w-texture baseline for 1B/3B/8B at
both pinned and floating clocks (3 reps each) and confirming each of the 6 (model × clock-mode)
cells has a recorded outcome.

**Acceptance Scenarios**:

1. **Given** 8da4w-texture PTEs exist (or are staged) for 1B/3B/8B, **When** the baseline is run at
   both pinned and floating clocks, **Then** the document reports per-rep prefill/decode tok/s (or
   a documented failure) for all 6 cells, with a caveated mean for the floating-clock cells.
2. **Given** all four quant-mode × clock-mode combinations (4w/8da4w × pinned/floating) are
   complete, **When** the final document is assembled, **Then** it presents them as four
   comparable tables covering the same three model sizes.

---

### Edge Cases

- What happens when a pinned run hits the SGPU job watchdog and crashes with
  `VK_ERROR_DEVICE_LOST` (already observed for 8B at pinned 4w)? → The crash is documented as an
  expected/known failure mode for that cell (shown as "CRASHED" in that rep's table cell, per
  FR-012), not silently retried or hidden; device liveness is re-verified, and the sweep proceeds
  immediately to the next rep/model/cell without pausing.
- What happens when sustained back-to-back runs heat the device enough to throttle floating clocks
  mid-sweep (already observed for 8B rep 2 under floating 4w)? → The affected rep is documented as
  a failure with the thermal reading at the time, and the mean for that cell is computed from the
  remaining valid reps rather than silently re-run without note.
- What happens if the 8da4w-texture PTE for a given model size isn't already staged on the device
  or on NFS? → It is pushed from the existing NFS/`.pte_out` staging location; no fresh export is
  performed as part of this study.
- What happens if the device's driver hash changes between the 4w and 8da4w portions of the study
  (shared-board drift)? → The hash is re-checked before each quant-mode's runs and any change is
  flagged in the document, since it would invalidate a same-driver comparison.
- What happens when a "pinned" rep's sysfs write succeeds but its throughput cross-check
  (FR-009) shows it actually ran at floating/boosted speed — the exact DVFS-boost-mistaken-for-
  pinned failure mode this workstream has hit before? → That rep's cell shows "DVFS-ARTIFACT" (not
  a number, not "CRASHED") per FR-012, with its measured throughput noted so a reader can see why
  it was excluded from the pinned mean/CoV; the sweep continues to the next rep rather than
  treating this as a crash or retrying silently.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The study MUST record the current Vulkan driver (`vulkan.samsung.so`) md5 hash on the
  M41 device (serial `000009b44fd4abd3`, host `xgpusw-debug07`) and state whether it matches any
  documented known-good reference hash for this SoC family.
- **FR-002**: The study MUST record the already-collected release/1.3 vanilla 4w-texture
  floating-clock baseline (1B/3B/8B, up to 3 reps each) as the starting dataset, including the one
  documented crash.
- **FR-003**: The study MUST produce pinned-clock (509/2730/663 MHz) 4w-texture baseline numbers
  for 1B, 3B, and 8B on M41, up to 3 reps each, using the same release/1.3 runner and 2048-token
  prompt as the existing floating dataset.
- **FR-004**: The study MUST produce floating-clock 8da4w-texture baseline numbers for 1B, 3B, and
  8B on M41, up to 3 reps each.
- **FR-005**: The study MUST produce pinned-clock 8da4w-texture baseline numbers for 1B, 3B, and 8B
  on M41, up to 3 reps each.
- **FR-006**: For any run that fails (watchdog crash, thermal-related crash, or other), the study
  MUST document the failure's error signature and suspected cause rather than dropping it silently
  or retrying without note, MUST confirm the device is still responsive, and MUST proceed to the
  next rep/model/cell without pausing for confirmation or re-running the failed rep — a crash is a
  recorded outcome for that cell, not a blocker to the rest of the sweep.
- **FR-007**: The study MUST present results as four tables (4w-pinned, 4w-floating, 8da4w-pinned,
  8da4w-floating), each covering all three model sizes with per-rep prefill/decode tok/s. For the
  two floating-clock tables, the mean MUST be shown alongside the per-rep values but explicitly
  labeled as potentially mixing cold-start peak with throttled steady-state performance (per the
  project constitution's documented -19% to -27% thermal drift for floating tiled configs) —
  never presented as an unqualified single number.
- **FR-008**: The study MUST record which storage type (texture, i.e. the T-tiled path with no
  coopmat) and which runner binary were used, so the numbers are unambiguous about what was
  measured.
- **FR-009**: Every run reported as "pinned" MUST be verified two ways before being trusted as
  pinned: (a) devfreq sysfs readback confirming the pin values took effect, and (b) an in-graph
  throughput cross-check confirming that run's prefill tok/s is no more than 70% of the
  already-collected floating number for the same (model, quant-mode) cell (a concrete threshold,
  chosen with headroom below the ~1.9× pinned-vs-floating clock ratio already observed on this
  device, so genuine noise doesn't false-positive). A run whose throughput exceeds that threshold
  despite a successful-looking sysfs write MUST NOT be reported as pinned — it is recorded with
  the `dvfs_artifact` outcome (FR-012) instead.
- **FR-010**: The study document MUST explicitly frame all M41 results as a secondary/cross-device
  reference baseline — the same treatment this workstream's constitution already gives the retired
  MiniPC's results — and MUST NOT present M41 numbers as directly comparable to, or a substitute
  for, Samsung M5 EVT1 headline numbers.
- **FR-011**: For every (model, quant-mode, clock-mode) cell with 2 or more valid reps, the study
  MUST report the coefficient of variation (CoV = stdev / mean, as a percentage) for both prefill
  and decode tok/s, so run-to-run variability on this device is visible alongside the mean/per-rep
  values — not just the raw numbers.
- **FR-012**: Each of the four results tables MUST show one column/row per rep (3 reps × 3 models),
  and every one of those 9 cells MUST contain one of exactly three things: the rep's numeric
  prefill/decode tok/s, the word "CRASHED" (with a pointer to that failure's documented error
  signature), or — for a pinned rep that failed FR-009's throughput cross-check — the word
  "DVFS-ARTIFACT" (with the measured throughput, so a reader can see it landed in the floating
  range). No rep is omitted from the table, and no table is presented with fewer than 9 cells
  shown.

### Key Entities

- **Run**: One execution of the release/1.3 baseline binary against a single (model size, quant
  mode, clock mode) combination; carries prefill tok/s, decode tok/s, and a pass/fail outcome.
- **Device State**: The environmental context a Run was captured under — driver md5 hash, clock-pin
  configuration (pinned values or floating range), and thermal reading at time of failure (if any).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A single document exists recording the M41 driver hash and whether it matches any
  documented known-good reference.
- **SC-002**: For each of the 4 quant-mode × clock-mode combinations, per-rep prefill and decode
  tok/s are reported for all 3 model sizes (fewer than 3 reps only when a rep failed, with the
  failure documented); for the 2 floating-clock combinations, an accompanying mean is shown with
  an explicit thermal-drift caveat rather than presented as an unqualified blended average.
- **SC-003**: Every crash encountered during the study is documented with its error signature and
  suspected cause, and confirmed not to have left the device in a broken/unresponsive state.
- **SC-004**: A reader unfamiliar with this session can reconstruct what was measured, on which
  hardware/driver, and under which clock policy, from the resulting document alone, without
  needing to replay this conversation.
- **SC-005**: Every run labeled "pinned" in the final document has a recorded sysfs pin-readback
  AND a throughput cross-check showing it was at or below 70% of the corresponding floating
  number; no run is labeled "pinned" on the strength of a successful-looking sysfs write alone,
  and any run that fails this check is labeled "DVFS-ARTIFACT" (FR-012) instead.
- **SC-006**: The document contains an explicit statement that M41 is a secondary/cross-device
  reference, not the active mission target, before or alongside its first results table — a reader
  cannot mistake these numbers for Samsung M5 EVT1 headline data.
- **SC-007**: All four tables (4w-pinned, 4w-floating, 8da4w-pinned, 8da4w-floating) are complete —
  every one of the 9 cells per table (3 models × 3 reps) shows a number or "CRASHED", with CoV
  reported per cell that has ≥2 valid reps — and delivered by end of day 2026-07-14. A crash in any
  cell delays only that cell, never the rest of the sweep.

## Assumptions

- The M41 device (serial `000009b44fd4abd3`, host `xgpusw-debug07`, Exynos s5e9965/ERD9965
  family) remains the target for the entire study; no driver flash is performed since no
  known-good hash is documented for this chip family.
- "Same release 1.3 vanilla baseline" means the `llama_main_rel1.3` runner (plain release/1.3
  branch, no coopmat/node-threshold features) against texture-storage PTEs (the T-tiled path) —
  consistent with how "baseline" was defined earlier in this session.
- The study covers all three model sizes already used this session (Llama 3.2 1B, Llama 3.2 3B,
  Llama 3.1 8B), each at 2048-token prefill + 1024-token decode.
- 8da4w-texture PTEs at `ctx3072` for all three model sizes are assumed to already exist in this
  workspace's `.pte_out`/NFS staging; if not present on-device they are pushed from there, not
  freshly exported.
- Pinned clocks use the workspace default 509/2730/663 MHz; floating means DVFS unpinned across
  each devfreq node's full hardware range, as already configured on this device.
- 3 reps per configuration matches the sample size already used this session and elsewhere in this
  workspace's specs, and is the minimum needed to compute a per-cell CoV as requested.
- The full 4-table deliverable (36 rep-slots total: 4 quant-mode × clock-mode combinations × 3
  models × 3 reps) is targeted for completion by end of day 2026-07-14. A crashed rep does not
  extend this deadline by triggering a retry — it is recorded as "CRASHED" and the sweep moves on.
- A crash (e.g., the pinned 8B watchdog crash, or a thermally-induced floating crash) is itself a
  valid, reportable result for that cell — the study does not require forcing a clean number by any
  means (e.g., no silent fallback to a different clock policy without documenting the substitution).
- Per the project constitution's Principle II ("Samsung M5 EVT1 Is the Only Active Target"), M41 is
  not the active mission target; this study's results are a secondary/cross-device reference
  baseline (the same treatment already given the retired MiniPC's results), not a substitute for or
  direct comparator against Samsung M5 EVT1 headline numbers.
- A "pinned" run additionally requires an in-graph throughput cross-check (not just a successful
  sysfs pin write) before being trusted as pinned, per the project constitution's Principle VII and
  its documented prior DVFS-boost-mistaken-for-pinned incident.
