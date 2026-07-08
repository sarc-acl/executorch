# Feature Specification: M5 EVT1 End-to-End WMMA Validation (Linear 4w/8da4w + SDPA)

**Feature Branch**: `015-m5-e2e-wmma-validation`

**Created**: 2026-07-05

**Status**: Draft

**Input**: User description: "Now, before today, I have added 8da4w and 4w linear shaders with WMMA, and added SDPA shaders with WMMA, and tested them on the miniPC. From today on we are focusing on M5 EVT1. I showed that e2e results on 8da4w and 4w was good. So we need to apply these to M5, get the same numbers for all 3 models. One thing to note is that we had some new minor improvements (e.g., the nimble optimiazation), also we also need to use the 128x64 tile size shader for this M5 (you can check the .shared-context/report-for-human) where i found this is optimal. Now get the e2e result on M5"

## Clarifications

### Session 2026-07-05

- Q: This repo's `linear_qw_coopmat.glsl` (4w) already has the fp16-accumulate + loop-flattening + vectorized-dequant changes from `specs/014` layered on top of the already-shipped 128x64 tile -- correctness-verified on M5 but explicitly not perf-tested by prior decision. `quant-dev`'s already-published 128x64 numbers (8B 110.6/3B 213.9/1B 565.3 tok/s) come from a shader WITHOUT those extra changes. Should this feature measure today's shader as-is, or isolate spec 014's changes out first for a clean match against `quant-dev`'s published figures? → A: Measure as-is (today's shader, all of spec 014's changes included). This answers the real question -- what this repo's current code actually delivers on M5 -- and doubles as the perf validation spec 014 deliberately deferred. Numbers are reported as directional/comparable against `quant-dev`'s figures, not a reproduction, since the shaders have diverged.
- Q: The user's explicit ask ("get the same numbers for all 3 models") only names 4w/8da4w, but SDPA WMMA was mentioned as prior MiniPC work done alongside them. Is SDPA-coopmat e2e on M5 in scope for this feature? → A: Yes, include it. Broadens scope to also produce M5 SDPA-coopmat e2e numbers, extending the existing partial M5 SDPA finding (only 1B fully measured at 2048-prefill; 8B/3B blocked by the known GPU watchdog issue at 2048) rather than leaving it MiniPC-only.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Confirm WMMA actually dispatches on M5 EVT1 from this repo's own build (Priority: P1) 🎯 MVP

As the contributor driving this workstream, I need to know that this
repo's exported `.pte` artifacts, run through this repo's own Android
build, actually dispatch the coopmat/WMMA kernels on the M5 EVT1 target --
not a silent tiled fallback -- before trusting any e2e tok/s number
produced on this specific device/build/driver combination, which has never
been exercised together before (this repo's shader code, its own build,
and the M5 EVT1 hardware).

**Why this priority**: Every WMMA finding cited by the user (4w/8da4w
linear, SDPA) was measured on a *different* codebase (`quant-dev`) and/or a
*different* device (`rocky-ryzen` MiniPC). Per this workstream's
constitution (Principle VI: verify with tools, never assume; Principle II:
M5 EVT1 is the only active target, MiniPC data is historical/comparative
only), none of that prior evidence substitutes for confirming dispatch on
this exact combination. `specs/014`'s own session already found and fixed
real dispatch/build problems (a stale library, a silently-skipped
correctness check) that would have produced misleading results if not
caught -- the same discipline applies here before any number is trusted.

**Independent Test**: Can be fully tested by exporting one (model, scheme)
configuration, running it on M5 EVT1 with a separate ETDump-enabled
capture, and confirming from the actual per-op kernel names that the
coopmat/WMMA kernel family (linear and, if in scope, SDPA) dispatched --
not the tiled fallback -- before any timing number from that configuration
is trusted.

**Acceptance Scenarios**:

1. **Given** one target model exported at `4w` with `Buffer` storage,
   **When** it is run on M5 EVT1 with ETDump capture, **Then** the trace
   confirms the linear coopmat kernel family dispatched for the linear
   layers, not the tiled fallback.
2. **Given** the same confirmation attempted for `8da4w` and (per this
   feature's scope) SDPA-coopmat, **When** each is checked, **Then** each
   independently confirms dispatch or is reported as a dispatch failure --
   never assumed from one scheme's success.

---

### User Story 2 - Measure linear (4w, 8da4w) e2e prefill/decode tok/s for all three models on M5 EVT1 (Priority: P2)

As the contributor driving this workstream, I need real, tool-confirmed
end-to-end prefill/decode tok/s for `4w` and `8da4w`, across all three
target models, captured on M5 EVT1 using this repo's own current shader
code (128x64 tile plus `specs/014`'s fp16-accumulate/loop-flattening/
vectorized-dequant changes, per this spec's Clarifications) -- so the
"good e2e results" already shown are backed by the actual target hardware,
not carried over from MiniPC or a different codebase.

**Why this priority**: This is the measurement the user explicitly asked
for. It depends on User Story 1's dispatch confirmation succeeding for
each configuration first.

**Independent Test**: Can be fully tested by running the standard
2048-prefill/1024-decode e2e capture procedure against one dispatch-
confirmed configuration and producing a directly comparable prefill/decode
tok/s pair, independent of the other five configurations.

**Acceptance Scenarios**:

1. **Given** a dispatch-confirmed `4w` or `8da4w` configuration, **When**
   e2e prefill/decode tok/s is measured, **Then** it uses the fixed
   2048-token prefill / 1024-token decode workload, pinned clocks by
   default, and a separately-captured ETDump run for dispatch confirmation
   (never the same run used for the reported number, per Principle IV).
2. **Given** all six (model, scheme) configurations, **When** measurement
   is complete, **Then** each has an M5 EVT1 prefill/decode tok/s pair
   recorded, or an explicit blocked/failed status with a stated reason --
   including the two configurations (`8da4w` 3B and 1B) for which **no
   prior M5 EVT1 e2e baseline of any kind exists** to compare against,
   which must be reported as new measurement, not a reproduction.

---

### User Story 3 - Measure SDPA-coopmat e2e prefill/decode tok/s for all three models on M5 EVT1 (Priority: P3)

As the contributor driving this workstream, I need SDPA-coopmat e2e
prefill/decode tok/s on M5 EVT1 for all three models -- extending the
existing partial finding (only 1B fully measured at 2048-token prefill;
8B/3B were blocked by the known GPU-watchdog issue at that prefill length)
to a complete set where possible, per this spec's Clarifications.

**Why this priority**: Explicitly brought into scope by this spec's
Clarifications, but after the linear numbers (User Story 2) since those
were the user's primary, explicitly-named ask.

**Independent Test**: Can be fully tested by running the standard e2e
capture procedure against one dispatch-confirmed SDPA-coopmat
configuration (combined with linear coopmat, matching the existing 1B
finding's stack) and producing a directly comparable prefill/decode tok/s
pair.

**Acceptance Scenarios**:

1. **Given** a dispatch-confirmed SDPA-coopmat configuration, **When**
   e2e prefill/decode tok/s is measured, **Then** it follows the same
   workload/clock/dispatch-confirmation discipline as User Story 2.
2. **Given** 8B or 3B hits the same 2048-token-prefill GPU-watchdog issue
   previously blocking this exact measurement, **When** this occurs,
   **Then** it is reported explicitly as blocked with that stated reason
   (and whatever shorter-prefill data can still be captured, clearly
   labeled as such) -- not silently omitted, not estimated from 1B.

---

### User Story 4 - Report all results against the known prior findings, with divergences named (Priority: P4)

As the contributor driving this workstream, I need a consolidated report
stating each configuration's M5 EVT1 e2e result alongside the relevant
prior finding it can be compared to (`quant-dev`'s 128x64 4w numbers, the
lone `8da4w` 8B figure, SDPA's 1B figure) -- explicitly noting where no
prior baseline exists at all, and where this repo's shader is known to
differ from the one that produced the prior number -- so a reader gets one
clear, honestly-scoped answer, not a table of numbers presented as if they
were directly reproducing something already proven on this exact codebase.

**Why this priority**: Turns the raw measurements from Stories 1-3 into
the actual deliverable; lowest priority only because it depends on them
completing first.

**Independent Test**: Can be fully tested by taking the completed
measurements and producing a report whose per-configuration comparisons
and divergence notes are directly traceable to those measurements and to
the specific prior-finding documents cited.

**Acceptance Scenarios**:

1. **Given** the full set of measured (or explicitly blocked) results,
   **When** the report is produced, **Then** each configuration states its
   M5 EVT1 tok/s, the prior finding it's compared against (if any) with
   that finding's own source document named, and whether the comparison is
   a like-for-like reproduction attempt or only directional (per this
   spec's Clarifications, `4w`/`8da4w` linear and SDPA are all directional
   given the shader/codebase divergence).
2. **Given** `8da4w` 3B/1B (no prior baseline anywhere) or 8B/3B SDPA at
   2048-prefill (previously watchdog-blocked), **When** these appear in
   the report, **Then** they are explicitly marked as new measurement /
   no-prior-baseline, never presented alongside the others as if a known
   number were being confirmed.

### Edge Cases

- What happens if a configuration's coopmat/WMMA kernel does not dispatch
  (silent tiled fallback)? Reported as a dispatch-check failure per User
  Story 1; no e2e tok/s number is reported for it (Principle VI).
- What happens if 8B or 3B at 2048-token prefill (with SDPA-coopmat
  enabled) hits the previously-observed GPU-watchdog issue again? Reported
  explicitly with that stated reason, per User Story 3's Acceptance
  Scenario 2 -- this workstream's `ET_VK_EXECUTE_NODE_THRESHOLD` mitigation
  (already present in this repo, unlike its uncommitted state in the
  `quant-dev` worktree) is applied first; if the watchdog issue still
  recurs despite it, that is reported as a real, current blocker, not
  silently worked around further or assumed fixed from `quant-dev`'s
  history.
- What happens if the M5 EVT1 driver has drifted since it was last
  verified? Per constitution Principle VIII, re-verify before any
  measurement in this feature -- do not assume the driver state recorded
  at the end of `specs/014`'s session is still current.
- What happens if an `8da4w` `.pte` does not yet exist for a model (only
  `4w` buffer/texture exports exist in `.pte_out/` as of this spec's
  writing)? Export it as part of this feature's own work -- not a
  blocker, just a prerequisite step, per FR-001.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST produce genuinely `Buffer`-storage `.pte`
  exports for all three target models at both `4w` and `8da4w`, at the
  fixed 2048-prefill/1024-decode (`_ctx3072`) workload, using this repo's
  actual storage-override mechanism (`backend.vulkan.storage_override:
  buffer`, per `research.md` Decision 6) -- **not** the three pre-existing
  `4w` files named `_buffer_ctx3072.pte`, which User Story 1's dispatch
  check found were produced with a non-functional mechanism
  (`export-pte.md`'s `ET_VK_FORCE_BUFFER`, which does not exist in this
  repo) and are internally `Texture3D` despite their name. All six
  Buffer-storage PTEs (three `4w` **re**-exports, three `8da4w` new
  exports) must be produced fresh with the corrected mechanism, and each
  verified via its own dispatch-confirm check (FR-002) before use --
  presence of a correctly-named file is not evidence of correct content.
- **FR-002**: Before any e2e tok/s number is reported for a configuration,
  a separate ETDump-enabled run MUST confirm the intended kernel family
  (linear coopmat, and SDPA-coopmat where in scope) actually dispatched --
  never assumed from the export or eligibility gate alone (Principle VI).
- **FR-003**: The system MUST measure e2e prefill/decode tok/s for every
  dispatch-confirmed configuration using the fixed 2048-token prefill /
  1024-token decode workload; clocks pinned by default (Principle VII),
  with the pin's effect verified (not merely commanded) before any number
  is trusted, per Principle VII's own GFLOP/s-cross-check requirement; **3
  repeated runs per configuration, reporting the mean and coefficient of
  variation (CoV)** -- matching this workstream's own established e2e
  methodology (`.shared-context/report-for-human/e2e-spec.md`'s "3-run
  means," not a single-shot capture); and the two-tier discipline of
  Principle IV (a separate dispatch-confirmation run, never one of the
  three reported-number runs).
- **FR-004**: Before any measurement, the M5 EVT1 driver identity MUST be
  re-verified against the known-good hash table (Principle VIII) -- not
  assumed current from a prior session.
- **FR-005**: This feature MUST cover all three target models at both `4w`
  and `8da4w` (six linear configurations) plus SDPA-coopmat for all three
  models (three additional configurations, per this spec's Clarifications)
  -- nine configurations total.
- **FR-006**: Every reported number MUST be measured against today's
  actual shader code in this repo (128x64 tile plus `specs/014`'s three
  changes, per this spec's Clarifications) -- not a reconstruction of the
  `quant-dev` or MiniPC shader state.
- **FR-007**: The final report MUST state, per configuration, its M5 EVT1
  tok/s alongside the specific prior-finding document it's compared
  against (if any), whether that comparison is a like-for-like
  reproduction attempt or only directional, and an explicit no-prior-
  baseline flag for `8da4w` 3B/1B and (if applicable) 8B/3B SDPA at
  2048-prefill.
- **FR-008**: Any configuration that fails dispatch confirmation (FR-002)
  or hits the GPU-watchdog issue (Edge Cases) MUST be reported explicitly
  with a stated reason -- never silently omitted or estimated from another
  configuration.

### Key Entities

- **M5 EVT1 Linear/SDPA Configuration**: One (model, scheme) pair for
  linear (`4w`/`8da4w`, six total) or one model for SDPA-coopmat (three
  total), each carrying its `.pte` export, dispatch-confirmation outcome,
  and e2e prefill/decode tok/s (or blocked status with reason).
- **Prior-Finding Reference**: A specific, already-existing result this
  feature's measurement is compared against (e.g. `quant-dev`'s 128x64 4w
  numbers, the lone `8da4w` 8B figure, SDPA's 1B figure) -- named by its
  source document, with an explicit like-for-like-vs-directional label and
  a no-prior-baseline flag where none exists.
- **M5 EVT1 E2E Validation Report**: The consolidated document covering
  all nine configurations (or their explicit blocked status), each
  compared to its Prior-Finding Reference where one exists.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Every one of the nine configurations has either a
  dispatch-confirmed, tool-verified M5 EVT1 e2e prefill/decode tok/s pair,
  or an explicit blocked/failed status with a stated reason -- none
  silently missing.
- **SC-002**: A reader of the final report can determine, for every
  configuration, its M5 EVT1 number, what it's compared against (if
  anything), and whether that comparison is a reproduction attempt or only
  directional -- without needing to consult any other document.
- **SC-003**: No e2e tok/s number in the report is presented without a
  kernel-dispatch-confirmed, separately-captured ETDump trace backing it,
  a verified (not just commanded) clock pin, and a 3-run mean with its
  CoV -- a single unreplicated run is never presented as a validated
  number.
- **SC-004**: `8da4w` 3B/1B and any watchdog-blocked SDPA configuration are
  never presented as if reproducing a known prior number -- the report
  makes clear, for each, that no such prior number exists.

## Assumptions

- This repo's `linear_qw_coopmat.glsl`/`linear_dq8ca_qw_coopmat.glsl`
  already ship the 128x64 tile geometry found optimal in
  `.shared-context/report-for-human/jira-tile-sweep.md` (4w:
  `WG_TILE_M=128, WG_TILE_N=64, WG_TILE_K=16, SG_GRID=2x2, SUBGROUP_SIZE=32`;
  8da4w: same tile with `WG_TILE_K=32, SUBGROUP_SIZE=64`, per each
  shader's own committed YAML, verified directly during this spec's
  drafting, not assumed) -- confirmed by direct inspection, so no
  shader/tile-geometry work is in scope for this feature; it is purely
  export + measure + report.
- Per this spec's Clarifications, today's shader (128x64 tile plus
  `specs/014`'s fp16-accumulate/loop-flattening/vectorized-dequant changes)
  is measured as-is; this feature does not isolate or revert any of those
  changes, and doubles as the real-world perf signal `specs/014` itself
  deliberately deferred.
- The "nibble/nimble optimization" the user referenced is `specs/014`'s
  vectorized `dequant_block` change -- confirmed via direct search of
  `.shared-context/report-for-human/` that no separate or prior M5 EVT1
  finding of this name exists there; it is this repo's own, not-yet-
  perf-measured work, covered by this feature's measurement per the
  Clarification above.
- `quant-dev`'s already-published numbers (128x64 4w: 8B 110.6/3B 213.9/1B
  565.3 tok/s; `8da4w` 8B: 85.1 tok/s; SDPA 1B: 763 tok/s combined stack)
  are read-only reference points, not re-derived or re-validated by this
  feature -- they come from a different, independently-evolved codebase
  (per `specs/014`'s own research.md Decision 1) and are cited for
  directional comparison only.
- `8da4w` has no prior M5 EVT1 e2e baseline at all for 3B or 1B (confirmed
  by search, not assumed absent), and no `8da4w`+128x64-tile-specific
  baseline exists for any model size -- this feature's `8da4w` numbers are
  new measurement, not reproduction, for every model except the one
  existing 8B point of comparison.
- SDPA-coopmat's existing M5 EVT1 finding covers 1B fully at 2048-prefill;
  8B/3B were only measured at 512-prefill due to the GPU-watchdog issue
  (`.shared-context/report-for-human/session-2026-06-23-sdpa-wmma-findings.md`)
  -- this feature attempts the full 2048-prefill measurement again with
  this repo's own already-committed `ET_VK_EXECUTE_NODE_THRESHOLD`
  mitigation, per Edge Cases, rather than assuming the same block recurs.
- Validation runs on Samsung M5 EVT1, this workstream's sole active target
  (constitution Principle II) -- not `rocky-ryzen` MiniPC.
- Scope matches the constitution's default benchmark scope (all three
  target models, both int4 schemes, fixed 2048-prefill/1024-decode
  workload) plus SDPA-coopmat per this spec's Clarifications.
