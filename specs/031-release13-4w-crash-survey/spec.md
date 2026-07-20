# Feature Specification: Release/1.3 Crash Survey on M5 EVT1 (4w + 8da4w, Floating + Pinned)

**Feature Branch**: `031-release13-4w-crash-survey`

**Created**: 2026-07-14

**Status**: Answered — extended same day, scope now covers both quant schemes and both clock
policies (see Extension below); original `4w`-only, floating-only scope fully superseded.

**Input**: User description: "idendify on M51, for vanila release 1.3, which will crash and will
will be normal" — followed mid-session by: "by end of the day, i want to see the report table of,
under the floating condition, for each 1/3/8B model, for 4w, 3 rep with CoV, what is the tok/s for
prefill and decode (annotate crash if it crash)." — followed by an **extension** later the same
day: "i want to see the report table of, both floating and pinned condition, for each 1/3/8B
model, for 4w and 8da4w, 3 rep with CoV, what is the tok/s for prefill and decode. For the
previously crashed entry, use the threadhold=32 and note that in the table" — refined again to
"actually we should retry all threadhold to 64" / "for all the previous 32 entry, recollect on
64 threashold."

## Extension (2026-07-14): scope now covers both quant schemes and both clock policies

The original scope (User Stories 1–2 below, `4w`-only, floating-only) is retained as-is and
fully answered. This same-day extension adds, using the identical methodology (coherence check →
vanilla attempt first → threshold fallback only on confirmed crash → 3 completed reps + CoV):

- **`4w` pinned-clock gap-fill**: 3B pinned and 8B pinned were never measured in the original
  scope; both are now covered.
- **`8da4w`**: all three models, both clock policies — a full second quant scheme added
  alongside `4w`, resolving the original spec's `4w`-only Assumption.
- **Threshold policy refinement**: the original single-cell workaround (`ET_VK_EXECUTE_NODE_THRESHOLD=32`,
  8B floating only) is superseded by a per-cell empirical policy: default fallback is now `64`
  (per explicit user direction mid-extension), falling back further to `32` only where `64` was
  directly confirmed insufficient (8B pinned, both quant schemes — the only cell(s) where `64`
  crashed).

See `results/report.md` for the full combined table (both quant schemes × both clock policies ×
all three models) and `research.md` for the extension's methodology decisions (why pinned is
riskier than floating, why 8B-pinned specifically needs the smaller threshold, run-order
reasoning for the new cells).

## Second extension (2026-07-16): sustained max-pinned clocks, 8B `4w` vanilla only

Follow-up user ask: pin GPU/MIF/INT to their hardware max (980000/5333000/934000, confirmed via
sysfs — see `../../.shared-context/instruction-for-ai/hardware/README.md`) instead of the
workspace's 509MHz-pinned default or floating, and re-test vanilla 8B `4w` (no threshold) — testing
whether a *sustained* max clock (vs. floating's intermittent up-to-980) avoids the watchdog crash.
It does not: 3/3 crashed, terminal. See `results/8b-4w-max-pinned-2026-07-16.md` for the full
write-up; `results/report.md`'s headline table and bottom line are updated with this cell. Scope
was intentionally narrow (8B `4w` only, one clock policy) — not a fourth full pass over all 12
cells.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Establish the crash/normal boundary across model sizes (Priority: P1)

A performance engineer needs to know, before trusting or reporting any floating-clock number on
vanilla `release/1.3` (no WMMA/coopmat fork additions), which model sizes actually complete a
standard e2e run on M5 EVT1 and which ones crash the device. Ad hoc single runs already showed
8B crashing (device dropped off `adb`, re-enumerated as `S5E9975_LK_Bootloader`) and 1B also
crashing on a repeat attempt, while 3B completed cleanly once — but each model has only one data
point so far, which is not enough to call a boundary.

**Why this priority**: Every other deliverable (the report table) depends on knowing, with
repeated evidence rather than a single anecdote, whether a given model reliably crashes, reliably
succeeds, or crashes intermittently under this configuration.

**Independent Test**: Can be fully tested by attempting 3 repetitions each of the 1B, 3B, and 8B
Llama models (4w quant, `ctx3072`, 2048-token prefill + 1024-token decode) on vanilla
`release/1.3` with GPU clocks floating, and recording a pass/crash outcome for every single
repetition attempted.

**Acceptance Scenarios**:

1. **Given** a model/rep combination completes the run and prints the runner's JSON stats line,
   **When** the survey records the outcome, **Then** it is logged as a successful rep with its
   prefill/decode tok/s.
2. **Given** a model/rep combination causes the device to drop off `adb` and re-enumerate as the
   bootloader, **When** the survey records the outcome, **Then** it is logged as a crashed rep
   (not silently skipped, not retried in place of the missing data point), the device is
   recovered via `fastboot reboot`, and the driver hash + clock floating-range are re-verified
   before the next attempt proceeds.
3. **Given** all 3 reps for a model crash, **When** the report is written, **Then** that model is
   reported as crashing (e.g., "3/3 crashed") rather than omitted from the table.

---

### User Story 2 - Produce the end-of-day report table (Priority: P1)

The engineer needs a single, self-contained table — deliverable by end of day — showing, for each
of 1B/3B/8B at 4w under floating clocks: prefill tok/s, decode tok/s, and the coefficient of
variation (CoV) across the completed reps, with any crashed rep explicitly annotated rather than
silently dropped from the denominator.

**Why this priority**: This table is the literal, explicitly requested end-of-day deliverable —
without it the crash/normal classification from User Story 1 has no reportable artifact.

**Independent Test**: Can be fully tested by reading the resulting `results/` document and
confirming it contains one row per model with prefill tok/s, decode tok/s, CoV (or "N/A —
insufficient completed reps"), and a crash annotation column, plus the raw per-rep numbers behind
the summary.

**Acceptance Scenarios**:

1. **Given** a model completed 3/3 reps, **When** the table is built, **Then** it shows mean (or
   median, consistent with this workstream's existing convention) prefill/decode tok/s and CoV
   computed from exactly those 3 reps.
2. **Given** a model completed fewer than 3 reps (some crashed), **When** the table is built,
   **Then** the tok/s columns are computed only from the completed reps (or marked N/A if zero
   completed), and a separate annotation states how many of the 3 attempts crashed.
3. **Given** the report is read by someone who was not in this session, **When** they read the
   table alone, **Then** they can tell, for each model, whether floating-clock `release/1.3`
   4w is safe to benchmark on M5 EVT1 without needing the raw chat history.

---

### Edge Cases

- What happens if a model crashes on rep 1 but succeeds on reps 2 and 3? → All 3 attempts are
  recorded individually; CoV is computed only over the successful reps; the crash count (e.g.
  "1/3 crashed") is reported alongside, not folded silently into the mean.
- What happens if `fastboot reboot` recovery itself fails (device does not re-enumerate on
  `adb` within a reasonable window)? → Stop the survey and escalate rather than continuing to
  retry blindly; this is a shared board and repeated unattended recovery attempts are
  out of scope for this spec.
- What happens if the on-device driver hash is found to have drifted to an unrecognized build
  mid-survey? → Halt measurement immediately (per this workstream's standing driver-drift rule),
  do not attribute any further crashes to model size until the driver identity is
  re-confirmed or reflashed.
- What happens if a "successful" rep's prefill/decode numbers look implausible (e.g. far below
  the model's already-observed floating-clock throughput) because the GPU throttled mid-run? →
  Report the number as-is with the raw per-rep table shown (not just a mean), so a reader can see
  cold-start-vs-throttled spread rather than a misleadingly smoothed single figure.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The survey MUST attempt exactly 3 repetitions of the standard e2e workload (2048-token
  prefill + 1024-token decode, `ctx3072` PTE, `--ignore_eos --temperature=0 --warmup=true`) for
  each of the 1B, 3B, and 8B Llama models at `4w` quantization, on the vanilla `release/1.3`
  worktree's runner (no WMMA/coopmat fork additions), with GPU clocks floating (devfreq
  `min_freq`/`max_freq` set to the hardware's full range, not pinned).
- **FR-002**: For every repetition, the survey MUST record a binary outcome — completed (with
  prefill tok/s and decode tok/s from the runner's own JSON stats line) or crashed (device
  dropped off `adb` / re-enumerated as bootloader) — before proceeding to the next repetition.
- **FR-003**: On a crashed repetition, the survey MUST recover the device via `fastboot reboot`
  (no flashing, no data wipe) and MUST re-verify both the driver `.so` md5 hash and the GPU
  clock floating-range via sysfs readback before attempting the next repetition or model.
- **FR-004**: The survey MUST verify the driver hash matches the documented default before each
  model's first repetition begins, consistent with this shared board's known drift risk.
- **FR-005**: The final report MUST present one row per model (1B/3B/8B) with: prefill tok/s
  (mean/median + CoV over completed reps), decode tok/s (mean/median + CoV over completed reps),
  and an explicit crash annotation (e.g. "2/3 crashed") — never omitting a model whose reps all
  crashed.
- **FR-006**: The report MUST also include the raw per-rep numbers (not just the summary row) so
  cold-start-vs-thermal-throttle spread within a model's completed reps remains visible.
- **FR-007**: The report MUST record the on-device driver hash and confirm it matched the
  documented default for the reps it presents, so the numbers are not silently attributed to a
  drifted/unknown driver.
- **FR-008** *(Extension)*: The survey MUST repeat FR-001–FR-007 for **both** `4w` and `8da4w`,
  and for **both** floating and pinned clocks, for every model — no (model, quant, clock) cell
  may be silently omitted from the final report.
- **FR-009** *(Extension)*: Where vanilla crashes on a given cell, the survey MUST first attempt
  the workspace-standard fallback threshold (`ET_VK_EXECUTE_NODE_THRESHOLD=64` as of this
  extension) before falling back further to `32` — and MUST NOT assume a threshold that worked
  on one (model, clock) cell also works on another without confirming it empirically on that
  specific cell.

### Key Entities

- **Benchmark Attempt**: one (model, quant ∈ {4w, 8da4w}, clocks ∈ {floating, pinned}, rep_index
  1–3, node_threshold ∈ {none, 32, 64}) combination run against vanilla `release/1.3` or
  `release13-node-threshold`; resolves to either a completed measurement (prefill/decode tok/s)
  or a crash event.
- **Crash Event**: a benchmark attempt that causes the M5 EVT1 board to drop off `adb` and
  re-enumerate as `S5E9975_LK_Bootloader`, requiring a `fastboot reboot` recovery before the
  survey can continue.
- **Model Row**: the report's unit of presentation — one of {1B, 3B, 8B} × {4w, 8da4w} ×
  {floating, pinned} (12 rows total post-extension) — aggregating its up-to-3 Benchmark Attempts
  into a summary (tok/s + CoV) plus a crash count and the node_threshold actually used.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: By end of day, a single report table exists covering all three model sizes (1B,
  3B, 8B) at 4w under floating clocks on vanilla `release/1.3`, with no model omitted.
- **SC-002**: Every model's classification (safe to benchmark under floating clocks vs. crashes)
  is backed by 3 attempted repetitions each, not a single anecdotal run.
- **SC-003**: A reader with no session context can determine, from the report alone, which model
  size(s) are safe to run under this exact configuration and which are not, including partial
  failure patterns (e.g. "crashed 1 of 3 times").
- **SC-004**: Every reported tok/s number is traceable to a specific rep with a confirmed-matching
  driver hash, so no number in the table is attributable to a drifted/unverified driver build.
- **SC-005** *(Extension)*: The final report covers all 12 (model × quant × clocks) cells, none
  omitted, each backed by 3 completed reps (via vanilla or a confirmed-necessary threshold
  fallback).
- **SC-006** *(Extension)*: For every cell where a threshold fallback was used, the report states
  which value (`32` or `64`) and that it was confirmed empirically on that cell — not carried
  over by assumption from a different model/clock combination.

## Assumptions

- "M51" in the original request refers to **M5 EVT1** (serial `0000088f8e579c33`, reached via
  `ssh yanwen.xu@sj1-dmckee-d01`) — the only device this session's investigation has touched;
  there is no separate "M51" device in this workspace's device roster.
- "Vanilla release 1.3" means the read-only `release-1.3/` worktree (upstream `release/1.3`, no
  WMMA/coopmat fork additions), matching how `specs/029-release-version-4w-baseline` used the
  term.
- ~~Scope is `4w` only for this spec~~ — **superseded by the Extension above**: `8da4w` was added
  same day, using the identical methodology. This bullet is kept (struck through) rather than
  deleted so the original scope decision remains visible in history.
- The workload is this workstream's standing default — 2048-token prefill + 1024-token decode,
  `ctx3072` PTE — not a custom shorter/longer budget.
- If a model crashes on all 3 attempted reps, the report states that plainly (e.g. "3/3
  crashed") rather than fabricating tok/s figures from a partial or corrupted run.
- Clocks are left floating between repetitions and models (not re-pinned mid-survey); clock
  policy is restored to the workspace's pinned default only after the full survey concludes.
