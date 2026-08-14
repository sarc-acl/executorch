# Feature Specification: RDNA3 Discrete GPU Release/1.3 Baseline

**Feature Branch**: `034-rdna3-dgpu-baseline`

**Created**: 2026-07-22

**Status**: Draft

**Input**: User description: "Earlier baseline measurement ran vanilla release/1.3 on M5 EVT1, M41, S25 Ultra, and the RDNA3 iGPU miniPC (4w/8da4w × floating/pinned, Llama 1B/3B/8B, 2048-token prefill + 1024-token decode). Now do the same baseline measurement on the AMD dGPU RDNA3, which needs to be accessed from the xtracer Linux box."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Floating-clock baseline on the RDNA3 discrete GPU (Priority: P1)

As the person maintaining the cross-device Release/1.3 baseline report, I need vanilla release/1.3 throughput numbers for Llama 1B/3B/8B in both 4w and 8da4w quantization on the RDNA3 discrete GPU (RX 7900 XTX, reached via the xtracer/xraytracing02 Linux host), at 2048-token prefill + 1024-token decode, so this device's numbers sit next to M5 EVT1/M41/S25 Ultra/RDNA3-iGPU in the same table and the existing cross-device questions (e.g. "does 8da4w beat 4w everywhere except M5 EVT1?") can be answered with one more data point.

**Why this priority**: This is the only strictly new data the request asks for — every prior device's floating-clock numbers already exist. Without this, there is no dGPU row to compare at all.

**Independent Test**: Can be fully tested by running all 6 model×quant cells (1B/3B/8B × 4w/8da4w) at floating clocks on the RDNA3 dGPU and producing a table with prefill/decode tok/s (median ± CoV, n=3 reps where no crash intervenes) — deliverable on its own even if pinned-clock measurement (User Story 2) turns out to be infeasible on this device.

**Acceptance Scenarios**:

1. **Given** the RX 7900 XTX host (xraytracing02) with a vanilla release/1.3 Linux runner built and the six existing 1B/3B/8B × 4w/8da4w `.pte` files staged, **When** each of the 6 cells is run for 3 reps at floating clock with a 2048-token prompt and 1024-token decode, **Then** each cell reports a prefill and decode tok/s figure with CoV, using the same median-of-3-reps convention as the other four devices' reports.
2. **Given** a cell crashes during a rep, **When** the crash is investigated, **Then** it is attributed to a specific cause (e.g. GPU hang/reset, OOM, driver fault) via host log cross-check before the cell is finalized as NR or partial-n, following the same crash-recovery-and-attribution discipline already used on M5 EVT1/M41 — never left as "unattributable" without at least attempting the check.
3. **Given** a model/quant cell completes its reps, **When** results are recorded, **Then** a short-prompt coherence check (sensible, non-garbled output) is confirmed for that cell before its timed numbers are reported as valid.

---

### User Story 2 - Pinned-clock baseline, if a pinning mechanism exists (Priority: P2)

As the same maintainer, I want a pinned-clock column for the RDNA3 dGPU matching the other devices' reports, but only if a genuine clock-pinning mechanism exists on this GPU/host (the RDNA3 iGPU miniPC and the no-root S25 Ultra both had to leave this column as `NR` for documented structural reasons) — so the report doesn't imply pinning was skipped by oversight when it may in fact be infeasible here too.

**Why this priority**: Valuable for apples-to-apples clock-normalized comparison against M5 EVT1/M41's pinned rows, but the floating numbers alone (User Story 1) already answer the primary "does this device behave like the other RDNA3 GPU" question, and pinning may not be achievable on discrete desktop GPU driver stacks the same way it is on mobile SoC devfreq nodes.

**Independent Test**: Can be tested independently by first determining whether a clock-pinning control exists for this GPU (e.g. a fixed-frequency mode exposed by the driver/tooling on this host); if one exists, run the same 6-cell sweep at the pinned clock and report it; if none exists, the story is still "done" by recording that determination and the reason, exactly as the S25 Ultra and RDNA3-iGPU reports already do for their own NR cells.

**Acceptance Scenarios**:

1. **Given** a clock-pinning mechanism is found for this GPU/host, **When** the same 6-cell sweep is run at a fixed clock, **Then** the pinned-clock table is populated the same way the floating one is (median ± CoV, n=3, crash-attributed).
2. **Given** no clock-pinning mechanism can be found, **When** the pinned column is reported, **Then** it is marked `NR` with the specific structural reason recorded (not left blank or silently omitted).

---

### User Story 3 - Cross-device synthesis (Priority: P3)

As the maintainer, once the RDNA3 dGPU numbers exist, I want them folded into the existing cross-device comparison (the "is 8da4w faster than 4w on every device except M5 EVT1?" style questions) so the new device's data point is actually used, not just filed away.

**Why this priority**: This is synthesis of already-produced data, valuable but strictly dependent on Stories 1–2 existing first — it adds no new measurement risk of its own.

**Independent Test**: Can be tested by taking the completed RDNA3 dGPU table and checking that the existing report's cross-device Q&A section is updated (or a new question is added) using this device's numbers, without needing any new measurement runs.

**Acceptance Scenarios**:

1. **Given** the RDNA3 dGPU floating-clock table is complete, **When** the cross-device comparison is updated, **Then** it states, for each model size, whether 8da4w beats 4w on this device — consistent in format with the existing M41/S25 Ultra answers.

### Edge Cases

- What happens if the RX 7900 XTX host is shared with other work at measurement time (as several mobile devices in this project are)? The device/driver state MUST be checked immediately before use and the check result recorded, even though this is a desktop host rather than a shared mobile board.
- What happens if the existing NFS-archived `.pte` files (built for the mobile Vulkan targets) are not directly usable on this host's Linux x86_64 runner? A fresh export for this target MUST be produced and archived rather than silently reusing a mismatched file.
- What happens if the vanilla runner crashes on a model size the way it did on M5 EVT1/M41/S25 Ultra (each had at least one crashing cell)? The same crash-recovery loop (retry, cross-check host logs, attribute cause) MUST be applied — a crash here is expected as a possible outcome, not a setup error.
- What happens if 8da4w on this device produces incoherent/gibberish decode text the way it did on M41's pinned 3B/8B cells? The throughput number MUST still be reported, with the correctness caveat noted separately, exactly as the M41 report already does.
- What happens if no clock-pinning control can be found for this GPU? The pinned column MUST be marked `NR` with the specific reason, not omitted from the table.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The measurement process MUST cover all 6 model×quantization cells (Llama 1B/3B/8B × 4w/8da4w) on the RDNA3 discrete GPU (RX 7900 XTX via the xtracer/xraytracing02 host), at a 2048-token prefill followed by 1024-token decode — matching the workload already used on M5 EVT1, M41, S25 Ultra, and the RDNA3 iGPU.
- **FR-002**: The process MUST report prefill and decode throughput (tok/s) as median ± coefficient of variation across completed reps (target n=3), consistent with the reporting convention used in the existing four-device baseline.
- **FR-003**: The process MUST attempt a floating-clock sweep of all 6 cells as the primary deliverable.
- **FR-004**: The process MUST determine whether a genuine clock-pinning mechanism exists for this GPU/host and, if so, run the same 6-cell sweep at a pinned clock; if not, mark the pinned column `NR` with the specific documented reason (mirroring the S25 Ultra no-root and RDNA3-iGPU precedents).
- **FR-005**: Any crash encountered during measurement MUST be cross-checked against host-side logs (e.g. kernel/driver logs) to attribute a specific cause before the affected cell is finalized, rather than being left as unattributed variance.
- **FR-006**: Every model×quant cell MUST pass a short-prompt output-coherence check before its timed reps are treated as valid.
- **FR-007**: The process MUST record enough environment provenance per cell (release/1.3 commit, GPU driver/Mesa version, `.pte` source and checksum, exact run command) that any reported number can be reproduced later, matching the "Reproduce" section convention already used for M5 EVT1 and M41.
- **FR-008**: Results MUST be delivered in the same per-device table shape (Clock × Quant × Model → Prefill/Decode tok/s with CoV and n, plus a crash-notes column) used for the other four devices, so the new row is directly comparable without reformatting.
- **FR-009**: Once the floating-clock table is complete, the existing cross-device comparison (e.g. "does 8da4w beat 4w on every device except M5 EVT1?") MUST be updated to include this device's answer.

### Key Entities

- **Measurement Cell**: One (model size × quantization scheme × clock mode) combination — e.g. "8B / 8da4w / floating" — the unit that gets a prefill tok/s, decode tok/s, CoV, rep count, and crash-notes value.
- **Device Under Test**: The RDNA3 discrete GPU (RX 7900 XTX) reached via the xtracer/xraytracing02 Linux host — distinct from the already-measured RDNA3 *integrated* GPU miniPC, even though both are RDNA3.
- **Crash Event**: A run that terminated abnormally; carries an attributed cause (GPU hang/reset, OOM, driver fault, or explicitly "unattributable" only after a genuine log check), matching the crash-log convention from the M5 EVT1/M41 report.
- **Baseline Report**: The consolidated cross-device table/Q&A document that this device's results extend.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A complete floating-clock table exists for all 6 model×quant cells on the RDNA3 dGPU, each cell showing prefill/decode tok/s with CoV, or an explicit crash-attributed NR.
- **SC-002**: 100% of reported cells carry enough recorded provenance (commit, driver/Mesa version, `.pte` checksum, command) to be reproduced without asking the original measurer for missing details.
- **SC-003**: 100% of crashes encountered during measurement are resolved to a specific attributed cause via a host-log cross-check, with zero cells left as "unattributable" without that check having been attempted.
- **SC-004**: The cross-device "8da4w vs 4w" comparison question can be answered for this device directly from the produced table, with no further measurement needed.
- **SC-005**: The pinned-clock column is either fully populated (6 cells) or explicitly marked `NR` with a documented structural reason — never silently blank.

## Assumptions

- The RDNA3 discrete GPU is the RX 7900 XTX (Navi 31, gfx1100) reachable via `ssh` to the xraytracing02 host, with `/sarc-c` NFS mounted there, as already used for prior desktop-GPU work on this project.
- The release/1.3 Linux x86_64 `llama_main` runner is built directly on the xraytracing02 host (native build), not cross-compiled, matching how this host was used previously.
- The 6 existing NFS-archived `.pte` files (1B/3B/8B × 4w/8da4w, ctx3072) are reusable as-is for this target; if a cell fails to load or behaves inconsistently with the mobile results for reasons traceable to `.pte` mismatch, a fresh export for this target is produced instead of forcing the existing file.
- Whether a "pinned clock" control exists for this GPU is discovered during measurement, not assumed in advance; the RDNA3 iGPU and S25 Ultra precedent of leaving this `NR` when no mechanism exists is an acceptable outcome here too, not a gap to force-fill.
- The node-threshold GPU-watchdog workaround built for the mobile SoC's job scheduler is assumed not to apply to this discrete GPU/driver stack; it is only reached for if a comparable watchdog-style crash pattern actually appears here.
- "Same baseline measurement" means reusing the existing methodology (workload, rep count, table format, crash/coherence discipline) — not redesigning it — so this device's results can be appended to the existing cross-device report rather than requiring a new report structure.
