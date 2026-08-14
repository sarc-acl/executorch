# Feature Specification: Buffer-Storage Release/1.3 Baseline (M51, M41, S25 Ultra)

**Feature Branch**: `035-buffer-storage-baseline`

**Created**: 2026-07-22

**Status**: M41 and S25 Ultra complete (floating + pinned where applicable; reports + cross-device synthesis written). **M51 handed off to the user mid-sweep 2026-07-22** — partial data only (1B/4w clean 3/3; 1B/8da4w crashes deterministically 3/3, new vs. texture, unresolved; 3B/4w 2/3 reps); not reflected in the cross-device synthesis. Executed directly (no speckit plan/tasks machinery), per user instruction.

**Input**: User description: "Do a fresh Release/1.3 vanilla baseline measurement, identical in
methodology to the existing texture-storage baseline, but using BUFFER-storage `.pte` files instead,
across the devices in this project: M5 EVT1 (M51), M41, Qualcomm S25 Ultra, and the RDNA3 discrete GPU
(RX 7900 XTX via xraytracing02). Execute this directly (build/run/measure) — do not go through speckit's
plan/tasks machinery, just use a spec file as documentation of intent like `specs/034-rdna3-dgpu-baseline`
already does, and write results directly into `results/`. **RDNA3 integrated-GPU miniPC is excluded
(unreachable).**"

## Scope

This spec covers the mobile/mixed device set that does **not** already have a buffer-storage baseline:
**M51, M41, S25 Ultra.** The RDNA3 discrete GPU (RX 7900 XTX) already has both texture and buffer
baselines complete in `specs/034-rdna3-dgpu-baseline/results/` (`report.md` / `report-buffer.md`) — this
spec folds that device's numbers into the final cross-device synthesis (User Story 3) rather than
re-running it. The RDNA3 **integrated**-GPU miniPC is out of scope — unreachable, no documented access
method exists anywhere in this repo (per the same gap `specs/034`'s own prep note recorded).

## Workload (identical across every device)

- Models: Llama 1B / 3B / 8B, quantization 4w and 8da4w (group size 128), `ctx3072` buffer-storage
  `.pte` files.
- Prefill: 2048 tokens (`p2048_exact.txt`, `--num_bos=1`). Decode: 1024 tokens (`--ignore_eos
  --temperature=0 --warmup=true`).
- Reps: n=3 per cell, report median ± CoV.
- Coherence check before any timed rep: `--prompt='The capital of France is' --seq_len=48
  --temperature=0 --warmup=false`, expect coherent "...Paris..." output.
- Runner: existing vanilla `release/1.3` binary `llama_main_rel1.3` (already staged on NFS) — storage
  type is a `.pte`-embedded property, not a build-time flag; no rebuild needed. `.pte` files: the 6
  existing buffer-storage files at `/sarc-c/gpusw/users/yanwen.xu/android-run/models/
  <model>_<quant>_buffer_ctx3072.pte` (already exported 2026-07-09 from `dev/executorch` with
  `ET_VK_FORCE_BUFFER=1`; vanilla `release/1.3`'s own export path has no buffer-override knob at all).

## User Scenarios & Testing

### User Story 1 — Floating-clock buffer baseline on M51, M41, S25 Ultra (Priority: P1)

As the maintainer of the cross-device baseline report, I need buffer-storage throughput numbers
(prefill/decode tok/s) for all 6 model×quant cells on each of these three devices at floating clocks, so
they sit next to the existing texture-storage numbers (`specs/029`/`030`/`031`) and the RDNA3 dGPU's own
buffer numbers (`specs/034`), enabling a same-shape texture-vs-buffer comparison on every device.

**Why this priority**: Floating is achievable on every device (including the no-root S25 Ultra), so it's
the one dataset guaranteed to complete across all three — the primary deliverable.

**Acceptance**: each of 3 devices × 6 cells reports prefill/decode tok/s (median±CoV, n=3, or a
crash-attributed NR), using each device's existing coherence-check and crash-attribution convention
(dmesg+meminfo cross-check on M51/M41; "observed signature, not diagnosed" on the no-root S25 Ultra).

### User Story 2 — Pinned-clock buffer baseline on M51 and M41 (Priority: P2)

As the same maintainer, I want a pinned-clock (509/2730/663 MHz) buffer column for M51 and M41 — the only
two devices in this set with a genuine clock-pinning mechanism — matching the pinned columns already
reported for texture storage in `specs/029`/`030`/`031`.

**Why this priority**: valuable for clock-normalized comparison, but strictly follows Story 1 (need the
floating baseline first, both as the primary deliverable and as the DVFS-artifact sanity check per M41's
own convention: a "pinned" run must be ≤70% of that cell's floating throughput or it's relabeled
`DVFS-ARTIFACT`).

**Acceptance**: pinned table populated for both devices (6 cells each), reusing/validating each device's
known `ET_VK_EXECUTE_NODE_THRESHOLD` fallback for the 8B (and some 3B) crash cells established on texture
storage (`specs/031` for M51, `specs/030` for M41) — confirmed empirically for buffer, not assumed to
transfer unchanged.

### User Story 3 — Cross-device buffer-vs-texture synthesis (Priority: P3)

As the maintainer, once buffer numbers exist for M51/M41/S25U, I want them combined with the RDNA3 dGPU's
already-complete buffer numbers into one cross-device view: per-device texture-vs-buffer delta, and an
updated "does 8da4w beat 4w" answer checked against buffer storage specifically.

**Acceptance**: each device's `results/report.md` contains a texture-vs-buffer table (skipped only where
a cell has no valid texture baseline to compare against, e.g. a deterministic texture crash), and a short
cross-device summary note is added covering all 4 devices (M51, M41, S25U, RDNA3 dGPU).

### Edge Cases

- A `.pte` exported from `dev/executorch` may not be release/1.3-compatible on a given device — always
  coherence-check per device, don't assume the RDNA3 dGPU's own compatibility confirmation transfers.
- Buffer storage may shift *which* cells crash relative to texture (S25 Ultra's Adreno backend has a
  documented history of buffer-specific crash fixes on a separate branch not present in vanilla
  `release/1.3`) — a new crash pattern here is an expected possible outcome, not a setup error.
- Buffer storage may change 8da4w's long-decode correctness failure *mode* without changing which model
  sizes are affected (already observed on the RDNA3 dGPU: garbled-text under texture became `!!!!`
  repetition under buffer) — check for this on every device, report as a caveat, don't discard the
  throughput number.
- If a device's driver/board state has drifted from its documented default, record the finding and the
  fingerprint; only reflash after explicit user sign-off (per the standing hold this workspace has used
  for M51's primary board since `specs/028`), not automatically.

## Requirements

- **FR-001**: Cover all 6 model×quant cells on M51, M41, and S25 Ultra using the existing buffer `.pte`
  files, matching the workload already used for texture storage.
- **FR-002**: Report prefill/decode tok/s as median±CoV across n=3 reps, or an attributed NR.
- **FR-003**: Run the floating-clock sweep on all three devices as the primary deliverable before
  attempting any pinned-clock work.
- **FR-004**: Run the pinned-clock sweep on M51 and M41 (509/2730/663 MHz), applying and empirically
  re-validating each device's known node-threshold crash workaround; mark S25 Ultra's pinned column `NR`
  (no root).
- **FR-005**: Attribute every crash via each device's own established convention (dmesg/meminfo on
  M51/M41; "observed, not diagnosed" on S25 Ultra) before finalizing a cell as NR.
- **FR-006**: Coherence-check every `.pte` on every device before trusting any timed rep from it.
- **FR-007**: Record per-cell provenance (commit, driver fingerprint, `.pte` path, exact command) for
  reproducibility, matching the existing reports' Reproduce section.
- **FR-008**: Deliver one `results/report.md` per device, same table shape as `specs/034`'s reports, plus
  a texture-vs-buffer comparison section per device.
- **FR-009**: Fold in the already-complete RDNA3 dGPU buffer numbers (`specs/034/results/report-buffer.md`)
  when producing the final cross-device "8da4w vs 4w under buffer storage" synthesis.
- **FR-010**: Log every command executed against real hardware to `.artifacts/cmd-log-2026-07-22.sh`.

## Assumptions

- The 6 existing buffer `.pte` files at `/sarc-c/gpusw/users/yanwen.xu/android-run/models/` (exported
  2026-07-09) are reusable as-is; a fresh export is only produced if a device's coherence check fails in
  a way traceable to `.pte`/release mismatch rather than a device-specific crash.
- `llama_main_rel1.3` (already staged on NFS) and, for the known 8B/3B crash cells,
  `llama_main_nodethresh` (the `release13-node-threshold` branch's runner) are reused as-is — no rebuild.
- "M51" and "M5 EVT1" are the same physical board (naming consolidated 2026-07-22); M41 as of 2026-07-22
  is the sole board `gpusw-m41-08` (serial `00000a34cdd4abd3`) — the `specs/030`-era M41 serial is retired.
- Driver/board drift, if found, is recorded and not auto-flashed without explicit user sign-off, following
  the standing precedent on M51's primary board.
