---

description: "Task list for End-to-End Texture3D vs. Buffer Storage Comparison"
---

# Tasks: End-to-End Texture3D vs. Buffer Storage Comparison

**Input**: Design documents from `/specs/006-e2e-storage-comparison/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md (all present)

**Tests**: Not requested as a separate automated suite — this feature's correctness
checks (the safety-property verification, the smoke-check) ARE tasks below, matching
how `001`/`004` validated their own work inline rather than via a separate test phase.

**Organization**: Tasks are grouped by user story. This is **real device work**
(build, export, GPU capture on `rocky-ryzen`), like `001`/`004`, and it includes a
small real product-code fix (restoring dead code), unlike any prior feature in this
workstream.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Paths are relative to the repository root

## Path Conventions

- `backends/vulkan/utils.py`, `backends/vulkan/_passes/tag_memory_meta_pass.py` — the dead-code fix
- `extension/llm/export/partitioner_lib.py`, `examples/models/llama/export_llama_lib.py` — the new CLI flag
- `specs/006-e2e-storage-comparison/scripts/compare_e2e_storage.py` — new analysis script
- `specs/006-e2e-storage-comparison/results/pte/` — new Buffer-storage `.pte` exports
- `specs/006-e2e-storage-comparison/results/e2e-storage-comparison-report.md` — the deliverable

---

## Phase 1: Setup

- [X] T001 Create `specs/006-e2e-storage-comparison/scripts/`, `specs/006-e2e-storage-comparison/results/pte/`, and `specs/006-e2e-storage-comparison/results/raw/` directories

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: The export-side mechanism every user story's `.pte` export depends on

**⚠️ CRITICAL**: No user story work can begin until this phase is complete AND its safety property is verified

- [X] T002 Restore the dropped `default_storage` check in `backends/vulkan/utils.py`'s `TensorRepSet.make_tensor_repr()` (or its caller in `backends/vulkan/_passes/tag_memory_meta_pass.py`) so ambiguous repsets honor the pass's storage preference instead of unconditionally returning `TEXTURE_3D` (research.md Decision 1). Add an inline comment documenting: what was dropped (cite `bedce91e7f4795869158b96ef479d92317b13871`, "Rewrite Memory Metadata Tagging Pass"), and why restoring it is safe (default value already `TEXTURE_3D`, matching today's behavior) — per Constitution Principle V
- [X] T003 Add a `--vulkan-storage-override {texture3d,buffer}` CLI flag to `examples/models/llama/export_llama_lib.py`, forwarded through `extension/llm/export/partitioner_lib.py`'s `get_vulkan_partitioner()` to `VulkanPartitioner`'s `storage_type_override` compile option, mirroring the existing `--vulkan-force-fp16` → `force_fp16` plumbing exactly (research.md Decision 2) (depends on T002)
- [X] T004 **Safety verification (hard gate)**: export one configuration (e.g. `llama-3.2-1b_4w`) *without* the new flag and confirm it is behaviorally identical to `001`'s existing `Texture3D` `.pte` for that configuration (same e2e prefill/decode tok/s within noise) — if this fails, STOP and fix before proceeding; the fix from T002/T003 must not change default export behavior (depends on T003)

**Checkpoint**: Foundation ready and proven safe — the `Buffer`-storage export path is available and default behavior is provably unchanged

---

## Phase 3: User Story 1 - Confirm a Buffer-storage model actually runs at all (Priority: P1) 🎯 MVP

**Goal**: Prove the Buffer-storage export mechanism works end-to-end for one configuration before committing to all six.

**Independent Test**: Export one configuration with `--vulkan-storage-override buffer`, run it, and confirm it completes and produces coherent output.

- [X] T005 [US1] Export `llama-3.2-1b_4w` with `--vulkan-storage-override buffer` → `specs/006-e2e-storage-comparison/results/pte/llama-3.2-1b_4w_buffer.pte` (depends on T004)
- [X] T006 [US1] Smoke-check (research.md Decision 3): run `llama_main` against it with the fixed prompt at `--temperature 0`, confirm it completes without crashing, `generated_tokens` matches the request, and output is coherent (not degenerate) — NOT a token-for-token match against the `Texture3D` variant (depends on T005)
- [X] T007 [US1] If the smoke-check fails, investigate the failure; fix if within reasonable scope, otherwise record this configuration as blocked with the actual error (depends on T006)

**Checkpoint**: US1 complete — the Buffer-storage export mechanism is proven to work (or its failure mode is understood) on one real configuration

---

## Phase 4: User Story 2 - Measure e2e prefill/decode tok/s for both storage types, all six configurations (Priority: P2)

**Goal**: Extend the proven mechanism to all six configurations and capture real e2e numbers.

**Independent Test**: Run the standard e2e capture procedure against both storage variants for one configuration and confirm a directly comparable prefill/decode tok/s pair, independent of the other five.

- [X] T008 [US2] Export the remaining 5 configurations with `--vulkan-storage-override buffer` → `specs/006-e2e-storage-comparison/results/pte/<model>_<scheme>_buffer.pte` (depends on T007)
- [X] T009 [US2] Smoke-check each of the 5 (same criteria as T006); watch specifically for `lm_head`-related allocation failures (research.md Decision 4) — record any failure with its actual error, do not retry with a silent workaround (depends on T008)
- [X] T010 [US2] Capture e2e prefill/decode tok/s for every configuration that passed its smoke-check, using `001`'s exact methodology (fixed 2048/1024 workload, 5 repeated runs, discard cold-start drift, no concurrent load) (depends on T007, T009)
- [X] T011 [US2] Record any blocked/failed configurations explicitly with their reason — never silently excluded from the six-configuration count (depends on T008, T009)

**Checkpoint**: US2 complete — every measurable configuration has both a `Texture3D` (from `001`) and `Buffer` (this feature) e2e prefill/decode tok/s pair

---

## Phase 5: User Story 3 - Report whether the microbenchmark-level finding holds at the e2e level (Priority: P3)

**Goal**: Turn the e2e measurements into an answer: does `004`'s "storage switch is basically free" finding hold on the real model?

**Independent Test**: Generate the report and confirm each configuration's consistency verdict traces directly to its e2e numbers and to `004`'s prior finding.

- [X] T012 [US3] Implement `specs/006-e2e-storage-comparison/scripts/compare_e2e_storage.py`: load `001`'s `Texture3D` e2e data and this feature's `Buffer` e2e data, compute relative diff per configuration, determine `consistent`/`diverges` against `004`'s microbenchmark-level finding for the same configuration (depends on T010)
- [X] T013 [US3] Implement the report renderer: overall statement (does `004`'s finding generalize?), per-configuration comparison table, and an explicit blocked/failed section (present even if empty) — per `contracts/e2e-storage-schema.md` (depends on T012)
- [X] T014 [US3] Run end to end → `specs/006-e2e-storage-comparison/results/e2e-storage-comparison-report.md` (depends on T011, T013)
- [X] T015 [US3] Self-review against SC-001 through SC-004: confirm every configuration appears (measured or explicitly blocked), no timing appears for a failed smoke-check, and each measured configuration's consistency verdict is stated (depends on T014)

**Checkpoint**: US3 complete — the report answers whether `004`'s finding generalizes to the real model

---

## Phase 6: Polish & Cross-Cutting Concerns

- [X] T016 Reproducibility spot-check: re-capture one `Buffer`-storage configuration's e2e numbers and confirm they match the original capture within noise, matching `001`'s established reproducibility discipline
- [X] T017 [P] Update `quickstart.md` with any corrections found during T004-T016 (if any were needed)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies
- **Foundational (Phase 2)**: Depends on Setup. BLOCKS all user stories — and its own safety-verification step (T004) is itself a hard gate before any user story starts
- **User Story 1 (Phase 3)**: Depends on Foundational (T004 passed)
- **User Story 2 (Phase 4)**: Depends on US1 (T007) — extends the proven single-configuration mechanism to the rest
- **User Story 3 (Phase 5)**: Depends on US2 (T010, T011)
- **Polish (Phase 6)**: Depends on US3

### Within Each User Story

- US1: export (T005) → smoke-check (T006) → fix-or-record (T007), strictly sequential
- US2: T008 (export remaining 5) and T009 (smoke-check them) are sequential per-configuration but the 5 configurations themselves can proceed in parallel with each other; T010 (capture) depends on both T007 and T009 since it only runs against configs that passed a smoke-check

### Parallel Opportunities

- Within T008/T009, the 5 remaining configurations' export+smoke-check can proceed in parallel with each other (different `.pte` files, no shared state)
- T017 (Polish) has no dependency on T016 and could run alongside it

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational — **including the hard safety-verification gate (T004)**
3. Complete Phase 3: User Story 1 — one configuration, Buffer-storage export proven to work
4. **STOP and VALIDATE**: the mechanism works (or its failure mode is fully understood) before spending device time on the other five

### Incremental Delivery

1. Setup + Foundational (+ safety verification) → mechanism ready and proven not to change default behavior
2. US1 → proven on one real configuration
3. US2 → all six configurations measured (or explicitly blocked)
4. US3 → the actual answer: does `004`'s finding hold at the e2e level?

---

## Notes

- No commits until the user explicitly asks, per repo convention.
- T004's safety verification is non-negotiable: if the dead-code restoration changes default (no-flag) export behavior for existing callers, that is a regression this feature must not ship, not just a risk to note.
- T009's `lm_head` allocation-failure watch is a real, flagged risk (research.md Decision 4), not a hypothetical — if it occurs, report it plainly per FR-006 rather than treating it as an implementation bug to silently work around.
