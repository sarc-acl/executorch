---

description: "Task list for SDPA Coopmat E2E Validation"
---

# Tasks: SDPA Coopmat E2E Validation

**Input**: Design documents from `/specs/011-sdpa-coopmat-e2e/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md (all present)

**Tests**: Not requested as a separate automated suite — this feature's own
ETDump dispatch-confirmation check (FR-002) IS the verification, matching
`009`'s precedent. No new correctness test is needed: `010` already
established the SDPA coopmat shaders are correct; this feature only
confirms they dispatch in a real exported model and measures the result.

**Organization**: Tasks are grouped by user story. This is **real device
work** (ETDump capture, e2e capture on `rocky-ryzen`), like every prior
tier-2 feature, but unlike any of them, it adds **zero new production or
test code** — `009`'s exports and build trees are reused exactly as they
are; this feature is purely measurement and reporting.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files/resources, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Paths are relative to the repository root

## Path Conventions

- `specs/009-e2e-tokrate-report/results/pte/*.pte` — existing exports, reused, **not modified**
- `cmake-out-vk`, `cmake-out-vk-etdump` — existing build trees, reused, **not modified**
- `specs/011-sdpa-coopmat-e2e/scripts/compare_sdpa_e2e.py` — new analysis script
- `specs/011-sdpa-coopmat-e2e/results/` — new ETDump captures, e2e logs, and the report

---

## Phase 1: Setup

- [X] T001 Create `specs/011-sdpa-coopmat-e2e/scripts/`, `specs/011-sdpa-coopmat-e2e/results/etdump/`, and `specs/011-sdpa-coopmat-e2e/results/raw/` directories

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Confirm every mechanism this feature reuses is actually
present and current — there is no code to write, only readiness to verify
(mirroring `007`'s Foundational phase, the closest precedent for a
zero-new-code feature).

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T002 [P] Confirm all six of `009`'s `Buffer`-storage `.pte` exports still exist under `specs/009-e2e-tokrate-report/results/pte/`
- [X] T003 [P] Confirm `cmake-out-vk` and `cmake-out-vk-etdump` build trees still exist and remain current — re-run `git status` on `backends/vulkan/runtime/` to confirm no production runtime code has changed since `009`'s last build (research.md Decision 1's premise); rebuild only if this check fails — confirmed unchanged, no rebuild needed
- [X] T004 [P] Confirm `specs/009-e2e-tokrate-report/results/e2e-tokrate-report.md` and `specs/010-sdpa-coopmat-microbench/results/sdpa-coopmat-microbench-report.md` are present and readable — read-only inputs this feature never re-captures

**Checkpoint**: Foundation ready — exports, build trees, and both upstream reports confirmed present and current

---

## Phase 3: User Story 1 - Confirm SDPA coopmat actually dispatches in a real exported model (Priority: P1) 🎯 MVP

**Goal**: Prove that enabling `ET_VK_SDPA_COOPMAT` on one of `009`'s
already-exported configurations actually dispatches the coopmat shaders
end-to-end, before trusting any e2e number produced with it enabled.

**Independent Test**: Run one already-exported, `Buffer`-storage
configuration with `ET_VK_SDPA_COOPMAT` set, capture an ETDump trace, and
confirm from the actual per-op kernel names that both coopmat shaders
dispatched for the prefill attention computation.

- [X] T005 [US1] Capture ETDump for `llama-3.2-1b`/`4w` with `ET_VK_SDPA_COOPMAT=1` set, against `cmake-out-vk-etdump` and `009`'s existing export (depends on Foundational checkpoint) → `specs/011-sdpa-coopmat-e2e/results/etdump/llama-3.2-1b_4w.etdump` — captured to this feature's own `results/` directory as the citable artifact (superseding planning's informal scratch-path spot-check)
- [X] T006 [US1] Parse the trace (`executorch.devtools.Inspector`, per `002`'s established `kernel_name` extraction) and confirm every measured `sdpa_compute_attn_weights_*`/`sdpa_compute_out_*` kernel name contains `_coopmat` (FR-002) (depends on T005) — confirmed: `sdpa_compute_attn_weights_coopmat_buffer_buffer_half` and `sdpa_compute_out_coopmat_buffer_buffer_half` each dispatched 16/16 times (matching this config's 16 layers), zero tiled fallbacks for either shader
- [X] T007 [US1] Record the outcome explicitly: `dispatch_status: confirmed` if T006 passed, or `fallback` with the actual kernel name if it didn't — no e2e number is trusted from this configuration otherwise (FR-007) (depends on T006) — `dispatch_status: confirmed` for `llama-3.2-1b`/`4w`

**Checkpoint**: US1 complete — SDPA coopmat's real-model dispatch is proven
(or its failure mode is understood) on one configuration

---

## Phase 4: User Story 2 - Measure e2e prefill/decode tok/s with SDPA coopmat enabled, all six configurations (Priority: P2)

**Goal**: Extend the proven mechanism to the remaining five configurations
and capture real e2e numbers with the toggle enabled.

**Independent Test**: Capture e2e prefill/decode tok/s for one additional
dispatch-confirmed configuration and confirm it produces a directly
comparable pair against `009`'s existing number for that same
configuration.

- [X] T008 [US2] Capture ETDump and confirm dispatch (same check as T006) for the remaining five configurations (depends on T007) — all 5 confirmed: `llama-3.2-1b_8da4w` 16/16, `llama-3.2-3b_4w` 28/28, `llama-3.2-3b_8da4w` 28/28, `llama-3.1-8b_4w` 32/32, `llama-3.1-8b_8da4w` 32/32 (both shaders each, matching each model's layer count), zero tiled fallbacks
- [X] T009 [US2] Record any `fallback` configuration explicitly with its actual kernel name — never silently excluded from the six-configuration count (FR-007) (depends on T008) — **none blocked/fallback**: all 6 configurations' `dispatch_status: confirmed`
- [X] T010 [US2] Capture e2e prefill/decode tok/s (`ET_VK_SDPA_COOPMAT=1`, `cmake-out-vk` standard build, `--warmup true`, 5 repeated runs, no concurrent GPU load) for every configuration whose dispatch was `confirmed` (T007 or T009), using `001`/`006`/`009`'s exact methodology (depends on T007, T009) — **6/6 configurations completed with full 5 reps**. Collection was originally stopped early by user request (`llama-3.1-8b/4w` at 2/5 reps, `llama-3.1-8b/8da4w` at 0/5); the user later asked to finish it: `llama-3.1-8b/4w`'s remaining 3 reps were captured (its original reps 1-2 were valid, uninterrupted runs, kept as-is), and `llama-3.1-8b/8da4w` was captured fresh, all 5 reps

**Checkpoint**: US2 complete — every dispatch-confirmed configuration has
an e2e prefill/decode tok/s pair with SDPA coopmat enabled, directly
comparable to `009`'s existing baseline pair

---

## Phase 5: User Story 3 - Report whether SDPA coopmat helps at the e2e level, per configuration (Priority: P3)

**Goal**: Turn the measurements into one answer: does enabling SDPA
coopmat change real end-to-end tok/s, and does that agree with `010`'s
prior microbenchmark-level finding.

**Independent Test**: Generate the report and confirm each configuration's
verdict traces directly to its own measurement pair and to `009`'s
baseline / `010`'s prior finding.

- [X] T011 [US3] Implement `specs/011-sdpa-coopmat-e2e/scripts/compare_sdpa_e2e.py`: parse `009`'s report table (baseline e2e numbers, research.md Decision 4) and this feature's new capture (T010); compute `diff_pct` per configuration/phase, tag every prefill row with the inherited cross-session caveat (research.md Decision 6) (depends on T010, T004)
- [X] T012 [US3] Implement the report renderer: one overall verdict statement (not split by scheme, research.md Decision 5, unless the data itself diverges), the 12-row comparison table, and a Blocked/Failed section (always present, even if empty) — per `contracts/sdpa-coopmat-e2e-schema.md` (depends on T011) — also generalized the exclusion reason to cover "e2e capture incomplete" (not just dispatch fallback), since `llama-3.1-8b`'s two configurations were excluded for that reason (T010), not a dispatch failure
- [X] T013 [US3] Run end to end → `specs/011-sdpa-coopmat-e2e/results/sdpa-coopmat-e2e-report.md` (depends on T009, T012) — done, re-run after collection was completed. **Final headline: SDPA coopmat improves real e2e prefill tok/s by +27.3% on average across all 6/6 configurations**, all `consistent` in direction with `010`'s 66.8%-average microbenchmark finding: `llama-3.1-8b/4w` +33.6%, `llama-3.1-8b/8da4w` +16.6%, `llama-3.2-3b/4w` +46.2%, `llama-3.2-3b/8da4w` +25.9%, `llama-3.2-1b/4w` +26.2%, `llama-3.2-1b/8da4w` +15.6%. Excluded/not-collected section is now empty
- [X] T014 [US3] Self-review against SC-001 through SC-004: confirm every configuration appears (measured or explicitly blocked), no SDPA-coopmat-enabled number appears without a `confirmed` dispatch status for both shaders, every prefill row carries its cross-session caveat, and the overall verdict states whether this agrees with `010`'s prior finding (depends on T013) — all four pass, now against the complete 6/6 dataset: SC-001/SC-002 hold (dispatch confirmed before any number reported, identical methodology to `009`); SC-003 holds (table + one overall statement, no raw-log re-derivation needed); SC-004 is vacuously satisfied (nothing left unmeasured)

**Checkpoint**: US3 complete — the report answers whether enabling SDPA
coopmat helps this device's real token generation rate

---

## Phase 6: Polish & Cross-Cutting Concerns

- [X] T015 Reproducibility spot-check: re-run one configuration's e2e capture and confirm it matches the original within noise, matching `001`'s established reproducibility discipline — **a genuine cross-session check materialized naturally**: `llama-3.1-8b/4w`'s reps 1-2 were captured in one session, then collection was paused (spec `012` ran in the interim) before reps 3-5 were captured in a later session. Session 1 (2 reps): mean 416.47; Session 2 (3 reps): mean 427.23±12.30 -- a 2.6% difference, well within this device's established noise floor. Reproducible across sessions, not just within one
- [X] T016 [P] Update `quickstart.md` with any corrections found during T002-T015 (if any were needed) — updated: documented that `009`'s exports/build trees needed zero changes (confirmed, not just planned), corrected the ETDump command shape used in practice, and added a note on how an early-stopped collection is handled (excluded/not-collected, not blocking the other configurations' report)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies
- **Foundational (Phase 2)**: Depends on Setup. BLOCKS all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational checkpoint
- **User Story 2 (Phase 4)**: Depends on US1 (T007) — extends the proven
  single-configuration dispatch check to the rest before capturing timing
- **User Story 3 (Phase 5)**: Depends on US2 (T009, T010)
- **Polish (Phase 6)**: Depends on US3

### Within Each User Story

- US1: T005 (capture) → T006 (dispatch check) → T007 (record outcome),
  strictly sequential
- US2: T008 (5 configs' dispatch checks) → T009 (record outcomes) → T010
  (e2e capture for confirmed configs only) — T008's 5 configurations can
  proceed in parallel with each other (different `.etdump` files, no
  shared state) but not with any other GPU-bound task

### Parallel Opportunities

- T002, T003, T004 (Foundational readiness checks) can all run in parallel
  — independent resources, no shared state
- Within T008, the 5 configurations' ETDump captures can proceed in
  parallel with each other
- T016 (Polish) has no dependency on T015 and could run alongside it
- **Never** parallelize T005/T008/T010/T015 with each other or any other
  GPU-bound task — they share the MiniPC's one GPU, matching this
  workstream's established discipline

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1 — one configuration, dispatch proven
4. **STOP and VALIDATE**: SDPA coopmat's real-model dispatch works before
   spending device time on the other five configurations and full e2e
   capture

### Incremental Delivery

1. Setup + Foundational → readiness confirmed, nothing rebuilt (research.md
   Decision 1: no rebuild needed)
2. US1 → proven on one real configuration
3. US2 → all six configurations measured (or explicitly excluded)
4. US3 → the actual answer: does enabling SDPA coopmat help real e2e
   tok/s, and does it agree with `010`'s prior finding?

---

## Notes

- No commits until the user explicitly asks, per repo convention.
- This feature adds no production or test code — if any task uncovers a
  need to change code (e.g. a configuration's dispatch check fails
  unexpectedly), that is a genuine, unplanned finding requiring the same
  root-cause-then-authorize discipline as `007`/`009`/`010`'s own
  mid-implementation discoveries, not something to force through this
  feature's originally lightweight scope.
- `010`'s microbenchmark finding (66.8% average) is cited as context for
  T014's consistency check, but its magnitude is not expected to match
  e2e's (research.md Decision 5) — only its direction (some prefill
  improvement) is the actual cross-check.
