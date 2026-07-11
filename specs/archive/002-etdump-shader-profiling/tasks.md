---

description: "Task list for ETDump E2E Shader Profiling Breakdown"
---

# Tasks: ETDump E2E Shader Profiling Breakdown

**Input**: Design documents from `specs/002-etdump-shader-profiling/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/profiling-report-schema.md, quickstart.md, and a completed `001-minipc-baseline-benchmarks` (six `.pte` files + `results/shapes.json` + `results/prompts/shared_2048.txt`)

**Tests**: Not requested in the feature spec — no separate test-writing tasks. The reconciliation/shape spot-checks (US1, Polish) are this feature's verification artifact.

**Organization**: Tasks are grouped by user story (US1 = prove the pipeline on one config, US2 = extend to all six, US3 = category rollup + report), matching `spec.md`'s priorities P1/P2/P3.

**A note on parallelism**: ETDump *capture* tasks (running `llama_main`) are never marked `[P]`, even across different configs — they exercise the same physical MiniPC GPU, and `001` already found concurrent CPU/GPU-heavy work contaminates timing. *Parsing* tasks (pure Python, no GPU) are marked `[P]` once their input `.etdump` files exist.

## Format: `[ID] [P?] [Story] Description`

## Path Conventions

- `specs/002-etdump-shader-profiling/results/` — all data this feature produces
- `specs/002-etdump-shader-profiling/scripts/` — the parsing/aggregation script
- `cmake-out-vk-profiling/` — separate, event-tracer-enabled build (does not touch `001`'s `cmake-out-vk`)
- Reused, not copied: `specs/001-minipc-baseline-benchmarks/results/{pte,shapes.json,prompts/shared_2048.txt,raw/*.json}`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: A separate, event-tracer-enabled build, per Research Decision 1.

- [X] T001 [P] Create the results scaffold: `specs/002-etdump-shader-profiling/results/`, `results/etdumps/`, `results/raw/`
- [X] T002 Confirm the `uv`-managed `.venv` is active (per constitution Environment & Build Bootstrap)
- [X] T003 Build a separate `cmake-out-vk-profiling/` with `-DEXECUTORCH_BUILD_DEVTOOLS=ON -DEXECUTORCH_ENABLE_EVENT_TRACER=ON` added to the Reference Build Recipe, `--target install`, then configure+build `examples/models/llama` (`llama_main`) against it; if linking fails on `etdump`/`flatccrt` symbols, add `target_link_libraries(llama_main PRIVATE etdump flatccrt)` to `examples/models/llama/CMakeLists.txt` and rebuild (Research Decision 1's flagged fallback) — **fallback was needed**: linking failed exactly as flagged (`undefined reference to ETDumpGen::...`); fixed by adding `if(EXECUTORCH_ENABLE_EVENT_TRACER)` block to `examples/models/llama/CMakeLists.txt` (links `etdump`/`flatccrt` and defines `ET_EVENT_TRACER_ENABLED`, which that CMakeLists never propagated from the root build on its own); verified with a smoke-test run that `--etdump_path` now actually writes a file

**Checkpoint**: event-tracer-enabled `llama_main` builds and runs; results directories exist.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Confirm `001`'s artifacts are usable as-is, and build the one script every story depends on.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T004 [P] Verify all six `.pte` files, `results/shapes.json`, and `results/prompts/shared_2048.txt` from `specs/001-minipc-baseline-benchmarks/` exist and are readable — no re-export or re-derivation needed (Research Decision 2)
- [X] T005 Write the ETDump parsing + aggregation script (`specs/002-etdump-shader-profiling/scripts/parse_etdump.py`): load via `Inspector(etdump_path=...)` with no `etrecord=` (Research Decision 3), iterate events, JSON-parse each event's embedded `name` field for kernel name / operator name / per-arg tensor `sizes` (Research Decision 2) to produce raw Kernel Invocation records, then aggregate by `(kernel_name, shape)` into Aggregated Kernel Entries (`total_time_us`, `invocation_count`, `pct_of_phase` — data-model.md) — validated against a real smoke-test `.etdump`: `event.name` is a JSON blob for every delegated dispatch (confirmed both the `{"kernel_name":...}` and richer `{"operator": {...}, "kernel_name":...}` forms); shape derived from the first/last TENSOR|TENSORREF args (input→M,K; output→N), matmul-detected by kernel-name substring (`linear`/`gemm`/`gemv`/`sdpa`/`bmm`); phase wall-clock parsed from the run's own `PyTorchObserver` stdout log, not `001`'s baseline number

**Checkpoint**: parsing script exists and is ready to run against any `.etdump` file — user story implementation can now begin.

---

## Phase 3: User Story 1 - Break down where one configuration's e2e time actually goes (Priority: P1) 🎯 MVP

**Goal**: A complete aggregated breakdown (name, total time, invocation count, shape, % of phase) for both prefill and a decode window of one configuration.

**Independent Test**: Run one already-exported configuration with ETDump enabled; confirm a breakdown listing every distinct kernel+shape invoked, with time, shape, and percentage of phase time.

### Implementation for User Story 1

- [X] T006 [US1] Capture prefill ETDump for Llama 3.2 1B / `4w`: `ET_VK_FORCE_TILED_LINEAR=1 llama_main --model_path .../llama-3.2-1b_4w.pte --prompt_file .../shared_2048.txt --num_bos 1 --temperature 0 --max_new_tokens 1 --seq_len 3072 --etdump_path results/etdumps/llama-3.2-1b_4w_prefill.etdump` (depends on T003, T004) — `prompt_tokens=2048`, `generated_tokens=0` confirmed
- [X] T007 [US1] Capture a short decode window (8 steps) ETDump for Llama 3.2 1B / `4w`, same flags but `--max_new_tokens 8 --etdump_path results/etdumps/llama-3.2-1b_4w_decode.etdump` (Research Decision 5; depends on T006 — run sequentially, not concurrently, on the shared GPU) — `generated_tokens=7` (same off-by-one pattern as `001`, fine for a "short window")
- [X] T008 [US1] Run the T005 script against both `.etdump` files; write `results/raw/llama-3.2-1b_4w.json` (both phases: `aggregated`, `category_rollup: []` placeholder until US3, `raw_invocations_path`) per `contracts/profiling-report-schema.md`, plus the raw per-invocation companion file(s) (depends on T005, T006, T007)
- [X] T009 [US1] Validate T008's output: confirm `attributed_pct` is a plausible majority for both phases (not e.g. <20%, which would indicate the parser is missing most events), spot-check a few `aggregated[].shape` entries against `results/shapes.json` for `llama-3.2-1b`, and record `phase_wall_clock_us_baseline` from `001`'s `results/raw/llama-3.2-1b_4w.json` alongside for comparison (FR-005/FR-006) — fix the T005 script if reconciliation or shapes look wrong — **two real bugs found and fixed here** (see research.md addenda): (1) a decode-window `.etdump` also contains the seeding prefill call in its own event block — summing everything gave 1575% over-attribution; fixed by classifying blocks via tiled/gemm vs. gemv/coop kernel-name markers, not blindly summing all blocks; (2) for dynamic-shape exports, the embedded tensor `sizes` reflect the static allocation bound, not the active M — M must come from which dispatch kernel fired (gemv⇒1, tiled/gemm⇒prefill's fixed 2048), not from the tensor JSON. After both fixes: prefill 99.29% attributed, decode 91.19% attributed, and every aggregated shape + invocation count matches `results/shapes.json`'s per-op catalog and the model's real layer count exactly (e.g. `(K=2048,N=8192)` count=32 at prefill = 16 layers × {gate,up})

**Checkpoint**: the pipeline is proven end-to-end on one configuration — this alone is a demoable MVP.

---

## Phase 4: User Story 2 - Extend the breakdown to all six baseline configurations (Priority: P2)

**Goal**: The same breakdown for all six (model × scheme) configurations.

**Independent Test**: Run the same profiling procedure from User Story 1 against each remaining configuration; confirm a breakdown exists for all six.

### Implementation for User Story 2

- [X] T010 [US2] Capture prefill ETDump for Llama 3.2 1B / `8da4w` → `results/etdumps/llama-3.2-1b_8da4w_prefill.etdump` (depends on T003, T004)
- [X] T011 [US2] Capture decode-window ETDump for Llama 3.2 1B / `8da4w` → `results/etdumps/llama-3.2-1b_8da4w_decode.etdump` (depends on T010)
- [X] T012 [US2] Capture prefill ETDump for Llama 3.2 3B / `4w` → `results/etdumps/llama-3.2-3b_4w_prefill.etdump` (depends on T003, T004)
- [X] T013 [US2] Capture decode-window ETDump for Llama 3.2 3B / `4w` → `results/etdumps/llama-3.2-3b_4w_decode.etdump` (depends on T012)
- [X] T014 [US2] Capture prefill ETDump for Llama 3.2 3B / `8da4w` → `results/etdumps/llama-3.2-3b_8da4w_prefill.etdump` (depends on T003, T004)
- [X] T015 [US2] Capture decode-window ETDump for Llama 3.2 3B / `8da4w` → `results/etdumps/llama-3.2-3b_8da4w_decode.etdump` (depends on T014)
- [X] T016 [US2] Capture prefill ETDump for Llama 3.1 8B / `4w` → `results/etdumps/llama-3.1-8b_4w_prefill.etdump` (depends on T003, T004)
- [X] T017 [US2] Capture decode-window ETDump for Llama 3.1 8B / `4w` → `results/etdumps/llama-3.1-8b_4w_decode.etdump` (depends on T016)
- [X] T018 [US2] Capture prefill ETDump for Llama 3.1 8B / `8da4w` → `results/etdumps/llama-3.1-8b_8da4w_prefill.etdump` (depends on T003, T004)
- [X] T019 [US2] Capture decode-window ETDump for Llama 3.1 8B / `8da4w` → `results/etdumps/llama-3.1-8b_8da4w_decode.etdump` (depends on T018)
- [X] T020 [P] [US2] Run T005's script for Llama 3.2 1B / `8da4w` → `results/raw/llama-3.2-1b_8da4w.json` + reconciliation/shape spot-check (T009-style) (depends on T005, T010, T011) — prefill 99.02%, decode 88.34% attributed
- [X] T021 [P] [US2] Same as T020 for Llama 3.2 3B / `4w` → `results/raw/llama-3.2-3b_4w.json` (depends on T005, T012, T013) — prefill 99.54%, decode 94.01% attributed
- [X] T022 [P] [US2] Same as T020 for Llama 3.2 3B / `8da4w` → `results/raw/llama-3.2-3b_8da4w.json` (depends on T005, T014, T015) — prefill 99.53%, decode 93.76% attributed
- [X] T023 [P] [US2] Same as T020 for Llama 3.1 8B / `4w` → `results/raw/llama-3.1-8b_4w.json` (depends on T005, T016, T017) — prefill 99.68%, decode 97.05% attributed
- [X] T024 [P] [US2] Same as T020 for Llama 3.1 8B / `8da4w` → `results/raw/llama-3.1-8b_8da4w.json` (depends on T005, T018, T019) — prefill 99.58%, decode 96.40% attributed

**Checkpoint**: all six `results/raw/<model>_<scheme>.json` files have a populated aggregated breakdown for both phases (or an explicit recorded gap, FR-010).

---

## Phase 5: User Story 3 - Summarize the breakdown into meaningful categories (Priority: P3)

**Goal**: A category-level rollup (attention projection, feed-forward, output/vocab projection, non-shader overhead) for every configuration, plus a consolidated report.

**Independent Test**: Take an existing per-shader breakdown and confirm it can be grouped into named categories whose percentages sum to the phase total, without re-running any profiling.

### Implementation for User Story 3

- [X] T025 [US3] Implement the category-rollup mapping (Research Decision 4) in `specs/002-etdump-shader-profiling/scripts/category_rollup.py`: match each `aggregated[].shape` against the per-model `ops` table in `001`'s `results/shapes.json` (`wq`/`wk`/`wv`/`wo` → attention projection, `w1_gate`/`w3_up`/`w2_down` → feed-forward, `lm_head` → output/vocab projection; shape-bearing but unmatched kernels named `sdpa*` → a new "attention (sdpa)" category, discovered from real data; `shape=None` → non-shader overhead) and apply it to `llama-3.2-1b_4w.json`'s existing data (no re-capture) as first validation — category percentages plus `unattributed` (`1 - attributed_pct`) sum to ~1.0 with no warning (depends on T009)
- [X] T026 [US3] Apply the T025 category-rollup logic to the remaining five raw JSON files (depends on T025, T020, T021, T022, T023, T024) — consistent pattern across all 6: feed-forward dominates (~40-54% prefill, ~33-51% decode), then attention/sdpa compute (~22-34%), then attention projection (~9-15%), non-shader overhead and lm_head are small
- [X] T027 [US3] Generate `specs/002-etdump-shader-profiling/results/profiling-report.md` (via `scripts/generate_report.py`): per (config, phase) category-rollup table, top-kernels-by-time table, and a reconciliation line (`attributed_pct`, profiled vs. `001` baseline phase total), linking to `001`'s `baseline-report.md` and to each raw JSON, per `contracts/profiling-report-schema.md`'s "Rendered summary" section (depends on T026) — **bug found and fixed**: the decode reconciliation line initially compared our 7-step profiled window against `001`'s full 1024-step baseline (`-99.5%`, meaningless); fixed by scaling the baseline to the same step count (`decode_window_steps / decode_tokens_per_sec`), yielding a real profiling-overhead comparison
- [X] T028 [US3] Self-review `profiling-report.md` (no outside reviewer available, matching `001`'s honest precedent): confirm a reader can determine, for any of the six configurations, what fraction of prefill/decode time is matmul vs. non-shader overhead from the report alone (SC-005); fix any clarity gaps found (depends on T027) — added a "Cross-model observations" section explaining the counterintuitive "profiled runs measure faster than baseline" finding (short window doesn't reach `001`'s thermally-throttled steady state) so a reader isn't misled into thinking this report's phase timings are a corrected throughput number

**Checkpoint**: all user stories complete — the profiling report is ready to inform where the WMMA workstream focuses next.

---

## Phase 6: Polish & Cross-Cutting Concerns

- [X] T029 [P] Update `quickstart.md` with any steps that diverged during execution (build flag corrections, decode-window step count actually used, etc.) — added the mandatory `EXECUTORCH_ENABLE_EVENT_TRACER` sub-project flag + `CMakeLists.txt` block (not just a "try if it fails" fallback), a smoke-test step, and the two shape/block-classification gotchas
- [X] T030 Reproducibility spot-check: re-capture one config's decode-window `.etdump` (e.g. Llama 3.2 1B / `4w`) once more and confirm the aggregated kernel set and rough time shares match the recorded breakdown (lighter-weight than `001`'s SC-004 since this is attribution, not throughput statistics) — re-capture matched to within 0.1 percentage point on every category (feed-forward 36.4%/36.4%, sdpa 30.3%/30.3%, output 11.4%/11.4%, attention proj 9.2%/9.2%) and `decode_token_per_sec` was bit-for-bit identical (60.8696)
- [X] T031 [P] Note any driver-workaround findings from enabling the event tracer on this hardware, per constitution Principle V — none found; the only issue was the CMake linking gap (T003/research.md Decision 1), which is a build-configuration fix, not a driver workaround. No commits made without explicit user go-ahead (matching `001`'s convention)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately.
- **Foundational (Phase 2)**: Depends on Setup (T003's build). BLOCKS all user stories.
- **User Story 1 (Phase 3)**: Depends on Foundational. Independent of US2/US3.
- **User Story 2 (Phase 4)**: Depends on Foundational; does not depend on US1 completing, though running US1 first is how the pipeline gets proven before scaling to 5 more configs.
- **User Story 3 (Phase 5)**: T025 depends only on US1 (T009); T026 depends on all of US2's parse tasks (T020-T024); T027/T028 depend on T026.
- **Polish (Phase 6)**: Depends on the user stories it touches.

### Within Each User Story

- US1/US2: capture prefill before capture decode for the same config (sequential, shared GPU); parse only after both captures for that config exist.
- US3: T025 (build the mapping, validate on one config) before T026 (apply to the rest) before T027 (report) before T028 (review).

### Parallel Opportunities

- Setup: T001 in parallel with T002/T003.
- Foundational: T004 in parallel with T003/T005 (independent of the build).
- US2's five parse tasks (T020-T024) run in parallel once their respective capture tasks are done.
- ETDump **capture** tasks (T006-T007, T010-T019) are never parallel with each other — single shared MiniPC GPU, per `001`'s hard-won lesson.

---

## Parallel Example: User Story 2 (after all ten captures are done)

```bash
Task: "Parse Llama 3.2 1B / 8da4w ETDumps into results/raw/llama-3.2-1b_8da4w.json"
Task: "Parse Llama 3.2 3B / 4w ETDumps into results/raw/llama-3.2-3b_4w.json"
Task: "Parse Llama 3.2 3B / 8da4w ETDumps into results/raw/llama-3.2-3b_8da4w.json"
Task: "Parse Llama 3.1 8B / 4w ETDumps into results/raw/llama-3.1-8b_4w.json"
Task: "Parse Llama 3.1 8B / 8da4w ETDumps into results/raw/llama-3.1-8b_8da4w.json"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1 — one configuration's complete breakdown
4. **STOP and VALIDATE**: reconciliation percentage is plausible, shapes match `001`'s catalog
5. This alone answers "where does this one configuration's time actually go?"

### Incremental Delivery

1. Setup + Foundational → parsing script ready
2. Add User Story 1 → pipeline proven on one config (MVP)
3. Add User Story 2 → all six configs profiled
4. Add User Story 3 → readable category rollups + consolidated report
5. Polish → repeatability confirmed, `quickstart.md` corrected

### Notes

- [P] tasks = different files, no dependencies — but never apply [P] to two ETDump capture tasks, which share the MiniPC's one GPU.
- [Story] label maps each task to US1/US2/US3 for traceability.
- Avoid: capturing the full 1024-step decode (Research Decision 5 explicitly rejects this), treating a low `attributed_pct` as acceptable without investigating, and silently coercing a non-matmul kernel's shape to a numeric placeholder instead of `null`.
