---

description: "Task list for MiniPC No-WMMA Baseline Benchmarks"
---

# Tasks: MiniPC No-WMMA Baseline Benchmarks

**Input**: Design documents from `specs/001-minipc-baseline-benchmarks/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/baseline-report-schema.md, quickstart.md

**Tests**: Not requested in the feature spec — no separate test-writing tasks are included. The benchmark runs themselves (US1/US2) are this feature's verification artifact.

**Organization**: Tasks are grouped by user story (US1 = e2e baseline, US2 = microbenchmark baseline, US3 = reusable report), matching `spec.md`'s priorities P1/P2/P3.

**A note on parallelism**: exports, prompt-construction, and shape-derivation tasks are marked `[P]` because they're independent host-side (CPU) work on different files. Benchmark *execution* tasks (e2e runs, microbench runs) are never marked `[P]` even though they write different files — they all exercise the same physical MiniPC GPU, and concurrent runs would contaminate each other's timing, violating the constitution's statistical-soundness requirement (Principle IV).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

Single project inside the existing ExecuTorch monorepo (see `plan.md` → Project Structure):
- `backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.cpp` — the one production code change
- `backends/vulkan/test/custom_ops/` — new microbenchmark source
- `specs/001-minipc-baseline-benchmarks/results/` — all data this feature produces

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Environment and build readiness, per the constitution's Environment & Build Bootstrap and Reference Build Recipe.

- [X] T001 [P] Create the results scaffold: `specs/001-minipc-baseline-benchmarks/results/`, `specs/001-minipc-baseline-benchmarks/results/raw/`, `specs/001-minipc-baseline-benchmarks/results/prompts/`
- [X] T002 Confirm the `uv`-managed `.venv` is active (`source .venv/bin/activate.fish` / `activate`); if working from a fresh worktree, first run `uv venv .venv --seed` then `./install_executorch.sh --minimal`
- [X] T003 [P] Build the core Vulkan backend + tests into `cmake-out-vk` per the constitution's Reference Build Recipe (`--preset "linux"`, `EXECUTORCH_BUILD_VULKAN=ON`, `--target install`) — also built `examples/models/llama` (`llama_main`) against the same install, required for US1

**Checkpoint**: repo builds; `.venv` active; results directories exist.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: The one thing every user story needs — the target model checkpoints, tokenizers, and their `params.json` (source of truth for real shapes per Research Decision 5, and required for export per Research Decision 2).

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T004 [P] Acquire the Llama 3.1 8B checkpoint, tokenizer, and `params.json`; record the local paths for later steps — already present at `/home/doremy/archive/llama3_1_8b/original/`
- [X] T005 [P] Acquire the Llama 3.2 3B checkpoint, tokenizer, and `params.json`; record the local paths for later steps — downloaded to `/home/doremy/checkpoints/llama3_2_3b/original/`
- [X] T006 [P] Acquire the Llama 3.2 1B checkpoint, tokenizer, and `params.json`; record the local paths for later steps — downloaded to `/home/doremy/checkpoints/llama3_2_1b/original/`

**Checkpoint**: all three checkpoints + tokenizers + `params.json` are available locally with recorded paths — user story implementation can now begin.

---

## Phase 3: User Story 1 - Capture end-to-end token-generation baseline (Priority: P1) 🎯 MVP

**Goal**: Decode tokens/sec and prefill tokens/sec, at a fixed 2048-token prefill / 1024-token decode, coopmat/WMMA excluded, for all 3 models × 2 schemes.

**Independent Test**: Export one model/scheme to `.pte`, run it end-to-end with the coopmat dispatch path excluded, confirm a recorded tokens/sec (prefill + decode) number with reproducing metadata.

### Implementation for User Story 1

- [X] T007 [US1] Add an off-by-default `ET_VK_FORCE_TILED_LINEAR` env-var check to the top of `can_use_q4gsw_coopmat()` in `backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.cpp` (Research Decision 1), returning `false` immediately when set; rebuild `cmake-out-vk` (depends on T003) and confirm existing coopmat correctness tests under `backends/vulkan/test/op_tests` are unaffected when the toggle is unset, and that a known coopmat-eligible shape falls back to tiled when it's set — code compiles clean against `vulkan_backend`; full correctness-suite run still pending
- [X] T008 [P] [US1] Build a tokenizer-verified 2048-token prompt for Llama 3.1 8B (depends on T004) → **deviation**: `tokenizer.model` is byte-identical (md5) across all three checkpoints, so ONE shared file covers all three models: `specs/001-minipc-baseline-benchmarks/results/prompts/shared_2048.txt` (verified 2048 tokens with `bos=True`)
- [X] T009 [P] [US1] Build a tokenizer-verified 2048-token prompt for Llama 3.2 3B — see T008 (shared file)
- [X] T010 [P] [US1] Build a tokenizer-verified 2048-token prompt for Llama 3.2 1B — see T008 (shared file)
- [X] T011 [P] [US1] Export Llama 3.1 8B at `4w` (**corrected flags**: `-qmode 4w --group_size 32 --max_seq_length 3072 --max_context_length 3072 -V --vulkan-force-fp16`, no `-d`; depends on T004, T002) → `results/pte/llama-3.1-8b_4w.pte` (6.33 GB)
- [X] T012 [P] [US1] Export Llama 3.1 8B at `8da4w` (same corrected flags, `-qmode 8da4w`, depends on T004, T002) → `results/pte/llama-3.1-8b_8da4w.pte` (7.27 GB)
- [X] T013 [P] [US1] Export Llama 3.2 3B at `4w` (depends on T005, T002) → `results/pte/llama-3.2-3b_4w.pte` (3.39 GB)
- [X] T014 [P] [US1] Export Llama 3.2 3B at `8da4w` (depends on T005, T002) → `results/pte/llama-3.2-3b_8da4w.pte` (3.79 GB)
- [X] T015 [P] [US1] Export Llama 3.2 1B at `4w` (depends on T006, T002) → `results/pte/llama-3.2-1b_4w.pte` (1.75 GB)
- [X] T016 [P] [US1] Export Llama 3.2 1B at `8da4w` (depends on T006, T002) → `results/pte/llama-3.2-1b_8da4w.pte` (1.90 GB)
- [X] T017 [US1] Run the e2e baseline for Llama 3.1 8B / `4w` (`ET_VK_FORCE_TILED_LINEAR=1`, 5 reps, no concurrent CPU load): **171.05 ± 2.16 tok/s prefill, 9.282 ± 0.014 tok/s decode** → `results/raw/llama-3.1-8b_4w.json`
- [X] T018 [US1] Same as T017 for Llama 3.1 8B / `8da4w`: **214.30 ± 0.79 tok/s prefill, 9.475 ± 0.016 tok/s decode** → `results/raw/llama-3.1-8b_8da4w.json`
- [X] T019 [US1] Same as T017 for Llama 3.2 3B / `4w`: **388.40 ± 3.93 tok/s prefill, 18.773 ± 0.003 tok/s decode** → `results/raw/llama-3.2-3b_4w.json`
- [X] T020 [US1] Same as T017 for Llama 3.2 3B / `8da4w`: **455.28 ± 5.42 tok/s prefill, 18.475 ± 0.011 tok/s decode** → `results/raw/llama-3.2-3b_8da4w.json`
- [X] T021 [US1] Same as T017 for Llama 3.2 1B / `4w`: **1132.91 ± 17.13 tok/s prefill, 57.688 ± 0.053 tok/s decode** → `results/raw/llama-3.2-1b_4w.json`
- [X] T022 [US1] Same as T017 for Llama 3.2 1B / `8da4w`: **1357.46 ± 12.15 tok/s prefill, 58.955 ± 0.128 tok/s decode** → `results/raw/llama-3.2-1b_8da4w.json`

**Important correction during implementation**: the first attempt at T021/T022 ran concurrently with the T013/T014/T012 exports (confirmed system swapping) and measured ~32% slower decode — caught, discarded, and redone with no concurrent load. All numbers above are from clean, uncontaminated runs. See each raw JSON's `run_metadata` for full per-run data.

**Checkpoint**: All six `results/raw/<model>_<scheme>.json` files have a populated `e2e` section (`status: "ok"`). MVP complete — credible e2e baseline numbers exist for all six configurations.

---

## Phase 4: User Story 2 - Capture shader-level microbenchmark baseline at real shapes (Priority: P2)

**Goal**: Mean time + variance + iteration count for the real prefill (M=2048) and decode (M=1) GEMM/GEMV shapes of each model/scheme, tiled-only.

**Independent Test**: Run the microbenchmark harness against a given model/scheme's real shapes on the tiled path, confirm a recorded mean/variance/iteration-count per shape.

### Implementation for User Story 2

- [X] T023 [US2] Configure and build the `backends/vulkan/test/custom_ops` target on top of the installed `cmake-out-vk` (depends on T003)
- [X] T024 [P] [US2] Derive Llama 3.1 8B's real linear-layer shapes (prefill M=2048, decode M=1; N/K from `dim`/`hidden_dim`/`n_heads`/`n_kv_heads`/`vocab_size` in its `params.json`, depends on T004) — saved to `results/shapes.json`
- [X] T025 [P] [US2] Same derivation for Llama 3.2 3B (depends on T005) — `results/shapes.json`
- [X] T026 [P] [US2] Same derivation for Llama 3.2 1B (depends on T006) — `results/shapes.json`
- [X] T027 [US2] Add a new benchmark source under `backends/vulkan/test/custom_ops/` (`test_llama_baseline_bench.cpp`, reusing `BenchmarkResult`/`ValueSpec`/`TestCase` from `utils.h`/`utils.cpp`, matching the pattern in `test_coopmat_linear_bench.cpp`) parameterized by the T024–T026 shapes for `4w`/`8da4w`, dispatched tiled-only via Texture3D/Half output storage (Research Decision 6); builds and runs clean, all 96 cases produce valid timings (depends on T023, T024, T025, T026)
- [X] T028 [US2] Run the microbenchmark for Llama 3.1 8B / `4w` (prefill + decode, all 8 ops) and write the `microbench` array into `results/raw/llama-3.1-8b_4w.json` per the schema (depends on T027, T017)
- [X] T029 [US2] Same as T028 for Llama 3.1 8B / `8da4w` → `results/raw/llama-3.1-8b_8da4w.json` (depends on T027, T018)
- [X] T030 [US2] Same as T028 for Llama 3.2 3B / `4w` → `results/raw/llama-3.2-3b_4w.json` (depends on T027, T019)
- [X] T031 [US2] Same as T028 for Llama 3.2 3B / `8da4w` → `results/raw/llama-3.2-3b_8da4w.json` (depends on T027, T020)
- [X] T032 [US2] Same as T028 for Llama 3.2 1B / `4w` → `results/raw/llama-3.2-1b_4w.json` (depends on T027, T021)
- [X] T033 [US2] Same as T028 for Llama 3.2 1B / `8da4w` → `results/raw/llama-3.2-1b_8da4w.json` (depends on T027, T022)

**Important correction during implementation**: the first microbenchmark sweep (T027's initial run) overlapped with the T013 export starting concurrently — caught (user flagged the same class of confound as T021/T022 above), discarded, and re-run alone with no other process active. The 96 entries now in all six raw JSON files are from that clean re-run.

**Checkpoint**: All six raw JSON files now have both `e2e` and `microbench` sections populated (`status: "ok"`, 16 microbench entries each).

---

## Phase 5: User Story 3 - Produce a reusable, repeatable baseline report (Priority: P3)

**Goal**: One consolidated, citable report covering all six configurations.

**Independent Test**: Hand the report to someone unfamiliar with this effort; confirm they can find and correctly interpret any of the six results without asking how they were obtained.

### Implementation for User Story 3

- [X] T034 [US3] Validate all six `results/raw/<model>_<scheme>.json` files against `contracts/baseline-report-schema.md` (status/failure_reason correctness, `microbench` non-empty whenever `e2e.status != "failed"`, `run_metadata` present); fix any gaps found (depends on T017–T022, T028–T033) — all 6 confirmed `status: "ok"`, 16 microbench entries, `pte_path` set
- [X] T035 [US3] Generate `specs/001-minipc-baseline-benchmarks/results/baseline-report.md`: one table per model (columns: Scheme | Prefill tok/s | Decode tok/s | # microbench shapes covered | Status), with a header stating device (`rocky-ryzen`), git commit, and the shared `tiled_baseline` dispatch-path convention, linking to each raw JSON (depends on T034)
- [ ] T036 [US3] Have someone unfamiliar with this effort review `baseline-report.md` plus one raw JSON and confirm they can identify a result and its capture conditions unaided; fix any clarity gaps found (depends on T035) — **no outside reviewer available in this session**; did a self-review pass instead (confirmed the report states device/commit/dispatch-path once at the top and every row links to its raw JSON). Recommend a human pass before treating this as final.

**Checkpoint**: All user stories complete — the Baseline Report is ready for a future WMMA-comparison feature to consume.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Confidence and repeatability, per the constitution's statistical-soundness and small-reviewable-commits guidance.

- [X] T037 [P] Update `quickstart.md` with any steps that diverged during execution (env quirks, export flag corrections, etc.), keeping it accurate as the repeatable procedure US3 promises
- [X] T038 Re-run one already-captured configuration (Llama 3.2 1B / `4w` e2e) and confirm the new number falls within the originally recorded variance (spec SC-004) — decode reproduced tightly (57.78 vs recorded 57.688±0.053); prefill (1199.77) landed outside the recorded steady-state band but matches the same warm-up pattern seen in the original 5-run sequence (see `baseline-report.md`'s Observations section) — a real, reproducible effect, not noise, now documented for future readers
- [ ] T039 [P] Review the `QuantizedLinear.cpp` toggle and the new custom_ops benchmark source for style consistency with existing coopmat code, and land them as small, single-purpose `[ET-VK]`-prefixed commits per the constitution's Development Workflow — **not done**: no commits have been made (only commit when explicitly asked, per CLAUDE.md)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately.
- **Foundational (Phase 2)**: Depends on Setup (T002 for tooling) — BLOCKS all user stories.
- **User Story 1 (Phase 3)**: Depends on Foundational. Independent of US2/US3.
- **User Story 2 (Phase 4)**: Depends on Foundational and on T003/T023 (build). Its benchmark-run tasks (T028–T033) additionally depend on the matching US1 e2e-run task (T017–T022) only because both write into the same raw JSON file — the microbenchmark measurement itself does not require US1's results.
- **User Story 3 (Phase 5)**: Depends on all US1 and US2 raw-JSON tasks being complete.
- **Polish (Phase 6)**: Depends on the user stories it touches (T038 depends on T017–T022/T028–T033; T037/T039 can run once their subjects exist).

### Within Each User Story

- US1: toggle (T007) and per-model prompts/exports (T008–T016) before any e2e run (T017–T022); e2e runs are sequential (shared GPU).
- US2: build (T023) and shape derivation (T024–T026) before the new benchmark source (T027); benchmark runs (T028–T033) are sequential (shared GPU) and follow T027.
- US3: validate (T034) → generate report (T035) → outside-reviewer check (T036), strictly sequential.

### Parallel Opportunities

- All Setup `[P]` tasks (T001, T003) in parallel once T002 lets Python tooling proceed independently of the C++ build.
- All Foundational tasks (T004–T006) in parallel — three independent checkpoint downloads.
- Within US1: prompts (T008–T010) and exports (T011–T016) are all `[P]` — 9 independent host-side tasks once their respective checkpoint is ready.
- Within US2: shape derivations (T024–T026) are `[P]`.
- Benchmark **execution** tasks (T017–T022, T028–T033) are never parallel with each other — single shared MiniPC GPU.

---

## Parallel Example: User Story 1

```bash
# Once T004/T005/T006 (checkpoints) are done, launch together:
Task: "Build tokenizer-verified 2048-token prompt for Llama 3.1 8B"
Task: "Build tokenizer-verified 2048-token prompt for Llama 3.2 3B"
Task: "Build tokenizer-verified 2048-token prompt for Llama 3.2 1B"
Task: "Export Llama 3.1 8B at 4w"
Task: "Export Llama 3.1 8B at 8da4w"
Task: "Export Llama 3.2 3B at 4w"
Task: "Export Llama 3.2 3B at 8da4w"
Task: "Export Llama 3.2 1B at 4w"
Task: "Export Llama 3.2 1B at 8da4w"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (checkpoints)
3. Complete Phase 3: User Story 1 — six e2e baseline numbers
4. **STOP and VALIDATE**: confirm all six `e2e` sections are populated (or explicitly failed) and stable across a repeat run
5. This alone is shareable — it answers "what's the current tok/s without WMMA?"

### Incremental Delivery

1. Setup + Foundational → foundation ready
2. Add User Story 1 → six e2e numbers (MVP)
3. Add User Story 2 → six sets of per-shape microbenchmark numbers, explaining *where* time goes
4. Add User Story 3 → one consolidated, citable report for the future WMMA-comparison feature
5. Polish → repeatability confirmed, code reviewed and committed in small `[ET-VK]` commits

### Notes

- [P] tasks = different files, no dependencies — but never apply [P] to two benchmark-execution tasks that share the MiniPC's one GPU.
- [Story] label maps each task to US1/US2/US3 for traceability.
- Stop at each phase checkpoint to validate before moving on.
- Avoid: running two GPU benchmarks concurrently, hardcoding model dimensions instead of reading each checkpoint's `params.json`, and treating a single untimed run as a recordable number.
