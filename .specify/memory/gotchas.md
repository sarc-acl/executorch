# Workstream Gotchas

## About This Document

Consolidated, hard-won operational lessons for this workstream (the
Vulkan cooperative-matrix/WMMA GEMM + SDPA effort governed by
`.specify/memory/constitution.md`). This doc exists because several of
these cost multiple hours to root-cause the first time, and were
scattered across individual `specs/NNN/research.md` files with nothing
pointing a new agent session toward them.

**Scope**: only *mechanism-level* findings belong here — a code path's
actual behavior, a build trap, a documented-but-nonexistent env var, a
naming collision between two similarly-named files. Never volatile,
time-sensitive facts (current driver hash, which clocks are pinned right
now, which teammate is using the device today) — those stay in
`.shared-context/ACTIVE-STATUS.md` / `README.md` §Conventions, per
constitution Principle X.

**This is a living document.** When a future session root-causes a new
multi-hour or repeat-mistake operational issue, append a new entry as
`G<N+1>` (next unused number — ids are never reused, even for an entry
later marked resolved), using the same symptom / root cause / fix-or-
workaround / citation / status shape as the entries below. Don't let a
new hard-won lesson rot back into a single spec's `research.md` the way
these ten did.

**Known risk**: this file, and the pointer block in this folder's root
`CLAUDE.md` that names it, can be silently lost if a future
`install_executorch.sh` re-sync (or similar tooling update) regenerates
`CLAUDE.md` from the stock upstream template. If `CLAUDE.md` ever again
reads as generic upstream content with no mention of this workstream's
constitution/target/`.shared-context/`, that regeneration is what
happened — restore the pointer block (see `specs/017-workstream-agent-housekeeping/`)
rather than treating its absence as intentional.

An entry's `status` is `open` (the underlying issue/mechanism is still
there — this is a workaround, not a fix) or `resolved as of <ref>` (the
underlying code/process issue was actually fixed; the entry stays for
history, not deleted).

---

### G1 — Android `install` can fail on an unrelated target, silently staling `libvulkan_backend.a`

- **Symptom**: `cmake --build <dir> --target install` fails with
  `ld.lld: error: .../libflatccrt.a(...) is incompatible with aarch64linux`
  while building `executor_runner`. The real target you care about
  (`vulkan_backend`) built fine — but because `install` failed, its `.a`
  never got copied to `<dir>/lib/`, so any downstream sub-build (e.g.
  `test_coopmat_linear_bench`, `llama_main`) silently links against a
  **stale** `lib/libvulkan_backend.a`, even though your source change
  compiled successfully.
- **Root cause**: a pre-existing, host-arch-built `third-party/flatcc/lib/libflatccrt.a`
  in the build tree is incompatible with the `aarch64linux` cross-build.
  This is unrelated to any `backends/vulkan/` source change — it recurs
  on a clean rebuild too.
- **Fix/workaround**: manually copy the freshly built library and force a
  relink: `cp <build-dir>/backends/vulkan/libvulkan_backend.a
  <build-dir>/lib/libvulkan_backend.a && rm -f <build-dir>/.../<target-binary>`,
  then rebuild the sub-target directly (skip `--target install`).
- **Citation**: this session's own build narrative (no prior spec — first
  documented here).
- **Status**: open.

### G2 — `ET_VK_FORCE_BUFFER` does not exist in this repo

- **Symptom**: following `.shared-context/instruction-for-ai/export-pte.md`'s
  documented `ET_VK_FORCE_BUFFER` env var to force buffer storage on
  export has no effect — the exported `.pte` still uses texture storage.
- **Root cause**: that env var is not implemented anywhere in this repo's
  source. The doc describes a mechanism this codebase never had (or had
  and removed).
- **Fix/workaround**: the real buffer-storage-override mechanism is
  `backend.vulkan.storage_override: buffer` in the export `config.yaml`.
- **Citation**: `specs/015-m5-e2e-wmma-validation/research.md` Decision 6.
- **Status**: open (the `.shared-context/` doc itself is out of this
  workstream's ownership to fix — see constitution Principle X's caveat).

### G3 — A non-editable `.venv` silently no-ops AOT/export Python code changes

- **Symptom**: you edit `export_llm` or another AOT Python source file,
  re-run the export, and the change has no effect — the old behavior
  persists with no error.
- **Root cause**: the active virtualenv has `executorch` installed as a
  regular (non-editable) package, so Python imports the installed copy,
  not your working-tree source.
- **Fix/workaround**: `pip install -e . --no-build-isolation` from the
  repo root, inside the correct venv, before trusting any AOT/export code
  change.
- **Citation**: `specs/015-m5-e2e-wmma-validation/research.md` Decision 6.
- **Status**: open.

### G4 — Exported `.pte` files must land directly in `/local/yanwen.xu/workspace/.pte_out`

- **Symptom** (historical): an export was redirected to ad hoc scratch
  locations (`/tmp`, then a job-specific NFS tmp dir) to work around disk
  space, and the result was never moved into this workspace's one
  canonical `.pte` location — easy to lose track of or leave orphaned.
- **Root cause**: `export_llm`'s `export.output_dir` config key is not
  honored (the file lands in the process's current working directory);
  without a standing rule, each session re-derives its own ad hoc
  location.
- **Fix/workaround**: `cd` into `/local/yanwen.xu/workspace/.pte_out`
  before invoking the export command — never export elsewhere and
  copy/move the result in afterward, and never use `/tmp` or a scratch
  dir even temporarily.
- **Citation**: Constitution v2.3.0 (Default Scope for Every Benchmark) —
  cross-referenced here, not duplicated.
- **Status**: resolved as of Constitution v2.3.0 — the rule now lives
  as a standing, principle-level requirement, not something re-derived
  per session.

### G5 — `/tmp` is small and this sandbox denies `rm -rf` even on your own scratch files

- **Symptom**: a scratch directory under `/tmp` fills the (20GB) `/tmp`
  filesystem, and `rm -rf` on it — even files you created yourself this
  session — is denied by the sandbox's permission system.
- **Root cause**: `/tmp` here is small and shared across parallel jobs;
  the permission system blocks broad recursive deletes regardless of
  ownership.
- **Fix/workaround**: use `mv` to relocate scratch out of `/tmp` (e.g.
  into `.artifacts/old-tmp-scratch/`) instead of deleting; better, write
  scratch directly to `.artifacts/` or a job-specific scratch dir from
  the start instead of `/tmp`.
- **Citation**: this session's own narrative (no prior spec).
- **Status**: open.

### G6 — ETDump's per-event `kernel_name` is not reliable dispatch evidence in the full LLaMA graph

- **Symptom**: an ETDump-based dispatch-confirmation pass on a full
  LLaMA-graph e2e run reports a tiled kernel name for an op, suggesting
  coopmat/WMMA never dispatched — but a direct wall-clock A/B against
  `ET_VK_FORCE_TILED_LINEAR`, and an independent isolated shader
  microbenchmark with its own kernel-name capture, both confirm coopmat
  genuinely did dispatch and ran.
- **Root cause**: still not located. Source-level reading of
  `DispatchNode::encode()` shows it reads `shader_.kernel_name` once, for
  both the actual GPU pipeline bind and the ETDump log call — they
  "should" always agree, yet empirically diverged for the same e2e path
  once (during this session's US2 dispatch-confirm step). A follow-up
  session ran the `ET_VK_DEBUG_ENCODE_DISPATCH` diagnostic (comparing the
  bind-time stderr print directly against `analyze_etdump_shaders.py`'s
  reading of the same run's `.etdp`) on 1B linear and SDPA — both agreed
  exactly (112/112, 16/16) with no misattribution. The bug is real (it
  happened once, reproducibly at the time) but is **not persistent or
  deterministic** across sessions/binaries on this build; this rules out
  "ETDump is fundamentally broken for this graph shape" without
  identifying what actually differed between the two sessions.
- **Fix/workaround**: never trust ETDump's `kernel_name` alone for a
  dispatch claim in the full graph context. Cross-check with at least one
  of: a direct wall-clock A/B against a forced-fallback path, an isolated
  shader microbenchmark with independent kernel-name capture (e.g.
  `test_coopmat_linear_bench.cpp`), or the `ET_VK_DEBUG_ENCODE_DISPATCH`
  bind-time print compared directly against the `.etdp` for the same run.
- **Citation**: `specs/015-m5-e2e-wmma-validation/research.md` Decision 7
  (and its reversal), Decision 8 (the diagnostic finally run).
- **Status**: open.

### G7 — Two similarly-named SDPA benchmark harnesses exist; only one is correct

- **Symptom**: running `test_coopmat_attention_bench` to benchmark SDPA
  coopmat shaders produces results that don't correspond to
  `SDPA.cpp`'s actual coopmat path, or crashes on unrelated shape
  assertions.
- **Root cause**: `test_coopmat_attention_bench.cpp` exercises a generic,
  unrelated `matmul_coopmat`/`coopmat_mm_ref` path — it is not the SDPA
  harness. `test_sdpa_coopmat_bench.cpp` is the correct one: it directly
  tests `sdpa_compute_attn_weights_coopmat`/`sdpa_compute_out_coopmat`.
- **Fix/workaround**: always use `test_sdpa_coopmat_bench.cpp` for SDPA
  coopmat benchmarking; treat `test_coopmat_attention_bench.cpp` as an
  unrelated, differently-scoped harness despite the similar name.
- **Citation**: `specs/016-m5-linear-sdpa-microbench/spec.md` Clarifications.
- **Status**: resolved (2026-07-06) — `test_coopmat_attention_bench.cpp`
  deleted (confirmed absent from `CMakeLists.txt`'s `add_operator_prototype`
  list via a full direct read, not a prefiltered grep, per G8's own lesson).
  `test_sdpa_coopmat_bench.cpp` is now the sole SDPA benchmark harness; the
  naming collision no longer exists.

### G8 — Don't conclude a CMake target "isn't wired in" from a prefiltered grep

- **Symptom**: concluded `test_sdpa_coopmat_bench` wasn't registered in
  `CMakeLists.txt` and planned wiring work for it — but it was already
  present (`add_operator_prototype(test_sdpa_coopmat_bench)`).
- **Root cause**: the grep used to check piped through a prefilter
  pattern (e.g. `grep -i "^#include\|BUILD\|CMakeLists"`) that excluded
  the actual line being searched for.
- **Fix/workaround**: when checking whether something is "wired into the
  build," grep the raw file directly for the exact symbol first, before
  trusting a prefiltered/piped grep's absence of a match.
- **Citation**: `specs/016-m5-linear-sdpa-microbench/tasks.md` T005.
- **Status**: open (a process discipline, not a code fix).

### G9 — The current production linear-coopmat retune predates hardware validation, and historical baseline numbers used a different dispatch mechanism

- **Symptom**: a "directional" comparison against `jira-tile-sweep.md`'s
  historical baseline numbers (110.6/213.9/565.3 tok/s) looks like an
  apples-to-apples regression or improvement check against the current
  production path, but isn't.
- **Root cause**: the current production linear-coopmat shader (128x64
  retune, fp16 accumulate, flattened loop, commit `133044739`) was
  committed with its own message stating it was not yet
  hardware-validated. The `jira-tile-sweep.md` baseline numbers were
  measured via a different dispatch mechanism entirely — the
  `.tmp-origcm` worktree's `ET_VK_Q4GSW_COOPMAT_VARIANT` toggle — not
  this repo's actual production `can_use_q4gsw_coopmat` code path.
- **Fix/workaround**: don't treat that historical baseline as a direct
  regression check for the current production path; re-measure on the
  actual production dispatch mechanism before drawing a conclusion.
- **Citation**: `specs/015-m5-e2e-wmma-validation/research.md` Decision 7's
  "post-completion lead".
- **Status**: open (the mismatch is a standing caveat about historical
  data, not something to "fix").

### G10 — M5 EVT1 is a shared device; confirm it's free before assuming so

- **Symptom**: started planning or running adb/build/flash work against
  M5 EVT1 without checking whether it was in use — risking interference
  with a teammate's in-flight investigation.
- **Root cause**: M5 EVT1 is shared, reference-class hardware (per
  constitution Principle VIII), not exclusively controlled by this
  workstream; a prior session's uninterrupted access does not imply the
  device is still free.
- **Fix/workaround**: confirm with the user before assuming the device is
  free for adb/build/flash work, rather than assuming continuity from a
  previous session.
- **Citation**: this session's own narrative (no prior spec); see also
  project memory `project-m5-device-sharing`.
- **Status**: open (a standing process discipline, not a one-time fix).

### G11 — `VK_ERROR_DEVICE_LOST` on SDPA-coopmat at long decode was host-side OOM, not a GPU crash

- **Symptom**: `ET_VK_SDPA_COOPMAT=1` at the full 2048-prefill/1024-decode
  workload crashed with `libc++abi: ... vkcompute::vkapi::Error ...
  vkQueueWaitIdle(queue().handle) returned -4` on 3B and 8B (both
  schemes), recorded as `blocked_reason` and not retried. Looked like a
  genuine GPU/driver device-lost defect specific to SDPA at scale.
- **Root cause**: on retry (after M5 EVT1 was free again), the crash did
  not reproduce at all via `adb shell`-launched runs even at the full
  1024-decode length -- but the *proper* 3-rep headline measurement
  (`--warmup=true`, matching this feature's established methodology)
  later failed silently on 8B (exit 0, no `PyTorchObserver` output).
  `dmesg` showed a real Android OOM kill of the runner process
  (`anon-rss:1971136kB, file-rss:2446176kB`); `/proc/meminfo` showed
  `MemAvailable` down to ~0.6-1.5GB out of 11.19GB total. Cause: this
  workstream's own on-device working directory
  (`/data/local/tmp/llama_vk`) had accumulated ~29GB of staged PTEs and
  `.etdp` traces across a long session with no cleanup between runs,
  leaving too little headroom to load another multi-GB PTE plus SDPA's
  extra coopmat buffers, especially under `warmup=true`'s doubled peak
  memory (a full extra prefill+decode pass before the timed one).
- **Fix/workaround**: periodically delete already-pulled/already-consumed
  `.etdp`/log files and PTEs not immediately needed from the on-device
  working directory during a long session; prefer `--warmup=false` for
  large-model SDPA-coopmat runs if memory is tight (accepted trade-off:
  loses the warmup pass's steady-state benefit, but avoids doubling peak
  memory). Check `/proc/meminfo`'s `MemAvailable` before a large-model run
  if a run fails with exit 0 and no expected output -- an OOM kill does
  not always surface as an obvious crash message in the captured
  stdout/stderr, only in `dmesg`.
- **Citation**: `specs/015-m5-e2e-wmma-validation/research.md` Decision 8;
  `specs/015-m5-e2e-wmma-validation/results/3b-results.md` and
  `8b-results.md` (original `blocked_reason` entries, now superseded).
- **Status**: resolved as of 2026-07-06 (root cause found and fixed by
  cleanup + `warmup=false`; the underlying accumulation risk itself is not
  eliminated -- a future long session could hit the same wall again, so
  this entry stays as a live caution, not purely historical).

### G12 — `ET_VK_EXECUTE_NODE_THRESHOLD` is only required for 3 of 6 (model x WMMA-state) e2e configs, and costs ~11% where it isn't

- **Symptom**: unclear whether the `ET_VK_EXECUTE_NODE_THRESHOLD` GPU-watchdog
  workaround (`ComputeGraph.cpp:181-195`) should be applied to every e2e run
  by default, or only to specific model/WMMA combinations -- applying it
  everywhere risks silently costing throughput where it isn't needed;
  skipping it anywhere it IS needed crashes the run outright.
- **Root cause**: a full 1B/3B/8B x WMMA-OFF(T-tiled)/WMMA-ON(full-stack)
  sweep at `THRESHOLD=32`, 2048-token prefill, found the requirement is NOT
  uniform across the 6 cases:

  | Model | WMMA | Prefill tok/s (THRESHOLD=32) | Speedup | Behavior without THRESHOLD |
  |---|---|---:|---:|---|
  | 1B | OFF (T-tiled) | 314.4 | -- | fine, no diff |
  | 1B | ON (full-stack) | 809.8 | 2.58x | fine, no diff (808.85) |
  | 3B | OFF (T-tiled) | 112.6 | -- | **fine, actually ~11% faster without it** (125.6) |
  | 3B | ON (full-stack) | 333.6 | 2.96x | **crashes without it** |
  | 8B | OFF (T-tiled) | 51.6 | -- | **crashes without it** |
  | 8B | ON (full-stack) | 152.7 | 2.96x | **crashes without it** |

  So `THRESHOLD=32` is strictly *required* for only 3 of the 6 configs (3B
  WMMA-ON, 8B WMMA-OFF, 8B WMMA-ON -- all three hit the GPU watchdog and
  crash without it). For 1B (either WMMA state) it's a no-op either way.
  For 3B WMMA-OFF specifically it is actively harmful if left on --
  splitting that config's command buffers costs ~11% throughput
  (112.6 vs 125.6 tok/s) for no benefit, since that config never approaches
  the watchdog's time budget in the first place.

  Where required, the workaround is also confirmed measurement-neutral, not
  just crash-preventing: all three WMMA-ON numbers here (809.8/333.6/152.7)
  and 8B's WMMA-OFF number (51.6) land within measurement noise of
  `specs/015-m5-e2e-wmma-validation`'s and `RESULTS-SUMMARY.md`'s previously
  published anchors (812.59/333.97/153.30 and 51.4 respectively); 1B's
  WMMA-OFF number (314.4) likewise matches the 312.7 T-tiled anchor.
- **Fix/workaround**: don't blanket-apply `ET_VK_EXECUTE_NODE_THRESHOLD=32`
  to every e2e run. Apply it only where a config is confirmed to crash
  without it (currently: 3B WMMA-ON/full-stack, 8B WMMA-OFF/T-tiled, 8B
  WMMA-ON/full-stack). Leave it unset for 1B (either way is fine) and
  specifically for 3B WMMA-OFF/T-tiled (measurably ~11% slower with it).
- **Citation**: this session's own measurement narrative (no prior spec);
  WMMA-ON anchor numbers from
  `specs/015-m5-e2e-wmma-validation/results/{1b,3b,8b}-results.md`;
  WMMA-OFF/T-tiled anchor numbers from
  `specs/018-m5-8da4w-t-tiled-baseline/{spec,data-model}.md` citing
  `RESULTS-SUMMARY.md`.
- **Status**: open (a standing per-config decision, not a one-time fix --
  the underlying GPU watchdog issue itself is still driver-side, per
  `ComputeGraph.cpp`'s own citation of `jira-tickets/001`).
