# Investigation: 4w T-Tiled Decode Regression, 2026-06-17 → July

**Date**: 2026-07-11 | **Target**: M5 EVT1 (primary, `0000088f8e579c33` @ sj1-dmckee-d01)
**Status**: Closed pending new evidence — every checkable hypothesis ruled out or refuted;
root cause narrowed to an unrecoverable variable (see Finding 8).

## Summary

`4w` T-tiled 1024-token steady-state decode throughput dropped **9–23%** between the
2026-06-17 "trusted anchor" measurement
(`report-for-human-archived-2026-07-08/4w-prefill2048-decode1024-3models.md`) and every
July 2026 measurement on this spec's canonical branch (`release13-node-threshold`), while
**prefill stayed within <1%** across the same comparison — a decode-specific effect, not a
general device or measurement drift.

| Model | 2026-06-17 anchor (decode tok/s) | July 2026 (decode tok/s, 4+ independent sessions) | Delta |
|---|---|---|---|
| 1B | 18.5 | 14.19–14.86 | **-19% to -23%** |
| 3B | 7.85 | 7.18995 | **-8.4%** |
| 8B | 4.42 | 3.96203 | **-10.4%** |

Eight hypotheses were tested, in order; the first six were ruled out or directly refuted by
experiment, the seventh is the true (structural) explanation for *why* it took this long to
localize, and the eighth is the one remaining, unrecoverable candidate.

## Finding 1: Not a decode-length mismatch

There is prior history of exactly this failure mode elsewhere in this project — the
archived `RESULTS-SUMMARY.md` explicitly flags its own earlier decode numbers as a
mislabeled ~128-token short window rather than genuine 1024-token steady state. Checked
both eras directly: the 2026-06-17 anchor doc's own methodology block confirms
`--max_new_tokens=1024 --ignore_eos`; every July run used the same
`_ctx3072.pte` + `--max_new_tokens=1024 --ignore_eos` convention. **Both are genuine
1024-token measurements.** Ruled out.

## Finding 2: Not the driver

Controlled A/B, branch/runner held fixed (`release13-node-threshold`, binary
`llama_main_nodethresh`, md5-verified byte-identical to the local build), same
`llama3_2_1b_4w_texture_ctx3072.pte`, pinned 509/2730/663 MHz clocks, only the driver
varied:

| Driver | prefill tok/s | decode tok/s |
|---|---|---|
| `f14c51b6f8` (current standard, md5 `c9861e9906d0...`) | 311.578 | 14.1884 |
| `c0d117aaf2` (older known-good, md5 `ae546fa078...`) | 309.693 | 14.5284 |

+2.4%, wrong direction for a "driver regressed" story, an order of magnitude smaller than
the 9–23% gap. Ruled out.

## Finding 3: Not the `ET_VK_EXECUTE_NODE_THRESHOLD` value

A real historical bug exists in this area but does not apply to the branch used for every
current measurement — see the companion memory
[[node-threshold-blocking-fence-bug-history]] for the full 2026-06-22/23 finding (the
*original* `.tmp-origcm`/`VulkanBackend.cpp` implementation of this workaround added a
blocking `vkQueueWaitIdle`-equivalent every 16 nodes, causing an **independently-measured
~50% prefill regression** — 8B T-tiled 51.5→99.3 tok/s when removed). Checked whether
`release13-node-threshold`'s *reimplementation* (`ComputeGraph.cpp`,
`submit_cmd_to_gpu(VK_NULL_HANDLE, false)` at the split point) carries the same bug:

- **By code inspection**: that call is a plain non-blocking Vulkan submit — no
  `vkQueueWaitIdle` anywhere in the split path (`Context::wait_for_queue()`, the actual
  blocking-wait method, is a distinct, uncalled function).
- **Empirically**: re-ran the identical July PTE/binary/driver/clocks, only changing
  `ET_VK_EXECUTE_NODE_THRESHOLD` 16→32: prefill 311.578→312.148 (+0.2%), decode
  14.1884→14.5812 (+2.8%). Both within normal run-to-run noise.

`release13-node-threshold`'s threshold mechanism is not bugged this way; 16-vs-32 is not a
live confound here. Ruled out.

## Finding 4: Not `backends/vulkan/runtime` C++ source

Full git-ancestry trace from the shared merge-base (`dbcf6ac7f`) between `.tmp-origcm`'s
base commit (`1da18955a`) and `release13-node-threshold`'s base (`origin/release/1.3` HEAD
`e2f18eb23`). `release/1.3`'s own 10 commits since that point are 100%
release-engineering (version bumps, CI pins) — zero touch to `backends/vulkan/`.
`.tmp-origcm`'s and `quant-dev`'s uncommitted deltas at archive time
(`.archived-artifacts/{tmp-origcm,quant-dev}-2026-07-11/patches/tracked-modifications.diff`,
both checked line-by-line) touch only coopmat-gated code (`kernel_name.find("_coopmat")`,
`can_use_q4gsw_coopmat()`) or env-var-gated experiment hooks (`ET_VK_Q4_WEIGHT_BUFFER`) —
categorically inert on a texture-storage (T-tiled) export. Ruled out.

## Finding 5: Not upstream PR #16986 (directly tested, refuted)

The single strongest circumstantial candidate. Upstream commit `77df9b79a` ("New exported
program pass manager and exported program passes", merged 2026-05-26) rewrites
`exir/pass_manager.py`/`exir/program/_program.py` and is present in quant-dev's lineage
(confirmed via a log fingerprint — instantiating the legacy `PassManager` class now emits
`"PassManager is deprecated. Please use ExportedProgramPassManager instead."`; this string
appears **2x** in the 2026-06-22 quant-dev export log, **0x** in every July
`release13-node-threshold` export log) but is entirely absent from `release/1.3`
(`git merge-base --is-ancestor 77df9b79a origin/release/1.3` → no).

**Tested directly, not just cited as circumstantial**:
1. Isolated throwaway worktree off `origin/release/1.3` (detached HEAD,
   `.tmp-pr16986-abtest/`, fully removed afterward — never touched `release13-node-threshold`
   or any persistent worktree).
2. `git cherry-pick -x 77df9b79a` — applied clean, zero conflicts.
3. Sped this up by reusing `release13-node-threshold`'s already-built `.venv` (plain copy,
   1.5GB, ~7s on local NVMe) instead of a fresh `./install_executorch.sh` (which would have
   re-cloned every third-party submodule and rebuilt custom ops from scratch for no benefit,
   since the change under test is pure Python) — see memory
   [[feedback-reuse-existing-venv-for-code-ab-tests]]. Overlaid just the 4
   runtime-relevant changed files into the copied venv's site-packages; verified live that
   the deprecation warning now fired.
4. Re-exported `llama3_2_1b_4w_texture_ctx3072` with the exact same config as every July
   export. Export log showed the deprecation warning 2x, confirming the cherry-pick was
   genuinely exercised (not a no-op). Resulting PTE: same file size, **different md5**
   (`9adca6d3...` vs `3091d4d3...`) — the cherry-pick did change something in the exported
   artifact.
5. Pushed to the primary M5 EVT1, ran with the existing unmodified `llama_main_nodethresh`
   binary (pure AOT-side change, no rebuild needed), same driver/clocks.
   **Result: `decode_token_per_sec: 14.3874`** — indistinguishable from the July baseline
   cluster (14.19–14.86), nowhere near June's 18.5.

**PR #16986 is refuted as the cause.** It changes exported-PTE bytes but not decode
throughput.

## Finding 6: Not any other file in the 108-commit quant-dev-vs-release/1.3 diff

Following Finding 5's refutation, every remaining file in the 108-commit range
(`dbcf6ac7f..1da18955a`) touching export/quantization/graph-building/runtime-execution code
was individually checked and found inert for this specific `4w`/Vulkan/torchao/
`use_kv_cache=True` export configuration:

| File | Why it's inert here |
|---|---|
| `runtime/executor/memory_manager.h` | Additive multi-device accessor (`planned_buffer_devices()`), unused single-Vulkan-backend |
| `extension/llm/export/builder.py` | Entirely inside `pt2e_calibrate` (static PT2E calibration) — never called by our torchao-based export |
| `extension/llm/export/config/llm_config.py` | Adds `"8da8w"` to a QMODE string enum; irrelevant to `qmode=4w` |
| `examples/models/llama/model.py` | Only touches `get_example_inputs()`'s non-KV-cache branch; we always set `use_kv_cache=True` |
| `examples/models/llama/norm.py` | CoreML-specific RMSNorm dtype fix; we use the Vulkan backend |
| `examples/models/llama/eval_llama_lib.py`, `evaluate/eager_eval.py` | lm-eval harness, not exercised by export |
| `extension/llm/custom_ops/op_fallback.py` (new) + its use in `exir/passes/spec_prop_pass.py` | Qualcomm QNN/llama-sharding fallback op, `try/except ImportError`-guarded, matches no node in a plain Vulkan graph |
| `extension/llm/runner/irunner.h` | Adds `grammar`/`grammar_type` fields for constrained decoding, default-empty, unused by our CLI. **The only file touched anywhere under `extension/llm/runner/`** — the actual C++ generate/decode-loop implementation is byte-identical between the two lineages |
| `third-party/ao` (torchao) submodule pin | Identical SHA across `dbcf6ac7f`/`1da18955a`/`origin/release/1.3` |
| `install_requirements.py`'s torch pin (`"torch==2.12.0"`) | Identical string across all three points |

Ruled out, exhaustively, at the git-diff level.

## Finding 7: What actually differs between the two eras (structural summary)

The June-17 PTE was exported from the `quant-dev` worktree (Python 3.12, custom-ops loaded
directly from the source tree — a non-packaged install). Every July PTE was exported from
`release13-node-threshold` (Python 3.10, packaged `site-packages` install, torch
confirmed `2.12.0+cpu` — no `.dev` date suffix, i.e. a **stable** PyPI release, not a
nightly snapshot). The export **command/config parameters are identical** in both eras
(`use_kv_cache=True`, `use_sdpa_with_kv_cache=True`, `force_fp16=True`, same
qmode/group_size/max_seq/max_context) — this was never a "someone passed a different flag"
story. PyTorch 2.12.0 GA shipped 2026-05-13, before the June 17 anchor, so a stable install
in either era *could* have resolved to the same build — checking the GA date didn't
resolve the question either way.

## Finding 8: The one remaining, unrecoverable candidate

Whether quant-dev's `install_requirements.py` was invoked with `use_pytorch_nightly=True`
or `False` is a **per-invocation flag, not something tracked in git** — if the two eras
differed on this flag (or landed on different nightly-index snapshots, if both used
nightly), that's a real difference this investigation cannot see from source code alone.
Searched exhaustively for any surviving record: `pip freeze` output, a lock file, any
`.artifacts`/NFS log printing a `.devYYYYMMDD`-suffixed torch version string from mid-June —
**none exists anywhere in the workspace.** `quant-dev/executorch/.venv` itself no longer
exists (removed at an unknown point before this investigation began). torchao's exact
June-era version is similarly unrecoverable (no equivalent "GA date" check available, and
no version string survived).

**This is now believed to be the true cause, but it is unprovable with available
evidence.** Every hypothesis checkable from git history, source code, or direct experiment
has been checked; this is the only one that isn't.

## Recommendation

Closed pending new evidence. If this needs to be fully resolved: find any record (a chat
log, a note, a screenshot) of what flags were passed to `install_requirements.py`/
`install_executorch.sh` when `quant-dev`'s venv was originally set up, or accept the
category-level explanation (package build provenance at install time, not a source commit)
as final. Going forward: **snapshot `pip freeze` (or at minimum keep the export log, which
records the venv path) whenever a `.pte` that might need reproducing later gets exported** —
this investigation only got as far as it did because two export *logs* (not the PTEs
themselves) happened to survive in `.artifacts/` and NFS `results_ctx3072/`.

## Related

- Memory: [[decode-regression-june-vs-july]] (index into this investigation),
  [[node-threshold-blocking-fence-bug-history]], [[feedback-reuse-existing-venv-for-code-ab-tests]],
  [[dbuf-variant-differs-by-quant-scheme]] (an unrelated but adjacent finding from the same
  session — corrected a stale workspace `CLAUDE.md` line).
- This spec's own canonical numbers (`data-model.md`) and the secondary-device cross-check
  (`results/secondary-m5-evt1-release13-node-threshold-2026-07-11.md`) are what first
  surfaced the discrepancy against the archived `4w-prefill2048-decode1024-3models.md`
  anchor.
- `specs/024-8da4w-slower-than-4w` — an unrelated investigation from the same session
  (why `8da4w` is slower than `4w` on the tiled path specifically, not about the
  across-time regression documented here).
