# Feature Specification: 4w/1B Baseline Perf Across release-1.1/1.2/1.3

**Feature Branch**: `029-release-version-4w-baseline`

**Created**: 2026-07-14

**Status**: Answered (report-grade numbers collected, 3 reps each)

**Input**: User request — after cloning upstream `release/1.1` and `release/1.2` as new
worktrees (mirroring the existing `release-1.3/` read-only reference clone) and building the
`llama_main` Vulkan runner for all three, compare stock (T-tiled/texture, no coopmat —
vanilla upstream doesn't have the fork's WMMA additions) `4w`/Llama-3.2-1B prefill+decode
perf across the three release versions.

## Context

`release-1.1/`, `release-1.2/`, `release-1.3/` are read-only vanilla-upstream reference
clones in this workspace (see top-level `CLAUDE.md` worktree table) — none of them carry the
`yanwen/dev-1.3` fork's WMMA coopmat/`ET_VK_FORCE_BUFFER` additions, so `4w` on these three is
always the stock T-tiled/texture op path. This spec answers: does the *baseline* (non-coopmat)
Vulkan perf change release-to-release, independent of any of this workstream's own
optimizations?

## Method

- **Device**: M5 EVT1, primary board (`0000088f8e579c33` via `sj1-dmckee-d01`).
- **Driver**: reflashed to `f14c51b6f8` (md5 `c9861e9906d03fa2c7d48b804e1a1c80`) before
  measurement — device was found drifted to an unrecognized build (md5 `3880e697df87…`)
  beforehand. Re-verified after flash.
- **Clocks**: pinned 509/2730/663 MHz (`pin_freqs.sh`); GPU devfreq `min_freq=max_freq=509000`
  confirmed.
- **Model/quant**: Llama 3.2-1B, `qmode=4w`, `group_size=128`, texture storage (no
  `ET_VK_FORCE_BUFFER` — doesn't exist on these branches), `ctx3072`
  (`max_seq_length=max_context_length=3072`).
- **PTEs**: exported per-release, from each release's own venv/checkout — a single PTE
  exported from `dev/` (release/1.3-based) was tried first and crashes (SIGABRT) on
  `release-1.1`/`release-1.2`'s runners; only `release-1.3`'s own runner loads it. Not
  cross-version-compatible — see `results/` for the crash detail. Configs archived at
  `.shared-context/scripts/export-configs/llama3_2_1b_4w_texture_ctx3072_rel1.1.yaml` /
  `_rel1.2.yaml`.
- **Workload**: 2048-token prefill (`p2048_exact.txt`, 2047 tokens + `--num_bos=1`) + 1024
  new tokens (`--max_new_tokens=1024`), `--warmup=true`, `--temperature=0`, 3 reps per
  release. This is `result-and-report/README.md`'s row-4 "Report-grade e2e" bar (3+ reps,
  real warmup, pinned+verified clocks).
- **`release-1.1` gotchas** (its `llama_main` predates some CLI flags — see
  `results/` for detail):
  - No `--prompt_file`/`--ignore_eos` flags (added in 1.2) — used a `--prompt="$(cat ...)"`
    wrapper script instead; ran to full length via greedy-decode repetition rather than an
    explicit ignore-eos flag.
  - Its pinned `extension/llm/tokenizers` submodule commit (`37e1c7ed13…`, vs 1.2/1.3's
    `6cbb882d9b…`/`0b10f027bc…`) tokenizes the stock `p2048_exact.txt` into one fewer token —
    used a release-1.1-specific prompt file (`p2048_exact_rel1.1.txt`, one word longer) to hit
    exactly 2048 prompt tokens too, for a fair comparison.

## Results

See `results/4w-1b-baseline-comparison-2026-07-14.md` for the full writeup (raw per-rep JSON,
medians, cross-version trend).

**Headline** (median of 3 reps, tok/s):

| Release | Prefill | Decode |
|---|---|---|
| 1.1 | 294.8 | 13.70 |
| 1.2 | 305.6 | 14.13 |
| 1.3 | 314.3 | 14.62 |

Monotonic improvement release-to-release (prefill +3.7% then +2.8%; decode +3.1% then
+3.5%) — consistent with general upstream runtime/backend improvements, not attributable to
anything in this workstream's own changes (none of which are present on these vanilla
branches).
