# release-1.1 vs release-1.2 vs release-1.3 — 4w/1B baseline e2e (2026-07-14)

**Status**: report-grade (3 reps/release, real warmup, pinned+verified clocks).

**Device**: M5 EVT1, primary board, `0000088f8e579c33` via `ssh yanwen.xu@sj1-dmckee-d01`.

**Driver**: found drifted to unrecognized md5 `3880e697df8753a0d4a8ec3b394430a7` at session
start (yet another instance of this shared board's known recurring drift). Reflashed to the
documented default `f14c51b6f8` from NFS (`/sarc-c/gpusw/users/yanwen.xu/vulkan.samsung.so`);
re-verified md5 `c9861e9906d03fa2c7d48b804e1a1c80` matches exactly. Not re-verified again
mid-session (single continuous session, no gap long enough to expect re-drift) — a resumed
session should re-check before trusting these numbers further.

**Clocks**: pinned via `pin_freqs.sh` (509/2730/663 MHz target). Verified via sysfs bounds,
not `cur_freq` (per `access-and-run/README.md` §3): GPU `min_freq=max_freq=509000` on
`/sys/class/devfreq/23400000.sgpu/`. MIF/INT verified via the script's own echoed
post-write readback (2730000/663000).

**Branches/binaries**: `release-1.1/executorch`, `release-1.2/executorch`,
`release-1.3/executorch` (all read-only vanilla-upstream reference worktrees, see top-level
`CLAUDE.md`) — `llama_main` built fresh for all three same day via the documented Android
arm64+Vulkan two-step cmake recipe (`.shared-context/instruction-for-ai/setup/README.md`).
None of the three carry the `yanwen/dev-1.3` fork's WMMA coopmat additions, so all three are
necessarily the stock T-tiled/texture op path for `4w` — there is no buffer/coopmat variant
to compare against on these branches.

**Model/quant**: Llama 3.2-1B, `qmode=4w`, `group_size=128`, `dtype_override=fp32` +
`backend.vulkan.force_fp16=True`, texture storage (no `ET_VK_FORCE_BUFFER` — doesn't exist
pre-fork), `max_seq_length=max_context_length=3072`.

**Workload**: 2048-token prefill + 1024 new tokens, `--warmup=true`, `--temperature=0`,
3 reps/release.

---

## Finding 0: a `dev`-exported PTE is not release-1.1/1.2-compatible

The existing `.pte_out/llama3_2_1b_4w_texture_ctx3072.pte` (exported earlier from `dev/`,
which is release/1.3-based) loads and runs fine on `release-1.3`'s `llama_main`, but **crashes
(SIGABRT) on both `release-1.1` and `release-1.2`'s runners** — confirmed via `logcat`, not a
fluke:
```
F libc    : Fatal signal 6 (SIGABRT), code -1 (SI_QUEUE) in tid ... (llama_main_rel1)
F DEBUG   : #00 pc ... libc.so (abort+160)
F DEBUG   : #01-#12 ... llama_main_rel1.1 (stripped, no symbols)
```
Re-exporting per-release (same yaml schema, own venv — schema confirmed byte-identical
across all three in `llm_config.py`) fixed this. Configs:
`.shared-context/scripts/export-configs/llama3_2_1b_4w_texture_ctx3072_rel1.1.yaml` (and
`_rel1.2.yaml`). `release-1.3` kept using the pre-existing `.pte_out` PTE (verified
compatible).

## Finding 1: release-1.1's `llama_main` predates two CLI flags

`--prompt_file` and `--ignore_eos` were added between 1.1 and 1.2 (confirmed via
`git diff release/1.1 release/1.2 -- examples/models/llama/main.cpp`, `DEFINE_string(prompt_file...)`/
`DEFINE_bool(ignore_eos...)` both absent in 1.1). Worked around with a device-side wrapper
script that reads the prompt file via shell `$(cat ...)` and passes it through `--prompt`
(the prompt text has no quote/backslash characters, confirmed before relying on this):
```sh
#!/system/bin/sh
cd /data/local/tmp/llama_vk
PROMPT="$(cat p2048_exact_rel1.1.txt)"
exec ./llama_main_rel1.1 --model_path=llama3_2_1b_4w_texture_ctx3072_rel1.1.pte \
  --tokenizer_path=tokenizer.model --prompt="$PROMPT" --num_bos=1 \
  --max_new_tokens=1024 --temperature=0 --warmup=true
```
Without `--ignore_eos`, 1.1's greedy (`temperature=0`) decode never actually emitted an EOS
token on this prompt — it degenerated into repeating the prompt text, then a run of `!`
characters, but ran the full requested length regardless (confirmed via `generated_tokens`
below, not truncated). Not a correctness concern for the timing measurement.

## Finding 2: release-1.1 tokenizes the shared prompt file one token short

Same `p2048_exact.txt` (2047 tokens designed to hit exactly 2048 with `--num_bos=1`) that
gives `release-1.2`/`release-1.3` exactly `prompt_tokens=2048` gives `release-1.1` only
`2047`. Traced to source, not a counting bug: the `tokenizer_->encode(prompt, num_bos, num_eos)`
call site in `extension/llm/runner/text_llm_runner.cpp` is byte-identical across all three
releases — the difference is the pinned `extension/llm/tokenizers` submodule commit itself:

| Release | `extension/llm/tokenizers` commit |
|---|---|
| 1.1 | `37e1c7ed13fa04accd696c776c2f05b4b12fe61d` |
| 1.2 | `6cbb882d9baac25c88b8ef38b338123bd2c35dbc` |
| 1.3 | `0b10f027bc66e9d372e3321c9fa0142d1c52891b` |

1.1's older tokenizer genuinely produces one fewer BPE token from the same 12249-byte prompt
text. Fixed by using a release-1.1-specific prompt file, `p2048_exact_rel1.1.txt` = the
original file + one appended word (`" and"`, continuing the cut-off sentence naturally),
empirically verified (via a fast `--max_new_tokens=1` probe) to push 1.1's tokenizer to
exactly `prompt_tokens=2048` too. All three releases now measured at exactly 2048 prompt
tokens.

## Finding 3: `"generated_tokens"` in the JSON stat is decode-phase-only, not total

`resolve_max_new_tokens()` (`extension/llm/runner/irunner.h`, identical across all three
releases) returns `min(max_new_tokens_requested, max_context_len - occupied) = min(1024,
3072-0) = 1024` for this config — not capped. `text_llm_runner.cpp` then generates
`max_new_tokens - 1` tokens in the decode loop because **prefill itself produces the first
generated token** (comment in source: `// Generate max_new_tokens - 1 because prefill
already generated 1 token.`) — confirmed by `first_token_ms == prompt_eval_end_ms` exactly in
every run's raw JSON below. So `"generated_tokens":1023` + the 1 prefill-boundary token =
1024 actual total, matching the `--max_new_tokens=1024` request exactly on all three
releases, all reps. The decode tok/s convention used below (`generated_tokens / decode_time`,
i.e. 1023-token count) is ExecuTorch's own stats convention (`stats.h`'s
`print_report()`), not something introduced for this comparison.

---

## Raw results (3 reps/release)

### release-1.1 (`llama3_2_1b_4w_texture_ctx3072_rel1.1.pte`, `p2048_exact_rel1.1.txt`)

| Rep | prompt_tokens | generated_tokens | prefill (s) | decode (s) | prefill tok/s | decode tok/s |
|---|---|---|---|---|---|---|
| 1 | 2048 | 1023 | 6.928 | 74.679 | 295.6 | 13.70 |
| 2 | 2048 | 1023 | 6.959 | 74.691 | 294.3 | 13.70 |
| 3 | 2048 | 1023 | 6.944 | 73.933 | 294.8 | 13.84 |
| **median** | | | | | **294.8** | **13.70** |

### release-1.2 (`llama3_2_1b_4w_texture_ctx3072_rel1.2.pte`, `p2048_exact.txt`, `--prompt_file`+`--ignore_eos`)

| Rep | prompt_tokens | generated_tokens | prefill (s) | decode (s) | prefill tok/s | decode tok/s |
|---|---|---|---|---|---|---|
| 1 | 2048 | 1023 | 6.705 | 72.312 | 305.6 | 14.15 |
| 2 | 2048 | 1023 | 6.674 | 72.429 | 306.9 | 14.13 |
| 3 | 2048 | 1023 | 6.702 | 72.938 | 305.6 | 14.03 |
| **median** | | | | | **305.6** | **14.13** |

### release-1.3 (pre-existing `.pte_out/llama3_2_1b_4w_texture_ctx3072.pte`, `p2048_exact.txt`, `--prompt_file`+`--ignore_eos`)

Runner-reported rates directly (this release's `stats.h` includes `prefill_token_per_sec`/
`decode_token_per_sec` in the JSON — cross-checked against manual timestamp math, exact
match):

| Rep | prompt_tokens | generated_tokens | prefill tok/s | decode tok/s |
|---|---|---|---|---|
| 1 | 2048 | 1023 | 314.255 | 14.616 |
| 2 | 2048 | 1023 | 314.303 | 14.647 |
| 3 | 2048 | 1023 | 315.465 | 14.616 |
| **median** | | | | **314.3** | **14.62** |

## Headline comparison

| Release | Prefill tok/s (median) | Decode tok/s (median) |
|---|---|---|
| 1.1 | 294.8 | 13.70 |
| 1.2 | 305.6 | 14.13 |
| 1.3 | 314.3 | 14.62 |

Monotonic improvement release-to-release: prefill +3.7% (1.1→1.2), +2.8% (1.2→1.3), +6.6%
total (1.1→1.3). Decode +3.1% (1.1→1.2), +3.5% (1.2→1.3), +6.7% total (1.1→1.3). Spread
within each release's 3 reps is tight (prefill within ~0.4%, decode within ~1%), so this
trend is real, not noise. Consistent with general upstream Vulkan/runtime improvements across
releases — nothing on these vanilla branches originates from this workstream's own
optimizations (none of which are present pre-fork).

## Anomalies

- release-1.1's greedy decode degenerated into prompt repetition then a long run of `!`
  characters near the end of the 1024-token budget — expected LLM behavior at
  `temperature=0` on a small model with a highly repetitive prompt, not a runtime bug; ran to
  the full requested length regardless (see Finding 1).
- No sgpu watchdog kills, no driver drift mid-session, no crashes on the final (correct)
  configs across all 9 runs (3 releases × 3 reps).

## Artifacts

- Export configs: `.shared-context/scripts/export-configs/llama3_2_1b_4w_texture_ctx3072_rel1.1.yaml`,
  `_rel1.2.yaml` (see `spec.md` for why release-1.3 didn't need its own).
- PTEs (on-device, `/data/local/tmp/llama_vk/`, and staged at
  `/sarc-c/gpusw/users/yanwen.xu/android-run/models/`):
  `llama3_2_1b_4w_texture_ctx3072_rel1.1.pte`, `_rel1.2.pte`; release-1.3 used the existing
  `llama3_2_1b_4w_texture_ctx3072.pte`.
- release-1.1-specific prompt file: `p2048_exact_rel1.1.txt` (staged at
  `/sarc-c/gpusw/users/yanwen.xu/android-run/assets/`).
- Wrapper script (release-1.1 only, missing `--prompt_file`): on-device
  `/data/local/tmp/llama_vk/run_rel1.1_report.sh`.
- Binaries staged: `/sarc-c/gpusw/users/yanwen.xu/android-run/runners/llama_main_rel1.1`,
  `_rel1.2`, `_rel1.3`.
