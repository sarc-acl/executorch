# M5 EVT1 E2E Smoke Test — `dev` (`yanwen/dev-1.3`), all 3 models x 4w/8da4w

Status as of 2026-07-11. **Single-run smoke test, not a 3-run mean** — this
confirms the post-migration `dev` branch (WMMA linear + SDPA coopmat ported
2026-07-09, SDPA coopmat default-on since `specs/026-sdpa-8da4w-defaults-e2e`)
produces coherent, in-range e2e numbers after a fresh `llama_main` rebuild.
Not a substitute for this spec's own 3-run-mean rows above — treat as a
build/dispatch sanity check, not a report-grade headline figure.

## Setup

- Branch: `dev` (`yanwen/dev-1.3`) @ `573d44dac` ("[ET-VK] Enable SDPA coopmat
  by default on capability-eligible devices"), freshly rebuilt `llama_main`
  this session (see gotcha below).
- Device: M5 EVT1 (`0000088f8e579c33` @ `sj1-dmckee-d01`). Driver verified
  `f14c51b6f8` (md5 `c9861e9906…`) before the run — already the documented
  default, no reflash needed.
- Clocks: pinned 509/2730/663 MHz (`pin_freqs.sh`).
- Workload: 2048-token prefill (`p2048_exact.txt` + `--num_bos=1`) + 1024-token
  decode, `--ignore_eos --temperature=0 --warmup=true`,
  `ET_VK_EXECUTE_NODE_THRESHOLD=16`.
- PTEs: `<model>_<qmode>_buffer_ctx3072.pte` from `.pte_out/` (exported
  2026-07-09, buffer storage — coopmat-eligible). No env var override needed;
  SDPA coopmat dispatches by default on this branch for any coopmat-capable
  buffer PTE + coopmat-built runner.
- Coherence check passed first (1B/4w, short prompt): `"The capital of France
  is Paris..."`, no crash.
- Raw output: `raw/2026-07-11-dev-branch-smoke-test.log`.

## Results

| Config | Prefill tok/s | Decode tok/s | Duration (load→inference end) |
|---|---|---|---|
| 1B `4w` | 797.8 | 13.79 | 218.6s (3m 39s) |
| 1B `8da4w` | 731.7 | 13.13 | 162.5s (2m 42s) |
| 3B `4w` | 336.1 | 6.81 | 315.1s (5m 15s) |
| 3B `8da4w` | 289.3 | 6.49 | 331.6s (5m 32s) |
| 8B `4w` | 153.2 | 3.81 | 568.9s (9m 29s) |
| 8B `8da4w` | 130.3 | 3.71 | 588.1s (9m 48s) |

## Cross-check against this spec's existing 3-run-mean data

Every number above lands within ~1-2% of this spec's existing **"+
SDPA-coopmat (full-stack)"** rows (`1b-results.md` / `3b-results.md` /
`8b-results.md`), which is exactly what's expected now that SDPA coopmat is
default-on rather than opt-in via `ET_VK_SDPA_COOPMAT=1`:

| Model / qmode | This smoke test | Prior "+SDPA-coopmat" 3-run mean |
|---|---|---|
| 1B `4w` prefill | 797.8 | 812.59 / 769.35 (two prior measurements, see `1b-results.md` UPDATE 2) |
| 1B `8da4w` prefill | 731.7 | 723.00 |
| 3B `4w` prefill | 336.1 | 333.97 |
| 3B `8da4w` prefill | 289.3 | 286.31 |
| 8B `4w` prefill | 153.2 | 153.30 |
| 8B `8da4w` prefill | 130.3 | 130.05 |

Decode numbers show the same agreement (e.g. 8B `4w` 3.81 vs prior 3.79; 8B
`8da4w` 3.71 vs prior 3.67). This consistency is itself useful evidence: it
confirms the `dev` branch's default-on coopmat path reproduces the same
performance as the old opt-in env-var path on `quant-dev`, with no
regression from the migration.

## Build gotcha hit this session (new, not yet in `setup/README.md`)

Rebuilding `vulkan_backend`/`executor_runner` after a `cmake .
-Bcmake-out-android-vk --preset llm ...` re-configure failed with:
```
ld.lld: error: third-party/flatcc/lib/libflatccrt.a(builder.c.o) is incompatible with aarch64linux
```
Cause: `third-party/CMakeLists.txt`'s `flatcc_ep` (builds the **host**-arch
`flatcc` CLI tool) and the `flatccrt` target (builds the **target**-arch
runtime lib actually linked into the runner) both write into the same
in-source-tree path `third-party/flatcc/lib/libflatccrt.a` (upstream's own
design — see the comment at that file's line ~161). Re-running the top-level
`cmake --build ... --target install` let `flatcc_ep` rebuild and clobber the
aarch64 lib with a host x86-64 one after `flatccrt` had already produced the
correct one, since `install` doesn't force `flatccrt` to always run last.

**Fix used:** `cmake --build cmake-out-android-vk --target flatccrt -j$(nproc)
--clean-first` (forces flatccrt to rebuild for the target arch, confirmed via
`file`/`ar p ... | file -` showing `ARM aarch64`), then build/install the
remaining targets (`executor_runner`, step 2's `llama_main`) **without**
re-triggering `flatcc_ep` (it's cached once `flatcc_ep`'s own build product
is up to date, so a second `--target install` right after doesn't reclobber
it). If this recurs, verify the arch of `third-party/flatcc/lib/libflatccrt.a`
before any relink: `ar p third-party/flatcc/lib/libflatccrt.a builder.c.o |
file -` must say `ARM aarch64`, not `x86-64`.
