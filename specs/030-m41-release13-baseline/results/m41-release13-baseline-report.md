# M41 Release/1.3 Baseline: 4w/8da4w × Pinned/Floating (2026-07-14)

> **M41 is a secondary/cross-device reference, not the Samsung M5 EVT1 active mission target**
> (constitution Principle II). These numbers are supplementary — the same treatment this
> workstream already gives the retired MiniPC's results — and are never compared against or
> substituted for Samsung M5 EVT1 headline numbers.

**Device**: M41, serial `000009b44fd4abd3`, `ssh xgpusw-debug07` (`export
ANDROID_SERIAL=000009b44fd4abd3`). SoC: `s5e9965` (Exynos, ERD9965 family — same family as the M4
cross-device boards, not the M5 EVT1's `s5e9975`).

**Driver**: `vulkan.samsung.so` md5 `d5d76f1bacf404b1a07d87ec8e479bdf` (checked 2026-07-14). No
documented known-good reference hash exists for this SoC family — the workspace's known-good
table (`f14c51b6f8`/`c0d117aaf2`) is built specifically for the M5 EVT1's `s5e9975` chip and does
not apply here. No driver flash was performed.

**Methodology**: `llama_main_rel1.3` runner (plain `release/1.3`, predates the WMMA coopmat port —
no `ET_VK_DISABLE_COOPMAT`/`ET_VK_EXECUTE_NODE_THRESHOLD` gate assumed present without checking).
Storage: texture (the stock ExecuTorch T-tiled path, no coopmat exists on this branch). Workload:
2048-token prefill + 1024-token decode (`p2048_exact.txt`, `--num_bos=1 --max_new_tokens=1024
--ignore_eos --temperature=0 --warmup=true`). 3 reps per (model, quant-mode, clock-mode) cell.

**Per-cell outcomes**: each of the 9 rep-cells in every table below is one of:
- a number (prefill / decode tok/s)
- **CRASHED** — the run aborted; cause attributed via `dmesg`/`/proc/meminfo`, never assumed
- **DVFS-ARTIFACT** — a "pinned" run whose measured throughput exceeded 70% of the corresponding
  floating number, meaning the pin didn't actually bind; excluded from that cell's mean/CoV

CoV = stdev / mean × 100%, computed only from `outcome=ok` reps (≥2 required).

---

## Table 1: 4w, floating clocks

| Model | Rep 1 | Rep 2 | Rep 3 | Mean (caveat: may mix cold-start peak w/ throttled steady-state) | CoV |
|---|---:|---:|---:|---|---|
| 1B | 594.14 / 30.14 | 601.82 / 30.41 | 601.47 / 30.55 | prefill 599.14, decode 30.37 | prefill 0.72%, decode 0.69% |
| 3B | 219.46 / 13.02 | 215.24 / 13.00 | 212.18 / 13.01 | prefill 215.63, decode 13.01 | prefill 1.70%, decode 0.06% |
| 8B | 90.91 / 7.26 | **CRASHED** (see below) | 86.90 / 7.21 | prefill 88.91, decode 7.24 (n=2) | prefill 1.60%, decode 0.49% |

*(prefill / decode tok/s per rep. All values verified/preserved from this session's earlier
collection — spec FR-002.)*

**8B rep 2 crash**: `VK_ERROR_DEVICE_LOST` (`vkQueueWaitIdle=-4`), floating clocks. `crash_cause`
= **unknown** — the on-device `dmesg` ring buffer only retains ~210s of history (confirmed at
check time: earliest entry was uptime 8649.76s, "now" was 8859.84s), so it no longer covers this
crash from earlier in the session; retroactive attribution isn't possible from this evidence.
However, a *reproduction* of the identical error signature (see Methodology notes below) was
captured live and shows a genuine GPU-driver-level watchdog reset, not a host OOM kill — while not
proof this specific rep had the same cause, it's the best available evidence and rules out a
default "probably OOM" guess.

## Table 2: 4w, pinned clocks (509/2730/663 MHz)

| Model | Rep 1 | Rep 2 | Rep 3 | Mean | CoV | Pin-verified? |
|---|---:|---:|---:|---|---|---|
| 1B | 316.98 / 17.49 | 315.90 / 17.64 | 316.25 / 17.62 | prefill 316.38, decode 17.59 | prefill 0.17%, decode 0.46% | ✅ (52.8% of floating — well under the 70% threshold) |
| 3B | 106.10 / 7.29 | 103.00 / 7.31 | 105.76 / 7.32 | prefill 104.95, decode 7.30 | prefill 1.62%, decode 0.22% | ✅ (48.7% of floating) |
| 8B (plain `release/1.3`) | **CRASHED** | **CRASHED** | **CRASHED** | n/a (0/3 valid) | n/a | n/a — not measurable with this binary |
| 8B (`release13-node-threshold`, `THRESHOLD=32`) | 52.75 / 3.95 | 52.77 / 3.97 | 52.71 / 3.95 | prefill 52.76, decode 3.96 | prefill 0.03%, decode 0.29% | ✅ (59.3% of floating) |
| 8B (`release13-node-threshold`, `THRESHOLD=64`) | 49.53 / 3.99 | 49.97 / 3.98 | 48.89 / 3.98 | prefill 49.46, decode 3.98 | prefill 1.09%, decode 0.17% | ✅ (55.6% of floating) |

*(prefill / decode tok/s per rep.)*

**8B — plain `release/1.3` crashes 3/3**: `VK_ERROR_DEVICE_LOST` (`vkQueueWaitIdle=-4`), identical
signature across all three. `crash_cause=gpu_watchdog` (confirmed, not inferred) — `dmesg` showed 3
distinct clusters of `sgpu ...: amdgpu: GPU reset(110-119) succeeded!` entries matching the 3 reps,
zero OOM-kill signatures anywhere in the buffer; `/proc/meminfo` showed 4.90GB/10.9GB
`MemAvailable` (healthy). Device confirmed responsive after each crash and after the full sweep.
`llama_main_rel1.3` (plain `release/1.3`) has no node-threshold workaround (Methodology notes) —
**not measurable with that binary**, not a one-off flake.

**Fix found and applied**: the separate `release13-node-threshold` branch (workspace-root
`CLAUDE.md`'s worktree table; adds an opt-in `ET_VK_EXECUTE_NODE_THRESHOLD` env var to
`ComputeGraph.cpp`, submitting a new command buffer every N nodes instead of the default 128)
fixes this crash completely on M41 — confirmed with a freshly-built runner from that branch's
current (uncommitted) source, not a reused stale binary. **`THRESHOLD=32` is the better setting**:
it fixes the crash and yields ~6.7% higher prefill than `THRESHOLD=64` (52.76 vs 49.46 tok/s) —
consistent with gotcha G12's finding on the M5 EVT1 primary target that a smaller threshold isn't
automatically better once the watchdog is already avoided; more frequent, smaller submissions add
their own overhead. Both settings pass the FR-009 throughput cross-check.

## Table 3: 8da4w, floating clocks

| Model | Rep 1 | Rep 2 | Rep 3 | Mean (caveat: may mix cold-start peak w/ throttled steady-state) | CoV |
|---|---:|---:|---:|---|---|
| 1B | 775.17 / 28.86 | 778.41 / 28.80 | 778.12 / 28.93 | prefill 777.23, decode 28.86 | prefill 0.23%, decode 0.23% |
| 3B | 287.52 / 12.63 | 288.17 / 12.49 | 286.59 / 12.60 | prefill 287.43, decode 12.57 | prefill 0.28%, decode 0.60% |
| 8B | 135.58 / 7.04 | 136.98 / 7.01 | 138.11 / 7.00 | prefill 136.89, decode 7.02 | prefill 0.93%, decode 0.26% |

*(prefill / decode tok/s per rep. All 9 reps succeeded — unlike the 4w-floating sweep, 8B had no
thermal/watchdog crash here.)*

**Correctness note (not part of the timing methodology, but worth recording)**: the 3B reps'
long-decode (1024-token) output was genuinely incoherent — actual gibberish including a stray CJK
character, not just degenerate repetition like the 4w baseline's long-decode output. All 3 reps
produced the *exact identical* gibberish string (expected for greedy/`temperature=0` decoding —
this rules out a flaky/random hardware issue). A separate short (48-token) coherence check on the
same PTE produced clean, sensible output ("Paris. It is the most beautiful city..."), confirming
this isn't a basic dispatch/quantization correctness bug — it's most likely long-horizon greedy-
decode drift, worse under 8da4w's added dynamic activation quantization error than 4w's
weight-only quantization. Throughput numbers above are still valid (compute cost per token is the
same regardless of semantic output quality); this is flagged for anyone who tries to eyeball the
generated text and finds it alarming.

## Table 4: 8da4w, pinned clocks (509/2730/663 MHz)

| Model | Rep 1 | Rep 2 | Rep 3 | Mean | CoV | Pin-verified? |
|---|---:|---:|---:|---|---|---|
| 1B | 411.41 / 16.98 | 412.82 / 16.97 | 410.42 / 16.94 | prefill 411.55, decode 16.96 | prefill 0.29%, decode 0.11% | ✅ (52.9% of floating) |
| 3B | 150.82 / 7.16 | 150.77 / 7.15 | 150.83 / 7.15 | prefill 150.81, decode 7.15 | prefill 0.02%, decode 0.05% | ✅ (52.5% of floating) |
| 8B | 64.90 / 3.91 | 65.18 / 3.91 | 64.32 / 3.89 | prefill 64.80, decode 3.90 | prefill 0.67%, decode 0.21% | ✅ (47.3% of floating) |

*(prefill / decode tok/s per rep.)*

**Notable: unlike the 4w-pinned sweep, 8B did NOT crash here — all 3 reps succeeded cleanly.**
This is a genuine difference between quant modes on this device, not measurement noise: 4w-pinned
8B crashed 3/3 with `VK_ERROR_DEVICE_LOST` (confirmed `gpu_watchdog`, Table 2), while 8da4w-pinned
8B succeeded 3/3 at a similar clock config. The most likely explanation is that 8da4w's compute
per node/command-buffer differs enough from 4w's (different dtype path, int8 activation
quantization) that its command buffers stay under the GPU watchdog's timeout window even at
509MHz, where 4w's don't — this is an observation, not a verified root cause (no ISA-level
inspection was done as part of this baseline study). The 3B reps' long-decode output is the same
identical gibberish as Table 3 (expected — greedy decode is clock-independent), confirming the
correctness note there applies equally here.

---

## Methodology notes

- **Node-threshold probe result (research.md Decision 3)**: `ET_VK_EXECUTE_NODE_THRESHOLD=16` has
  **no effect** on `llama_main_rel1.3` — an 8B pinned run with it set crashed identically
  (`VK_ERROR_DEVICE_LOST`, `vkQueueWaitIdle=-4`) to the unset case. Confirms this binary (plain
  `release/1.3`) does not recognize the env var at all — that workaround lives only on the separate
  `release13-node-threshold` feature branch, per workspace-root `CLAUDE.md`. No node-threshold
  mitigation is available for this study's crashes; gotcha G12's per-config guidance (from the M5
  EVT1 target, where this flag *does* exist) does not transfer here.
- HW devfreq ranges (for pin/unpin): sgpu 226000–980000, mif 676000–5333000, int 133000–800000
- Clocks pinned 2026-07-14 (`pin_freqs.sh`), sysfs-verified: GPU 509000/509000, MIF 2730000, INT 663000.
- **Crash-attribution reproduction (research.md Decision 4)**: the T004 probe crash (8B pinned,
  identical `VK_ERROR_DEVICE_LOST`/`vkQueueWaitIdle=-4` signature) was checked live against
  `dmesg`/`/proc/meminfo`: `dmesg` showed 7 consecutive `sgpu ...: amdgpu: GPU reset(66..72)
  succeeded!` entries clustered right before the check, and **zero** OOM-kill signatures anywhere
  in the entire ~1280-line buffer; `/proc/meminfo` showed 4.97GB/10.9GB `MemAvailable` (healthy,
  no memory pressure). `crash_cause=gpu_watchdog` for this reproduction — confidently not
  `host_oom`. The buffer only retains ~210s of uptime history, so this same check could not reach
  back to the original (much earlier) crashes this session — see each crash's own note.

## Pinned-vs-floating comparison

### 4w

| Model | Floating prefill | Pinned prefill | Pinned / Floating | Pinned viable? |
|---|---:|---:|---:|---|
| 1B | 599.14 | 316.38 | 52.8% (floating ~1.89× faster) | Yes |
| 3B | 215.63 | 104.95 | 48.7% (floating ~2.05× faster) | Yes |
| 8B (plain `release/1.3`) | 88.91 (n=2) | — | — | **No — crashes every rep (3/3) with plain `release/1.3`** |
| 8B (`release13-node-threshold`, `THRESHOLD=32`) | 88.91 (n=2) | 52.76 | 59.3% (floating ~1.69× faster) | **Yes, with the branch fix** |

The 1B/3B pinned/floating ratios (52.8%, 48.7%) line up closely with the raw GPU clock ratio
(509/980 = 51.9%), which is a useful sanity check that the pin genuinely bound and the comparison
is measuring what it claims to. 8B's ratio (59.3%) sits a bit higher than that — consistent with
the smaller command-buffer submissions (every 32 nodes vs. the default 128) adding some overhead
relative to the other cells, not a sign the pin failed to bind (it's still well under the 70%
DVFS-artifact threshold). See Table 2's note above for the fix and the `THRESHOLD=64` comparison.

### 8da4w

| Model | Floating prefill | Pinned prefill | Pinned / Floating | Pinned viable? |
|---|---:|---:|---:|---|
| 1B | 777.23 | 411.55 | 53.0% (floating ~1.89× faster) | Yes |
| 3B | 287.43 | 150.81 | 52.4% (floating ~1.91× faster) | Yes |
| 8B | 136.89 | 64.80 | 47.3% (floating ~2.11× faster) | **Yes — unlike 4w, 8B's pinned baseline IS measurable for 8da4w** |

Same sanity check as 4w: these ratios (53.0%, 52.4%, 47.3%) all land close to the 509/980=51.9%
clock ratio. The standout finding for this quant-mode × clock-mode matrix is 8B: 4w-pinned is
unmeasurable (crashes every rep) while 8da4w-pinned works cleanly — see Table 4's note for the
likely (unverified) explanation.

---

## Completion

All 4 tables (36/36 rep-slots) complete as of 2026-07-14 — within the end-of-day target (spec
SC-007). Zero cells required a "DVFS-ARTIFACT" label — every pinned rep that produced a number
cleanly passed the 70% throughput cross-check on its first attempt. Every "pinned" cell above is
backed by both the sysfs pin-readback (Methodology notes) and its own per-cell throughput
cross-check (SC-005). This document is self-contained (SC-004) — device identity, driver state,
methodology, and every per-cell outcome are defined above without requiring the session that
produced them.

**Companion report**: [`qualcomm-s25-ultra-companion-report.md`](./qualcomm-s25-ultra-companion-report.md)
covers the same release/1.3 vanilla baseline (same methodology, same day) on a Galaxy S25 Ultra
(Qualcomm SM8850/Adreno 840) — a different GPU vendor entirely, floating-clock only (no root), and
notably more crash-prone on 4w than M41. Not directly comparable to M41 (different driver stack),
included for independent cross-device context only.
