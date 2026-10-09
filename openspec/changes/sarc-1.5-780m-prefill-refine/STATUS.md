# STATUS: 780M prefill campaign

## Round 3 (2026-10-08): `fused3sb` and `780m-final` (closing task, not a tuning round)

Updated 2026-10-09 02:36 UTC. **Not closed: one owner decision is open ("Decision needed from the owner (open,
2026-10-09)" below the table).** Round 3 is measured on the replacement build `head4` as the owner decided
(option (a), 2026-10-08 22:55 UTC), and the reported numbers were recomputed by the review of 2026-10-09; but two
predicates of the R6 validity rule are not evidenced for any timed run, and the owner's decision kept the
validity rule as it is. Nothing is running and nothing is queued; no measurement is started until the owner
answers. Chains `chain27.sh` (23:03 to 23:27 UTC) and `chain28.sh` (23:27 to 02:08 UTC) ended with `DONE`
(`results/780m/round3/chain27.status`, `chain28.status`). Only the hold watcher is alive (and the `nvtop` of
the ssh session described below, which is not the campaign's).

**Which build each number comes from:** the part "Replacement build `head4`" directly below is `head4`
(`c639d4760`, recursive export from object stores) and is what closes the round. Everything from "State before
the decision" downwards was measured on `head3` (submodules copied from the working copy), is kept as written,
and is evidence for `head3` only; its item table and its "Decision needed from the owner" are superseded by the
part below.

| item | state on `head4` |
|---|---|
| A.3 build of the branch head | `head4` = `c639d4760`, exported from object stores only, no local patch; `spirv_golden.py` PASS (53 shipped variants); 1,469 of 1,469 SPIR-V files of `head2` (candidate 11's gate) byte-identical, 2 new (the `fused3sb` pair); 1,471 of 1,471 identical to `head3` |
| A.4 gate with `fused3sb` | `verify.sh` rc = 0, equal to candidate 11's gate apart from the tok/s figures; tiers `all` / `extended` / `full` 12 passes each: 192 cases, 0 failed, 0 mismatches, `pairing=ok`; next token SAME in 18 of 18 items (six cells, three prompts); SDPA output byte-identical to `fused3` in 21 of 21 cases (and the 5 of tier `fused`) |
| A.5 session `c11` with `fused3` against `c11` with `fused3sb` | **-0.12 % geomean** (cells -0.41 to 0.00 %), 60 of 60 timed runs valid on the recorded predicates: inside the band |
| B.1 dispatch of `780m-final` alone | equals candidate 11's gate: `verify.out` identical line for line (tok/s set aside); the tiers dispatch the `fused3sb` pair; three-kernel path as `c11` |
| B.2 session `780m-final` against `dev/1.5` | **+33.92 % geomean** (+23.62 to +48.43 %), 60 of 60 timed runs valid on the recorded predicates; round 2 measured +33.82 %: inside the band |
| C | `proposal.md` section "Round 3" says which build each number comes from; `check.sh --no-build` PASS (02:10 UTC, output below; run by the actor only, it compiles the two selector tests); committed and pushed. **Round not closed: R6 validity, see the decision below** |
| **recommended configuration** | **`ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=780m-final`** |

### Decision needed from the owner (open, 2026-10-09)

**R6 validity is not fully evidenced for the timed runs of round 3, on `head4` or on `head3`.** R6: "A run is
valid only with rc 0, 2048 prompt tokens, 0 generated tokens, no other GPU workload before, during or after,
enough clock samples inside the prefill window (5 ...), a median clock at or above the calibrated floor, and no
thermal throttle reason." The decision of 2026-10-08 22:55 UTC says "Nothing else changes: thresholds,
tolerances, prompts, the gate, the number of runs"; it does not waive a predicate. "60 of 60 valid" in this file
means the predicates `tools/e2e5.sh` records. Missing:

1. **No other GPU workload *during* a run.** `e2e5.sh` lists processes matching its pattern once before and
   once after each run (empty in all 168 rows of the two `head4` sessions); nothing is sampled while the runner
   executes.
2. **No thermal throttle reason.** The sampler writes clock, busy, power and temperature every 0.1 s; no
   throttle status is read.
3. A defect of the tool, without effect on these counts: `e2e5.sh` rejects a run for too few clock samples only
   below 2 (`clock_unsampled`), where R6 asks for 5. Every timed row of the four sessions has at least 5
   (recomputed from the `.clk` files), so no row was accepted that the rule's 5 would reject.

What the saved data and the host do show (read 2026-10-09 02:20 to 02:35 UTC, nothing started on the GPU):

- Clock: per-run median 2749.5 to 2800 MHz (item A) and 2730 to 2800 MHz (item B) against the top level of
  2799 MHz and the floor of 2700; lowest single sample inside a timed prefill window 2636 MHz (A), 2643 MHz (B).
  Peak temperature 92 C (A), 94 C (B). `hwmon` exposes no critical-temperature threshold for this device
  (`temp1_crit` absent).
- The device does expose a throttle status that the tools never read: `gpu_metrics` (format 2.1, 120 bytes),
  field `throttle_status` (u32 at offset 108); it reads `0x00000000` now, idle at 44 C. It is a current value,
  not a history: it says nothing about the runs after the fact.
- `thermal_throttling_logging` of the device reads "enabled, with interval 60 seconds". The kernel log of this
  boot has no throttling or thermal line after boot; its last line is of 2026-10-08 23:12:39 UTC (`hrtimer:
  interrupt took 1974 ns`), before the first `head4` session. I have not established that the driver emits such
  a line for this APU, so the silence is an indication, not evidence of absence.
- Processes on the host during the sessions (journal 23:27 to 02:09 UTC): hourly cron, `dnf-makecache` at
  00:14:45 UTC (2 s, 455 ms CPU; during the tiers of item A, not during a timed session), ssh logins. The
  gdm greeter (`gnome-shell`, since boot) holds the display on this GPU as in every session of the campaign.
- **Found now, not reported before: an `nvtop` (3.3.2) was started in an interactive ssh session (pts/0, from
  100.77.177.104) at 01:18:45 UTC and is still running.** It was alive during the whole of item B on `head4`
  (timed session 01:26 to 01:59, `verify.sh`, the tier passes); item A and its gate (23:35 to 00:56) ended
  before it started. I did not start it and have not touched it. It is a monitor: it holds `/dev/dri/renderD128`
  open (DRM client 1041, 2 MiB GTT, 12 KiB VRAM) and its `fdinfo` shows no `drm-engine-*` line, i.e. no engine
  time; 34 s of CPU time in 64 minutes. The guard's pattern does not name `nvtop`, so `others` / `others_post`
  are empty with it running. Whether an idle monitor counts as "other GPU workload" is the owner's to say
  (lesson L25 speaks of recording one, not of aborting). Item B's numbers with it running: +33.92 %, parent arm
  within 0.14 % of round 2's.

**The owner's choice:**

- **(a) a ruling on the missing predicates:** accept the timed runs of `head4` on the recorded predicates plus
  the indications above (as the sessions of rounds 1 and 2 were accepted, which have the same two gaps), and say
  whether item B stands with `nvtop` running; or
- **(b) authorisation for replacement timed sessions with monitoring** (recommended if the rule is to hold as
  written): items A and B again on `head4` (same binaries, about 35 minutes each plus cooling; the gate, tiers
  and byte comparison do not depend on these predicates and would not be repeated unless the owner asks), with
  `tools/e2e5.sh` changed, campaign-local, as follows and nothing else: the 0.1 s sampler also writes
  `throttle_status` (and the temperature) from `gpu_metrics`; a run with any non-zero status in its window is
  invalid with reason `throttle`; the process list of the guard and the DRM clients readable to this user
  (`/proc/*/fdinfo`, engine time per client) are sampled during the run and a foreign client with engine time
  makes the run invalid; the clock-sample minimum becomes the rule's 5. Thresholds, floor, band, prompts and run
  counts unchanged; the existing sessions stay on record as they are. Limits to state now: the greeter's
  `fdinfo` belongs to another user and cannot be read without sudo, so "no foreign engine time" would cover
  this user's processes plus the device-wide `gpu_busy_percent`; and the meaning of the bits of
  `throttle_status` on this APU would be recorded as raw values, not interpreted. The owner would also need to
  say whether `nvtop` may keep running.

State 02:36 UTC, after the second review of this record: the task file still ends with the owner's note of
00:20 UTC, no answer to this choice; nothing was started and nothing is queued; `nvtop` (PID 245296) is still
running. The review recomputed both `head4` tables, the export, the SPIR-V comparison, the gates and, this time,
the fp64 reference figures of the 26 dumps, and keeps the round open on this point only. It could not read
`/proc/245296/fdinfo` (permission denied in its context), so "no engine time for `nvtop`" is the actor's reading
alone (repeated 02:36 UTC: still no `drm-engine-*` line), and it does not accept the kernel-log and `fdinfo`
indications as a substitute for monitoring during the runs. Neither do I: they are context for the owner's
ruling, not evidence of validity.

Also still open, not blocking by itself: the specification sentence the owner's note of 2026-10-09 00:20 UTC
asks for (the `vulkan-docs` server failed to connect again at 02:22 UTC, `CONNECTION_CLOSED`); and the review's
not-checked list (`check.sh --no-build` was run by the actor only; the RX 7600's numbers quoted in `proposal.md` are that
campaign's, not checked here).

### Replacement build `head4` (2026-10-08 23:00 UTC to 2026-10-09 02:08 UTC)

Evidence: `results/780m/round3/head4/`, `results/780m/sessions/{r3a-fused3sb-head4,r3b-final-dev15-head4}/`,
`results/780m/fused/kernel-time-steady-r3-fused3sb-head4.csv`; raw data `<artifacts 10-08>/{src,build}/head4`,
`raw/head4/`, `stage/*-head4/`. Scripts as run: `round3/chain27.sh`, `chain28.sh`, `tools/export_recursive.sh`.

**Export (R5).** `tools/export_recursive.sh c639d4760`: `git archive` of the commit from this clone's object
store; the 23 submodules and their 7 nested ones each fetched at exactly the commit its parent tree pins (URL
from the `.gitmodules` of that parent commit, `fetch --depth 1 <url> <sha>`, `fsck`) into bare repositories under
`<artifacts 10-08>/submodules/`, and written from there with `read-tree` + `checkout-index`. Nothing is read from
a working tree. Manifest: `round3/head4/build-head4.EXPORT-MANIFEST` (31 lines: path, pinned commit, store, URL,
tree, file count, content hash). Checked twice: by the chain (`export-check.txt`, 30 of 30 trees
`MATCHES-COMMIT`), and again after the build by walking the tree of `c639d4760` through the gitlinks into the
stores and hashing every exported file as a git blob: 45,204 blobs expected, 45,204 equal, 0 different, 0
missing; the 30 pins of the manifest equal the gitlinks. The 17 files in the export directory that no tree
names are what the build wrote there afterwards (14 `.pyc`, `third-party/flatcc/{bin/flatcc,lib/*.a}`).

**Did `head3` differ?** (`round3/head4/submodules-head4-vs-head3.txt`, `diff -r` per submodule, nested ones
included):

| submodule | `head4` export against the directory `head3` was built from |
|---|---|
| 21 of 23 (`FACTO`, `mlx`, `Vulkan-Headers`, `VulkanMemoryAllocator`, `volk`, `FP16`, `FXdiv`, `XNNPACK`, `cpuinfo`, `pthreadpool`, `eigen`, `shim`, `ao`, `flatbuffers`, `gflags`, `googletest`, `ios-cmake`, `json`, `pocketfft`, `prelude`, `pybind11`) | identical, file for file |
| `extension/llm/tokenizers` | no file differs; `head3` has two untracked directories more (`build`, `pytorch_tokenizers.egg-info`) |
| `third-party/flatcc` | no file differs; `head3` has two untracked directories more (`bin`, `lib`) |

Outside the submodules `head3` has four `__pycache__` directories under `codegen/` more. So every tracked file
`head3` was built from equals the pinned commits; it carried leftover untracked directories that `head4` does
not. The binaries of the two builds are not byte-identical (`llama_main` sha256 `578935be...` on `head4`,
`705ac48f...` on `head3`; `round3/head4/binaries-sha256.txt`); all 1,471 SPIR-V files are.

**Build.** `sarc/tools/build.sh --llama` and `--llama --traced --no-tests` of the export, in the campaign's
container, rc = 0 both (23:03 to 23:11 UTC; nothing else ran). `spirv_golden.py`: `spirv golden: PASS (53
shipped variants)` for `llama/` and `backend/`. Against `head2`: 1,469 files, identical 1,469, different 0,
missing 0, new 2 (`round3/head4/spirv/`). The stage directories of both sessions, both arms and `verify.sh`
hold the `head4` binaries (same sha256).

**Kernel names** (`round3/head4/dispatch.txt`): as the table "Kernel names" below, with `head4` for `head3`, in
every row. `verify.out` under `780m-refine3` + `c11` and under `780m-final` alone: 0 differing lines against
candidate 11's gate and against the same stage on `head3` (tok/s set aside); against the parent snapshot
`s0-parent-verify` lines 2 and 3 differ (the linear kernel names), as for candidate 11. Smoke, tier `all`: no
environment -> release QK^T / softmax / attn*V, `fused=-`; `c10` and `780m-final` + `c10` -> the `fused3` pair;
`c11`, `780m-final` with and without `ET_VK_SARC_UNVERIFIED=1` -> the `fused3sb` pair; fused node off under
`c11` and under `780m-final` -> `sweep_t128x64k32g22s64nf`, softmax `..._780m_r3`, `sweep_t64x64k32g42s32`.

**Item A, `r3a-fused3sb-head4`** (session 23:35 to 00:08 UTC; start 43 C after a 430 s wait). Same binary in
both arms; parent `c11` with `ET_VK_SARC_780M_SDPA_FUSED` naming the `fused3` pair, candidate `c11` as committed.
Recomputed from `runs.csv`, the per-run logs and the clock files:

| cell | `c11` with `fused3` | `c11` with `fused3sb` | difference | repeat spread parent / candidate | next token (`prompt_2048` / `prompt_check` / `r1304`) |
|---|---:|---:|---:|---|---|
| 1B 4w | 3842.40 | 3835.21 | -0.19 % | 0.19 / 0.19 % | SAME / SAME / SAME |
| 1B 8da4w | 3764.71 | 3764.71 | 0.00 % | 0.18 / 0.18 % | SAME / SAME / SAME |
| 3B 4w | 1458.69 | 1458.69 | 0.00 % | 0.07 / 0.14 % | SAME / SAME / SAME |
| 3B 8da4w | 1422.22 | 1416.32 | -0.41 % | 0.48 / 0.49 % | SAME / SAME / SAME |
| 8B 4w | 640.60 | 640.60 | 0.00 % | 0.53 / 0.44 % | SAME / SAME / SAME |
| 8B 8da4w | 628.99 | 628.22 | -0.12 % | 0.21 / 0.25 % | SAME / SAME / SAME |
| geomean | | | **-0.12 %** | | |

Inside the +-2 % band in every cell (`head3`: -0.01 %). Gate, candidate environment, the timed binary:
`verify.sh` unmodified, one invocation, rc = 0 (00:49 to 00:54 UTC; correctness rc = 0, 12 of 12
production-diff ALL PASSED, default vs tiled SAME on both prompts, decode 31 tokens); tiers `all` / `extended` /
`full`, 12 passes each: 48 / 96 / 48 passed, 0 failed, `mismatches=0` in 192 of 192, `pairing=ok` in 192 of 192
kernel lines, only the `fused3sb` pair dispatched; control with the table kernels 1 pass per tier, 16 of 16.
SDPA output `fused3` against `fused3sb`, byte for byte: **identical in 26 of 26 dumps** (the 21 cases of
`all`, `extended`, `peaked`, `full` and the 5 of `fused`), and each of the 52 dumps identical to its `head3`
dump; the 26 `[sdpa-error]` lines equal between the arms and equal to `head3`'s (8B head configuration, S =
2048: rms 1.033182e-05, max 3.693156e-04). Fused kernel time at a steady clock (median of 3 runs):
2266 -> 2268 us a layer (1B, +0.07 %), 3221 -> 3214 (3B, -0.21 %), 4051 -> 4062 (8B, +0.25 %). Warm traces, one
run per arm, candidate second without cooling: copy / view / other -1.2 to +2.2 %, linear GEMM (same kernels)
+1.2 to +2.0 %.

**Item B, `r3b-final-dev15-head4`** (session 01:26 to 01:59 UTC). Same binary in both arms; parent with no
environment, candidate `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=780m-final`. The wait for 43 C ended at
its 1,800 s limit with the device at 44 C (it had just run the traces); the runs start at 44 to 51 C, as in
item A (44 to 50 C).

| cell | `dev/1.5` dispatch | `780m-final` | gain | `head3` | round 2 (`s9-final-dev15`) | published `dev/1.5` | repeat spread parent / candidate | next token (`prompt_2048` / `prompt_check` / `r1304`) |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 1B 4w | 2694.74 | 3835.21 | **+42.32 %** | +42.32 % | +42.51 % | 2698.29 | 0.26 / 0.37 % | SAME / SAME / SAME |
| 1B 8da4w | 2540.94 | 3771.64 | **+48.43 %** | +48.16 % | +48.35 % | 2544.10 | 0.25 / 0.18 % | SAME / SAME / SAME |
| 3B 4w | 1140.95 | 1458.69 | **+27.85 %** | +27.71 % | +27.85 % | 1151.21 | 0.06 / 0.14 % | SAME / SAME / SAME |
| 3B 8da4w | 1049.18 | 1405.63 | **+33.97 %** | +34.04 % | +33.72 % | 1051.33 | 0.10 / 0.07 % | SAME / SAME / SAME |
| 8B 4w | 517.56 | 639.80 | **+23.62 %** | +23.09 % | +23.42 % | 525.80 | 0.40 / 0.28 % | SAME / SAME / SAME |
| 8B 8da4w | 487.39 | 628.41 | **+28.94 %** | +28.94 % | +28.73 % | 489.13 | 0.26 / 0.15 % | **DIFFER** / SAME / SAME |
| geomean | | | **+33.92 %** | +33.77 % | +33.82 % | | | |

Against round 2: +0.10 points in the geomean, cells within 0.26 points; parent arm within 0.14 % and candidate
arm within 0.19 % of round 2's. The parent arm is within 1.57 % of the published numbers (8B 4w; the others
within 0.9 %). The one differing item, 8B 8da4w on `prompt_2048.txt`, is candidate 8's: **ACCEPTED
(reference-error rule, owner decision 2026-10-04)**, evidence `results/780m/probe/`, `results/780m/sdpa-error/`;
the output of each arm in this cell is byte-identical to the same arm of `s9-final-dev15` and of `head3`. No new
differing item. `verify.sh` with the `780m-final` environment (01:59 to 02:04 UTC): rc = 0, the lines of
candidate 11's gate. One pass of each tier: 4 / 8 / 4 passed, 0 failed, 0 mismatches, `pairing=ok`.

**Validity, and what it does and does not show.** Each session has 60 timed rows, 5 per arm per cell, parent
first on odd repeats; 60 of 60 are valid in both on the predicates `e2e5.sh` records: rc 0, 2,048 prompt tokens,
0 generated, no other GPU process found before the run and none after it (`others`, `others_post`: empty in all
168 rows), at least 5 clock samples in the prefill window, median clock at or above 2700 MHz (item A 2749.5 to
2800, peak 92 C; item B 2730 to 2800, peak 94 C). **Not recorded, as in rounds 1 and 2:** GPU processes during a
run (the list is taken before and after, not sampled), and a thermal throttle reason (the sampler has clock,
busy, power and temperature; the device's `gpu_metrics` has a `throttle_status` field, which the tools do not
read). This is the open decision above; the count is not a statement of full R6 validity. Invalid rows, all kept:
none timed; of the 24 untimed next-token runs per session (no `--warmup`, compared by output only) 4 in item A
and 6 in item B carry `clock_low` (2608 to 2684 MHz).

**Page cache (D5, as decided).** `e2e5.sh` and `trace.sh` read the model file before the first process of each
cell and record `fincore` residency per run; `verify.sh` is run once with the six files read first
(`verify-residency.txt`: all six fully resident before and after it, in both items; before the read of item A
the 1B 4w file was 67 % resident). Residency just before each process: 100.00 % in 159 of 168 session rows;
99.86 to 99.90 % in 9 rows of item A, all 8B 4w (about 4 to 6 MB of 4.17 GB not resident), whose load times
(2,922 to 3,090 ms) are inside that cell's range. Model load times, all 168 rows: 1B 345 to 383 ms, 3B 734 to
899 ms, 8B 2,922 to 3,268 ms; **no slow load and no abort** (rc 0 in all 168; the 12 trace runs completed, each at 100.00 % residency). For
comparison, on `head3`, where the file was not re-read per cell: item A 350 to 11,940 ms with one slow load
(`prefill-8b-8da4w-parent-r1`, 11,940 ms, the first process of its cell; the other runs of the cell 3,100 to
3,300 ms; rc 0), item B 345 to 3,265 ms.

**Checks (02:10 UTC).** `sarc/tools/check.sh --no-build`, unedited, with the `head4` evidence files in the tree:

```
== 1 zone rule vs origin/release/1.5
== 2 twin wrappers
== 3 test_sarc_select
test_sarc_select: PASS (1240 checks, 31 rows, 0 candidates, dev zone absent, unverified off)
[sarc_dev] overrides active: unverified=1 variant= dq8ca_variant=
test_sarc_select: PASS (1433 checks, 31 rows, 122 candidates, dev zone linked, unverified on)
check.sh: PASS
```

Since `c639d4760` (the commit that was built) the branch changed only files of this change directory
(`STATUS.md`, `proposal.md`, `tools/`, `results/`): `git diff --name-only c639d4760 HEAD` lists nothing outside
it, so the head builds the same sources. `tools/rgp_chunks.py` is untracked and in `.git/info/exclude`.

**Owner note of 2026-10-09 00:20 UTC (`vulkan-docs` MCP server).** No shader was written or changed after the
note (the `fused3sb` files are the RX 7600 campaign's blobs, taken unchanged in `8d909b3fb`). The server did not
connect in the session that wrote this (`CONNECTION_CLOSED`), so the barrier reasoning in `proposal.md` is not
yet backed by a quoted specification sentence; it stands as written on 2026-10-08, from reading, not from the
server.

### Record: the start of the replacement, as written 2026-10-08 23:03 UTC

**Running now (started 23:03 UTC, detached, one after the other): `chain27.sh`, then `chain28.sh`** (copies in
`results/780m/round3/`; status files `<artifacts 10-08>/logs/chain27.status`, `chain28.status`).
`chain27.sh`: check of the export `src/head4` (written 23:00 UTC by `tools/export_recursive.sh c639d4760`: `git
archive` of the commit, the 23 submodules and 7 nested ones fetched at their pinned commits into bare
repositories under `<artifacts 10-08>/submodules/` and written from there; nothing from a working tree), builds
`head4` and `head4-traced`, `spirv_golden.py`, SPIR-V against `head2` and `head3`, staging, dispatch smoke, SDPA
output byte comparison, kernel time. `chain28.sh`: item A on `head4` (timed session, tiers 12 passes, `verify.sh`,
traces), then item B (timed session, `verify.sh`, tiers). About 2.5 hours; no build overlaps a timed session (one
chain). Tools changed for it, as decided: `tools/e2e5.sh` and `tools/trace.sh` read the model file before each
cell and record its `fincore` residency per run; `e2e5.sh` also records the runner's model load time and the GPU
processes after each run, and compares the next token on `r1304.txt` too (1,792 tokens; the open point 3 below).
`verify.sh` is not edited: one invocation, the six files read first, residency recorded before and after.
Everything below this paragraph was measured on `head3` and is evidence for `head3` only.

State before the decision, as written 22:31 UTC: not closed, blocked on an owner decision (build provenance, R5); see "Decision needed from the owner" at the end of this section. Task file: `~/hmz-sarc/CAMPAIGN-round3.md`. Raw data of this round:
`rocky-ryzen:~/hmz-sarc/.artifacts/780m-prefill-refine-2026-10-08/`; evidence in `results/780m/round3/` and
`results/780m/sessions/{r3a-fused3sb,r3b-final-dev15}/`; summary in `proposal.md`, "Round 3".

As of 22:31 UTC nothing ran and nothing was queued. The two detached chains of this round (`chain25.sh` 19:44 to 20:07 UTC, `chain26.sh`
20:08 to 22:09 UTC; copies and their status files in `results/780m/round3/`) have ended. Only the hold watcher
(`hold.sh watch`, started again 19:44 UTC; it did not survive the reboot) is alive; it starts no GPU job.

| item | state |
|---|---|
| A.1 the two `fused3sb` files of `b3bb758e38` | taken unchanged (same blobs), commit `8d909b3fb`. Diff against `fused3`: the header comment and 13 inserted `subgroupBarrier();` lines, nothing else (listed below). **The kernel has 13 `memoryBarrierShared()` calls, not 14** as the task file says; the fourteenth text match is the sentence in the new header |
| A.2 `kFused780mOnePassSb`, `c11` uses it | commit `8d909b3fb`; `c8`, `c9`, `c10` unchanged |
| B.1 dev profile `780m-final` | commit `c639d4760`; `c11` still works |
| A.3 build of the branch head | **does not satisfy R5** (submodules copied from the working copy, not exported from object stores). `head3` = export of `c639d4760` for the repository itself, no local patch; `spirv_golden.py` PASS (53 shipped variants); all 1,469 SPIR-V files of `head2` (candidate 11's gate) byte-identical, 2 new (the `fused3sb` pair) |
| A.4 gate with `fused3sb` | **passed on `head3`** (to be repeated on a compliant build; model files not re-read at each model change): `verify.sh` rc = 0 and equal to candidate 11's gate apart from the tok/s figures; tiers `all` / `extended` / `full` 12 passes each, 192 cases, 0 failed, 0 mismatches, `pairing=ok`; next token SAME in 12 of 12 items; SDPA output **byte-identical to `fused3` in 21 of 21 cases** (and in the 5 of tier `fused`) |
| A.5 session `c11` with `fused3` against `c11` with `fused3sb` | **-0.01 % geomean** (cells -0.09 to +0.07 %), 60 of 60 timed runs valid: inside the band |
| B.1 dispatch of `780m-final` alone | equals candidate 11's gate: `verify.out` identical line for line (34 lines, tok/s set aside); SDPA tiers dispatch the `fused3sb` pair of the same tokens; the three-kernel path has the `780m-refine3` QK^T / attn*V and softmax `780m_r3` (table below) |
| B.2 session `780m-final` against `dev/1.5` | **+33.77 % geomean** (+23.09 to +48.16 %), 60 of 60 timed runs valid; round 2 measured +33.82 %: inside the band |
| C `proposal.md`, `check.sh --no-build`, commits, push | **open until the blocker is resolved**; section "Round 3 (2026-10-08): fused3sb and 780m-final"; `check.sh: PASS` (output below); branch pushed |
| **recommended configuration** | **`ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=780m-final`** (measured on `head3` only) |

Host as found 2026-10-08 19:38 UTC: idle, no campaign process, no `HOLD`, branch = origin at `90fe4d013`; kernel
6.12.0-211.62.1.el10_2, RADV PHOENIX Mesa 25.2.7, GPU `auto` (800 / 1100 / 2799 MHz), 39 C. Booted 2026-10-07
05:58 UTC (the task file says 03:48 UTC; the journal has boots at 02:48 and 05:58 UTC). No profiler variable in
any process of this round.

### The shader diff, every differing line (`results/780m/round3/shader-diff-fused3-fused3sb.txt`)

`diff glsl/sarc_dev/sarc_dev_780m_sdpa_fused3.glsl glsl/sarc_dev/sarc_dev_780m_sdpa_fused3sb.glsl`: 21 added
lines, none removed, none changed.

- Header, 8 comment lines inserted after line 9 (lines 10 to 17 of the new file): the paragraph that says this
  is the RX 7600 campaign's copy of the 780M's file for merge-plan item M2a, and the line `Original header:`.
- 13 lines `subgroupBarrier();`, each directly after a `memoryBarrierShared();` and with its indentation:

  | after line of `fused3` | line of `fused3sb` | where |
  |---:|---:|---|
  | 261 | 270 | pass A, after `qk_block`, before the lanes read their scores |
  | 272 | 282 | pass A, end of the block loop body |
  | 275 | 286 | pass A, after the store of the lane's row maximum to `Rsh` |
  | 279 | 291 | pass A, after the read of the row's `Rsh` slots |
  | 291 | 304 | pass B, after `qk_block` |
  | 309 | 323 | `ONLINE`, after the store of the block maximum to `Rsh` |
  | 313 | 328 | `ONLINE`, after the read of the row's `Rsh` slots |
  | 324 | 340 | `ONLINE`, after the store of the rescale divisors to `Dsh` |
  | 332 | 349 | `ONLINE`, after the `coopMatLoad` of the divisor tile and the rescale |
  | 357 | 375 | pass B, after the lanes stored e to `Psh`, before `av_block` |
  | 359 | 378 | pass B, after `av_block` |
  | 364 | 384 | after the store of the lane's row sum to `Rsh` |
  | 372 | 393 | after the store of the row sums to `Psh`, before the final `coopMatLoad` |

  In `fused3sb` no `memoryBarrierShared();` is left without a `subgroupBarrier();` on the next line. Every one of
  them is in subgroup-uniform control flow (block loops bounded by the workgroup's row tile, `if (SEGS > 1u)` a
  constant, the rescale branch taken on `subgroupAny`).
- yaml: the header comment, the family name (`sarc_dev_780m_sdpa_fused3sb`), and 2 variants instead of 33
  (`..fused3sb_d64_t32x32g11s32rko`, `..fused3sb_d128_t16x64g11s32rko`) whose parameters equal their `fused3`
  twins; defaults and `generate_variant_forall` identical.

Nothing but barriers and the header differs, so the task's stop condition does not apply. Read for
unsynchronised shared writes before the gate: no location has two writers (a lane stores to `Psh[e_idx + i]`,
`Rsh[e_row * SEGS + e_seg]`, and the `Dsh` / `Psh` divisor slots `j` of its row with `j mod SEGS = e_seg`);
every read of a slot another lane or a `coopMatStore` wrote follows a memory barrier and a subgroup barrier.

### Build `head3` (19:44 to 19:52 UTC)

Export of `c639d47605ca1f2b5314e4b6afc6a58752386796`: `git archive` of the commit into
`<artifacts 10-08>/src/head3/executorch`; the 23 submodule directories copied from the working copy with the
pinned commit and a content hash each in the manifest (`results/780m/round3/build-head3.EXPORT-MANIFEST`),
because this clone has no submodule object stores (`.git/modules` is empty; the earlier builds of the campaign
built the working copy in place, with the same directories). Built with the export's own `sarc/tools/build.sh
--llama` (and `--traced --no-tests`) in `localhost/et-vk-build:rocky10`; rc = 0 for both. `STATUS.md` was the
only modified file in the working copy at that time and is not part of the export.

- `sarc/tools/spirv_golden.py` on `head3` (`llama/` and `backend/` shader directories): `spirv golden: PASS (53
  shipped variants)`.
- Every `.spv` of `head2` (the build of candidate 11's gate and of the final sessions of round 2), `llama/` and
  `backend/`: 1,469 files, identical = 1,469, different = 0, missing = 0. New in `head3`:
  `sarc_dev_780m_sdpa_fused3sb_d64_t32x32g11s32rko_buffer_buffer_half.spv` (sha256 `01ee25a0...`) and
  `..fused3sb_d128_t16x64g11s32rko_buffer_buffer_half.spv` (`745587fc...`) (`round3/spirv/`).

### Kernel names (`results/780m/round3/dispatch.txt`)

| environment (build) | `verify.sh` `linear 4w` / `linear 8da4w` lines | SDPA kernels on the tiers | three-kernel path (fused node switched off with an empty `ET_VK_SARC_780M_SDPA_FUSED`) |
|---|---|---|---|
| candidate 11's gate: `780m-refine3` + `c11` (`head2`) | 4w `t128x128k32g24s32f32cbt` x 2, `t128x256k32g42s32f32cbt` x 1, `t256x128k32g18s32f32cbt` x 6, `..g24..cbt` x 2, `..g28..cbt` x 1 (texture3d), release tile x 12 (buffer); 8da4w `zpg_t256x64k64g48s32afmb1` x 6 and `zpg_bt_t128x64k32g22s32` x 6 (texture3d), `zpg_bt_..` x 12 (buffer) | `fused3_d64_t32x32g11s32rko`, `fused3_d128_t16x64g11s32rko` (tiers last run for `h1-c10`) | |
| `780m-refine3` + `c11` (`head3`) | the same two lines, character for character | `fused3sb_d64_t32x32g11s32rko`, `fused3sb_d128_t16x64g11s32rko` | QK^T `sweep_t128x64k32g22s64nf`, softmax `..._780m_r3`, attn*V `sweep_t64x64k32g42s32` |
| `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=780m-final`, `ET_VK_SARC_780M_PROFILE` unset (`head3`) | the same two lines | the same `fused3sb` pair | the same three |
| `ET_VK_SARC_DEV_PROFILE=780m-final` without `ET_VK_SARC_UNVERIFIED` (`head3`) | not run through `verify.sh` | the same `fused3sb` pair | |
| `780m-refine3` + `c11` + `ET_VK_SARC_780M_SDPA_FUSED` naming the `fused3` pair (`head3`; the parent arm of `r3a`) | same linear kernels (same profile) | the `fused3` pair | |
| `780m-refine3` + `c10`; `780m-final` + `ET_VK_SARC_780M_PROFILE=c10` (`head3`) | | the `fused3` pair in both | |
| no environment (`head3`) | | release QK^T / softmax / attn*V, `fused=-` | |

(The 4w line was read from the file; the `x n` counts are the production-diff cases per kernel.) `verify.out`
under `780m-final` and under `c11` on `head3` each equal candidate 11's `verify.out` line for line (34 lines)
once the tok/s figures are set aside. So `780m-final` alone dispatches what candidate 11's gate dispatched,
with the two fused kernel names changed by item A and nothing else.

### Item A: `r3a-fused3sb` (20:09 to 21:23 UTC)

Same binary (`head3`) in both arms on `ET_VK_SARC_DEV_PROFILE=780m-refine3 ET_VK_SARC_780M_PROFILE=c11`; parent
arm with `ET_VK_SARC_780M_SDPA_FUSED=fused3_d64_t32x32g11s32rko,fused3_d128_t16x64g11s32rko`, candidate arm as
committed. Model files read into the page cache first. Start 43 C after a 90 s wait, 5 valid runs per arm,
interleaved; tok/s recomputed from `runs.csv`:

| cell | `c11` with `fused3` | `c11` with `fused3sb` | difference | repeat spread parent / candidate | next token (2048 / unaligned prompt) |
|---|---:|---:|---:|---|---|
| 1B 4w | 3842.40 | 3842.40 | 0.00 % | 0.19 / 0.19 % | SAME / SAME |
| 1B 8da4w | 3764.71 | 3764.71 | 0.00 % | 0.18 / 0.18 % | SAME / SAME |
| 3B 4w | 1459.73 | 1458.69 | -0.07 % | 0.07 / 0.14 % | SAME / SAME |
| 3B 8da4w | 1412.41 | 1413.39 | +0.07 % | 1.10 / 0.89 % | SAME / SAME |
| 8B 4w | 640.60 | 641.00 | +0.06 % | 1.21 / 0.75 % | SAME / SAME |
| 8B 8da4w | 629.57 | 628.99 | -0.09 % | 0.25 / 0.12 % | SAME / SAME |
| geomean | | | **-0.01 %** | | |

60 timed runs, 60 valid (rc 0, 2048 prompt tokens, 0 generated, no other GPU process, at least 5 clock samples,
median clock 2774 to 2800 MHz against the floor of 2700; run starts at 43 to 48 C, peak 90 C). Of the 12 untimed
next-token runs on the unaligned prompt, 3 carry `clock_low` (2641 to 2698 MHz; no warm-up, as in every session
of round 2); they are compared by output only.

Gate (candidate environment, the timed binary):

- `verify.sh --models 1b,3b,8b --schemes 4w,8da4w --pdiff`, unmodified: rc = 0. Against candidate 11's gate:
  identical apart from the tok/s figures. Against the parent snapshot `s0-parent-verify`: lines 2 and 3 (the
  linear kernel names) differ, the other 32 are identical apart from tok/s: correctness rc = 0, `linear <scheme>
  rc=1` as on the parent, 12 of 12 production-diff cases ALL PASSED, default vs tiled SAME on both prompts,
  decode 31 tokens.
- SDPA tiers `all`, `extended`, `full`, 12 passes each: 48 / 96 / 48 cases passed, 0 failed, `mismatches=0` in
  all 192, `pairing=ok` in every kernel line, `fused3sb` the only SDPA kernel dispatched; control with the table
  kernels 1 pass per tier, 16 of 16 (`sessions/r3a-fused3sb/sdpa-correctness/`).
- SDPA output, `fused3` against `fused3sb`, byte for byte (`round3/bitwise/bitwise-fused3sb.txt`): **IDENTICAL
  in 21 of 21 cases** of tiers `all`, `extended`, `peaked`, `full` (up to 16.8 MB each, the three production
  head configurations at S = 2048 among them) and in the 5 cases of tier `fused`. Error against the fp64
  reference, both arms, e.g. 8B head configuration at S = 2048: rms 1.033182e-05, maximum 3.693156e-04 in
  both (`round3/bitwise/sdpa-error.txt`, 26 cases, no line differs between the arms). Not an arithmetic
  change: neither owner decision of 2026-10-04 is used.
- Fused kernel time at a steady clock, copy pass included (40 + 10 runs, 3 runs per arm,
  `results/780m/fused/kernel-time-steady-r3-fused3sb.csv`): 2268 -> 2271 us a layer (1B), 3219 -> 3210 (3B),
  4050 -> 4049 (8B).
- Warm traces, one run per arm: copy / view / other (holds the copy pass and the fused kernel) -0.4 to +1.3 %,
  linear GEMM (same kernels) +0.8 to +1.7 %: the candidate arm is traced second without cooling, as in round 2.

### Item B: `r3b-final-dev15` (21:30 to 22:09 UTC)

Same binary (`head3`) in both arms: parent with no environment (nothing selected: the release table kernels,
what `dev/1.5` dispatches; shown for the hooks by `h0-control` in round 2), candidate with
`ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=780m-final`. Model files read into the page cache first. Start
43 C after a 391 s wait, 5 valid runs per arm, interleaved; recomputed from `runs.csv`:

| cell | `dev/1.5` dispatch | `780m-final` | gain | round 2, candidate 11 (`s9-final-dev15`) | published `dev/1.5` | repeat spread parent / candidate | next token (2048 / unaligned prompt) |
|---|---:|---:|---:|---:|---:|---|---|
| 1B 4w | 2694.74 | 3835.21 | **+42.32 %** | 2691.20 -> 3835.21, +42.51 % | 2698.29 | 0.13 / 0.19 % | SAME / SAME |
| 1B 8da4w | 2540.94 | 3764.71 | **+48.16 %** | 2537.79 -> 3764.71, +48.35 % | 2544.10 | 0.12 / 0.18 % | SAME / SAME |
| 3B 4w | 1142.22 | 1458.69 | **+27.71 %** | 1140.95 -> 1458.69, +27.85 % | 1151.21 | 0.17 / 0.07 % | SAME / SAME |
| 3B 8da4w | 1052.96 | 1411.44 | **+34.04 %** | 1049.72 -> 1403.70, +33.72 % | 1051.33 | 0.21 / 0.07 % | SAME / SAME |
| 8B 4w | 519.93 | 640.00 | **+23.09 %** | 517.43 -> 638.60, +23.42 % | 525.80 | 0.23 / 0.34 % | SAME / SAME |
| 8B 8da4w | 487.39 | 628.41 | **+28.94 %** | 487.85 -> 628.03, +28.73 % | 489.13 | 0.10 / 0.25 % | **DIFFER** / SAME |
| geomean | | | **+33.77 %** | +33.82 % | | | |

60 timed runs, 60 valid (median clock 2748 to 2800 MHz, run starts at 44 to 50 C, peak 93 C); 4 of the 12
untimed unaligned-prompt runs carry `clock_low`. Against round 2: geomean -0.05 points, cells within 0.33
points; the parent arm is within 1.12 % of the published numbers (`sarc-1.5-e2e-benchmark/results/cells.csv`).

The differing item, 8B 8da4w on `prompt_2048.txt`, is candidate 8's (**ACCEPTED (reference-error rule, owner
decision 2026-10-04)**; evidence `results/780m/probe/`, `results/780m/sdpa-error/`): the generated output of
each arm in this cell is byte-identical to the same arm of `s9-final-dev15` (parent ends ` the`, candidate
`V`), and identical to the `fused3` arm of `r3a`. No new differing item.

`verify.sh` with the `780m-final` environment: rc = 0, the 34 lines of candidate 11's gate. One pass of each
SDPA tier with it: 4 / 8 / 4 passed, 0 failed, `pairing=ok` (`sessions/r3b-final-dev15/`).

### Checks (22:11 UTC)

`sarc/tools/check.sh --no-build`, unedited:

```
== 1 zone rule vs origin/release/1.5
== 2 twin wrappers
== 3 test_sarc_select
test_sarc_select: PASS (1240 checks, 31 rows, 0 candidates, dev zone absent, unverified off)
[sarc_dev] overrides active: unverified=1 variant= dq8ca_variant=
test_sarc_select: PASS (1433 checks, 31 rows, 122 candidates, dev zone linked, unverified on)
check.sh: PASS
```

The counts are those of the close of round 2 (1240 / 31 and 1433 / 122): the round adds no candidate row.
`git diff --name-status 90fe4d013 HEAD` outside this change directory: `A glsl/sarc_dev/sarc_dev_780m_sdpa_fused3sb.glsl`,
`A glsl/sarc_dev/sarc_dev_780m_sdpa_fused3sb.yaml`, `M impl/sarc_dev/Overrides.cpp`: dev zone only. No
release-zone file, nothing under `sarc/tools` or `sarc/golden`, no tolerance, prompt, threshold or tool of the
campaign changed in this round (`tools/` is byte-identical to the copy that ran). `tools/rgp_chunks.py` stays
untracked. (Correction, 22:59 UTC: commit `b3ec12057` added `tools/rgp_chunks.py` to git by mistake, against the
task file's "Leave it untracked"; found by the review. It was taken out of the index again with `git rm --cached`
in the next commit (`3942401f5`), the local file kept, history not rewritten. The commit after that, `aead6ed25`,
added it once more through a `git add` of the whole change directory; taken out again in the commit that follows it,
and the path is now in this clone's `.git/info/exclude` so that a directory-wide `git add` cannot pick it up. The
file is in the trees of `b3ec12057` and `aead6ed25` only.)

### Decision needed from the owner

**Answered: the owner decided option (a) on 2026-10-08 22:55 UTC (task file, last section); points 1 to 3 were
carried out on `head4` (top of this file). Point 4 (validity predicates that are not evidenced) was not waived and
is the decision that is open now, at the top of this file. The text below is the question as it was put on
2026-10-08.**

**Round 3 is not closed. Two findings of the review of 2026-10-08 block it; nothing further is queued until the
owner answers** (task file, "If something takes a different turn ... stop"). Nothing is running; no build and no
GPU job was started after the review.

1. **BLOCKING: build `head3` does not satisfy R5.** R5 asks for an export of one commit "and, recursively, of
   every submodule at the commit it pins, taken from the git object stores, never from the live working tree".
   The repository itself was exported with `git archive` (all 11,406 blobs of `c639d4760` equal the export, checked
   by the reviewer), but the 23 submodules were copied with `cp -a` from the working copy
   (`round3/chain25.sh` lines 28 to 31; `source=working-copy-directory` on every line of
   `round3/build-head3.EXPORT-MANIFEST`). This clone has no submodule object stores (`.git/modules` does not
   exist), so it cannot be shown that those directories are the pinned commits. Calling this "nothing blocks"
   in the first version of this section was wrong. Everything measured on `head3` (gate `r3a-fused3sb`, session
   `r3b-final-dev15`, the SPIR-V comparison, the byte comparison) is kept as measured and is **evidence for
   `head3` only**, not for a build that satisfies R5.

   The owner's choice:
   - **(a) replacement build, recommended.** Fetch the 23 submodules (and their nested ones) at the pinned
     commits into bare repositories under `<artifacts 10-08>/submodules/` (network access to the upstream URLs
     of `.gitmodules`; `git ls-remote` answers from this host, checked 2026-10-08 for two of them; the working
     copy is not touched), export `c639d4760` recursively from object stores only, build it under a new tag
     (`head4`), and compare each exported submodule tree with the directory `head3` used, so that it is known
     whether `head3` differed at all. Then, on `head4`, everything again: `spirv_golden.py`, the SPIR-V
     comparison with `head2`, the dispatch comparison, the byte comparison of the SDPA output, the gate of item A
     (`verify.sh`, tiers `all` / `extended` / `full` at 12 passes, next token), the session of item A and the
     session, `verify.sh` and tiers of item B. Cost as measured today: export and build about 10 min plus the
     fetch, pre-gate checks 15 min, item A 75 min, item B 40 min; about 2.5 hours of device time.
   - **(b) a dated owner exception** appended to the task file, accepting `head3` with submodules from the
     working copy (as every build of rounds 1 and 2 was made).

2. **Protocol deviation, R6 / owner decision 2026-10-06 (D5): the model file was not read into the page cache
   at each model change.** `chain26.sh` (lines 22 and 26) reads all six model files once before the gate of
   item A and once before the session of item B. Inside them the model changes without another read: the cell
   loop of `tools/e2e5.sh` (lines 114 to 128), `sarc/tools/verify.sh` and `tools/trace.sh`. What is on record:
   rc = 0 in all 144 runs of the two sessions and no abort line in any log; the host has 28 GB of RAM for 12.4 GB
   of model files, and `fincore` showed all six fully resident when checked after everything (about 22:20 UTC). Whether a file
   was resident at each model change was not recorded, so a slow load cannot be excluded for any single run.

   For a replacement validation I would, with the owner's agreement (these are the campaign's own tools under
   `tools/`, listed as changes in `proposal.md`; nothing under `sarc/tools`, no threshold, tolerance or prompt):
   - `tools/e2e5.sh` and `tools/trace.sh`: `cat <model> > /dev/null` before the first process of each cell, the
     same for both arms, and the `fincore` residency of the model recorded per run;
   - `sarc/tools/verify.sh` is not edited. It would be run once per model (`--models 1b`, then `3b`, then `8b`),
     with the read before each, and its three outputs concatenated in model order for the line-by-line
     comparison; or, if the owner prefers the single invocation of the snapshot, run as before with the six
     files read first and their residency recorded before and after. **Which of the two is the owner's call.**

3. **Open, second review of 2026-10-08: parent-versus-candidate next token on two prompts, not three.** R7 asks
   for all six cells on the timed, the real-text and the unaligned prompt. `tools/e2e5.sh` (lines 126 to 131)
   compares `prompt_2048.txt` and the 1,972-token `prompt_check.txt`; nothing compares parent and candidate on
   `r1304.txt`. `verify.sh` runs that prompt only as default versus tiled, for 1B. The tables above head the
   second column "unaligned prompt", as the tables of round 2 do: it is `prompt_check.txt`. Every session of
   rounds 1 and 2 has the same two-prompt coverage. For a replacement validation I would add the six-cell
   parent-versus-candidate runs on `r1304.txt` for both items with a campaign-local script (the same runner
   command as `e2e5.sh`'s check runs, same lock and guard), **unless the owner rules that two prompts suffice.**
4. **Not evidenced, same review:** the sessions record other GPU processes once before each run (not during or
   after it) and record temperatures and the clock, but no thermal throttle reason; "60 of 60 valid" means the
   predicates `e2e5.sh` records, as in rounds 1 and 2. The absolute error values against the fp64 reference are
   the test binary's log lines (the second reviewer did not reconstruct them; the first did). `check.sh
   --no-build` was run by the actor only.

**State 2026-10-08 22:31 UTC: the task file still ends without an owner decision; nothing is queued and nothing
runs.** With no exception granted, the plan is 1 (a) with 2 and 3 corrected, started only on the owner's word.

For the record, not blocking:

5. The kernel has 13 barrier pairs, not 14 (above). Nothing else differs, so the work was not stopped for it.
6. The task defines the `dev/1.5` arm of item B as the branch head with no profile; that is what was measured
   (same binary). The pristine `dev/1.5` build `ef079ac41` of round 2 was not re-timed; the arm agrees with it
   within 0.48 % per cell.

Everything below this section is the record of round 2 as it was closed on 2026-10-07.

# STATUS (round 2, closed): 780M prefill campaign, round 2 (parameter space + beyond)

Updated 2026-10-06 18:10 PDT (2026-10-07 01:10 UTC). Parent for this round: profile `780m-refine3` (build `topic-r1`).
Artifacts: `rocky-ryzen:~/hmz-sarc/.artifacts/780m-prefill-refine-2026-10-04/` (new raw data) and
`.../780m-prefill-refine-2026-10-03/` (earlier builds and sessions).

## State

| item | state |
|---|---|
| candidate 7 (softmax r3) | **complete**: gate passed, bit-identical to its parent, six cells +2.39 to +6.78 %, geomean **+4.10 %** over `780m-refine3` |
| candidate 8 (fused SDPA kernel, Part 2) | **ACCEPTED (reference-error rule, owner decision 2026-10-04)**: +7.84 to +20.14 %, geomean **+13.16 %** over candidate 7, 60 valid runs; gate finished 00:04 PDT; one next-token item differs (8B 8da4w, `prompt_2048.txt`); record below. It needs `hooks/sdpa-fused-hook.patch`, so adopting it is the owner's decision |
| igpu-roofline `fast` plan | finished 20:15 PDT; matrix roofs 14.766 TFLOP/s (fp16 -> fp32) and 14.379 TOP/s (int8) |
| candidate 9 (candidate 8 + one-pass fused kernel + 4w kernel per shape) | **gate passed** (finished 04:10 PDT): +4.50 to +5.45 % in the 4w cells, +1.06 to +1.80 % in the 8da4w cells (inside the band), geomean **+3.19 %** over candidate 8; every next-token item SAME; record below |
| 4w, Part 1 | **done** (final table in "Part 1, 4w", second confirmation): seeded sample of 2,500, three refinement rounds, a geometry scan; 260 configurations measured in full, the 41 in a top ten each with at least 5 full measurements and 36 of 36 production-diff passes; profile `refine10` is the fastest or within 0.62 % of it on all twelve shapes, 5.3 to 6.1 % less 4w linear time per layer than `780m-refine3` |
| candidate 10 (candidate 9 with the refined 4w table, profile `refine10`) | **gate passed, +0.47 % geomean: inside the band, not a gain** (4w cells +0.94 / +1.43 / +1.16 %, 8da4w cells 0.00 / -0.73 / +0.06 %); byte-identical 4w output; first candidate under 2 % |
| 8da4w, Part 1 | **done** (792 of 792 production-diff passes ALL PASSED): all 2,238 survivors screened, 49 measured in full, 22 five times; best kernel per shape below ("Part 1, 8da4w"), 2.1 to 3.1 % less 8da4w linear time per layer than `780m-refine3` |
| candidate 11 (candidate 10 + the 8da4w kernel per shape, profile `c11`) | **gate passed, +1.20 % geomean** (8da4w cells +2.02 / +2.82 / +2.36 %, 4w cells 0.00 / 0.00 / +0.06 %): the second consecutive candidate under 2 %, **stop rule met**; kept as the final configuration |
| production-diff passes | **done**: 2,700 of 2,700 passes ALL PASSED (75 configurations x 3 models x 12), the six kernels of the final profile among them |
| QK^T / attn*V, Part 1 | **done**: all 1,724 + 470 survivors measured (0 failures), 44 configurations five times with 12 correctness passes (0 failed cases); best kernel per head dimension in "Part 1, SDPA": QK^T 9 to 30 % and attn*V 8 to 10 % faster than the `780m-refine3` choices on the three-kernel path; not in a profile, not gated |
| campaign | **closed 2026-10-07 00:48 UTC**: stop rule met, Part 1 complete for all four families, final configuration measured, branch pushed |
| final configuration (candidate 11), measured directly | **+33.82 % geomean over `dev/1.5`** (+23.42 to +48.35 %), **+24.12 % over `780m-refine3`** (+14.56 to +35.83 %); section "Final configuration". **Since round 3 (2026-10-08) the final configuration is `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=780m-final`**: candidate 11 with the `fused3sb` pair, +33.77 % geomean over `dev/1.5` measured directly (`r3b-final-dev15`, section "Round 3" at the top) |
| release-zone hooks (owner decision 2026-10-05) | committed (`b969e8f1c`, `1c8861aa7`, dev side `8518659ef`); candidate 10 reproduced from the committed build: gate passed, **+22.64 %** geomean over `780m-refine3` measured directly (section "Committed build") |

Chained over the three sessions (`t1-recheck`, `c7-softmax-r3`, `c8-fused3`; not a direct measurement, that comes
at the end): candidate 8 is about +27 % geomean over `dev/1.5` (1B 4w 2698 -> 3625 tok/s, 8B 8da4w 488 -> 606).

## Coordinator hold (owner decision 2026-10-06 00:35 UTC): in place since 2026-10-05 17:50 PDT

- The coordinator creates and removes **`/home/doremy/hmz-sarc/.artifacts/HOLD`** (on `rocky-ryzen`).
- The queue answers with **`/home/doremy/hmz-sarc/.artifacts/HELD`**: one line
  `HELD <UTC time> <what it would start next>` per waiting queue, written only when nothing of the campaign is
  running; removed when `HOLD` is gone. While `HOLD` exists nothing new starts; the poll is once a minute; a unit
  that is running when `HOLD` appears finishes normally; afterwards the queue continues where it was (every step
  is resumable from its CSV).
- Mechanism (`tools/hold.sh`): every unit holds a shared lock on `.artifacts/.hold-busy` while it runs and looks
  at `HOLD` again after taking it (and after taking the gpu-lab lock), so `HELD` can only be written when an
  exclusive lock on that file is free: no unit is running and none can start. A watcher (`hold.sh watch`,
  detached, pid in `<artifacts 10-05>/logs/hold-watch.out`'s process) writes `HELD` when `HOLD` appears while the
  queue is idle or waiting on a marker, so the coordinator is never left waiting.
- Units, and how long one lasts at most:

  | unit | where the hold is checked | lasts |
  |---|---|---|
  | one sweep configuration (`sweep_space.py`: screen, confirmation timing, SDPA enumeration) | before every configuration | 3 to 40 s; hard limit 600 s per process |
  | one microbench job through `gl.sh` (production-diff pass, SDPA correctness pass, SDPA perf run) | before every job | seconds; up to 3.5 min (`--sdpa-tier=full`) |
  | one warm trace set (`trace.sh`, 12 runs) | at its start | about 5 min |
  | one `verify.sh` (not editable, so wrapped as a whole) | at its start | 8 to 15 min |
  | one build (`build-both.sh`, `build_space.sh`; CPU only) | at its start | 2 to 25 min |
  | **one timed session** (`e2e5.sh`: 5 valid runs x 6 cells x 2 arms, interleaved) | at its start; the cool start is taken again after a hold | 35 min measured, **at most about 50 min: the longest unit** |

  A timed session is not split: a foreign workload between its interleaved runs would make it a different
  session, and it could not be resumed without discarding it.
- The existing `PAUSE` marker is **not** this mechanism and is unchanged: it lives in the dated directory, is set
  by the campaign's own gates against its own sweep, writes no `HELD`, and the confirmation steps ignore it.
  `HOLD` is checked in addition, by everything, and `--ignore-pause` does not skip it.
- Tested 2026-10-06 00:44 to 00:46 UTC with the test name (`SARC_HOLD_NAME=HOLD-TEST`, same code path, 2 s
  poll): a unit running when `HOLD-TEST` appeared finished; no `HELD-TEST` while it ran; then
  `HELD 2026-10-06T00:44:40Z test unit B (would start next)`; unit B did not start until `HOLD-TEST` was removed,
  then ran within 1 s and `HELD-TEST` was gone. Watcher alone: `HELD-TEST` with "queue idle" after 5 s, removed
  when the test file was removed. The sweep's Python path was tested the same way (below, "Running now").
- The running queue picked it up at a job boundary without being killed: the confirmation repeat that was running
  (`full-r4`, started before the change) finished with the old code; `full-r5` and everything after it start as
  new processes with the hold. While `HELD` exists I start no GPU job and no build by hand either.

## Host hang and reboot, 2026-10-05 08:53 PDT (recorded 10:45 PDT, before anything was restarted)

The host stopped responding at 08:53 PDT (last journal entry of that boot 08:53:42) and was rebooted by hand at
09:34 PDT. Same kernel (6.12.0-211.61.1.el10_2), same driver (RADV PHOENIX, Mesa 25.2.7), GPU at `auto` with the
800 / 1100 / 2799 MHz levels as before.

**Owner decision 2026-10-05: no profiler captures on this device.** The RGP request of the same morning is
withdrawn. No `MESA_VK_TRACE*`, `RADV_THREAD_TRACE_*` or `RADV_PROFILE_PSTATE` in any process; `tools/gl.sh`
refuses to start a job that carries one. Evidence stays ETDump, shader-clock phase timing, kernel timing and
igpu-roofline.

What was running when the host went down (from `<artifacts>/rgp/`, `raw/dq/screen.{csv,out}`,
`logs/sweep-pauses.txt`; `<artifacts>` = `.../780m-prefill-refine-2026-10-04`):

| job | state at 08:53 | lost |
|---|---|---|
| `rgp/take.sh` (RGP captures with `MESA_VK_TRACE=rgp`, per submit, 512 MB thread-trace buffer, cache counters; one microbench SDPA run each, under the gpu-lab lock) | captures 1 and 2 (three kernels, release softmax and softmax r3) finished at 08:53:17 and 08:53:23. Capture 3 (fused two-pass) started 08:53:26 and its run returned rc = 0; capture 4 (fused one-pass) had started (its three files are stamped 08:53:31 to 08:53:33) when the machine hung | the files of captures 3 and 4 are zero-length (never reached the disk). Nothing from any capture is used anywhere in this change |
| `chain6e.sh`: 8da4w screen of the 2,238 survivors, sharing the GPU run by run with the captures | 261 configurations done, last row 08:53:25 | the process. `raw/dq/screen.csv` is intact (783 rows, 21 fields each, all dispatched); the sweep resumes from it |
| queued behind it in the same chain: 8da4w confirmation, production-diff passes, QK^T / attn*V enumeration | not started | the queue (restarted below) |

Not lost: every gated session (the candidate-10 gate finished at 08:49 and was committed at 08:50); the staged
binaries of `c7` to `c10` still match the sha256 in their `STAGE.md`; `git fsck` is clean; no result file written
near the hang is truncated.

Where the capture material is (kept as evidence, nothing calls it):

- `<artifacts>/rgp/take.sh.DISABLED-by-owner-2026-10-05` and `cap.sh.DISABLED-by-owner-2026-10-05`: the capture
  scripts, renamed and not executable, with `README-DISABLED.txt`. No chain script, tool or timer references
  them (checked with grep over `<artifacts>/*.sh`, both `tools/` directories; no crontab, no user timers).
- `<artifacts>/rgp/<capture>/`: the capture files and what was extracted from the first two.
  `/tmp/rgpmb_2026.10.05_08.53.3{1,2,3}.rgp`: the files RADV was writing for capture 4, left where they are.
- `<artifacts>/fx/rgpmb`: a copy of the microbench under another name (so that the capture files were named
  after it); an ordinary binary, not used by anything else.
- `tools/rgp_chunks.py` in this directory (untracked, not committed): a reader for `.rgp` files. It sets no
  variable and starts no GPU job; nothing calls it.

## Running now

**Nothing. The campaign is closed** (2026-10-07 00:48 UTC; review follow-up on the 4w evidence 00:56 to 01:07 UTC,
"Part 1, 4w", second confirmation): no detached job of the campaign is running except
the hold watcher (`hold.sh watch`, which only answers a coordinator `HOLD` with `HELD` "queue idle").

- Stop rule met: candidates 10 (+0.47 %) and 11 (+1.20 %) are two consecutive gated candidates under 2 % geomean.
- Final configuration: candidate 11, `ET_VK_SARC_DEV_PROFILE=780m-refine3 ET_VK_SARC_780M_PROFILE=c11` on the
  committed branch: **+33.82 % geomean over `dev/1.5`, +24.12 % over `780m-refine3`** ("Final configuration").
- Part 1 is complete for all four families: 4w ("Part 1, 4w"), 8da4w ("Part 1, 8da4w"), QK^T and attn*V
  ("Part 1, SDPA").
- Part 2: softmax r3 (candidate 7), the fused attention node (candidates 8, 9); upstream operator costs reported
  in `proposal.md`, not changed.

Production-diff passes (12 per configuration and model, texture3d, M = 2048, the real shapes, the
configuration's own kernel on 4 of 4 shapes in every pass; 8da4w with non-zero zero-points): **all done, all
passed.**

| list | configurations | passes | result |
|---|---:|---:|---|
| 8da4w confirmation (`results/780m/space/confirm-8da4w/pdiff.csv`) | 22 | 792 | 792 ALL PASSED |
| 4w, first confirmation (`confirm-4w/pdiff.csv`) | 35 | 1,260 | 1,260 ALL PASSED |
| 4w, second confirmation (`confirm2-4w/pdiff.csv`; the 16 not in the first list) | 16 | 576 | 576 ALL PASSED |
| 4w, review follow-up (`confirm3-4w/pdiff.csv`; the two top-ten configurations the second confirmation left out) | 2 | 72 | 72 ALL PASSED |

The six kernels of the final profile `c11` are in those lists with 36 of 36 passes each: 4w
`t256x128k32g18s32f32cbt`, `..g24..cbt`, `..g28..cbt`, `t128x128k32g24s32f32cbt`, `t128x256k32g42s32f32cbt`;
8da4w `t256x64k64g48s32afmb1`; and the `780m-refine3` 8da4w kernel (`bt_t128x64k32g22s32afmb2`).

Faults of my own queue on 2026-10-06, found and corrected:

- The second 4w confirmation list (four of the five final 4w kernels) was not in the production-diff queue,
  which only knew the first list. Its passes were run separately (10:33 to 11:46 UTC, table above).
- The SDPA enumeration's first 36 runs (10:30 and 11:46 to 11:50 UTC) were recorded with `dispatched = 0`,
  `NOT_DISPATCHED`: `sweep_space.py` did not parse the kernel line of the batch binaries rebuilt on 2026-10-05,
  which has a `fused=` field. Nothing was ranked from them; they are in
  `<artifacts 10-04>/superseded/sdpa-steady-kernel-line-not-parsed/`, the parser is fixed, and the enumeration
  restarted from zero.
- Found by the reviewer, corrected 2026-10-07: the second 4w confirmation took 6 per shape instead of 10, so two
  top-ten configurations had 2 full measurements and no production-diff pass; the committed copy of
  `confirm2-4w/full-r4.csv` was 48 rows short; and `proposal.md` still gave the `refine9` kernels as the 4w
  result. All three are corrected ("Part 1, 4w"); no selected kernel and no reported tok/s changes.
- My estimates of the enumeration's end (22:00, then 22:30 UTC) were wrong: 1,796 runs, not 1,724, at 22.6 s a
  run. It ended at 23:15 UTC; the repeat stage at 00:48 UTC.

The coordinator's hold was used once: `HOLD` 2026-10-06 00:53:06 UTC, `HELD` 00:53:12 UTC, `HOLD` removed about
02:08 UTC; the queue removed `HELD` and continued with the configuration it had named, by itself.

### After the reboot: re-check and A/A (`<artifacts 10-05>/stage/t4-recheck`, `t5-aa`; 10:28 to 11:38 PDT)

`t4-recheck`, pristine `dev/1.5` build against build `topic-r1` with `780m-refine3`, 5 valid runs per arm, cool
start: 1B 2694.74 -> 2832.64 (+5.12 %), 2537.79 -> 2828.73 (+11.46 %); 3B 1146.70 -> 1192.08 (+3.96 %), 1060.04 ->
1168.28 (+10.21 %); 8B 522.32 -> 545.99 (+4.53 %), 488.43 -> 549.80 (+12.56 %) (4w, 8da4w); geomean **+7.92 %**
(before the reboot: +7.91 %). Clock 2771 to 2800 MHz, all next-token items SAME. The device is as it was.

`t5-aa`, candidate 10 (build `fused9`) on both arms, 3 repeats: cells -0.09 to +0.19 %, geomean 0.00 %.
(`summary.csv` of that session says INCOMPLETE because the tool expects 5 runs; the medians are from `runs.csv`.)

### The hook commits, and the edit that was interrupted (found and fixed 11:15 to 11:43 PDT)

The control session was stopped by the owner's coordinator in the middle of the dev-zone edit that follows the
two hook commits (`b969e8f1c` softmax variant name, `1c8861aa7` fused attention node). The working tree was
inspected before anything else: the edit was textually complete but **not working**. The first build of it
(`build/wt1`) selected the softmax variant and never fired the fused node (`raw/wt1-smoke/`: `fused=-` with
profiles `c8` and `c10`, and with the explicit variable). Cause: `Sdpa780mFused.cpp`, now its own file under
`780m/`, registered its two functions from its own static initializer; the shared `Registrar` of `Overrides.cpp`
starts from an empty `Override` and ran afterwards. Fixed inside the 780m block (the pair is kept there and
applied by whichever initializer runs last), rebuilt (`build/wt2`), and checked with every profile
(`raw/wt2-smoke/`): no profile -> release kernels and release softmax; `c7` -> softmax `..._780m_r3`; `c8` ->
`fused3_..rk`; `c9`, `c10` -> `fused3_..rko`; the names of the measured builds. `check.sh --no-build` passes.
Committed as `8518659ef`. Nothing was measured with the broken build.

## Final configuration, measured directly (2026-10-05 21:07 to 22:34 PDT)

Candidate 11 = committed branch head with `ET_VK_SARC_DEV_PROFILE=780m-refine3 ET_VK_SARC_780M_PROFILE=c11`
(build `head2` of `a8fffa5ea`, no local patch). Two sessions, each from a cool start (45 C and 43 C after the 30
min wait; the device idled at 44 to 45 C that evening), 5 valid runs per arm, arms interleaved, clock 2728 to
2800 MHz: `s9-final-dev15` (pristine `dev/1.5` build `ef079ac41`, no environment) and `s10-final-refine3`
(`780m-refine3`, same binary as the candidate). Evidence: `results/780m/sessions/{s9-final-dev15,s10-final-refine3}/`.

| cell | `dev/1.5` | candidate 11 (s9) | gain over `dev/1.5` | `780m-refine3` | candidate 11 (s10) | gain over `780m-refine3` |
|---|---:|---:|---:|---:|---:|---:|
| 1B 4w | 2691.20 | 3835.21 | **+42.51 %** | 2828.73 | 3842.40 | **+35.83 %** |
| 1B 8da4w | 2537.79 | 3764.71 | **+48.35 %** | 2824.83 | 3771.64 | **+33.52 %** |
| 3B 4w | 1140.95 | 1458.69 | **+27.85 %** | 1188.62 | 1458.69 | **+22.72 %** |
| 3B 8da4w | 1049.72 | 1403.70 | **+33.72 %** | 1166.95 | 1412.41 | **+21.03 %** |
| 8B 4w | 517.43 | 638.60 | **+23.42 %** | 540.23 | 640.00 | **+18.47 %** |
| 8B 8da4w | 487.85 | 628.03 | **+28.73 %** | 549.21 | 629.19 | **+14.56 %** |
| geomean | | | **+33.82 %** | | | **+24.12 %** |

Repeat spread 0.07 to 0.37 % (3B 8da4w candidate in s9: 1.11 %). Next token against either parent: SAME in eleven
of twelve items; 8B 8da4w on `prompt_2048.txt` DIFFERS, the item of candidate 8 (**ACCEPTED (reference-error
rule, owner decision 2026-10-04)**, `results/780m/probe/`, `results/780m/sdpa-error/`). The `dev/1.5` arm agrees
with `sarc-1.5-e2e-benchmark/results/cells.csv` and with the re-check after the reboot within 1 %.

Where it comes from (warm traces of both arms of `s9`, ms per prefill; `dev/1.5` -> candidate 11):

| | 1B 4w | 1B 8da4w | 3B 4w | 3B 8da4w | 8B 4w | 8B 8da4w |
|---|---:|---:|---:|---:|---:|---:|
| total | 758 -> 537 | 801 -> 544 | 1799 -> 1419 | 1949 -> 1460 | 3998 -> 3247 | 4237 -> 3304 |
| QK^T + softmax + attn*V | 234 -> 0 | 235 -> 0 | 397 -> 0 | 397 -> 0 | 600 -> 0 | 600 -> 0 |
| copy / view / other (holds the fused kernel and its copy pass) | 61 -> 97 | 36 -> 71 | 134 -> 225 | 86 -> 177 | 228 -> 361 | 137 -> 268 |
| linear GEMM | 375 -> 354 | 401 -> 343 | 1088 -> 1015 | 1183 -> 999 | 2854 -> 2570 | 3017 -> 2553 |
| elementwise (upstream) | 68 | 68 | 129 | 131 | 244 | 243 |

Attention is 198 / 306 / 467 ms less (three kernels replaced by the fused node: 234 -> 36, 397 -> 91, 600 -> 133
ms), linear GEMM 21 / 73 / 284 ms less in 4w and 58 / 184 / 464 ms less in 8da4w.

Percent of the re-measured roofs (run `2026-10-04-fast-prefill-refine2`), candidate arm of that trace, by shape:
4w linear 9.6 to 12.0 TFLOP/s = 65 to 81 % of `matrix_fp16_fp32` (14.766); 8da4w linear 9.9 to 12.1 TOP/s = 69 to
84 % of `matrix_int8` (14.379); the fused attention kernel about 71 % of `matrix_fp16_fp32` (kernel timing,
"Candidate 8").

## Committed build: candidate 10 reproduced from the branch alone (owner decision 2026-10-05, release-zone hooks)

Build `head1` = branch head `8518659ef`, clean tree, no local patch (`<artifacts 10-05>/build/head1`). Evidence in
`results/780m/sessions/{h0-control,h1-c10}/` and `results/780m/hooks/`.

**Nothing selected: every dispatch as before.**

- `test_sarc_select`: PASS, 1240 checks / 31 rows (release tables) and 1432 checks / 121 candidates (dev zone),
  the counts of the parent. `spirv_golden.py` on `head1`: PASS (53 shipped variants).
- All 1,325 shaders of the pristine `dev/1.5` build (`build/parent`) are byte-identical in `head1`.
- `h0-control`: `head1` with no environment, unmodified `verify.sh`, against the pristine `dev/1.5` control
  (`s0-parent-verify`): the 34 lines are identical once the measured tok/s figures are set aside (kernel names,
  return codes, 12 production-diff cases ALL PASSED, default vs tiled SAME on both prompts, decode 31 tokens).
  `--sdpa-correctness-only` three times: 4 of 4, release QK^T / softmax / attn*V names, `fused=-`.

**Candidate 10 dispatches what was measured.** Selected by `ET_VK_SARC_780M_PROFILE=c10` on top of
`ET_VK_SARC_DEV_PROFILE=780m-refine3`. All 1,468 SPIR-V binaries of the measured build (`fused9`) are
byte-identical in `head1` (the six softmax variants under their new names). Kernel names in `verify.sh`
(`linear 4w` / `linear 8da4w` lines) and in the SDPA tiers (`fused3_d64_t32x32g11s32rko`,
`fused3_d128_t16x64g11s32rko`) are those of the candidate-10 gate; `verify.out` of `h1-c10` equals that gate's
line for line apart from the tok/s figures.

**Gate, once, on the committed build (`h1-c10`, 12:03 to 13:22 PDT):** SDPA tiers `all`, `extended`, `full`, 12
passes each: 48 / 96 / 48 cases passed, 0 failed, 0 mismatches, pairing ok in every line; `verify.sh` rc = 0 as
above. Next token against the parent: SAME in eleven of twelve items; 8B 8da4w on `prompt_2048.txt` DIFFERS,
the item already recorded for candidate 8 (**ACCEPTED (reference-error rule, owner decision 2026-10-04)**,
evidence `results/780m/probe/`, `results/780m/sdpa-error/`; the kernels are byte-identical, so that evidence
stands).

**Timed session against the parent (`780m-refine3`), same binary in both arms, started at 44 C, 5 valid runs per
arm.** This is also the first direct measurement of candidate 10 against the round's parent; until now the
figure was chained over four sessions (1.0410 x 1.1316 x 1.0319 x 1.0047 = +22.13 %).

| cell | `780m-refine3` | candidate 10, committed build | gain | candidate 10 through the patches (`c10-q4-refine10`) | spread parent / candidate |
|---|---:|---:|---:|---:|---|
| 1B 4w | 2828.73 | 3835.21 | **+35.58 %** | 3842.40 | 0.28 / 0.19 % |
| 1B 8da4w | 2824.83 | 3690.09 | **+30.63 %** | 3690.09 | 0.28 / 0.18 % |
| 3B 4w | 1190.01 | 1459.73 | **+22.67 %** | 1459.73 | 0.06 / 0.07 % |
| 3B 8da4w | 1166.29 | 1379.12 | **+18.25 %** | 1360.80 | 0.57 / 1.46 % |
| 8B 4w | 542.09 | 641.40 | **+18.32 %** | 642.21 | 0.16 / 0.28 % |
| 8B 8da4w | 549.50 | 615.20 | **+11.96 %** | 615.20 | 0.43 / 0.42 % |

Geomean **+22.64 %** over `780m-refine3`. Committed build against patch build: -0.19 to +1.35 %, geomean +0.17 %
(different sessions; the A/A floor is +-0.2 %, and 3B 8da4w is the cell with the 1.4 % repeat spread in both).
Traces (warm, one run per arm): 718.2 -> 536.2 ms (1B 4w), 719.2 -> 552.0, 1719.6 -> 1417.5, 1742.7 -> 1484.7,
3809.2 -> 3250.7, 3741.0 -> 3364.6 ms (8B 8da4w); in 1B 4w the three attention kernels (46.5 + 115.3 + 42.1 ms)
are gone, the copy family grows by 35.6 ms (copy pass + fused kernel are counted there), linear GEMM 366.6 ->
352.9 ms.

`sarc/tools/check.sh --no-build`: PASS. Its zone rule compares against `release/1.5`, where the release zone is
allowed and `SDPA.cpp` is already listed in `sarc/HOOKS`, so it does not flag the two entry points; they are the
two commits named above, permitted by the owner decision of 2026-10-05 and by nothing else.

### What the clock does to the microbench (found 2026-10-04, affects how the sweeps are read)

The GPU clock idles at 800 MHz and needs sustained load to reach 2800 MHz. In `llama_main` it is at 2800 MHz
throughout; in the microbench it is not always:

- SDPA suite, 3 warm-up + 5 timed runs of 3 to 15 ms each: the timed runs are still on the rising clock. The
  coefficient of variation over the 5 runs is 7 to 9 % (median of the 330 enumeration rows measured so far) and a
  compute-bound kernel reads up to 1.7 times slower than at 40 + 10 runs, while a memory-bound one barely moves.
  `ET_VK_SDPA_PERF_RUNS` (added) sets the run counts; the enumeration will use 20 + 8. The 55 runs done with 3 + 5
  stay in `raw/space/results.csv` and are not used for ranking.
- Linear suite, full measurement (3 + 5): steady, coefficient of variation 0.5 to 1 %. Longer runs only heat the
  device (300 + 5: +5 %).
- Linear suite, screening mode (1 + 2): the first timed run is still rising (12 to 15 % between the two timed
  runs). On 14 configurations across the ranking the screening value is 3 to 13 % above the steady one, by a
  different amount per configuration; the order is kept overall (Spearman 0.987) but not among the first five
  (`results/780m/space/clock-check.csv`). The random sample and refinement round 1 were screened this way, so their
  single values scatter by about +-5 %: fine for the parameter-importance table (averages over many
  configurations), not for choosing among near-ties. Therefore the confirmation takes everything within 8 % of the
  fastest (not the 10 best by rank), and the 8da4w screen uses 3 + 3 runs.

### Pauses and overlaps (for whoever audits the timings)

- Builds are CPU work and the screening sweep does not look at `PAUSE`, so the sweep process was stopped
  (`SIGSTOP`, between two runs) for each build and continued afterwards: 03:33:14 to 03:34:33, 03:41:31 to
  03:41:33, 03:41:53 to 03:43:11, 03:49:09 to 04:05:35 and 04:13:42 to 04:13:45 UTC (`logs/sweep-pauses.txt`).
- Candidate-8 correctness and kernel-timing runs took the gpu-lab lock between sweep runs from 03:28 to 04:16 UTC.
  They cannot overlap a sweep run, but they warm the device, and the sweep waits until it is back under 62 C
  before each run: that is why round 1 is slow, not a change in what it measures.
- During the candidate-8 gate: a build (05:29 to 05:34 UTC) and correctness / kernel-timing runs of the one-pass
  variant and the four-arm logits run (05:35 to 05:57 UTC) were slotted in while the gate ran its SDPA
  correctness tiers, after the timed session had ended (05:28 UTC). Nothing ran next to the session.
- The gate of candidate 9 was first started at 08:59 UTC and did not run: the sweep process had been stopped
  while it held the gpu-lab lock, so the session timed out on the lock after 15 min (`gpu-lab lock busy`).
  Nothing was measured; the empty outputs are in `<artifacts>/superseded/c9-gate-lock-held-by-stopped-sweep/`.
  It was restarted at 09:18 UTC with the sweep stopped while the gate script itself held the lock.
- One build of mine started while the roofline was still running and was stopped after about 15 s (03:05:28 to
  03:05:42 UTC, `-j6`). It overlapped the confirm runs of `mem_write`, `mem_copy`, `mem_triad` and
  `sharedbw_fp16`. The matrix roofs used below were confirmed earlier in the run and repeat within 0.1 %; the
  shared-memory fp16 read / write roofs show a 15 % repeat range and are not used.

## Roofs, re-measured (igpu-roofline `fast` plan, 2026-10-04 19:46 to 20:15 PDT)

Run `~/igpu-roofline/campaigns/780m/2026-10-04-fast-prefill-refine2` (code `dbdd193e`, RADV PHOENIX, Mesa 25.2.7,
GPU clock DVFS-governed, not pinned). Confirmed medians (3 repeats each):

| roof | value | repeat range | earlier value (`sarc-1.5-e2e-benchmark/evidence/roofline.md`) |
|---|---:|---:|---:|
| `matrix_fp16_fp32` (4w linear, SDPA) | 14.766 TFLOP/s | 0.1 % | 14.772 |
| `matrix_int8` (8da4w linear) | 14.379 TOP/s | 0.1 % | 14.393 |
| `global_read` / `global_write` / `global_copy` | 86.6 / 77.6 / 71.5 GB/s | 0.0 to 0.1 % | |
| MMA fed from shared memory, one tile pair per 1 / 2 / 4 / 8 multiply-adds | 2.42 / 4.82 / 9.65 / 14.73 TFLOP/s | | |
| MMA fed from a cache-resident buffer, same | 4.01 / 7.89 / 10.75 / 12.80 TFLOP/s | | |

The last two rows are what shaped candidate 8: operand loads, not the matrix unit, limit a kernel that loads a
tile pair for fewer than about 8 multiply-adds, and contiguous tiles in a cache-resident buffer load faster than
tiles in shared memory.

## Candidate 7 (softmax r3, through the uncommitted hook): ACCEPTED, bit-identical, +4.10 % geomean

Session `c7-softmax-r3` (`results/780m/sessions/c7-softmax-r3/`, raw data `<artifacts>/stage/c7-softmax-r3`),
started 18:13 PDT at 45 C after the full 30 min wait. Both arms are the same binary (build `hook3` = the branch +
`hooks/softmax-name-hook.patch` in a scratch tree) with `ET_VK_SARC_DEV_PROFILE=780m-refine3`; the candidate arm
adds `ET_VK_SARC_780M_SOFTMAX=r3`. Tok/s, median of 5 valid runs per arm, arms interleaved:

| cell | parent (`780m-refine3`) | candidate (+ softmax r3) | gain | repeat spread parent / candidate | next token (2048 / unaligned prompt) |
|---|---:|---:|---:|---|---|
| 1B 4w | 2828.73 | 3020.65 | **+6.78 %** | 0.28 / 0.29 % | SAME / SAME |
| 1B 8da4w | 2824.83 | 3011.76 | **+6.62 %** | 0.28 / 0.15 % | SAME / SAME |
| 3B 4w | 1190.70 | 1231.51 | **+3.43 %** | 0.12 / 0.18 % | SAME / SAME |
| 3B 8da4w | 1170.29 | 1206.12 | **+3.06 %** | 0.40 / 0.35 % | SAME / SAME |
| 8B 4w | 541.23 | 554.41 | **+2.44 %** | 0.71 / 0.11 % | SAME / SAME |
| 8B 8da4w (repeat, 21:46 PDT, start 44 C) | 548.47 | 561.56 | **+2.39 %** | 0.56 / 0.36 % | SAME / SAME |

Geomean of the six cells **+4.10 %**, each outside the +-2 % band and far above the A/A floor (+-0.23 %).
The parent arm agrees with `t3-aa` within 0.5 % (2828.73 / 2828.73, 1192.08 / 1172.97, 544.10 / 549.50).
The 8B 8da4w cell is from the repeat (`sessions/c7-softmax-r3/repeat-8b8da4w/`, both arms interleaved, 5 valid
runs each, cool start); in the first session 8 of its runs carry `other_gpu_process` (below) and its 4 valid runs
per arm read 548.0 to 549.1 against 562.0 to 562.5 tok/s, the same gain.

Gate items (all with the candidate's environment):

| item | result |
|---|---|
| `verify.sh --models 1b,3b,8b --schemes 4w,8da4w --pdiff` | correctness rc = 0; 12 of 12 production-diff cases ALL PASSED; default vs tiled SAME on the real-text and the unaligned prompt; decode 31 tokens; `linear <scheme> rc=1` as on the parent. Status lines identical to the parent's (`s8-r3final/verify.out`) |
| next token parent vs candidate | SAME in all six cells on both prompts |
| SDPA tiers, 12 passes each | `all` 4 of 4, `extended` 8 of 8, `full` 4 of 4 in every pass, 0 mismatches, `pairing=ok` in every case; control with the table kernels 1 pass each, same |
| traces (one warm run per arm) | softmax 115.1 -> 70.5 ms (1B), 150.2 -> 93.3 ms (3B), 229.1 -> 142.3 ms (8B); QK^T and attn*V unchanged |

Where the time is after candidate 7 (`sessions/c7-softmax-r3/trace/families.csv`, candidate arm, ms; 4w / 8da4w):

| family | 1B | 3B | 8B |
|---|---|---|---|
| linear GEMM | 373 / 351 | 1081 / 1030 | 2799 / 2625 |
| softmax | 70 / 70 | 93 / 93 | 142 / 142 |
| QK^T | 47 / 46 | 129 / 127 | 191 / 187 |
| attn*V | 42 / 42 | 77 / 78 | 119 / 119 |
| elementwise (`mul`, `sigmoid`, `add`; upstream) | 68 / 68 | 129 / 130 | 244 / 243 |
| copy / view / other (upstream) | 61 / 36 | 135 / 86 | 229 / 137 |
| 8-bit activation quantize (upstream) | - / 43 | - / 101 | - / 167 |
| RMSNorm + RoPE (upstream) | 14 / 14 | 40 / 41 | 60 / 61 |
| total dispatch | 680 / 675 | 1694 / 1695 | 3797 / 3693 |

Single traced runs: the linear rows of the two arms differ by up to 3 % (8B 4w 2716 against 2799 ms) without any
linear kernel having changed, so differences of that size between sessions are not resolved by a trace.

Why the cell was repeated: in the first session its runs 4 to 8 (eight runs) are marked `other_gpu_process`. The process the session saw was pid 576199,
`/bin/bash`: an interactive monitoring shell of this campaign whose command line contained the name of the
runner binary as text. `e2e5.sh` looks for other GPU processes with `pgrep -af`, which matches full command lines,
so the shell was counted for the 6 minutes it was alive. No GPU process ran (the lock holder was the session; the
flagged runs read 548.0 to 549.4 and 561.9 to 562.5 tok/s, the same as the valid ones). The flag is not removed
and the tool is not changed; the cell is repeated instead. Pitfall for anyone watching a session: do not put the
runner or microbench binary names in a shell command while `e2e5.sh` is running.

Bitwise comparison against the parent (`results/780m/softmax/bitwise-c7.txt`): the raw fp16 SDPA output of
every correctness case of the tiers `all`, `extended`, `peaked` and `full` (21 cases, up to 16.8 MB each, the three
production head configurations at S = 2048 among them) is **byte-identical** between the parent kernels and the
parent kernels with softmax r3. Candidate 7 therefore does not use the near-tie or the reference-error rule.

## Candidate 8 (Part 2): fused SDPA kernel `sarc_dev_780m_sdpa_fused3`: ACCEPTED (reference-error rule, owner decision 2026-10-04)

After candidate 7, QK^T + softmax + attn*V are still 160 of 680 ms on 1B, 299 of 1694 ms on 3B and 452 of
3797 ms on 8B, and all three kernels are bound by the traffic of the S x S attention matrix (QK^T writes it,
the softmax reads and rewrites it, attn*V reads it; about 550 MB per layer on 1B by count of bytes), not by
arithmetic. The fused kernel never writes that matrix. For a block of query rows of one head it walks the context
in blocks, twice: pass A computes the scores (fp32 accumulate, scaled, rounded to fp16, exactly as the QK^T kernel
does) and keeps the row maxima; pass B computes the scores again, e = exp(score - max) in fp16, the row sums in
fp32 and `acc += e V` in fp32; the output is `acc / sum`.

### Gate (session `c8-fused3`, `results/780m/sessions/c8-fused3/`)

Started 21:59 PDT at 44 C. Both arms are the same binary (build `fused5` = the branch + both hook patches in a
scratch tree) with `ET_VK_SARC_DEV_PROFILE=780m-refine3 ET_VK_SARC_780M_SOFTMAX=r3` (= candidate 7); the candidate
arm adds `ET_VK_SARC_780M_SDPA_FUSED=fused3_d64_t32x32g11s32rk,fused3_d128_t16x64g11s32rk`. Tok/s, median of 5
valid runs per arm, arms interleaved; recomputed from `runs.csv`:

| cell | parent (candidate 7) | candidate 8 | gain | repeat spread parent / candidate | next token (2048 / unaligned prompt) |
|---|---:|---:|---:|---|---|
| 1B 4w | 3020.65 | 3624.78 | **+20.00 %** | 0.15 / 0.18 % | SAME / SAME |
| 1B 8da4w | 3011.76 | 3618.37 | **+20.14 %** | 0.15 / 0.18 % | SAME / SAME |
| 3B 4w | 1232.25 | 1376.34 | **+11.69 %** | 0.06 / 0.20 % | SAME / SAME |
| 3B 8da4w | 1201.88 | 1344.71 | **+11.88 %** | 1.24 / 0.79 % | SAME / SAME |
| 8B 4w | 556.07 | 601.12 | **+8.10 %** | 0.46 / 0.59 % | SAME / SAME |
| 8B 8da4w | 561.71 | 605.74 | **+7.84 %** | 0.19 / 0.21 % | **DIFFER** / SAME |

Geomean **+13.16 %**; 60 timed runs, none rejected; median clock 2783 to 2800 MHz; start temperature 44 to 50 C,
peak 93 C. The unaligned prompt (1972 tokens) is not served by the fused kernel, so its checks compare the same
kernels in both arms.

The one differing item, 8B 8da4w on `prompt_2048.txt` (2048 times the token " the"): the model is undecided there.
Next-token logits of the top candidates in the four arms (`results/780m/probe/c8-differ-8b-8da4w-prompt_2048.csv`):

| token | parent tiled | parent default | candidate tiled | candidate default |
|---|---:|---:|---:|---:|
| 118 (a byte token) | 11.461 (p 0.0126) | 11.500 (p 0.0129) | 11.125 | 11.156 |
| 53 `V` | 11.297 | 11.234 (p 0.0099) | 11.422 (p 0.0119) | 11.367 (p 0.0113) |
| 5531 `cal` | 11.047 | 11.180 | 11.102 | 11.016 |
| top-2 margin | 0.164 | 0.266 | 0.297 | 0.211 |

The parent's top token has probability 1.3 % and leads by 0.27 logit (0.16 in its tiled arm); the candidate moves
token 118 by -0.34 and token 53 by +0.13. Largest logit difference over the vocabulary: candidate default against
parent default 0.66, parent tiled against parent default 0.83; KL 8.2e-3 against 4.6e-3 nat.

Record under the second owner decision of 2026-10-04 (arithmetic changes are judged against a reference). No
threshold was chosen here; the two in item 3 are the decision's.

1. **Gate criterion, error against the reference** (`results/780m/sdpa-error/c8-fused3.csv`; fp64 CPU reference
   computed from the fp16 inputs the device saw; parent = candidate 7 kernels, same inputs). Production shapes
   (tier `full`):

   | case | elements | rms parent -> candidate | maximum parent -> candidate |
   |---|---:|---|---|
   | 1B head configuration, S = 2048 | 4,194,304 | 1.968e-5 -> 1.029e-5 (x 0.52) | 8.44e-4 -> 3.86e-4 (x 0.46) |
   | 3B head configuration, S = 2048 | 6,291,456 | 2.000e-5 -> 1.035e-5 (x 0.52) | 9.52e-4 -> 3.57e-4 (x 0.37) |
   | 8B head configuration, S = 2048 | 8,388,608 | 1.979e-5 -> 1.036e-5 (x 0.52) | 1.039e-3 -> 3.69e-4 (x 0.36) |
   | 8B head configuration, S = 1024, `input_pos` 1024 | 4,194,304 | 9.62e-6 -> 4.79e-6 (x 0.50) | 9.51e-5 -> 4.05e-5 (x 0.43) |

   The candidate's rms and maximum error are not larger than the parent's on every production shape: **met**.
   Outside the production shapes: lower in all 8 `extended` cases (rms x 0.51 to 0.53); in the 5 `peaked` cases rms
   x 0.81 to 0.91, maximum lower in 4 and higher in 1 (`peaked_tiny_gqa_s256`, S = 256, 2 heads: 1.74e-3 ->
   1.98e-3).
   Correctness tiers with the candidate's environment, 12 passes each, 0 mismatches in every pass: `all` 4 of 4,
   `extended` 8 of 8, `full` 4 of 4, `peaked` 5 of 5, `fused` 5 of 5; the fused kernel the only SDPA kernel
   dispatched, `pairing=ok`; one control pass of `all`, `extended` and `full` with the table kernels.
2. **Evidence.** The logits at the differing position for the four arms: the table above. The real-text
   comparison (`results/780m/probe/c8-real-text-compare.csv`; 32 prompts of 128 to 1920 tokens cut from six real
   texts by the fixed recipe of `tools/probe_prompts.py`, next-token distribution after each prompt, 24 runs of
   `logits_probe`):

   | cell | pair | top-1 differs | KL mean / max (nat) | largest logit difference | perplexity ratio |
   |---|---|---:|---|---:|---:|
   | 1B 4w | candidate vs parent (default arms) | 0 of 32 | 2.2e-5 / 1.3e-4 | 0.11 | 1.0029 |
   | | parent tiled vs parent default | 0 | 3.9e-5 / 2.6e-4 | 0.11 | 1.0044 |
   | 3B 4w | candidate vs parent | 0 | 1.5e-5 / 1.1e-4 | 0.09 | 0.9993 |
   | | parent tiled vs parent default | 0 | 3.2e-5 / 5.9e-4 | 0.11 | 0.9993 |
   | 8B 4w | candidate vs parent | 0 | 2.1e-5 / 2.0e-4 | 0.07 | 1.0003 |
   | | parent tiled vs parent default | 0 | 2.6e-5 / 2.5e-4 | 0.11 | 1.0014 |
   | 1B 8da4w | candidate vs parent | 2 | 3.3e-2 / 0.39 | 3.69 | 0.951 |
   | | parent tiled vs parent default | 1 | 3.7e-2 / 0.55 | 3.45 | 0.896 |
   | 3B 8da4w | candidate vs parent | 5 | 3.4e-2 / 0.45 | 3.32 | 0.912 |
   | | parent tiled vs parent default | 3 | 3.6e-2 / 0.47 | 2.91 | 0.937 |
   | 8B 8da4w | candidate vs parent | 4 | 2.6e-2 / 0.30 | 2.63 | 0.990 |
   | | parent tiled vs parent default | 4 | 6.3e-2 / 0.58 | 2.81 | 0.902 |

   (Perplexity of the true next token over the 32 prompts, first arm over second; the csv also has the
   candidate-tiled vs candidate-default rows.) In the 4w cells the candidate is closer to its parent than the
   parent's two linear arms are to each other. The 8da4w cells are three orders of magnitude noisier in every
   pair, including the two parent arms (the activations are re-quantized to 8 bit from their own range at every
   linear layer); the candidate against its parent is of the same size as that spread in each of them (top-1
   differences 2 / 5 / 4 against 1 / 3 / 4, mean KL 0.033 / 0.034 / 0.026 against 0.037 / 0.036 / 0.063).
3. **Gross-divergence check** (reject if, in any cell, the mean KL of candidate-default against parent-default
   exceeds 0.5 nat or the top-1 token differs on more than one third of the prompts): largest mean KL 0.034 nat,
   at most 5 of 32 prompts: **passed**.
4. **Next-token items that differ**: one. 8B 8da4w, `prompt_2048.txt`, parent against candidate (the e2e session).
   Every other item is SAME: parent vs candidate in the other 11 cell x prompt checks, and `verify.sh` default vs
   tiled on the real-text and the unaligned prompt.

The rest of the gate: unmodified `verify.sh --models 1b,3b,8b --schemes 4w,8da4w --pdiff` with the candidate's
environment: correctness rc = 0, 12 of 12 production-diff cases ALL PASSED, default vs tiled SAME on both
prompts, decode 31 tokens, `linear <scheme> rc=1` as on the parent. Traces (one warm run per arm,
`sessions/c8-fused3/trace/`): total dispatch time 672.5 -> 563.9 ms (1B 4w), 1657 -> 1499 ms (3B 4w), 3700 ->
3458 ms (8B 4w); QK^T + softmax + attn*V 158.9 / 296.2 / 449.4 ms are gone and the copy pass + fused kernel add
45.3 / 120.7 / 184.2 ms (they are in the trace tool's "copy/view/other" family); the other families are
unchanged within 2 %. Kernel time per layer at a steady clock after the gate, three runs
(`results/780m/fused/kernel-time-steady-c8-gate.csv`): 9.96 -> 2.82 ms (1B), 10.24 -> 4.32 to 4.36 ms (3B),
13.51 -> 5.54 to 5.66 ms (8B).

### Structures

Three structures were built and measured; the third is the candidate.

| structure | what a workgroup is | kernel time per layer, 1B / 8B head configuration |
|---|---|---|
| three kernels (candidate 7) | | 9.97 / 13.52 ms |
| `fused`: K and V staged in shared memory, 4 to 6 barriers a block | 4 to 8 subgroups, 64 or 128 rows | 3.84 / 11.7 ms |
| `fused2`: staging double-buffered (1 barrier a block), score rows private to a subgroup, V transposed | 2 to 8 subgroups | not faster than `fused` (first 5 runs: 6.2 against 5.9 ms on 1B) |
| `fused3`: one subgroup, no staging, no barrier; Q tiles in registers, K from the cache buffer, V from a transposed copy | 1 subgroup, 16 or 32 rows | 2.55 / 9.95 ms |
| `fused3`, packed (`...rk`): K and V read from tile-packed copies in which a 16 x 16 operand tile is 512 contiguous bytes | same | **2.80 / 5.57 ms, copy pass included** (2.46 / 4.84 ms without it) |

(Steady GPU clock: 40 warm-up and 10 timed runs, median; `results/780m/fused/kernel-time-steady.csv`. With the
default 3 + 5 runs of the microbench the clock is still rising and the fused kernels read up to 1.7 times slower:
`kernel-time-development.csv`, where only values in the same column compare.)

What the measurements say about why:

- The disassembly shows two 128-bit shared-memory loads per lane for every operand tile; `fused` and `fused2` load
  1.0 to 1.5 tiles per multiply-add, which the roofline table above puts at a small fraction of the matrix roof.
  Removing the row padding from the shared tiles made `fused2` 2.6 times slower (15.5 against 5.9 ms): shared
  memory loads were the limit, not the barriers.
- Keeping the Q tiles in registers and reading K straight from the buffer halves the loads per multiply-add.
  Reloading Q from the buffer for every block instead costs 2.2 to 3.0 times (1B 10.2 against 4.6 ms, 8B 44.7
  against 15.0 ms, first-5-run values): loads from the cache buffers are only cheap when a tile is contiguous. In
  the caches a 16 x 16 tile is 16 runs a row stride apart; the packed copies make it one run, which is what
  brought head_dim 128 from 9.95 to 4.96 ms (16 x 16 tile, steady clock).
- The packed `fused3` runs 3.1 M (1B) and 6.3 M (8B) multiply-adds a layer in 2.46 and 4.84 ms: about 71 % of the
  matrix roof. Without pass A the same kernel takes 1.58 / 3.12 ms (measurement-only variant `...rkm1`): a
  one-pass (running-maximum) form would save at most that, about 2 % end to end; not attempted.
- The copy pass (`sarc_dev_780m_sdpa_kvt`, 4 to 16 MB a layer) costs 0.35 (1B) to 0.7 ms (8B), three times what
  the bytes cost at the copy roof; a 16 x 4 workgroup instead of 8 x 8 is slower (2.36 against 2.27 ms on 1B);
  not tuned further.

### One-pass variant (`...rko`, built and measured, a later candidate)

`ONLINE` keeps a running row maximum instead of pass A: when a block raises the maximum of a row, that row's
accumulator and sum are scaled by exp(old - new) (an element-wise divide of the accumulator tiles by a divisor
tile from shared memory, only in subgroups where some row moved). Build `fused6`: correct on tiers `all`,
`extended`, `peaked` and `fused` (one pass each, 0 mismatches), error against the fp64 reference equal to or
slightly below the two-pass kernel's. Kernel time per layer, copy pass included, steady clock: 1B 2.80 -> 2.27 ms,
3B 4.33 -> 3.22 ms, 8B 5.54 -> 4.04 ms. End to end that is about +1.5 to +2 %, so it is not gated alone: it goes
into the next candidate together with the Part 1 winners per shape.

Variants chosen (`ET_VK_SARC_780M_SDPA_FUSED=fused3_d64_t32x32g11s32rk,fused3_d128_t16x64g11s32rk`): for head_dim
64, 32 rows x 32-column blocks (2.80 ms; 16 x 32 reads 2.77 to 2.79 ms, a tie); for head_dim 128, 16 rows x 64
(8B 5.56 to 5.58 ms, 3B 4.32 ms; 16 x 16, 16 x 32 and 32 x 32 are 1.5 to 4 % behind on both models, twice).

- Files: `glsl/sarc_dev/sarc_dev_780m_sdpa_{fused,fused2,fused3,vt,kvt}.{glsl,yaml}`,
  `impl/sarc_dev/Sdpa780mFused.cpp`, the 780m block of `Overrides.cpp`.
- It needs `hooks/sdpa-fused-hook.patch` (about 70 added lines in the release zone and `SDPA.cpp`, not applied on
  this branch; described in `hooks/README.md`): nodes appended after the three SDPA nodes, and an empty dispatch
  for those three when the fused node serves the call. It serves prefill calls whose S and `input_pos` are
  multiples of the row tile and the context block; decode, unaligned prompts and `ET_VK_DISABLE_COOPMAT` keep the
  three kernels. It also serves aligned lengths the 128-row QK^T tile does not (64, 192, ...), which the stock path
  runs on the tiled kernels.
- It changes the arithmetic (the three kernels round e / sum to fp16 before attn*V; this kernel rounds e and
  divides the fp32 accumulator), so it is judged by the reference-error rule of the second owner decision.
- Test additions (`test/sarc_dev/test_llama_microbench.cpp`, no tolerance and no existing case changed): the fused
  kernel is recognised by the SDPA cases (it must then be the only SDPA kernel dispatched);
  `ET_VK_SDPA_ERROR_REPORT=1` prints rms and maximum error against an fp64 reference computed from the fp16
  inputs; tier `peaked` (Q scaled by 8, so a few context positions carry most of a row's weight; the existing
  tiers have near-uniform attention); tier `fused` (S = 32, 64, 192, 320, shapes only the fused kernel takes);
  `ET_VK_SDPA_PERF_RUNS` for the steady-clock timing. `test/sarc_dev/probe/logits_probe` writes the next-token
  logits of token prompts, for the real-text comparison (`tools/probe_{prompts,run,compare}.*`).

Results so far (one pass each unless said; `<artifacts>/fx/`, `<artifacts>/stage/c8-fused3/`):

- Correctness, chosen variants: tiers `all` 4 of 4, `extended` 8 of 8, `peaked` 5 of 5; tier `fused` 12 passes,
  5 of 5 each; 0 mismatches, the fused kernel the only SDPA kernel dispatched.
- Error against the fp64 reference, three kernels -> fused: rms 1.9e-5 to 4.6e-5 -> 0.95e-5 to 2.4e-5 on the 8
  `extended` cases (about half), maximum lower in all 8; on the 5 `peaked` cases rms 2.6e-4 to 2.7e-4 -> 2.2e-4 to
  2.4e-4, maximum lower in 4 and higher in 1 (`peaked_tiny_gqa_s256`: 1.74e-3 -> 1.98e-3; both arms round the
  same fp16 scores there). The `full` tier, which holds the production shapes the rule names, runs in the gate.
- End to end, one run per arm, not a session: 1B 4w 3020.65 -> 3624.78 tok/s (+20.0 %). Projected from the
  kernel times and the candidate-7 traces: about +20 % (1B), +12 % (3B), +8 % (8B).

## Candidate 9: candidate 8 + one-pass fused kernel + 4w kernel per shape: gate passed, +3.19 % geomean

Session `c9-online-q4` (`results/780m/sessions/c9-online-q4/`), started 02:18 PDT at 44 C. Both arms are the same
binary (build `fused7` = the branch + both hook patches) with candidate 8's environment; the candidate arm uses
the one-pass variants (`ET_VK_SARC_780M_SDPA_FUSED=fused3_d64_t32x32g11s32rko,fused3_d128_t16x64g11s32rko`) and
`ET_VK_SARC_780M_PROFILE=refine9`. Tok/s, median of 5 valid runs per arm, arms interleaved:

| cell | parent (candidate 8) | candidate 9 | gain | repeat spread parent / candidate | next token (2048 / unaligned prompt) |
|---|---:|---:|---:|---|---|
| 1B 4w | 3631.21 | 3806.69 | **+4.83 %** | 0.35 / 0.19 % | SAME / SAME |
| 1B 8da4w | 3624.78 | 3690.09 | +1.80 % | 0.18 / 0.00 % | SAME / SAME |
| 3B 4w | 1377.27 | 1439.21 | **+4.50 %** | 0.14 / 0.14 % | SAME / SAME |
| 3B 8da4w | 1348.26 | 1362.61 | +1.06 % | 1.24 / 1.21 % | SAME / SAME |
| 8B 4w | 601.29 | 634.06 | **+5.45 %** | 0.18 / 0.43 % | SAME / SAME |
| 8B 8da4w | 606.10 | 615.75 | +1.59 % | 0.33 / 0.33 % | SAME / SAME |

Geomean **+3.19 %**; 60 timed runs, none rejected; clock 2800 MHz. The 8da4w cells only have the one-pass fused
kernel (their linear kernel is unchanged) and are inside the +-2 % band: by the campaign's rule that part is not
a gain by itself. The 4w cells have both changes and are outside the band.

The two parts, measured separately before the gate:

- 4w kernels (`refine9`): the raw output of all 12 production-diff cases is byte-identical to `780m-refine3`
  (`results/780m/space/bitwise-4w-refine9.txt`), so this part does not change the arithmetic.
- One-pass fused kernel: it does (rescaling instead of a first pass), so it is judged against the reference. On
  the four production shapes its rms error is 0.996 to 0.998 times candidate 8's and its maximum is equal or lower
  (`results/780m/sdpa-error/c9-online-full-precheck.csv`); the gate repeats this.

Gate (finished 04:10 PDT), all with the candidate's environment:

| item | result |
|---|---|
| `verify.sh --models 1b,3b,8b --schemes 4w,8da4w --pdiff` | correctness rc = 0; 12 of 12 production-diff cases ALL PASSED; default vs tiled SAME on the real-text and the unaligned prompt; decode 31 tokens; `linear <scheme> rc=1` as on the parent; status lines identical to candidate 8's. The 4w texture3d cases dispatch the `refine9` kernels |
| next token parent vs candidate | SAME in all six cells on both prompts: **no differing item** |
| SDPA tiers, 12 passes each | `all` 4 of 4, `extended` 8 of 8, `full` 4 of 4, `peaked` 5 of 5, `fused` 5 of 5 in every pass, 0 mismatches, the fused kernel the only SDPA kernel dispatched, `pairing=ok`; controls with the table kernels 1 pass each |
| error against the fp64 reference (`results/780m/sdpa-error/c9-online-q4.csv`) | production shapes: rms 0.996 to 0.998 times candidate 8's, maximum equal or lower: **not larger on any of them**. All 17 cases of `extended`, `peaked` and `full`: rms ratio 0.992 to 1.000 |
| real-text comparison, 32 prompts, six cells (`results/780m/probe/c9-real-text-compare.csv`) | 4w cells: top-1 differences 0, mean KL 1.6e-5 to 2.2e-5 (candidate 8 tiled vs default: 1.9e-5 to 3.9e-5). 8da4w cells: top-1 differences 3 / 4 / 1 of 32, mean KL 0.028 / 0.020 / 0.017 (candidate 8 tiled vs default: 3 / 3 / 2 and 0.045 / 0.027 / 0.027). Gross-divergence check passed (limits 0.5 nat, one third of the prompts) |
| traces, one warm run per arm (`sessions/c9-online-q4/trace/`) | 4w linear GEMM 367.5 -> 350.9 ms (1B, -4.5 %), 1049 -> 1018 ms (3B, -2.9 %), 2659 -> 2551 ms (8B, -4.1 %); 8da4w GEMM unchanged (+0.1 / +0.1 / -0.5 %); copy pass + fused kernel -10 / -31 / -48 ms in both schemes; totals 560.0 -> 533.3, 1481 -> 1420, 3382 -> 3228 ms (4w) |
| kernel time per layer, steady clock, three runs (`results/780m/fused/kernel-time-steady-c9-gate.csv`) | copy pass + fused kernel 2.82 -> 2.26 ms (1B), 4.31 -> 3.22 ms (3B), 5.54 -> 4.05 ms (8B) |

The candidate changes the SDPA arithmetic but no next-token item differs, so nothing has to be excused; the
reference-error evidence is recorded anyway. The 8da4w cells' gain (+1.06 to +1.80 %) comes from the one-pass
kernel alone and is inside the band; whether that kernel is kept for 8da4w is therefore a matter of preference
for the simpler two-pass form, not of a measured gain.

## Part 1, 4w: the best kernel per shape within the existing bodies (`results/780m/space/confirm-4w/`)

Search: 2,444 of 2,500 random survivors screened, 991 neighbours of the 22 best screened, then the 95
configurations within 8 % of the fastest on some screened shape measured in full (all twelve shapes, 3 + 5 runs),
and the 10 fastest per shape (35 configurations) five times. Repeat spread of the leaders 0.1 to 0.4 %.
Kernel time relative to the fastest of the shape (`summary.csv`):

| shape (N, K) | fastest | `780m-refine3` choice | shipped `t128x128..c` | `t128x256..cbt` | `t256x128..g18..bbt` | `..g24..bbt` | `..g28..bbt` |
|---|---|---:|---:|---:|---:|---:|---:|
| 1B wq_wo (2048, 2048) | `t128x256k32g42s32f32cbt` 1586 us | +2.9 % | +2.6 % | **0** | +5.6 % | +3.2 % | +0.4 % |
| 1B wk_wv (512, 2048) | `t128x128k32g42s32f32xpiiw` 439 us | +0.3 % | **+0.3 %** | +5.1 % | +9.3 % | +9.9 % | +5.2 % |
| 1B w1_w3 (8192, 2048) | `t256x128k32g28s32f32bbt` 6019 us | +4.6 % | +6.1 % | +3.2 % | +9.1 % | +5.8 % | **0** |
| 1B w2 (2048, 8192) | `t256x128k32g18s32f32bbt` 5887 us | +7.5 % | +11.1 % | +5.4 % | **0** | +6.4 % | +5.8 % |
| 3B wq_wo (3072, 3072) | `t256x128k32g24s32f32bbt` 3410 us | +3.9 % | +5.2 % | **+1.4 %** | +2.0 % | 0 | +2.6 % |
| 3B wk_wv (1024, 3072) | `t128x256k32g42s32f32cbt` 1203 us | +2.7 % | +5.1 % | **0** | +7.9 % | +4.7 % | +3.9 % |
| 3B w1_w3 (8192, 3072) | `t256x128k32g24s32f32bbt` 9136 us | +3.3 % | +4.5 % | **+1.5 %** | +1.9 % | 0 | +4.2 % |
| 3B w2 (3072, 8192) | `t256x128k32g18s32f32bbt` 8596 us | +6.9 % | +11.7 % | +4.5 % | **0** | +3.4 % | +3.0 % |
| 8B wq_wo (4096, 4096) | `t256x128k32g18s32f32bbt` 5980 us | +3.6 % | +9.4 % | +1.1 % | **0** | +1.3 % | +4.3 % |
| 8B wk_wv (1024, 4096) | `t128x256k32g42s32f32cbt` 1590 us | +1.7 % | +8.2 % | **0** | +1.8 % | +2.0 % | +3.4 % |
| 8B w1_w3 (14336, 4096) | `t256x128k32g18s32f32bbt` 20766 us | +4.5 % | +7.5 % | +2.0 % | **0** | +1.4 % | +4.3 % |
| 8B w2 (4096, 14336) | `t256x128k32g18s32f32bbt` 20408 us | +5.3 % | +9.7 % | +2.5 % | **0** | +2.3 % | +0.9 % |

Bold = what profile `refine9` runs. Its rule (`Overrides.cpp`, 780m block): N below 1024 keeps the shipped tile;
N >= 2048 with K >= 4096 takes the 256-row tile `t256x128k32g18s32f32bbt`; N >= 8192 with K below 3072 takes
its 16-subgroup grid `..g28..`; everything else from N = 1024 takes `t128x256k32g42s32f32cbt`. Where two kernels
are within 2 % the rule keeps the one it already uses (3B wq_wo and w1_w3: `..cbt` instead of `..g24..`).
Per layer (2 wq_wo + 2 wk_wv + 2 w1_w3 + w2) against `780m-refine3`: best per shape -4.76 / -4.03 / -4.26 %
(1B / 3B / 8B); `refine9` -4.75 / -3.05 / -4.26 %.

What the confirmed set says about the parameters near the optimum (`local-effects.csv`: pairs of confirmed
configurations that differ in one parameter, full measurement, ratio of kernel times over all shapes):

| parameter | change | median | range | reading |
|---|---|---:|---|---|
| `IMG_A` | off -> on | 1.000 | 0.997 to 1.009 | does not matter |
| `IMG_W` | off -> on | 1.000 | 0.987 to 1.009 | does not matter |
| drain mode | band / pooled full / in Ash | 0.993 to 1.003 | 0.976 to 1.030 | does not matter (the three usable modes are within 3 %) |
| texel-wise staging | bx -> plain | 1.003 | 0.866 to 1.180 | no effect on average; interacts with the tile (-13 to +18 %) |
| `B_COLMAJOR` | off -> on | 0.993 | 0.815 to 1.047 | decides the 256-row tile (up to -18 %), neutral to +5 % elsewhere |
| `FRAG_LAYOUT` | off -> on | 0.968 | 0.958 to 0.986 | -1 to -4 % on the 12 pairs present; it cannot be combined with `B_COLMAJOR`, and those pairs are 5 % or more behind the leaders |
| `SH_F16V4` | off -> on | 1.028 | 1.023 to 1.035 | always slower |
| tile M | 128 -> 256 | 1.013 | 0.862 to 1.188 | by shape: better for large K, worse for small N |
| tile N | 128 -> 256 | 0.970 | 0.833 to 1.080 | by shape, with M and the grid |
| grid X / Y | 2 -> 4, 1 -> 2, 4 -> 8 | 0.985 to 1.012 | 0.917 to 1.108 | by shape |

Beyond the parameter space, tried on the winners and not pursued: a slab layout of both operands in shared
memory (`tools/gen_sl.py`: A as `FRAG_LAYOUT` has it, B column-major as [k16 slab][n][16 k], no padding; the
layout the 8da4w zpg kernel uses). It is 8 to 25 % slower than the padded column-major kernels of the same tile on
all twelve shapes (`results/780m/space/slab/kernel-time.txt`); the generated files are not kept in the tree.

Tile K (32), subgroup size (32) and the fp32 accumulator have no alternative within 14 % of the leaders
(refinement round 1).

After candidate 9 the local search was continued until it stopped moving (`results/780m/space/{refine2,refine3,
geo-cbt}/`, full measurement on all twelve shapes):

- Round 2, the 51 untested single-parameter neighbours of the winners above: the in-Ash drain instead of the
  band drain on the 256-row tiles (`t256x128k32g{18,24,28}s32f32cbt`) is 0.3 to 2.3 % faster on 9 of the 12
  shapes; nothing else moves.
- Round 3, the 29 untested neighbours of those: `t128x128k32g24s32f32cbt` (a 2 x 4 grid on the shipped tile
  size, column-major B) is the fastest on the two small 1B shapes (wk_wv 420 against 440 us, wq_wo 1544 against
  1586 us); `IMG_A` / `IMG_W` on the leaders are within 0.3 %.
- Geometry scan: the 85 surviving geometries with the leading flag set (fp32 accumulator, in-Ash drain,
  column-major B; subgroup 32, tile K 16 or 32, M and N at least 64, 2 to 32 MMA tiles per subgroup) that no
  earlier stage had measured: none enters the first five on any shape.

Second confirmation (`results/780m/space/confirm2-4w/`, `confirm3-4w/`, `final-4w/`; corrected 2026-10-07 after
review). The second confirmation of 2026-10-05 (`chain12.sh`) took the **6** fastest per shape of everything
measured in full, plus the `refine9` kernels (28 configurations), four more times; the campaign asks for 10. Over
all full measurements two further configurations are in a top ten (`t256x128k32g24s32f32bibt` and `..biwbt`: 3B
wq_wo and w1_w3, ranks 8 to 10) and had 2 measurements and no production-diff pass. They were given three more
full measurements and 12 production-diff passes x 3 models on 2026-10-07 00:56 to 01:07 UTC (`chain24.sh`, same
commands, lock and hold as the confirmations): 72 of 72 passes ALL PASSED; they stay at ranks 8 to 10 on those
two shapes, 1.8 to 2.3 % behind the fastest. The committed `confirm2-4w/full-r4.csv` had been copied two minutes
before that repeat finished (288 of 336 rows); it is now the complete file, and `confirm2-4w/summary.{csv,txt}`
is regenerated from all 16 full-measurement files (`tools/q4_final.py`).

State of the evidence (`final-4w/coverage.csv`): 260 configurations measured in full; **41 are among the ten
fastest of some shape, each with at least 5 full measurements and 12 of 12 passes on each model, 0 failed
passes.** Repeat spread of the 120 top-ten rows: at most 1.5 % for 117. The other three are one configuration,
`t128x128k32g24s32f32cbt` on 1B wq_wo, wk_wv and w1_w3 (8.7, 6.2, 8.8 %): its fifth measurement (2026-10-05
15:06:10 UTC, 90 s before a build started) reads 1671 / 445 / 6526 us where the other four read 1537 to 1544 /
419 to 420 / 6000 to 6024 us. I have not identified the cause; the measurement is kept, the median is unchanged.

**Within the existing kernel bodies the best 4w kernel per shape on this device and driver is** (median over
all full measurements; `final-4w/per-shape.csv`):

| shape (N, K) | fastest | tied within 2 % | profile `refine10` (final) | behind the fastest | against `780m-refine3` | `refine9` behind the fastest | release table kernel behind the fastest |
|---|---|---:|---|---:|---:|---:|---:|
| 1B wq_wo (2048, 2048) | `t128x128k32g24s32f32cbt` 1539 us | 4 | the same | 0 | -5.70 % | +3.03 % | +5.70 % |
| 1B wk_wv (512, 2048) | `t128x128k32g24s32f32cbt` 420 us | 1 | the same | 0 | -4.62 % | +4.84 % | +4.84 % |
| 1B w1_w3 (8192, 2048) | `t256x128k32g28s32f32cbt` 5915 us | 7 | the same | 0 | -6.08 % | +1.73 % | +7.89 % |
| 1B w2 (2048, 8192) | `t256x128k32g18s32f32biwbt` 5872 us | 6 | `t256x128k32g18s32f32cbt` | +0.62 % | -6.54 % | +0.09 % | +11.24 % |
| 3B wq_wo (3072, 3072) | `t256x128k32g24s32f32cibt` 3337 us | 7 | `t256x128k32g24s32f32cbt` | +0.10 % | -5.61 % | +3.63 % | +7.38 % |
| 3B wk_wv (1024, 3072) | `t128x256k32g42s32f32cbt` 1202 us | 12 | the same | 0 | -2.66 % | 0 | +5.15 % |
| 3B w1_w3 (8192, 3072) | `t256x128k32g24s32f32ciwbt` 8973 us | 10 | `t256x128k32g24s32f32cbt` | +0.03 % | -4.76 % | +3.28 % | +6.25 % |
| 3B w2 (3072, 8192) | `t256x128k32g18s32f32cibt` 8530 us | 6 | `t256x128k32g18s32f32cbt` | +0.09 % | -6.87 % | +0.65 % | +12.44 % |
| 8B wq_wo (4096, 4096) | `t256x128k32g18s32f32cibt` 5852 us | 3 | `t256x128k32g18s32f32cbt` | +0.09 % | -5.41 % | +2.17 % | +11.73 % |
| 8B wk_wv (1024, 4096) | `t256x128k32g18s32f32cbt` 1581 us | 20 | the same | 0 | -2.14 % | +0.51 % | +8.75 % |
| 8B w1_w3 (14336, 4096) | `t256x128k32g18s32f32cibt` 20152 us | 3 | `t256x128k32g18s32f32cbt` | +0.27 % | -6.67 % | +2.94 % | +10.72 % |
| 8B w2 (4096, 14336) | `t256x128k32g18s32f32cibt` 20270 us | 12 | `t256x128k32g18s32f32cbt` | +0.15 % | -5.31 % | +0.49 % | +10.32 % |

Where the fastest differs from the `refine10` kernel it is its `IMG_A` / `IMG_W` or band-drain twin, within
0.62 %: tied, and the profile keeps the simplest. Its rule (`Overrides.cpp`, 780m block): K >= 4096 with
N >= 1024 -> `t256x128k32g18s32f32cbt`; K = 3072 with N >= 2048 -> `..g24..cbt`; N = 8192 with K below 3072 ->
`..g28..cbt`; other K below 3072 -> `t128x128k32g24s32f32cbt`; otherwise N >= 1024 -> `t128x256k32g42s32f32cbt`.

Per layer (`final-4w/per-layer.txt`), against `780m-refine3`: best per shape -6.25 / -5.35 / -6.12 %, **profile
`refine10` -6.09 / -5.30 / -5.93 %**, `refine9` -4.76 / -2.99 / -4.18 %, the release table kernel +1.60 / +2.10 /
+3.91 % (1B / 3B / 8B). The raw output of the 12 production-diff cases is byte-identical to `780m-refine3`
(`results/780m/space/bitwise-4w-refine10.txt`). Everything above this paragraph in this section (the first
confirmation and profile `refine9`) is the earlier stage, kept as measured; this table is the final result.

## Candidate 10: candidate 9 with the refined 4w table (`refine10`): gate passed, +0.47 %, inside the band

Session `c10-q4-refine10` (`results/780m/sessions/c10-q4-refine10/`), started 08:19 PDT at 43 C; same binary
(build `fused9`) in both arms, candidate 9's environment with `ET_VK_SARC_780M_PROFILE=refine10` instead of
`refine9`:

| cell | parent (candidate 9) | candidate 10 | gain | repeat spread parent / candidate | next token |
|---|---:|---:|---:|---|---|
| 1B 4w | 3806.69 | 3842.40 | +0.94 % | 0.00 / 0.38 % | SAME / SAME |
| 1B 8da4w | 3690.09 | 3690.09 | 0.00 % | 0.36 / 0.18 % | SAME / SAME |
| 3B 4w | 1439.21 | 1459.73 | +1.43 % | 0.14 / 0.00 % | SAME / SAME |
| 3B 8da4w | 1370.82 | 1360.80 | -0.73 % | 1.47 / 1.41 % | SAME / SAME |
| 8B 4w | 634.84 | 642.21 | +1.16 % | 0.15 / 0.41 % | SAME / SAME |
| 8B 8da4w | 614.83 | 615.20 | +0.06 % | 0.30 / 0.27 % | SAME / SAME |

Geomean **+0.47 %**, every cell inside the +-2 % band: by the campaign's rule not a gain, and the first of the
two candidates under 2 % that end the campaign. `verify.sh`: correctness rc = 0, 12 of 12 production-diff cases
ALL PASSED, default vs tiled SAME on both prompts, decode 31 tokens, `linear <scheme> rc=1` as on the parent; no
SDPA kernel changes, so the SDPA tiers were not repeated. Traces: 4w linear GEMM 350.6 -> 346.8 ms (1B, -1.1 %),
1015.9 -> 995.4 ms (3B, -2.0 %), 2538.4 -> 2508.4 ms (8B, -1.2 %). The 4w cells move by what the kernel timing predicts (+0.9 /
+1.7 / +1.4 %); the 8da4w cells run the same kernels in both arms and scatter by -0.7 to +0.1 %.

## Candidate 11: candidate 10 + the 8da4w kernel per shape (`c11`): gate passed, +1.20 % geomean, the second candidate under 2 %

Session `c11-dq-refine11` (`results/780m/sessions/c11-dq-refine11/`), committed head `a8fffa5ea` (build `head2`),
the same binary in both arms: `ET_VK_SARC_780M_PROFILE=c10` (parent) against `c11`, both on
`ET_VK_SARC_DEV_PROFILE=780m-refine3`. Started 2026-10-05 19:54 PDT at 44 C after the full 30 min wait (the device
idled at 45 C that evening), 5 valid runs per arm, arms interleaved:

| cell | candidate 10 | candidate 11 | gain | repeat spread parent / candidate | next token (2048 / unaligned prompt) |
|---|---:|---:|---:|---|---|
| 1B 4w | 3835.21 | 3835.21 | 0.00 % | 0.19 / 0.19 % | SAME / SAME |
| 1B 8da4w | 3690.09 | 3764.71 | **+2.02 %** | 0.18 / 0.18 % | SAME / SAME |
| 3B 4w | 1457.65 | 1457.65 | 0.00 % | 0.07 / 0.21 % | SAME / SAME |
| 3B 8da4w | 1369.90 | 1408.53 | **+2.82 %** | 1.21 / 1.17 % | SAME / SAME |
| 8B 4w | 639.60 | 640.00 | +0.06 % | 0.25 / 0.25 % | SAME / SAME |
| 8B 8da4w | 614.28 | 628.80 | **+2.36 %** | 0.24 / 0.18 % | SAME / SAME |

Geomean **+1.20 %**. The three 8da4w cells are just outside the band, the 4w cells run the same kernels and do
not move. By the campaign's rule this is the second consecutive gated candidate under 2 % geomean (candidate 10:
+0.47 %): **the stop rule is met.** Candidate 11 is kept as the final configuration: its 8da4w cells gain more
than the band and nothing loses.

Gate: unmodified `verify.sh` rc = 0; its output equals candidate 10's line for line except the `linear 8da4w`
kernel line (6 of the 12 texture3d cases now on `..._zpg_t256x64k64g48s32afmb1`); 12 of 12 production-diff cases
ALL PASSED (8da4w with non-zero zero-points); default vs tiled SAME on both prompts; decode 31 tokens. No SDPA
kernel changes against the parent, so the SDPA tiers were not repeated. This candidate does not change
arithmetic precision (int8 x int4 products in exact integer accumulation, as its parent).

Where the gain comes from (kernel timing of the confirmation, 5 repeats, spread 0.1 to 1.2 %; "Part 1, 8da4w"
below): the six `wq_wo` / `w1_w3` shapes are 1.8 to 3.8 % faster with the 256 x 64 tile, which is 1.6 / 2.6 /
2.4 % of the 8da4w linear time per layer (8B / 3B / 1B). With linear at 69 / 73 / 81 % of the 8da4w prefill that
predicts +1.6 / +1.9 / +1.3 % (1B / 3B / 8B); the session measured +2.0 / +2.8 / +2.4 %, 0.4 to 1.1 points more
than the kernel timing explains. I have no measurement that accounts for that difference (3B 8da4w is the cell
with 1.2 % repeat spread). The warm traces of this gate do not settle it: the candidate arm was traced after the
parent arm without cooling and its unchanged kernels read 1.7 to 2.3 % slower (4w linear 346.9 -> 354.9 ms with
identical kernels), so they only show the changed shapes relative to that drift: 1B `wq_wo` 1.539 -> 1.477 ms,
`w1_w3` 6.006 -> 5.832 ms; 3B 3.421 -> 3.293, 9.366 -> 8.997 ms; 8B `wq_wo` 6.361 -> 6.181 ms, `w1_w3` 21.66 ->
21.91 ms (while the unchanged 8B `w2` went 21.53 -> 21.72 ms).

## Part 1, 8da4w zpg: the best kernel per shape within the existing bodies (`results/780m/space/{dq,confirm-8da4w}/`)

**All 2,238 statically surviving configurations were measured** (of 165,888 combinations; no sampling was
needed): screened on `wq_wo` of the three models with 3 warm-up + 3 timed runs (mode validated on 64
configurations: rank correlation 0.998 to 1.000). Every one dispatched its own kernel; none failed. Then the 49
configurations within 8 % of the fastest on some shape were measured in full (twelve shapes, 3 + 5 runs), and the
10 fastest per shape (22 configurations) five times.

**Within the existing 8da4w kernel bodies, the best configuration per shape on this device and driver is:**

| shapes | kernel | against the `780m-refine3` kernel (`bt_t128x64k32g22s32`, `A_MAP_FULL`, 2 blocks) | tied within 2 % |
|---|---|---|---|
| `wq_wo` (N = K = 2048 / 3072 / 4096) | `t256x64k64g48s32afmb1` | -3.4 / -3.7 / -1.8 % | 6 / 1 / 9 configurations; the refine3 kernel only on 8B |
| `w1_w3` (N = 8192 / 8192 / 14336) | `t256x64k64g48s32afmb1` | -3.4 / -3.8 / -2.5 % | 8 / 3 / 9; the refine3 kernel on none |
| `w2` (K = 8192 / 8192 / 14336) | the refine3 kernel (tied) | best is -1.5 / -1.6 / -1.7 % (`bt_t128x128k32g14s64afmb1`, `bt_t128x64k32g14s32afmb2`) | 10 / 8 / 7, the refine3 kernel among them |
| `wk_wv` 3B, 8B (N = 1024) | the refine3 kernel (tied) | best is -1.3 / -1.2 % | 12 / 6 |
| `wk_wv` 1B (N = 512, K = 2048) | 14 configurations tied, the refine3 kernel not among them | best -3.4 % (`bt_t64x64k32g22s32afmb1`); the 256 x 64 tile -2.9 % | not selected: 0.1 % of the layer |

Per layer (2 `wq_wo` + 2 `wk_wv` + 2 `w1_w3` + `w2`): the best kernel per shape is 2.9 / 3.1 / 2.1 % less 8da4w
linear time than `780m-refine3` (1B / 3B / 8B); profile `c11` (one kernel added, for N >= 2048 with K <= 4096)
gets 2.4 / 2.6 / 1.6 %. The shipped tile (`t128x64k32g42s32`, release table) is 28 % behind the refine3 kernel.
The 12 production-diff passes of the 22 confirmed configurations on the three models: 792 of 792 ALL PASSED
(`results/780m/space/confirm-8da4w/pdiff.csv`).

Response surface (`results/780m/space/dq/importance/`; y = log kernel time, mean of the three screened shapes;
slowest configuration 17.5 times the fastest):

| parameter | levels | effect when everything else is chosen well (best of level against the best) | matters |
|---|---|---|---|
| `WG_TILE_M` | 32, 64, 128, 256 | 32: +31.6 %; 64: +3.6 %; 128: +0.9 %; 256: best | yes |
| `WG_TILE_N` | 32, 64, 128, 256 | 64: best; 128: +0.9 %; 32: +5.6 %; 256: +17.8 % | yes |
| `WG_TILE_K` | 16, 32, 64 | 64: best; 32: +0.9 %; 16: +4.3 % | yes |
| `SG_GRID_Y` | 1, 2, 4, 8 | 8: best; 4: +0.9 %; 2: +2.6 %; 1: +5.6 % | yes |
| `SG_GRID_X` | 1, 2, 4, 8 | 4: best; 1: +0.9 %; 2: +1.8 %; 8: +2.7 % | only 8 |
| `A_MAP_FULL` | off, on | on: best; off: +2.6 % | yes |
| `A_BLOCKS` (`A_MULTI_BLOCK`) | off, 1, 2, 4 | 1: best; 2: +1.3 %; off: +1.5 %; 4: +1.9 % | no (tied) |
| `SUBGROUP_SIZE` | 32, 64 | 32: best; 64: +0.9 % | no |
| the `bt` weight staging | off, on | off: best; on: +0.9 % | no |
| MMA shape | | this device exposes 16 x 16 x 16 only | - |

- The parameters do not act independently: an additive model of the nine explains only 47 % of the variance.
  The largest pair interactions (R^2 gained by the pair's cells): tile M x tile N 0.081, tile N x `A_BLOCKS`
  0.078, tile M x grid Y 0.072, tile M x tile K 0.057, tile K x `A_BLOCKS` 0.049, tile M x `bt` 0.039. In all of
  them the best level of one changes with the other.
- What the interactions encode is one derived quantity: activation slots per thread, M * K / 16 / (threads of
  the workgroup). At 1 (every thread unpacks exactly one slot) the best configurations sit; 2 costs +1.3 %, 4
  +1.9 %, 0.5 +2.9 %, 0.25 +7.5 %, 0.125 and below +45 % to +173 % (threads that have nothing to unpack still
  take part in the staging).
- That is why the winner is not a neighbour of the earlier choice: 256 x 64 with K = 64 on a 4 x 8 grid of
  32-wide subgroups is 1024 threads for 1024 slots, with a 32 x 16 tile per subgroup (two MMA tiles), against
  128 x 64, K = 32, 2 x 2 grid (128 threads, 2 slots each, 64 x 32 per subgroup).

## Part 1, SDPA QK^T and attn*V: the best kernel per shape within the existing bodies (`results/780m/space/{sdpa,confirm-sdpa}/`)

**All statically surviving configurations were measured**: 1,724 QK^T (of 9,216 combinations) and 470 attn*V (of
4,608), 1,796 runs from 2026-10-06 11:51 to 23:15 UTC at a steady clock (20 warm-up + 8 timed runs, coefficient of
variation 0.2 to 0.5 %). Each run: the 8 cases of the extended correctness tier, then the op time per model at
S = 2048. Every configuration ran its own kernel with 0 mismatches and pairing ok; none failed. Repeat stage: the
10 fastest per model and family plus the table and `780m-refine3` kernels (44 configurations), five timing
repeats (spread at most 1.5 % for 122 of 129 rows) and **12 correctness passes each: 308 separate passes + the 5
of the timing repeats, 0 failed cases** (`confirm-sdpa/passes.csv`; three attn*V configurations fit only
head_dim 128 and run their kernel in 4 of the 8 cases, two configurations in 7 of 8).

**Within the existing kernel bodies, the best configuration per shape on this device and driver is** (op time per
layer at S = 2048, median of 5):

| family | shape | kernel | op time | tied within 2 % | `780m-refine3` choice | release table kernel |
|---|---|---|---:|---:|---:|---:|
| QK^T | head_dim 64 (1B, 32 heads) | `pk_t64x32k16g22s32nf` | 2,661 us | 11 | `t128x64k32g22s64nf` 2,911 (+9.4 %) | `t128x64k32g22s64` 4,735 (+77.9 %) |
| QK^T | head_dim 128 (3B, 24 heads) | `pk_t64x64k32g42s32nf` | 3,317 us | 5 | 4,233 (+27.6 %) | 5,589 (+68.5 %) |
| QK^T | head_dim 128 (8B, 32 heads) | `pk_t64x64k32g42s32nf` | 4,221 us | 6 | 5,486 (+30.0 %) | 7,219 (+71.0 %) |
| attn*V | head_dim 64 (1B) | `ml_t32x32k32g21s32` | 2,449 us | 2 | `t64x64k32g42s32` 2,639 (+7.8 %) | `t64x64k32g22s64` 2,740 (+11.9 %) |
| attn*V | head_dim 128 (3B) | `ml_t64x64k32g22s32` | 2,460 us | 1 | 2,692 (+9.4 %) | 2,964 (+20.5 %) |
| attn*V | head_dim 128 (8B) | `ml_t64x64k32g22s32` | 3,290 us | 2 | 3,617 (+9.9 %) | 3,980 (+21.0 %) |

The choice follows the head dimension, not the model: the head_dim 64 winners are 9 to 13 % behind on head_dim
128 and the other way round.

**These kernels are not in any profile and were not gated end to end.** With the fused attention node
(candidates 8 to 11) QK^T and attn*V only serve the calls it does not take (unaligned prompts, decode), so on the
benchmark prompt they would change nothing; they are a result for the three-kernel path: QK^T + softmax r3 +
attn*V per layer 9.97 -> 9.53 ms (1B), 10.23 -> 9.09 ms (3B), 13.55 -> 11.95 ms (8B), against 2.26 / 3.22 / 4.05 ms
for the fused kernel. Adopting them would be a new candidate: the QK^T winners accumulate in different chunks
(K = 16 on head_dim 64) and read packed K, an arithmetic change that needs the SDPA tiers at 12 passes, the
error against the fp32 reference and the real-text comparison of the owner decisions of 2026-10-04.

Response surface, QK^T (`sdpa/importance-qk/`; 1,724 configurations, mean over the three models; slowest 14
times the fastest; best of level against the best):

| parameter | effect | matters |
|---|---|---|
| `NO_MASK_FILL` ("nf") | on: best; off: +51.5 % | yes, the largest |
| packed K staging ("pk") | on: best; off: +8.9 % | yes |
| `WG_TILE_M` | 64: best; 32: +2.6 %; 128: +3.6 %; 256: +18.9 % | yes |
| `WG_TILE_N` | 64: best; 32: +3.2 %; 128: +3.6 %; 256: +12.0 % (per head_dim: 32 on head_dim 64, 64 on 128) | yes |
| `WG_TILE_K` | 32: best; 16: +0.3 %; 64: +6.5 % (16 on head_dim 64, 32 on 128) | only 64 |
| `SG_GRID_X`, `SG_GRID_Y` | 1 to 4: within 0.8 %; 8: +4.1 / +5.6 % | only 8 |
| `SUBGROUP_SIZE` | 32 and 64 equal (0.0 %) | no |

attn*V (`sdpa/importance-av-d128/`, 470 configurations on 3B and 8B; `importance-av-d64/`, the 268 whose tile N
divides 64, on 1B):

| parameter | head_dim 128 | head_dim 64 | matters |
|---|---|---|---|
| `WG_TILE_M` | 64: best; 32: +1.5 %; 128: +9.3 %; 256: +23.9 % | 32: best; 64: +4.1 %; 128: +12.0 %; 256: +27.4 % | yes |
| `WG_TILE_N` | 64: best; 128: +1.5 %; 32: +8.6 % | 32: best; 64: +4.1 % | yes |
| `WG_TILE_K` | 32: best; 64: +5.9 % | 32: best; 64: +4.1 % | yes |
| `SG_GRID_Y` | 2: best; 1: +1.5 %; 4: +8.2 %; 8: +17.1 % | 1: best; 2: +1.1 %; 4: +4.3 %; 8: +12.7 % | yes |
| `SG_GRID_X` | 2: best; 4: +1.5 %; 8: +6.6 %; 1: +6.9 % | 2: best; 1: +1.1 %; 4: +4.1 % | yes |
| `SUBGROUP_SIZE` | 32: best; 64: +6.6 % | 32: best; 64: +2.3 % | yes |
| multi-load ("ml") | on: best; off: +9.4 % (only 24 configurations exist without it) | on: best; off: +7.0 % | yes |

- Interactions: the additive model explains 58 % (QK^T), 37 % and 47 % (attn*V) of the variance. The pairs that
  carry the rest are the same in all three: grid X x grid Y (0.073 to 0.085), tile M x grid Y (0.049 to 0.149),
  tile N x grid X or tile M x grid X (0.025 to 0.033), grid x subgroup size (0.02 to 0.05).
- As for the linear families, they reduce to derived quantities: the MMA tiles a subgroup owns (2 to 8 is within
  1 % for QK^T, 2 to 4 within 3 % for attn*V; 32 costs +8 to +30 %, 64 three to five times the time, 128 eleven
  times) and the threads of the workgroup (64 to 256; 32 threads +11 to +32 %, 1024 threads +7 to +27 %).

## Can the sweep slot in between two timed runs of a session? No (checked 2026-10-04 18:44 PDT, during `c7`)

Asked by the owner after seeing `sweep_space.py` alive and the GPU at 79 C during the candidate-7 session.

- `e2e5.sh` takes the gpu-lab lock once, at the start of the script (`exec 9>>lock; flock 9`), and holds it until
  it exits: the lock is per session, not per run. `lslocks` during the session shows `e2e5.sh` (pid 565053) as the
  holder of `~/.cache/gpu-lab/lock-00000000-c400-...`. `sweep_space.py` takes the same lock for each of its own
  runs, so it cannot run while a session holds it.
- Independently of the lock, the sweep waits while `<artifacts>/PAUSE` exists; `gate2.sh` creates it before the
  cool-down wait and removes it when the whole gate is done. `PAUSE` has existed since 17:42 PDT.
- The enumeration process seen alive (pid 537928) is that paused sweep. Its last result row is stamped
  21:15:48 UTC (14:15 PDT), four hours before the session's first run (01:13:58 UTC); `raw/space/results.csv` and
  `sweep.out` have not been written since. No microbench process ran during the session.
- 79 C is the in-run temperature of `llama_main` itself (the runs of this session end at 60 to 73 C and peak
  higher; the earlier sessions peaked at 88 to 95 C). The session started at 45 C after the full 30 min wait
  (`prestart.txt`: the device did not get below 45 C this evening; `t3-aa` started at 42 C).

Conclusion: the session is not affected and is not repeated. The one way a sweep job could run next to a
session would be a session script that locks per run; none of the session tools does.

## Owner decisions in force (read from `CAMPAIGN.md`, section "Owner decisions", 2026-10-04)

- Next-token near-ties: a `DIFFER` no longer rejects by itself if the logits of all four arms, a real-text
  comparison on at least 32 prompts and the parent-tiled vs parent-default floor are measured; recorded as
  `ACCEPTED (near-tie, owner decision 2026-10-04)`. Not used so far: no `DIFFER` has occurred in this round.
- Arithmetic changes are judged against the fp32 reference (rms and maximum error not larger than the parent's
  on the production shapes), with a gross-divergence check; recorded as `ACCEPTED (reference-error rule, ...)`.
  A change meant to be bit-identical does not use this rule and must show bit-identical output. Candidate 7
  (softmax r3) is meant to be bit-identical on every element that is read; the bitwise comparison of the SDPA
  output against the parent is still owed. Candidate 8 (fused SDPA) changes the arithmetic and uses this rule.
- Large parameter spaces: the method below.

## Owner decision, 2026-10-04 (relayed): how the 4w family is searched

The full 4w space (26.8 days) is not wanted and the staged plan (geometries first, flags afterwards) is dropped
before any of its rows were measured. Instead:

1. cost per configuration broken down, and a cheap screening mode validated against the full measurement by rank
   correlation on at least 50 configurations;
2. a uniform random sample of 2,500 of the 125,712 survivors, seed 20261004
   (`tools/sample_space.py`; survivor list sha256 `5b0cd644...605d75`, `enum_space.py --list 4w`), screened;
3. parameter importance and pair interactions from that sample, and a check of the staged plan's independence
   assumption against it;
4. local refinement around the best 20 sampled configurations (one or two parameters at a time);
5. the best 10 per shape get the full measurement (5 repeats, 12 correctness passes).

8da4w, QK^T and attn*V stay full enumerations.

### Cost of one 4w configuration (`raw/cost/cost-1.csv`, one configuration, texture3d, correctness off)

| mode | shapes | runs per shape | wall |
|---|---:|---|---:|
| no case (process start, Vulkan device, exit) | 0 | | 0.05 s |
| full measurement | 12 | 3 warm-up + 5 timed | 16.3 s |
| setup only | 12 | 0 + 1 | 12.1 s |
| `w1_w3` of each model | 3 | 3 + 5 | 5.3 s |
| `w1_w3` of each model | 3 | 1 + 2 | 4.6 s |
| `wq_wo` of each model | 3 | 1 + 2 | 2.2 s |

About 74 % of the full measurement is per-shape setup in the test harness (host-side generation, packing and
upload of up to 58.7 million 4-bit weights per shape, graph build, pipeline creation): 0.26 to 2.2 s per shape,
growing with N x K. The 84 extra runs cost 4.2 s. Process start is negligible and one shader module is compiled
per process, so neither is worth optimising. A run without warm-up reads 1.5 to 2.2 times too slow, so the
screening mode keeps one warm-up run.

### Screening mode against the full measurement (`results/780m/space/validate/`)

The first 64 configurations of the random sample (a prefix of the draw, so itself a uniform sample), each
measured in full (12 shapes, 3 + 5 runs; 24.6 s a configuration on this sample) and in five candidate modes.
Spearman rank correlation of the mode's kernel time with the model's linear time per layer from the full
measurement (2 `wq_wo` + 2 `wk_wv` + 2 `w1_w3` + `w2`), per model (8B / 1B / 3B), and how many of the full top 10
are in the mode's top 10 and top 20:

| mode | s per configuration | rho, 8B / 1B / 3B | top 10 in top 10 | top 10 in top 20 |
|---|---:|---|---|---|
| `wq_wo`, 1 + 1 runs | 2.4 | 0.996 / 0.995 / 0.990 | 10 / 10 / 9 | 10 / 10 / 10 |
| **`wq_wo`, 1 + 2 runs** | 2.7 | 0.995 / 0.996 / 0.997 | 10 / 9 / 9 | 10 / 10 / 10 |
| `wq_wo`, 1 + 3 runs | 3.1 | 0.995 / 0.997 / 0.997 | 10 / 10 / 9 | 10 / 10 / 10 |
| `w1_w3`, 1 + 2 runs | 6.5 | 0.995 / 0.998 / 0.998 | 10 / 9 / 9 | 10 / 10 / 10 |
| `wk_wv`, 1 + 2 runs | 1.5 | 0.992 / 0.992 / 0.996 | 10 / 10 / 10 | 10 / 10 / 10 |

Against every single shape of the full measurement the screening shape has rho 0.972 to 1.000 (lowest for `w2`,
the K = 8192 / 14336 shape). All 64 configurations dispatched their own kernel on all 12 shapes; the kernel-time
coefficient of variation over the 5 timed runs has median 0.5 % and 90th percentile 2.2 %.

Chosen: `--op=wq_wo --runs=1,2` (N = K = 2048 / 3072 / 4096). `wk_wv` is cheaper and ranks as well here, but its
N = 512 / 1024 is the one shape where the wide tiles were measured to lose in round 1, so it is not used alone.
What the agreement does not show: the sample spans a 10x range of kernel times, so a high rho is expected; the
top-10 overlap is the sharper check, and the confirmation stage measures the leaders on all 12 shapes anyway.


## Done in this round

| step | session | result |
|---|---|---|
| baseline re-check, cool start (40 C) | `t1-recheck` | `dev/1.5` 2698.29 / 2544.10 (1B 4w / 8da4w), 1147.98 / 1043.30 (3B), 523.65 / 488.32 (8B) tok/s: within 0.8 % of `sarc-1.5-e2e-benchmark/results/cells.csv` (2698.29 / 2544.1, 1151.21 / 1051.33, 525.80 / 489.13). `780m-refine3` over `dev/1.5`: +4.98 / +11.19, +3.84 / +11.98, +3.99 / +12.44 %, geomean +8.00 %. The flagged 8B 4w cell, re-measured cool: **+3.99 %** (523.65 -> 544.54). Binaries of 2026-10-03, not rebuilt. |
| A/A, warm start | `t2-aa` (superseded) | started 90 s after the re-check (idle reference 44 C, run starts at 49 C): a warm-up, not used as the noise floor. Geomean +0.01 %, cells within +-0.14 %. |
| A/A, cool start (42 C, the idle temperature after 30 min) | `t3-aa` | **noise floor**: geomean +0.01 %, cells -0.23 to +0.19 %, repeat spread at most 0.52 %, clock 2797 to 2800 MHz, next token SAME in all cells on both prompts. Per cell (tok/s, 4w / 8da4w): 1B 2828.73 / 2828.73, 3B 1192.08 / 1172.97, 8B 544.10 / 549.50. |

### The random sample: 2,500 4w configurations screened (`results/780m/space/rand/`)

Seed 20261004, `wq_wo` of each model with 1 + 2 runs, 15:29 to 17:40 PDT. After the first 136 configurations the
cooling limit between runs was raised from 55 to 62 C (it cost 4.5 s a configuration); the 64 validation
configurations re-screened at 62 C differ from their 55 C values by a median of +0.08 % (5th to 95th percentile
-2.9 to +3.3 %, which is the repeatability of this mode). 2,444 configurations were timed; no process failed and
no Vulkan context was lost. The other 56 (2.2 %) kept the table kernel on all shapes: the selector's shared-memory
model (`impl/sarc/Select.cpp`, release zone) refuses them although the shader fits (it does not know the band
drain or `FRAG_LAYOUT`); reaching them needs a hook.

The space is mostly bad and sharply peaked: the median configuration is 2.23x slower than the sample's fastest,
the slowest 248x; 0.1 % are within 2 % of the fastest, 0.3 % within 10 %, 10 % within 1.5x.

Fastest of the sample (geomean kernel time of the three `wq_wo` shapes): `bx_t128x256k32g42s32f32xpiw` 3612 us,
`t128x128k32g42s32f32ciw` 3659 us (the shipped tile + `IMG_W`), `bx_t128x64k16g22s32f32xpiw` 3687 us,
`t256x128k32g42s32f32bbt` 3718 us. The shipped tile and the refine3 wide tile themselves are measured in the same
mode in refinement round 1 (they are not in the sample).

Parameter importance (`importance/importance.csv`, `levels.csv`; preliminary, the local effects around the optimum
come from the refinement). "Range" is between the geomean times of the best and the worst level over the whole
sample; "unique R^2" is what the additive model of log time loses without the parameter; "best-of-level spread"
is how much slower the fastest sampled configuration of the worst level is than the fastest of the best level
(what the parameter still costs when everything else is chosen well; it is a minimum over a sample, so about 3 %
of it is noise):

| parameter | range over the sample | unique R^2 | best-of-level spread | fastest level / levels of the fastest 5 % |
|---|---:|---:|---:|---|
| `WG_TILE_M` | 380 % | 0.254 | 47 % | 128 (63 % of the fastest 5 %), then 64, 256; 32 never |
| `WG_TILE_N` | 272 % | 0.154 | 35 % | 128, 64, 256 all present; 32 never |
| `SG_GRID_Y` | 118 % | 0.108 | 23 % | 2, then 4, 1 |
| `SG_GRID_X` | 29 % | 0.100 | 16 % | 2 and 4 |
| ACC (accumulator) | 129 % | 0.080 | 52 % | fp32: 90 % of the fastest 5 %, best in all 21 geometry strata |
| `WG_TILE_K` | 144 % | 0.028 | 27 % | 32 and 16; 64 is 27 % behind at best |
| `SUBGROUP_SIZE` | 0.6 % | 0.013 | 19 % | 32 (84 % of the fastest 5 %): no effect on average, large near the optimum |
| CSH (drain) | 46 % | 0.002 | 38 % (full without pool) | in Ash, band and pooled full within 3 % of each other |
| `IMG_A` | 4.2 % | 0.000 | 6.9 % | off |
| `SH_F16V4` | 2.3 % | 0.000 | 6.5 % | off |
| `FRAG_LAYOUT` | 6.0 % | 0.000 | 4.4 % | off |
| `B_COLMAJOR` | 9.5 % | 0.000 | 2.9 % | either |
| `IMG_W` | 4.2 % | 0.000 | 2.9 % | either |
| texel-wise staging (`bx`) | 1.2 % | 0.000 | 1.3 % | either: does not matter |
| derived: MMAs per subgroup, (M/Y/16) x (N/X/16) | 4485 % | (eta^2 0.605 alone) | | 8 and 16 (4 to 32 usable); 64 and more are 11 to 25x slower on average |
| derived: threads per workgroup | 340 % | (eta^2 0.062) | 46 % | 256, then 128 |

One derived quantity explains more of the variance (60 %) than all 14 yaml parameters additively (63 %): how
many 16 x 16 MMA tiles one subgroup owns. That is why M, N and the grid look important and interact.

Pair interactions (`importance/pairs.csv`; gain in R^2 over the additive model, noise level about 0.001):
tile M x ACC 0.038, tile N x ACC 0.034, tile K x ACC 0.032, M x grid Y 0.021, M x N 0.019, M x K 0.016,
N x K 0.014, K x CSH 0.007, N x grid X 0.006. No pair of two boolean options is above 0.003.

The staged plan's independence assumption, checked (`importance/independence.txt`): it does not hold. The
additive model explains 63.1 % of the variance of log time; geometry x geometry pairs add 14.4 points,
geometry x option pairs add 12.3 points, option x option pairs 1.2 points (all pairs: 87.9 %). The options do not
interact with each other, but they do interact with the tile geometry about as strongly as the geometry
parameters interact among themselves. For the accumulator the interaction changes the size of the effect, not
its sign (fp32 is best in every stratum). For the drain mode and the boolean options the best level changes
with the geometry (drain: a different best level in 10 of 21 strata; `FRAG_LAYOUT` 5, `IMG_A` 5, `IMG_W` 8), so
"best geometry first, then best flags on it" could have missed combinations; at the size of those effects
(1 to 10 %) that is exactly the margin this campaign is looking for.

### Refinement round 1 (`results/780m/space/refine1/`, 20:21 to 21:41 PDT, screening mode)

1,002 configurations that differ from one of the 22 best sampled ones in one parameter (and in two for the best
five); 991 dispatched their kernel on the three screened shapes. `tools/refine_summary.py`:

- Fastest by geomean of the three shapes: `t256x128k32g24s32f32bbt` 3533 us, `t256x128k32g18s32f32bbt` 3535,
  `bx_t128x256k32g42s32f32xpi` 3578, `t128x256k32g42s32f32cbt` 3598; the sample's best was 3612 and the
  refine3 tile `t128x256k32g42s32f32c` reads 3623 (+2.5 %). All of that is inside the screening scatter (+-5 %):
  the round found no geometry or option that clearly beats what the sample already had. The full measurement
  decides (queued).
- Fastest alternative level of each parameter, relative to its centre (median over the centres; below 1 = faster):
  accumulator 12.0 (fp32 everywhere), subgroup size 1.49, tile K 1.20, grid X 1.19, grid Y 1.17, tile M 1.14,
  tile N 1.14, `FRAG_LAYOUT` 1.06, `B_COLMAJOR` 1.02, texel-wise staging 1.01, drain mode 1.01, `IMG_W` 1.00,
  `IMG_A` 1.00, `SH_F16V4` 0.96 (4 centres only). So around the best configurations every geometry parameter is at
  a local optimum (the nearest alternative costs 14 to 49 %), and the options are flat within the scatter.

## Part 1: parameter space (static count, `tools/enum_space.py`)

| family | combinations | device | flags | geometry | shared memory | shape | survivors | tile geometries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4w | 75,497,472 | 64,487,424 | 10,515,456 | 292,692 | 76,188 | 0 | 125,712 | 590 |
| 8da4w zpg | 165,888 | 133,632 | 7,632 | 22,002 | 384 | 0 | 2,238 | 594 |
| SDPA QK^T | 9,216 | 3,840 | 0 | 2,832 | 820 | 0 | 1,724 | 628 |
| SDPA attn*V | 4,608 | 1,920 | 0 | 2,023 | 28 | 167 | 470 | 446 |

Columns "device" to "shape" count the combinations removed by each static rule, in that order.
The rules replay without a rejection on the 64 existing variants this device runs (`enum_space.py --check`).

Plan (`tools/plan_space.py`; `results/780m/space/sweep-manifest.csv`): 8da4w (2,238), QK^T (1,724) and attn*V
(470) in full, batches b00 to b07. The 4w rows of that manifest (the staged plan, batches b07 to b10) are not run.

Timing sample, 30 configurations drawn at random (`results/780m/space/sample30-results.csv`, 20:21 to 20:33 UTC):

| family | sample | s per configuration | planned | projected |
|---|---:|---:|---:|---:|
| SDPA (one QK^T and one attn*V token per run: 8 correctness cases, then timing) | 10 runs, 20 tokens | 12.5 per run | 1,724 runs | 6.0 h |
| 8da4w | 10 | 15.8 | 2,238 | 9.8 h |
| 4w, staged (dropped, see the owner decision) | 10 | 18.4 (15.3 to 26.4) | 1,642 | 8.4 h |
| total without 4w | | | | 15.8 h, plus cooling waits and the 06:40 to 07:40 pause |
| 4w, all 125,712 survivors | | 18.4 | 125,712 | **642 h (26.8 days): over the 48 h limit, not started** |

The first sample is in `<artifacts>/superseded/sample30-kernel-name-not-recognised/`: the linear family names
lacked `linear_`, which the microbench needs to report the kernel time.

Found by the sample: `CSH_BAND` 4w tiles whose `SG_GRID_Y * 16 * WG_TILE_N * 2` reaches 64 KiB are refused by
the selector's shared-memory check (`impl/sarc/Select.cpp`, release zone; it does not know the band drain), so
they keep the table kernel and are recorded with `dispatched=0`.

Linear configurations are timed with the microbench's correctness gate off (the gate only knows the table
kernels' small shapes); correctness is checked in the repeat stage. SDPA configurations are checked in every run.

## Part 2: softmax (measured in the microbench; `raw/softmax/`, build `hook2`)

`sarc_sdpa_attn_weights_softmax` op time per layer at S = 2048, profile `780m-refine3` for QK^T and attn*V, two
runs each; all variants reached through `hooks/softmax-name-hook.patch` applied in a scratch tree:

| variant | what | 1B (32 heads) | 3B (24 heads) | 8B (32 heads) | extended correctness tier |
|---|---|---:|---:|---:|---|
| release | three loads of the row prefix, exp twice | 7.19 / 7.21 ms | 5.42 / 5.41 ms | 7.21 / 7.22 ms | 8 of 8 |
| `r1` | one load kept in registers, exp once | 5.94 / 5.95 ms | 4.46 / 4.46 ms | 5.94 / 5.93 ms | 8 of 8 |
| `r2` (dropped) | r1 + in-wave tree reductions, no shared memory | 5.95 ms | 4.45 ms | 5.93 ms | 8 of 8 |
| `r3` | r1 + zero fill bounded to the K-chunks the SARC attn*V kernels read | 4.41 / 4.43 ms | 3.32 / 3.30 ms | 4.45 / 4.42 ms | 8 of 8 |
| `m1` (measurement only, wrong results) | r3 without exp | 4.42 / 4.42 ms | 3.32 / 3.31 ms | 4.42 / 4.45 ms | 0 of 8, as intended |

- r3 is -38.6 % on the kernel (r1 -17.6 %). Removing exp entirely changes nothing (m1 = r3), and neither do the
  barriers (r2 = r1): the kernel is bound by memory traffic, about 284 MB per layer at about 64 GB/s with r3.
- r3 depends on its neighbour: the release softmax comment says attn*V stages every chunk below `context_len`,
  but the SARC attn*V kernel already stops at the chunk holding column `a + M - 1 + input_pos` of its row tile
  (`useful_chunks`), so zeros past that chunk are never read. r3 keeps the full fill unless S and `input_pos` are
  multiples of 256 and head_dim a multiple of 64. With an attn*V that reads whole rows (upstream kernel,
  `ET_VK_DISABLE_COOPMAT`) r3 would be wrong; the microbench pairing check now fails such a pairing.
- Expected end to end (16 / 28 / 32 layers): about -44 ms (1B), -59 ms (3B), -89 ms (8B).

## Closing checks (2026-10-07 00:55 UTC; `check.sh --no-build` run again 01:10 UTC after the review follow-up: PASS)

- `sarc/tools/check.sh --no-build`: **PASS** (zone rule against `release/1.5`, twin wrappers, `test_sarc_select`
  1240 checks / 31 rows on the release tables and 1433 checks / 122 candidates with the dev zone).
- `sarc/tools/spirv_golden.py` on the final build (`head2`): spirv golden: PASS (53 shipped variants). No shipped SPIR-V changed.
- `git diff --name-status ef079ac41 HEAD` outside the dev zone and this change directory: exactly four files,
  `impl/SDPA.cpp`, `impl/sarc/SdpaCoopmat.cpp`, `impl/sarc/SdpaCoopmat.h`, `impl/sarc/Select.h`: the two entry
  points committed under the owner decision of 2026-10-05 (`b969e8f1c`, `1c8861aa7`; diffs in `proposal.md`).
  `check.sh` does not flag them (its zone rule allows the release zone, and `SDPA.cpp` is listed in
  `sarc/HOOKS`); they are permitted by that decision and by nothing else. No gate, golden, tolerance, prompt or
  tool under `sarc/tools/` was edited; nothing was promoted.
- `tools/rgp_chunks.py` stays untracked (a reader for `.rgp` files from before the owner's decision; it sets no
  variable, starts no GPU job, nothing calls it).

## Next (nothing is queued; these are the owner's decisions)

1. Whether candidate 11 is adopted: it rests on the fused attention entry point (`1c8861aa7`), which is committed
   but subject to the owner's review before any promotion, and on the softmax variant hook (`b969e8f1c`).
   Without the fused node the branch still offers `c7` (softmax r3, bit-identical output, +4.10 % over
   `780m-refine3`) and the 4w / 8da4w kernels per shape.
2. Whether the QK^T / attn*V kernels per head dimension ("Part 1, SDPA") become a candidate for the three-kernel
   path (unaligned prompts): 0.4 to 1.6 ms a layer there, nothing on the benchmark prompt; an arithmetic change
   that needs its own gate.
3. Outside both zones, reported and not changed: the upstream operators (elementwise, copies, the 8da4w
   activation quantize) are now 17 to 27 % of the prefill; packing K and V where the cache is written would
   remove the fused kernel's copy pass.

## Blocking

Nothing.
