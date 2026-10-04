# STATUS: sarc-1.5-4070ti-prefill-refine

**2026-10-04 20:15 UTC, gpu-dev-4004. RUNNING. The write fence is lifted; builds, the baseline, the A/A session,
the parent control and the per-op traces are done. No candidate has been gated yet; the stop rule is not met;
the branch is not pushed.**

## Now

- Running: igpu-roofline `fast` on the current driver (615.71.09), results in
  `.artifacts/4070ti-prefill-refine/roofline/2026-10-04-fast/` (detached, under the gpu-lab lock).
- Next: kernel-level SDPA screen of the 13 dev variants (`tools/sdpa_screen.sh`), then candidate 1
  (`4070ti-refine1`, QK^T without mask fill + attn*V, both subgroup 32) through `tools/gate_sdpa.sh`; phase
  timing of the 4w `ga` tiles and of zpgtr with the PROF twins before any linear sweep.
- Blocking: nothing.

## Built

| tag | commit | note |
|---|---|---|
| `parent` | `6a7cc8cc6` pristine | spirv_golden PASS (53 shipped variants) |
| `topic1` | `3e2b9dc4d` + `tools/local-hook-nvidia-sdpa.patch` (not committed) | spirv_golden PASS (53 shipped variants) |

Both from `git archive` trees with `sarc/tools/build.sh` in `localhost/et-vk-build:rocky10` through the
docker shim; provenance in `.artifacts/4070ti-prefill-refine/build/<tag>.src.txt`.

## Baseline and A/A (session `s1-aa`, parent vs `topic1` without environment, record-only clock)

tok/s, median of 5 valid runs per arm, arms interleaved:

| cell | parent | topic, no env | ratio | `cells.csv` (dev/1.5) |
|---|---:|---:|---:|---:|
| 1B 4w | 19692.3 | 19692.3 | 1.0000 | 19692.3 |
| 1B 8da4w | 20898.0 | 20898.0 | 1.0000 | 20898.0 |
| 3B 4w | 8714.9 | 8714.9 | 1.0000 | 8752.1 |
| 3B 8da4w | 9660.4 | 9615.0 | 0.9953 | 9660.4 |
| 8B 4w | 4481.4 | 4481.4 | 1.0000 | 4491.2 |
| 8B 8da4w | 5007.3 | 4995.1 | 0.9976 | 5031.9 |

The baseline agrees with `cells.csv` within 0.5 % in every cell. A/A geomean 0.9988 (-0.12 %), largest cell
difference 0.47 %, repeat spread at most 2.1 % (1B 8da4w, timer quantisation: 97 ms against 96 or 98 ms).
Next token SAME in all six cells on the four prompts. 60 timed runs, all rc 0.
1B quantises as announced: all ten 1B 4w runs read 104 ms = 19692.3 tok/s. One timer step is 1 %.

Clock: the per-run median of `nvidia-smi clocks.gr` is 2565 to 2790 MHz, differs by cell and by repeat without
any effect on the rate, and one run that followed an idle period still showed the ramp (675 MHz median, same
rate). `calibrate_clock.py` now writes one device-wide threshold, floor(0.97 x lowest per-cell median) =
**2502 MHz** (`results/4070ti/clkmin.json`); 1 of 60 calibration runs is below it. This replaces the per-cell
rule with a 3 % spread limit, which refused this session. My choice; say so if another rule is wanted.

## Parent control (`s0-parent-verify`)

Unmodified `sarc/tools/verify.sh --models 1b,3b,8b --schemes 4w,8da4w --pdiff`, no environment: correctness
rc=0, `linear 4w rc=1` and `linear 8da4w rc=1` (24 confirmed + 24 `unexpected_coopmat` cases, the known
texture3d report), 12 of 12 production-diff ALL PASSED, default vs tiled SAME on the check and the unaligned
prompt for 1B 4w and 8da4w, decode 31 tokens, 22 of 22 runner calls rc 0. `gate_check.py verify`: ACCEPT.
SDPA correctness on the parent (recorded only): 0 mismatches in every case, `qk_coopmat=NO` (stock kernels).

Two things happened on the way, both kept:
- First attempt aborted by my own watcher (`superseded/zombie-sighting/`): one sighting of a `llama_main`
  pid during `verify.sh`. Reproduced with a dummy: a runner of ours between exit and `wait` has no readable
  environment and was reported as foreign. `common.sh` now judges such a process by its parent (ours ->
  ignored, foreign parent -> still reported; both tested). No other user was logged in and no GPU client was
  listed. I read it as our own runner, not as a foreign process; the control was run again from scratch.
- Second attempt: `verify.sh` complete, but `gate_check.py` rejected it because it demanded a `[correctness]`
  summary line that the microbench prints only on failure. Fixed (completeness is now the rank-3 verdict block
  at the end of the log; regression test added, 29 tests); the check was re-run on the same files. The first
  verdict is kept as `gate.done.first-check`.

## Where the time goes (parent, warm ETDump, ms per 2048-token prefill; `s1-aa/trace`)

| family | 1B 4w | 1B 8da4w | 3B 4w | 3B 8da4w | 8B 4w | 8B 8da4w |
|---|---:|---:|---:|---:|---:|---:|
| linear GEMM | 34.2 (33 %) | 26.7 (28 %) | 98.1 (42 %) | 75.6 (36 %) | 242.6 (53 %) | 185.0 (45 %) |
| QK^T (stock) | 16.3 | 16.0 | 42.7 | 41.8 | 65.3 | 63.1 |
| attn*V (stock) | 21.7 | 21.4 | 39.6 | 39.1 | 60.2 | 59.7 |
| softmax (stock) | 13.6 | 13.6 | 17.9 | 17.9 | 27.3 | 27.3 |
| attention total | 51.6 (51 %) | 51.0 (54 %) | 100.2 (43 %) | 98.8 (47 %) | 152.8 (33 %) | 150.1 (37 %) |
| copy/view | 7.8 | 4.6 | 17.7 | 11.2 | 29.7 | 18.1 |
| elementwise | 6.9 | 6.9 | 12.6 | 12.9 | 25.6 | 26.4 |
| 8-bit quantize | - | 3.4 | - | 8.9 | - | 23.7 |
| total dispatch | 102.1 | 94.1 | 232.8 | 211.5 | 457.2 | 409.4 |

Attention is the largest block on 1B and 3B and a third of 8B, all of it stock kernels. One fp16 attention
matrix is 268 MB per layer on 1B (32 heads x 2048 x 2048): QK^T writes it, softmax reads and rewrites it,
attn*V reads it. My reading, to be checked against the fresh roofs: these three kernels are bound by memory
traffic, not by arithmetic, so the gain should come from not writing and not reading the masked half.

## Incidents

- 20:01 UTC: the traced `8b 4w parent` runner wrote its ETDump and then spun at 93 % CPU without printing the
  stats line (GPU idle, `nvidia-smi` answering, no Xid in the kernel log). Killed by me with SIGTERM after
  10 min; the ETDump is complete (same size and dispatch count as the candidate arm's) and was analysed. One
  such hang in about 200 runner calls so far; the e2e notes report rare exit-time failures on this card.
- No device loss, no foreign GPU process.

## Open

- igpu-roofline: the tool is on this host only as `~/.cache/igpu-roofline/fleet-fast-20260926/`; it is run from
  there with its existing environment and a new results directory under `.artifacts`. Nothing in it is edited.
- ETDump analysis runs with a venv under `.artifacts` (`executorch` 1.5.1 wheel + CPU torch), `TRACE_PY`.
- `gate*.sh`, `screen.sh`, `sdpa_screen.sh` and the PROF decode have not run end to end yet.
- The softmax kernel name is fixed in the release zone (`impl/sarc/SdpaCoopmat.cpp`); no dev variant can
  replace it.

## Per-cell numbers against the parent

No candidate yet. Baseline above.

## SDPA reachability

SDPA is not reachable from the dev zone on this device: the profile only replaces an existing table choice and
the SDPA hooks in `impl/sarc/SdpaCoopmat.cpp` need a release-table row. As the campaign allows, the candidate
builds use `tools/local-hook-nvidia-sdpa.patch` (two `kUnverified` rows in `table_nvidia.cpp`, applied to
the candidate's archived source tree only, never committed, recorded in the build provenance) with
`ET_VK_SARC_UNVERIFIED=1`. The parent build is the pristine `6a7cc8cc6` without it.

# Tool history (reviews of the gate tools, before the first measurement)

## Tool defect from the fifth review (of `819e03685`), fixed

**A crashed or newly fallen-back linear case can no longer hide behind the parent's `rc=1`.**
`gate_check.py verify` parses every case of `linear-<scheme>.json`: the 24 expected identities (3 models x 4
projections x buffer/texture3d) each exactly once; no case crashed or without a kernel; `ok` true, positive
times, a known dispatch state. Per case, the shape and the (variant, dispatch) pair must equal the parent
control's, so the parent's existing anomaly (`unexpected_coopmat` on texture3d, the reason for
`linear <scheme> rc=1` in this device's recorded logs) passes only as the same anomaly. The cases of
`correctness.log` are compared the same way (coopmat kernel or not, per case name). `test_gate_check.py`
(28 tests) holds the review's reproduction (the genuine 24-case report in both arms, candidate case 0
crashed), a new tiled fallback in the JSON and one in `correctness.log` behind an unchanged rc=1, and
missing, duplicated and invalid cases. The same fallback present in the parent too is accepted.

## Tool defect from the fourth review (of `06f0e1243`), fixed

**`gate_check.py verify` reads the evidence behind `verify.out`, for the candidate and the parent control.**
- Exit statuses: `verify.sh` is not edited. The stage directory's `llama_main` is now `tools/llama_main_rc.sh`,
  which runs the real runner (`verify-bin/llama_main`) with the same arguments, environment and streams,
  returns its status and appends it to `verify-runs.jsonl`. Dry-tested with a stand-in runner: status passed
  through (0, 139), a `timeout` kill forwarded and recorded (143). Consequence: the `llama_main` hash in
  `verify/env.txt` is the wrapper's; the runner's own hash is in `STAGE.md`.
- Each of the 22 runner calls of `verify.sh` (12 prefill, 8 check/unaligned, 2 decode) needs exactly one
  recorded status of 0, stats and the expected prompt tokens. Prefill: positive rate equal to the one in
  `verify.out`, 0 generated tokens. Default vs tiled: recomputed from the two logs with `nexttoken.py`
  (failed or empty outputs are INVALID). Decode: generated tokens, positive rate, text after the prompt.
  Microbench: every case of `correctness.log`, every shape of the 12 production-diff logs, the linear JSONs
  (see the fifth review).
- `test_gate_check.py` holds the review's case (summary lines intact, every check/unaligned log
  empty), failed default-vs-tiled runs, and a teardown failure after the stats were printed (complete log,
  recorded status 139). All rejected.
- Run over the recorded verify logs of this device in `sarc-1.5-4w-port/results/4070ti/sarc`, the parsers for
  prefill, decode, correctness, production-diff and the linear JSON raise nothing; the only findings are the
  ones expected there (no recorded exit statuses, no 8da4w).
- **Changed rule, found on those logs:** the shipped state of this device has had `correctness rc=1` (a rank-3
  case that does not dispatch coopmat, all numeric cases PASSED). The earlier checker demanded rc=0 and would
  have rejected the parent control. Now `correctness rc` must fit its log and equal the parent's, with the same
  case count and the same set of cases without coopmat; any case not PASSED is still a rejection.
- Open: a strict rc 0 for every runner call may meet the exit-time crashes reported for this device in the
  e2e benchmark notes. If the parent itself crashes at exit, that is a finding to report, not to waive.

## Tool defects from the third review (of `1cfadaa2b`), fixed

1. **Timed runs are validated one by one.** `gate_check.py session` no longer counts rows by their `valid`
   flag. A timed run counts only if its log and its model/scheme/build/repeat identity are unique and agree,
   rc is 0, the rate is finite and positive, prompt tokens are 2048, generated tokens 0, no foreign GPU process,
   at least 2 clock samples and the median clock at or above the cell's calibrated threshold. A row marked
   valid that fails any of these is a finding. With the logs (`--require-logs`, as in the gates) every timed
   run is recomputed from its log and clock samples with `runrow.py`, the same code `e2e5.sh` now uses to
   write the row, and must equal it. `test_gate_check.py` (15 tests) holds both cases from the review: repeats
   2 to 5 failed (rc=134, no rate, 17 prompt tokens, 4 generated, no clock samples, empty logs) with valid=1
   kept, and five copies of each arm's r1 row. Both are rejected with and without the logs; so is a rate
   edited in `runs.csv` alone.
2. **Foreign GPU processes during a job.** `e2e5.sh`, `trace.sh` and `gl.sh` look for them every 0.5 s while
   the job runs and once at its end, and abort with exit 76 on what was captured, without asking again; the
   overlapped run stays in `runs.csv` as invalid and the sightings in `logs/<run>.others`. The gates watch the
   unmodified `verify.sh` the same way from outside (`verify.others`). Tested live with dummy processes: a
   tagged one is ignored; an untagged one that had already exited when the job ended is still reported and
   ends the tool with 76. Limit: a process living less than the 0.5 s between two looks can be missed, and
   the polling itself (one `nvidia-smi pmon` per look) runs during timed runs, for both arms alike.

## Tool defects from the second review (of `fb875ea13`), fixed

1. **Next token, every cell.** After the timed runs of a cell `e2e5.sh` runs parent and candidate once each on
   `prompt_real_2048.txt` (2048 tokens, aligned real text), `prompt_check.txt` (1972) and `r1304.txt` (1792)
   and compares each pair, plus the first timed pair, with `nexttoken.py`. A row is SAME or DIFFER only when
   both runs have rc 0, the expected prompt tokens, an output that begins with the prompt and a non-empty
   token after it; otherwise it is `INVALID:<reasons>`. Two empty outputs are never SAME. `nexttoken.csv`
   keeps rc, prompt tokens, the token and the output hashes. Checked on recorded 4070 Ti logs of
   `sarc-1.5-4w-port` (token ` intimidation` on `prompt_check`; 1792 prompt tokens on `r1304.txt`).
2. **`gate_check.py session`** no longer trusts the SAME strings. Per cell and prompt it requires the row, the
   tracked prompt's hash, both runs in `runs.csv` with rc 0 and the expected prompt tokens, equal non-empty
   outputs and tokens, and in the gates (`--require-logs`) it recomputes every row from the logs.
   `tools/test_gate_check.py` (synthetic sessions, no GPU) includes the review's case, 60 valid timed
   rows with all token runs rc=134, no output and SAME written: rejected, with and without the logs.
3. **Normal clock.** `e2e5.sh` needs either `--calibrate` (record-only, for the baseline and A/A sessions) or
   `--clkmin-file`; the gates always pass `results/4070ti/clkmin.json` and refuse to start without it. The
   file is written by `calibrate_clock.py` from the calibration sessions: per cell, floor(0.97 x the median of
   the per-run median clock), refused below 10 usable runs or above 3 % spread. Each row of `runs.csv` records
   the threshold applied; `gate_check.py` rejects a session whose rows do not carry the calibrated value.
   The 0.97 and the per-cell rule are my choice, to be revisited when real clock samples exist.
4. **`gl.sh`** now also looks for a foreign GPU process after the job (exit 76). The previous STATUS said so
   before it was true. The gates and `screen.sh` pass 75 and 76 up unchanged (`GATE_ABORTED`).
5. **One candidate environment.** The gates take the environment from the staged `cand/env` (what `e2e5.sh`
   and `trace.sh` read) for the SDPA passes and `verify.sh` too; they refuse to start if `cand/env` and
   `cand-traced/env` differ or if an environment given on the command line is not the staged one.
   `gate_check.py env` then checks what actually ran: the `ET_VK_*` variables recorded by `verify.sh`, and the
   profile banner in every candidate log (session, SDPA, trace) and its absence in the parent's.

## Tool defects from the first review (of `13defbe7b`), fixed

1. `stage.sh` no longer overwrites the tools path; it stages the unaligned prompt `tools/r1304.txt` (the file
   the earlier 4070 Ti campaigns used, taken unchanged from `~/.cache/et-e2e/sarc15-r4/`, sha256 `881de104…`,
   checked at staging) and refuses (exit 77) when a binary, a traced binary, `test_llama_microbench` or a prompt
   is missing, or when the session already has results.
2. Foreign GPU processes end a measurement (exit 76) in every path: `e2e5.sh` (before and after each run; an
   overlapped run stays in `runs.csv` as invalid), `trace.sh` (which now also cools before each run), `gl.sh`,
   and around `verify.sh` in the gates (see the third review for the monitoring during a job). "Ours" is decided by a tag in the process environment
   (`SARC_CAMPAIGN_TAG`, inherited by everything the tools start), not by the program name. Tested with two
   dummy processes named `llama_main`: the tagged one is ignored, the untagged one is reported and stops the tool.
3. Device loss: every temperature read is checked; `gpu_gone` writes a marker in the artifact directory, appends a
   `DEVICE LOST` section to this file and exits 70; every tool refuses to start while the marker exists, checks
   the card after each child (`gl.sh`, `e2e5.sh`, `trace.sh`, the gates) and passes 70 up without retrying.
   `gate.sh` / `gate_sdpa.sh` stop at the first failed step and write `GATE_ACCEPTED`, `GATE_REJECTED <step>` or
   `GATE_ABORTED` to `gate.done`. Acceptance is decided by `gate_check.py` from the result files:
   - `verify`: see the fourth review above;
   - `sdpa`: 12 passes per tier, 8 (`extended`) and 4 (`full`) cases each, `mismatches=0`, both coopmat kernels
     dispatched, `pairing=ok` on every case;
   - `session`: see the second review above.
   Checked against the 780M campaign's recorded evidence (`s8-r3final`, `s0-parent-verify`, `r1-sdpa-ext`):
   accepted as recorded, rejected after a production-diff rc, a `SAME` line, a `pairing=ok` or a pass log was
   altered or removed.
4. The inherited generators are gone from this change (they still exist, unmodified, in the 780M change).
   Two generators for this device replace them and write only `4070ti`-named files and
   `// >>> 4070ti <id>` … `// <<< 4070ti <id>` blocks in `impl/sarc_dev/Overrides.cpp` (`tools/devzone.py`
   enforces both; a second run changes nothing):
   - `gen_4070ti_sdpa.py`: families `sarc_sdpa_qk_coopmat_4070ti`, `…_4070ti_pk`, `sarc_sdpa_av_coopmat_4070ti`,
     `…_4070ti_ml`, 13 variants, profiles `4070ti-qk-*`, `4070ti-av-*` and `4070ti-refine1`;
   - `gen_4070ti_prof.py`: phase-timing twins of the shipped 4w `ga` tiles and of the zpgtr kernel.
   Not carried over: `gen_bt.py`, `gen_bx.py`, `add_batch1..3.py`. They produce variants of the 780M's zpg and
   `f32c` kernels, which this device does not run; linear sweep generators for the `ga` and zpgtr kernels will be
   written on `devzone.py` once phase timing says what to sweep. **This is a deviation from "adapt every
   generator"; say so if they are wanted anyway.**
5. `build-both.sh <tag> <commit> [local patch]` never builds the working copy: `mktree.sh` (local, reads this
   working copy only, fetches nothing) archives the commit and its 30 pinned submodules into
   `src/<tag>/executorch`, made read-only; the build runs the `build.sh` of that tree. `<tag>.src.txt` records
   commit, tree hash, local patch hash, image id, binary hashes and the `spirv_golden.py` result; a non-zero
   golden comparison fails the build. The parent is `build-both.sh parent 6a7cc8cc6`. `mktree.sh` was run once
   into the scratch directory (842 MB, 30 submodules) and removed.

## Generated dev-zone content (compiles; never run on the GPU)

- 20 new variants, all compiled with the container's glslc (shaderc v2023.8) through `gen_vulkan_spv.py`.
- The four SDPA shaders are byte-identical from `#version` on to the 780M campaign's dev twins
  (`qk_coopmat_sweep`, `qk_coopmat_pk`, `av_coopmat_sweep`, `av_coopmat_ml`); only the variant lists differ.
  Static pruning: subgroup 32 only, fp16 MMA 16x16x16, at most 1024 invocations, shared memory at most
  49152 bytes (`vulkaninfo`, driver 615.71.09; this removes QK^T 128x128 and 256x64 tiles), tiles dividing
  2048 and head_dim 64 / 128.
- `impl/sarc_dev/Overrides.cpp`: 5 marked blocks added, no line removed. `sarc/tools/check.sh --no-build`:
  PASS (31 rows, 127 candidates). Not done: a full build, and the shipped-SPIR-V comparison on it.
