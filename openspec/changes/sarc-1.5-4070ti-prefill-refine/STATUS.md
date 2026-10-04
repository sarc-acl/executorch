# STATUS: sarc-1.5-4070ti-prefill-refine

**2026-10-04, gpu-dev-4004. BLOCKED before the first measurement. Nothing has been built with `build.sh`, no GPU
job has run, there is no `runs.csv` and no `verify.out`. The stop rule is not met; the branch is not pushed.**

## Blocking (unchanged, re-checked after the review)

The agent's shell on this host is confined (`TMPDIR=/tmp/hmz-fence-*`): only the working copy and that scratch
directory are writable. Denied with `Permission denied`, although owned by `doremy`:

| path | needed for |
|---|---|
| `~/hmz-sarc-4070ti/.artifacts/` | every build, source tree, log, ETDump and stage directory |
| `~/.cache/gpu-lab/lock-81a511a2-de7e-c3c8-f641-3562c315ffa7` (open for append) | the gpu-lab lock; the unmodified `sarc/tools/verify.sh` opens it with `>>` and exits 75 otherwise |
| `~/.cache`, `~/.docker`, `/tmp` | igpu-roofline results; docker client config (`DOCKER_CONFIG` in the scratch directory is used instead) |

The restriction was not bypassed and no raw output was put into the repository. **Needed from the owner:** write
access to `~/hmz-sarc-4070ti/.artifacts/` and to the lock file, and a writable results directory for
igpu-roofline (for example under `.artifacts/`); then restart the campaign.

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
   - `verify`: every status item of the candidate's `verify.out` (correctness, `linear <scheme> rc`, 12
     production-diff cases, default vs tiled on `prompt_check` and the unaligned prompt, decode) against the
     parent control produced by `parent_verify.sh`;
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

## SDPA reachability

SDPA is not reachable from the dev zone on this device: the profile only replaces an existing table choice and
the SDPA hooks in `impl/sarc/SdpaCoopmat.cpp` need a release-table row. As the campaign allows, the candidate
builds will use `tools/local-hook-nvidia-sdpa.patch` (two `kUnverified` rows in `table_nvidia.cpp`, applied to
the candidate's archived source tree only, never committed, recorded in the build provenance) with
`ET_VK_SARC_UNVERIFIED=1`. The parent build is the pristine `6a7cc8cc6` without it.

## Open before the baseline

- `results/4070ti/clkmin.json` does not exist yet: it comes from the baseline and A/A sessions.
- `trace.sh` needs a python with the ExecuTorch devtools (`TRACE_PY`); not looked for yet.
- igpu-roofline exists on the host only as `~/.cache/igpu-roofline/fleet-fast-20260926/`; no roof measured.
- 1B timer quantisation (1 ms timer, about 100 ms prefill): report ETDump dispatch time alongside.
- None of `build-both.sh`, `stage.sh`, `parent_verify.sh`, `e2e5.sh`, `trace.sh`, `gate*.sh`, `screen.sh` has run
  end to end; expect first-run fixes. In particular the prompt-echo rule of `nexttoken.py` was only checked on
  older logs of this device, and the profile banner check on no real log at all.

## Next steps once unblocked

1. `tools/build-both.sh parent 6a7cc8cc6`; `tools/build-both.sh topic <head> tools/local-hook-nvidia-sdpa.patch`.
2. `parent_verify.sh parent`; baseline of the six cells against `cells.csv` (4w 19692 / 8752 / 4491, 8da4w
   20898 / 9660 / 5032 tok/s) and the A/A session parent vs topic without environment, both with
   `session.sh <s> --calibrate`; then `calibrate_clock.py` -> `results/4070ti/clkmin.json`.
3. Per-op ETDump breakdown, phase timing with the PROF twins, igpu-roofline `fast`.
4. SDPA candidates (`4070ti-refine1` first) through `gate_sdpa.sh`, then 8da4w and 4w linear.

## Per-cell numbers against the parent

None.
