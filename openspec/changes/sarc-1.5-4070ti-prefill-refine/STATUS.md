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

## Tool defects from the review of `13defbe7b`, fixed

1. `stage.sh` no longer overwrites the tools path; it stages the unaligned prompt `tools/r1304.txt` (the file
   the earlier 4070 Ti campaigns used, taken unchanged from `~/.cache/et-e2e/sarc15-r4/`, sha256 `881de104…`,
   checked at staging) and refuses (exit 77) when a binary, a traced binary, `test_llama_microbench` or a prompt
   is missing, or when the session already has results.
2. Foreign GPU processes end a measurement (exit 76) in every path: `e2e5.sh` (before and after each run; an
   overlapped run stays in `runs.csv` as invalid), `trace.sh` (which now also cools before each run), `gl.sh`,
   and around `verify.sh` in the gates. "Ours" is decided by a tag in the process environment
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
   - `session`: six cells with 5 valid runs per arm and the next token SAME on both prompts.
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

- CLKMIN is 0 (record only) until the baseline and A/A sessions show the normal clock.
- `trace.sh` needs a python with the ExecuTorch devtools (`TRACE_PY`); not looked for yet.
- igpu-roofline exists on the host only as `~/.cache/igpu-roofline/fleet-fast-20260926/`; no roof measured.
- 1B timer quantisation (1 ms timer, about 100 ms prefill): report ETDump dispatch time alongside.
- None of `build-both.sh`, `stage.sh`, `parent_verify.sh`, `e2e5.sh`, `trace.sh`, `gate*.sh`, `screen.sh` has run
  end to end; expect first-run fixes.

## Next steps once unblocked

1. `tools/build-both.sh parent 6a7cc8cc6`; `tools/build-both.sh topic <head> tools/local-hook-nvidia-sdpa.patch`.
2. `parent_verify.sh parent`; baseline of the six cells against `cells.csv` (4w 19692 / 8752 / 4491, 8da4w
   20898 / 9660 / 5032 tok/s); A/A session parent vs topic without environment; set CLKMIN.
3. Per-op ETDump breakdown, phase timing with the PROF twins, igpu-roofline `fast`.
4. SDPA candidates (`4070ti-refine1` first) through `gate_sdpa.sh`, then 8da4w and 4w linear.

## Per-cell numbers against the parent

None.
