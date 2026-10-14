# Campaign rules (shared by every GPU kernel-tuning campaign)

What this file is for: the rules an actor and a reviewer follow in every campaign, on any device. They are
part of the task. The device-specific part is `templates/TASK.md`, filled in per device; the owner's rulings
that override single rules are in `OWNER-DECISIONS.md`; what happened when rules were missing is in
`LESSONS.md`. This file says what to do, not why it was learned.

Adapted from the shared rules of the five finished campaigns (Radeon 780M, Arc B580, Arc Pro B70,
RTX 4070 Ti SUPER, Jetson Orin Nano). Changes against that version are listed at the end.

Placeholders: `<host>` the GPU host, `<campaign-root>` a directory on it, `<artifact-dir>` =
`<campaign-root>/.artifacts` unless the task says otherwise, `<tag>` the device tag (`7900xtx`, `rx7600`, ...),
`<change>` = `openspec/changes/sarc-1.5-<tag>-prefill-refine`, `<lock-uuid>` the device's lock name,
`<parent>` the parent commit of the campaign.

## R1. Where the work lives

- Actor and reviewer both run on the GPU host, in the working copy `<campaign-root>/executorch` (a clone of the
  public fork; the directory must stay named `executorch`). Everything built, measured and recorded is on this
  host. Files outside the working copy (the artifact directory, the model directory, `~/.cache`) are read with
  shell commands.
- Several campaigns run at the same time on different hosts and branches. Stay on your own host, your own
  branch and your own device's content.
- If the device is measured over ssh from a build host (cross-built targets), the task file says which side
  runs what. Nothing is compiled on a device that is only a measurement target.

## R2. Read first

1. The task file, completely, including every "Owner decision" section at its end.
2. `LESSONS.md` and `OWNER-DECISIONS.md` of this package.
3. `sarc/README.md`, `sarc/SWEEP-PARAMETERS.md`, `sarc/HOOKS`.
4. The sibling campaign your branch was forked from: `openspec/changes/sarc-1.5-<sibling>-prefill-refine/`
   (`STATUS.md`, `proposal.md`, `tools/`). It is the worked example and its tools are your starting point.
5. `openspec/changes/sarc-1.5-e2e-benchmark/`: `CONTRIBUTING-A-GPU.md` (how to benchmark and deliver results),
   `REPORT.md` (method and pitfalls), `TECHNICAL-REPORT.md` sections 7 to 10, `results/cells.csv`, `kit/host/*.sh`,
   and `contrib/<tag>/NOTES.md` if your device has one.
6. `openspec/changes/sarc-1.5-sdpa-port/` (how the attention prefill kernels were introduced).
7. `backends/vulkan/runtime/graph/ops/impl/sarc/{Select.h,Select.cpp,SdpaCoopmat.cpp,table_*.cpp}` and
   `impl/sarc_dev/Overrides.cpp`.

## R3. Repository rules (enforced by `sarc/tools/check.sh`)

- Dev zone only: `backends/vulkan/runtime/graph/ops/{glsl,impl}/sarc_dev/`, `backends/vulkan/test/sarc_dev/`,
  `sarc/` (never `sarc/tools/` or `sarc/golden/`), and your own `<change>/`. No upstream file is edited. Nothing
  is promoted. The only release-zone edits allowed are the hooks of `OWNER-DECISIONS.md` D4, under its
  conditions, each as its own commit.
- A kernel that cannot be reached from the dev zone is first reached by a dev-zone registration: a file
  `impl/sarc_dev/<Device>Sdpa.cpp` (or `<Device>Linear.cpp`) that registers `kUnverified` rows which match only
  while `ET_VK_SARC_DEV_PROFILE` names one of your profiles. Every other configuration must select exactly what
  the release tables select (`test_sarc_select` unchanged).
- Never edit `verify.sh`, `check.sh`, `build.sh`, the goldens, test tolerances, the prompts or the measurement
  protocol to make a candidate pass or look faster. Adding tests is welcome.
- Keep your content separable: prefix profile names (`<tag>-refineN`) and new kernel families with your device
  tag; put new code in new files; where a shared dev-zone file must change (`impl/sarc_dev/Overrides.cpp`, the
  sweep yamls), append only, inside a block delimited by comments that name your device tag.
- Do not change another device's shipped SPIR-V. `sarc/tools/spirv_golden.py` must show the shipped variants
  unchanged in every build you measure.
- Commit on your topic branch as you go, at every consistent state, in the style of the existing commits on
  it. Never commit to, rebase or force-push `dev/1.5`. Push only your own topic branch, at the end or when the
  task file says so (`git push origin <your branch>`, no force). No pull request.
- Non-source outputs (builds, raw logs, ETDumps, binaries, venvs, temporary files) go under `<artifact-dir>`,
  never into the working copy and never under a temp directory. Small evidence files (CSV, summaries) go into
  `<change>/results/`.

## R4. First steps

1. Create `<change>/`. Copy the sibling's `tools/` into it and adapt what is hard-coded for the sibling's host:
   paths, sensors, the lock name, the GPU process check. Do not edit the originals. List what you changed in a
   table in `proposal.md`.
2. Write the comparison thresholds into `proposal.md` and commit them before the first measurement (R6).
3. Build the unmodified parent `<parent>` from an exported commit (R5) and store its unmodified `verify.sh`
   output once as the snapshot `s0-parent-verify`. This is a record, not a verification of the starting rows
   (`OWNER-DECISIONS.md` N2).
4. Re-measure the baseline for all six cells (1B, 3B, 8B x 4w, 8da4w). It must agree with the device's published
   numbers within 3 % per cell. A cell outside 3 % is explained before anything is optimised.
5. Run an A/A session (parent against the unmodified topic build with no profile) and calibrate from it: the
   noise band, the clock floor, the repeat count, the foreign-busy ceiling where the device has one.

## R5. Builds

- Build with `sarc/tools/build.sh --llama` (and `--traced` for ETDump) in the pinned container image. If the
  image is missing, build it from the `Containerfile` in the sibling's tools.
- The source of a build is an export of exactly one commit and, recursively, of every submodule at the commit
  it pins, taken from the git object stores, never from the live working tree. Record the commit and the export
  manifest beside the build. A build tag is built once; a rebuild takes a new tag.
- A build is usable only if its shipped SPIR-V matches `sarc/golden/spirv.json`.
- No build runs during a timed session, on the same host, by anyone.

## R6. Measurement protocol

- Runner: `llama_main`. Models: the six exported `.pte` files and `tokenizer.model` named in the task file,
  read-only. Do not re-export.
- Prefill only: `prompt_2048.txt` (2048 tokens), `--max_new_tokens 1 --temperature 0 --warmup`, a fresh process
  per run. Read the model file into the page cache before the first process of each cell and whenever the model
  changes, for both arms alike (`OWNER-DECISIONS.md` D5).
- A run is valid only with rc 0, 2048 prompt tokens, 0 generated tokens, no other GPU workload before, during
  or after, enough clock samples inside the prefill window (5; fewer only where the sensor cannot answer
  faster, stated in the task), a median clock at or above the calibrated floor, and no thermal throttle reason.
  An invalid run stays in `runs.csv` with its reason and is replaced.
- Baseline and candidate in the same session, arms interleaved (parent first on odd repeats, candidate first on
  even ones), median of 5 valid runs per cell. Sample clock, temperature and power during each run. Start every
  session from a cool device and cool between runs; end a cooling wait also when the temperature has stopped
  falling.
- A difference inside +-2 % is noise, not a gain.
- Thresholds are fixed before the data exists: the clock floor rule (97 % of the lowest per-run median clock of
  the valid A/A runs, one device-wide value), the repeat rule, the noise band, the baseline tolerance, the
  kernel-screen margin. Nobody changes a threshold after seeing a candidate's numbers.
- One GPU job at a time, under the device lock (`flock` on `~/.cache/gpu-lab/lock-<lock-uuid>`). If the device
  is busy with something you did not start, wait and report it; never force. Do not change clocks, power, fan or
  governor settings; record them as found.
- Roof values: re-measure with igpu-roofline (`fast` plan) on the current driver and say which run you used. If
  the tool is not on the host, set it up under `<artifact-dir>` or report rates without percent-of-roof.

## R7. Gate for every candidate

- Unmodified `sarc/tools/verify.sh --models 1b,3b,8b --schemes 4w,8da4w --pdiff` with the candidate's
  environment, on exactly the binaries that are timed: correctness, dispatched kernel names, production-diff
  for three models x buffer / texture3d (non-zero zero-points for 8da4w), default vs tiled next token on the
  real-text and the unaligned prompt, decode. Its output is compared line by line with the snapshot
  `s0-parent-verify`, rates removed.
- Next token parent vs candidate in all six cells on the timed, the real-text and the unaligned prompt.
- A candidate that changes an attention kernel also passes `test_llama_microbench --sdpa-correctness-only` with
  `--sdpa-tier=extended` and `--sdpa-tier=full`, 12 passes each, 0 mismatches, and the kernel pairing line must
  read `pairing=ok`.
- A candidate that changes arithmetic (another attention kernel, another accumulation precision or order) is
  judged by its error against the fp32 reference (`OWNER-DECISIONS.md` D3). A candidate that claims not to
  change arithmetic shows bit-identical output.
- The shipped-SPIR-V golden check passes on the candidate build.
- Every new or changed shader is read once for unsynchronised shared writes before it is gated (one elected
  lane stores a subgroup's reduced value; shared memory is read only after the barrier that follows its writes).
- A failed candidate is recorded with its reason and its numbers, never deleted. Bad or interrupted results
  move to `superseded/<reason>/`. A failed gate is diagnosed before it is run again.

## R8. How to work

- Locate before you sweep: per-operator ETDump breakdown and in-kernel phase timing first, then decide.
- Candidates come from three sources, in this order of expected yield: a kernel proven on a sibling device, a
  new kernel for the located bottleneck, a parameter search. The task file says which are in scope.
- Sweeps are run by a resumable script, detached, that appends one row per (configuration, round, shape) to a
  CSV and skips rows already present. Prune statically first (shared-memory limit, cooperative-matrix shapes the
  device exposes, divisibility of the real shapes). Before any run projected to take more than 48 hours, stop
  and report the count and the projection.
- Shapes are independent: choose the best kernel per shape, not one for all. A screen selects a kernel only
  when it is at least 3 % faster than the incumbent in every round; a tie keeps the incumbent.
- Long work runs detached on the host with a status file, so that it survives the loss of the control session.
  After any restart, check what is still running before starting anything.
- Every detached queue obeys the coordinator hold (`tools/HOLD.md`).
- Keep `<change>/STATUS.md` current after every candidate: date and time (UTC), what is running now, the latest
  per-cell numbers against the parent, the next step, anything blocking. A question for the owner goes under
  the heading "Decision needed from the owner". The owner and the coordinator read this file instead of
  interrupting the campaign.

## R9. Profiler tracing

Driver-level tracing (RGP / SQTT, `MESA_VK_TRACE*`, `RADV_THREAD_TRACE*`, `INTEL_MEASURE`, Perfetto GPU data
sources, vendor capture tools) is allowed (`OWNER-DECISIONS.md` N4), under these conditions:

- commit and push-safe state first: nothing uncommitted that you cannot lose, `STATUS.md` says what is running;
- from a detached job, one capture at a time, with a timeout;
- never during a timed session, a gate or a sweep, and never while another campaign measures on the same host;
- no kernel setting is changed for it (`perf_event_paranoid` and the like stay as found) unless the task file
  permits it;
- the first capture of each new kernel on a device is treated as able to hang the host: say in `STATUS.md`
  before it starts that a capture is running, so that a dead host is explained.

Always allowed: ETDump, the shader-clock phase timing built into the kernels, kernel timing with
`test_llama_microbench`, igpu-roofline, reading sensors.

## R10. Not allowed

Benchmark-derived constants; skipping work that only happens to be invisible on the benchmark prompt; reduced
precision without measured error; caching results across runs; searching for a kernel variant that happens to
keep a next token; touching other checkouts on the host; `sudo` beyond what the task file explicitly permits;
installing system packages; starting, stopping or changing services; retrying, rebooting or reloading a driver
after the device has disappeared; opening a pull request.

## R11. Done when

Two consecutive gated candidates each gain less than 2 % geometric mean over their parent, or the budget ends.
Then:

1. run the final verification of everything together (unmodified `verify.sh`, the attention tiers, the golden
   check, the reference-error evidence) on the build of the committed branch head, with no local patch;
2. run one final timed session of the final stack against the pristine parent;
3. update `proposal.md` and `STATUS.md`, run `sarc/tools/check.sh --no-build` and report its output;
4. commit, push your topic branch;
5. report per-cell tok/s against the parent and against the published numbers, where each gain came from (with
   the timing data that shows it), percent of the freshly measured roofs, the negative results, and what limits
   further progress.

## R12. Reviewer

You are on the same host and in the same working copy as the actor, with a fresh context each round. You may
be the same model as the actor, so do not weigh its account: redo the checks. In every round, yourself:

1. recompute every reported number (medians, gains, geometric means, validity counts) from `runs.csv` and the
   per-run logs, and state each beside the reported value;
2. run the shipped-SPIR-V golden check on the measured build;
3. list the files changed outside the dev zone (`git diff --name-status <parent> HEAD`) and under
   `sarc/tools`, `sarc/golden`, tolerances, prompts and threshold files; every release-zone entry needs an owner
   decision that covers exactly that edit;
4. read every new or changed shader for unsynchronised shared writes;
5. check that `verify.sh` output exists for exactly the binaries and environment that were timed and matches
   the parent snapshot, and that a candidate accepted under an owner decision is recorded as such;
6. state what you did not check, and why.

Do not start GPU jobs or builds. Say "done" only when the conditions of R11 hold and none of checks 1 to 5 is
in your not-checked list.

## Changes against the rules of the first five campaigns

| rule | before | now | source |
|---|---|---|---|
| R4.3 | no snapshot named | parent `verify.sh` output stored once; no verification pass of unverified starting rows | owner, N2 |
| R4.4 | "within a few percent" (one campaign fixed 5 %) | 3 % per cell, else explain | owner, N7 |
| R6 | model read from disk by the first process | page cache filled before each cell | owner, D5 |
| R3 | no release-zone edit at all | the hooks of D4 may be committed | owner, D4 |
| R7 | next-token equality with the parent | reference-error rule for arithmetic changes | owner, D3 |
| R7 | shaders judged by tests only | shader read for shared-write races before the gate | lesson L8 |
| R8 | no hold | every queue obeys the coordinator hold | owner, D6 |
| R9 | driver-level tracing forbidden on every device | allowed under conditions | owner, N4 |
| R12 | reviewer of another model family | same family, with a fixed list of checks to redo | owner, N6 |
