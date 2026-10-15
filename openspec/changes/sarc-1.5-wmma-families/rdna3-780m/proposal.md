# rdna3-780m bounded matrix

Base: `617897184529ff180af491ee0d5d7de778cbb6c6`. Actor: Codex.
This run compares existing RDNA3 RADV linear configurations from the reviewed inventory.
The final launch decision authorizes the finite matrix, local commits and independent review.
It supersedes the historical two-small-gains stop rule. No promotion, release-zone edits,
new shader bodies, parameter search, pushes or PRs belong to this stage.

## Reviewed input

Inventory commit `f393c3ff7ff4fc41e4651a5109a8509097c501f6`, source base as above.
Archive SHA256 `65d1e3ca0eb7e9dffa8e9fbbb1a100bf2cf609868720b9053de67ff8bf5247d8`.
RDNA3 manifest SHA256 `edcfa39ea7d8fac9c907a6a901754b96aa2d2894302b9c8ad623b90c2780d923`.
All 18 supplied file hashes match. All 53 source hashes match the pinned git objects
and this checkout. Inputs remain outside the checkout and read-only.
`tools/verify_input.py` checks the supplied provenance record without regenerating it.

## Controls and limits registered before measurements

Both arms use the approved base plus committed dev-only selectors. The control is
`ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=780m-final`. Candidate environments
retain that attention setup and change only the reviewed linear choices. Clear unrelated
exact-kernel, per-device profile, disable-coopmat and force-tiled variables first.
The invariant attention setup includes softmax `780m_r3`, the final `fused3sb` pair
and the inherited refine3 three-kernel fallback. Record actual names from the measured
binary; this source description is not dispatch evidence. Decode remains unchanged
and is covered by the full unmodified gate.

Use fresh `llama_main` processes, the unmodified 2048-token prompt, `--warmup`,
`--max_new_tokens 1 --temperature 0`. Fill the model page cache equally for both
arms before each cell and whenever the model changes. Interleave control first on
odd repeats and candidate first on even repeats. Retain every invalid/rejected row.
Require rc 0, 2048 prompt tokens, zero generated tokens, at least five clock samples
inside the measured window, no thermal throttle and no observed foreign GPU work
before, during or after, subject to the specific monitoring exception below.
Missing required sensor/own-user monitor samples are not a valid run.

Owner decision 2026-10-15 03:57 UTC permits own-user GPU-client accounting plus
device-wide busy sampling despite incomplete other-user client visibility. Every
result records `monitoring_visibility=own_user_plus_device_busy_other_users_unobservable`
and that decision date; the final assessment repeats the limitation. This replaces
the full-visibility launch blocker, leaving every other validity and acceptance
requirement unchanged. Observed foreign activity still invalidates the run.

Start with five valid repeats per arm/cell. In A/A only, if any arm/cell's rate
spread `(max-min)/median` exceeds 3%, use seven repeats for the subsequent matrix
and report the instability before candidates. Allow at most three replacement pairs
per cell, then report incomplete coverage rather than looping indefinitely.
Clock floor is one device-wide number: 97% of the lowest median clock among valid
A/A runs. Never recalibrate it from candidate results or relax it for another workload.
The noise band remains +/-2%. A/A does not narrow that band.

Remeasure the final linear configuration across all six cells; a rate differing by
more than 3% from the published final-profile reference must be explained before
candidates. Record the parent snapshot and A/A on the current exported-commit binary,
not historical artifacts. Keep the kernel-screen margin at 3%; no screen replaces
the required six-cell matrix and gates.

For each cell define `r = candidate median tok/s / same-session control median tok/s`.
The member margin requires every `r >= 0.92` and `exp(mean(log(r))) >= 0.97` across
six cells. Report spread and paired bootstrap uncertainty (20,000 resamples, seed 780,
95% percentile intervals); do not change acceptance limits based on the intervals.
One member cannot establish family acceptance. External member evidence stays pending.

Sample clock, temperature, power and GPU metrics every 0.1 seconds and monitor process
ancestry/DRM engine counters throughout each run. Stop samplers and their children,
then check process counts after the first jobs. Cool to idle+5 C, or until temperature
stops falling (three consecutive 5-second readings with no decrease), with a 120-second
maximum; record which condition ended the wait. No hardware policy is changed.

For metric format 2.1, pre-register thermal bits 4,5,6,9,10 and unknown bits above 12
as rejection bits (`0xFFFFE670`); record power/current bits without rejecting on them
alone. This follows the [upstream Linux v6.12 SMU13.0.4 bit table](https://github.com/torvalds/linux/blob/v6.12/drivers/gpu/drm/amd/pm/swsmu/inc/pmfw_if/smu13_driver_if_v13_0_4.h#L134).
This is a source-based interpretation, not a transferred historical owner exception.
The downstream host kernel's corresponding header has not been established; confirm
that interpretation before timing, and never silently label it an exact host-source check.
Device loss means stop and report; no retry, reset or driver reload.

## Preflight evidence and remaining work

Private raw inspection is under `<artifact-root>/preflight/`: `input-hashes.txt`,
`vulkaninfo.txt`, `device-profile.json`, `matrices.json`, `models.json`,
`sensor-latency.json`, `model-hashes.txt`, `hold-test.txt`, `host-facts.txt`, `compiler.txt`, and container/header records.
Infrastructure identifiers and private absolute paths are excluded from this directory.
The 14 enumerated matrix records include the manifest's subgroup-scope, non-saturating
16x16x16 fp16/fp16/fp32/fp32 and signed-int8/signed-int8/int32/int32 requirements.
The complete signature, limits and formats remain in the raw device profile.

The six model sizes match `kit/MODELS.md`. Read-only flatbuffer inspection confirms
serialized context/sequence length 2560, group-128 linear arguments, four-bit embedding
ops with group 32, fp16 texture3d width-packed projection IO and the twelve production
NK pairs. Unlike the kit's historical inference, group sizes and the embedding op
are explicit in these graphs. Runtime constant-method execution has not been performed.
Default weight-storage fields still require graph-initialization/dispatch confirmation.
No model was modified, re-exported or executed.

`results/preflight-candidates.csv` retains all 19 manifest candidates and source fingerprints.
All pass the queried declared MMA/shared-memory/subgroup/workgroup requirements.
Fifteen match predicted production storage; four buffer variants are retained for
alternate-storage correctness coverage. These are partial checks, not legal launch or
acceptance verdicts. Exact compiled allocation, actual packed storage/image extents,
origin predicates and preceding service conditions must be bound before launch.
All reported fit counts check alignment/storage only, not route reachability.

The existing pinned container is present; its compiler version is recorded privately.
At the original preflight, no source export, shader compilation, build, model execution,
gate, A/A or timed run had started. No profiles are claimed runnable yet. No timing projection is claimed
without measured job durations; report any projection above 48 hours before queueing.
The full unmodified gate and golden checks remain required on exact measured binaries.

The first parent build completed from the exact unmodified base. Both backend and
runner shader directories pass the unchanged 53-variant golden. An after-build
git-blob audit covers 50,464 files across 31 root/submodule trees with zero
mismatches. tools/inspect_modules.py checks all 40 exact modules for 19 manifest
candidates and their aliases across both builds: compiled Workgroup payloads equal
the source declarations and fit the queried limit, and actual matrix multiply-add
operands/results match the required types/scope/shapes. Unsupported payload types
or unresolved lengths stop the check. This does not establish driver LDS allocation,
runtime storage, origin reachability or GPU correctness. Raw audit paths and the
small results/parent-build.json record are indexed in STATUS.md.

The next CPU-only metadata helper links the preserved build's extension_module_static
and its CPU dependencies, without the Vulkan backend. It accepts only the nine named
constant methods, each with zero inputs/backends/instructions, and never loads forward.
tools/check_metadata.sh exports the helper sources from the committed git object,
records the helper and runtime commits/manifest, and checks the ELF for Vulkan
dependencies/symbols. tools/validate_metadata.py rechecks model hashes, executes
the constants and compares them to the saved serialized metadata. Group size and
embedding metadata remain graph/tensor evidence tied to identical bytes rather
than invented runtime constant fields. This adds no patch to the parent build.

## Copied adapters

| Adapter | Source | Changes |
|---|---|---|
| `tools/export_recursive.sh` | final 780M campaign tool | Infer this working copy from the script location instead of its historical checkout constant; recursive pinned-object export logic unchanged |
| `tools/hold.sh` | final 780M campaign tool | Infer this run's artifact root beside the checkout, retaining `SARC_HOLD_DIR` override; remove historical root from comments; hold logic unchanged |

Other historical measurement adapters were read but are not installed as runnable
queues at the original preflight. They require this run's calibrated floor, richer
provenance rows, child cleanup and the current owner ruling before use.
Never run their historical chains or tune/search scripts in this matrix stage.

After the monitoring decision, tools/queue.py adds a finite detached command queue
with atomic terminal-state reporting and failure preservation. Every step uses the
copied hold adapter. tools/build_parent.sh adds the initial pinned recursive export,
existing-container build and unmodified golden comparisons for backend and runner.
The image, compiler, manifest, orchestration commit and binary hashes remain under
the artifact root. A fresh tag is required after any failed build. No source or
shader patch is applied to the exported base. The first queue performs no model GPU
work and has a four-hour timeout. Foreground success, command-failure and missing-
executable checks establish status reporting and fail-stop behavior; their private
logs are retained under queue-tests-20261015/.

Coordinator hold: `<artifact-root>/HOLD` and `<artifact-root>/HELD`, shared busy lock
`<artifact-root>/.hold-busy`. A separately named dry-run hold blocked a `true` job,
then released it and removed its acknowledgement; the real HOLD was untouched.
Future smallest units are one build, gate or interleaved session; their maximum
durations are pending current-build calibration. The initial build queue is bounded
by a four-hour timeout; its status file is indexed in STATUS.md.

## Historical monitoring blocker and resolution

Current inspection confirms that DRM counters of active display users cannot be read.
Own-user engine counters plus device-wide busy percent do not prove the absence of
other users' GPU engine activity during the runner. The historical owner permitted
that limitation, but this run explicitly does not inherit historical exceptions.
The original question was recorded in STATUS.md. Independent preparation and its review
are complete. The 2026-10-15 recovery instruction authorized one assessment of existing
readable counter alternatives, without inheriting the historical exception.

That assessment found the display users' fdinfo still inaccessible, no per-client
activity counters in the inspected DRM sysfs directories, and inaccessible debug DRM
client/fence paths and tracing data. Device busy percent and metrics remain readable,
but combining them with own-user engine time and process names cannot distinguish
overlapping foreign work during runner load. No inaccessible interval is certified
clean. Private evidence is `preflight/recovery-monitoring-20261015.json` under the
artifact root; its hash is recorded in STATUS.md. No build or GPU workload was started.

Owner decision 2026-10-15 03:57 UTC resolves this blocker by explicitly accepting the
limited monitoring described above. Resume preparation and the finite device matrix;
do not request a duplicate preflight review. Fully observable other-user accounting
is no longer a launch prerequisite. Preserve the raw assessment and disclose the
limitation on every result. Services, permissions and host policy stay as found.
The matrix remains incomplete until the required measured evidence exists.
