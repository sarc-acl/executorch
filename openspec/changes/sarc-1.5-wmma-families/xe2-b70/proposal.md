# Xe2 B70: bounded existing-kernel matrix

Actor: Codex. Base `617897184529ff180af491ee0d5d7de778cbb6c6`.
This stage covers the reviewed Xe2 inventory only. No family-wide acceptance,
promotion, upstream port, new bodies, parameter search, push or PR is included.
The 2026-10-14 launch decision replaces the historical two-small-gains stop rule
with completion of the finite legal matrix within the two-day local cap.
Historical instructions to stop services or change hardware policy do not apply.

## Reviewed inputs and finite configuration list

Inventory commit `f393c3ff7ff4fc41e4651a5109a8509097c501f6`, source at the base.
Archive SHA256 `65d1e3ca0eb7e9dffa8e9fbbb1a100bf2cf609868720b9053de67ff8bf5247d8`;
Xe2 manifest SHA256 `5904caa4c1977f256d58b4c1c111e7fdcb2e54bc312676b40c90da6bcf048306`.
Archive, all 18 per-file hashes and all 53 pinned source-object hashes were checked.
Input review is source-only; it is not device legality or performance acceptance.

`profiles.json` and `results/candidate-coverage.csv` map every one of the 12
storage-specific entries to seven kernel-base profiles. The supplied model graphs
use texture3d prefill input/output. Buffer entries therefore receive alternate-storage
production-diff correctness coverage, not fabricated buffer model timings.
Every profile is compared across all six supplied cells, including unchanged cells.
The n13 and n26 picks retain their original weight-size predicates, and the k32
4w pick retains N<=512. Remaining shapes and the other scheme retain the control.
Shared kernel bases may select both registered storage variants; no new shader is made.

Both arms use `ET_VK_SARC_UNVERIFIED=1` and
`ET_VK_SARC_DEV_PROFILE=b70-fused1-order-i-n26-w16m`.
Candidates additionally set `ET_VK_SARC_B70_WMMA_PROFILE` to a listed name.
The appended selector is inert without that name, the exact final base profile,
allow-unverified, a served prefill shape and the B70 table device match.
It chains after the existing selectors and preserves all attention override fields.
The fused pair and fallback QK/AV preferences remain the final campaign's ones;
no candidate changes attention, its softmax, fp linear or decode selection.

## Thresholds fixed before measurement

These rules are preregistered now; calibrated values will be appended after A/A
and before any candidate timing. Historical calibrated values are not imported.

| Item | Rule |
|---|---|
| Runner | Fresh llama_main, prompt_2048.txt, max_new_tokens=1, temperature=0, warmup; 2048 prompt tokens and zero generated tokens |
| Validity | rc=0, throughput present, no foreign workload before/during/after, at least five in-window clock samples, no thermal reason, median clock at or above calibrated floor |
| Clock floor | One device-wide value: 97% of the lowest valid A/A per-run median clock |
| Thermal rejection | thermal, prochot, ratl, vr_thermalert; pl1/pl2/pl4/vr_tdc recorded only |
| Foreign occupancy | No exemptions; nonzero sampled foreign engine time invalidates a run; missing surrounding busy samples or guard polls invalidates a run |
| Repeats | Five valid runs per arm/cell; seven if any A/A cell differs by more than 1% or any arm's range/median exceeds 3% |
| A/A consistency | Explain before candidates if any cell lies outside +/-2% or geomean outside +/-1% |
| Noise | +/-2%; an effect inside this band is not a gain |
| Baseline comparison | Explain any cell more than 3% from published final configuration; use same-session tuned controls for every candidate margin |
| Family margin | At most 3% six-cell geomean slowdown and at most 8% slowdown in any cell; report uncertainty without widening either bound |
| Screen margin | Existing R8 threshold remains 3% in every round if used; no separate tile/parameter screen is planned |
| Invalid evidence | Retain rows/logs/reasons; six extra interleaved replacement pairs maximum per cell, then explicit incomplete result |
| Correctness | Unmodified verify.sh, all models/schemes, pdiff, real and unaligned prompts, decode; exact binaries/environments match timing; parent snapshot comparison with expected linear-name changes recorded |
| Arithmetic | Bit identity for claimed staging/order-only changes; D1/D3 evidence if needed, with no historical exception extended |
| Stopping | Device loss stops without retry/reset; sustained foreign use is reported; holds and owner questions stop dependent work |

Reference numbers come from `sarc-1.5-b70-fused-port/STATUS.md` (s3-final 4w:
20686.90, 7968.87, 3482.99 tok/s) and `results/b70/order2/model.md` (m2 final
8da4w: 24674.7, 10039.2, 4853.1 tok/s). They are from different historical
sessions and are references only; same-session controls are mandatory here.

Seven configurations imply 42 cell comparisons, 420 timed processes at five
repeats (588 at seven), plus gates, next-token checks and dispatch evidence.
The historical adapters suggest a same-day matrix, but this is a projection,
not a promise. Record actual build/gate duration before queueing the matrix;
report any revised projection beyond 48 hours before queueing it.

## Adapter changes

Selected active tools were copied from `sarc-1.5-b70-fused-port/tools`.
Historical attention screens, roof/decode tuning, collector and campaign chains
are not launch tools for this stage. Original files and sarc/tools are unchanged.

| Copied adapter | Adaptation |
|---|---|
| host.sh | Resolve the deeper run directory through git, use own artifact root/TMPDIR, use approved base/final profile, load private unit/model constants from artifact-local host-config.sh, use own analysis environment |
| e2e5.sh, trace.sh | Model root from the private host configuration |
| gate.sh, parent_verify.sh | Pass the same private model root to unmodified verify.sh |
| Remaining copied active tools | Unchanged; inherit adapted constants from host.sh |

The inherited lock, conservative two-card foreign-client guard, in-run guard,
sampler and HOLD implementation are retained. No work targets the second card.
All descendants have the run's unique XE2_TOP ancestry. A dry hold check is required
before launch. Every job uses the device lock; no build overlaps a timed session.
Samplers terminate their own processes; inspect child counts after initial jobs.

## Prerequisite evidence and remaining work

Private read-only inspection, query output, sensor paths/latencies, current policy,
model paths/headers/serialized metadata and compiler/image facts are in the own
artifact preflight record. Unit identity and driver agree with the supplied task;
no readable primary DRM client was present at initial inspection.
The complete subgroup-scope matrix type/shape query covers the manifest requirements.
Declared shared memory and workgroup geometry fit the queried limits, and all twelve parent compiled-module capacities equal the source declarations.
Capabilities and image formats are supported in the recorded query; exact topic
module identity and correctness remain launch prerequisites.
Six file sizes and serialized context/cache constants agree with kit/MODELS.md.
Serialized Vulkan graph operators identify 4-bit embeddings at group 32 and
linear group 128; runtime constant-method execution has now confirmed the six metadata values for all files.
Pinned submodule objects were initialized recursively in this clone only.

`check.sh --no-build` passed (1240 release checks; 1921 dev checks). No shader,
release-zone file, sarc/tool, golden, prompt or tolerance was edited.
Fresh independent source/preflight review passed; see REVIEW-PREFLIGHT.md.
The focused CPU fixture passes seven profiles/twelve storage variants, including
unsupported storage, unrelated ops/devices, decode, invalid batch and attention-hook
invariance. The isolated HOLD dry run passed.

Scope conflict identified before launch: inherited parent_verify.sh adds SDPA tiers,
and gate.sh unconditionally uses trace.sh without D5 cache preparation. Those three
outer wrappers are retained as read-only examples and are inactive launch paths.
Bounded orchestration must call the unchanged verify.sh, preserve its full decode
checks, and prepare models before every cell/model change. No historical exception
or gate acceptance is inferred from a copied wrapper.
No build, gate or timing evidence is claimed by this document yet.

The parent normal/tests/traced build completed in 6m18s. Its llama shader
golden passes all 53 shipped variants, and the standard build wrapper also
reports BUILD_BOTH_OK. New matrix_verify.sh calls the unchanged six-cell gate
without historical SDPA tier extras, after filling all six supplied models
(which fit in RAM) and recording residency. Device-loss output blocks later jobs.
The metadata helper executes constant methods only and links no Vulkan backend.

Slowdown is preregistered explicitly as control tok/s divided by candidate tok/s
minus one (elapsed-time ratio), including the six-cell geometric mean. This is
a conservative reading of the 3%/8% limits. Uncertainty uses 20,000 independent-arm
bootstrap resamples of valid selected runs, seed7321, with 2.5/97.5 percentiles.
No data has been measured before this clarification.

The standalone dispatch collector fills the model cache at every model/cell change,
uses the exact exported schema/flatc and traced binaries, and extracts every shader
name with standard-library JSON parsing. The inherited Inspector/venv trace path
is not used. The standalone linear identity tool reuses the exact exported
microbenchmark's deterministic production inputs, uses nonzero activation zero
points, and compares every fp16 output bit for both registered storage modes on
all twelve production NK shapes. It changes no gate, tolerance or shader.
The finite queue requires an actor-recorded BASELINE_READY marker, the calibrated
repeat count and the exact manifest hash; it records every raw run with per-row
unit/driver/commit/binary/profile/session/sensor provenance. Incomplete sessions
stop and require an explicit owner resume; no interrupted session is overwritten.

The first diagnostic metadata helper was compiled from live helper source against
the parent SDK. Its preliminary output is retained but will not serve as final
runtime-metadata evidence: the helper is rebuilt from an immutable topic export
against that export's SDK before timing. No GPU measurement used the preliminary
helper. The normal candidate build and its shaders have always used exact exports.


Recovery on 2026-10-15 reuses the completed parent and topic-02 exact exports,
normal/traced builds and exported helper binaries. No unchanged source is rebuilt.
The run-local baseline_queue.sh prepares the parent snapshot, topic gate and
five-repeat A/A only; an actor must assess its evidence before candidate launch.
The detached job_runner.py publishes atomic queued/running/completed/failed JSON,
records orchestration provenance separately from binary source commits, clears
inherited ET_VK/ETVK controls and tags all descendants by ancestry. A signal stops
only its own process group. e2e5.sh records the sampler process count after every
run and stops on a leftover. The pending dispatch collector fix adds the inherited
device lock, and the finite queue checks complete raw rows rather than a marker
alone before skipping a prior result. Historical adapters and sarc/tools remain
unchanged. These are protocol repairs under the 2026-10-15 recovery instruction.

The parent/A/A queue is projected below 48 hours from the historical adapter
workload and completed six-minute builds; it makes no candidate launch decision.
Actual gate and session durations will set the finite-matrix projection before
that queue is launched. No new timing threshold or historical acceptance exception
is imported by the recovery.


Before launching the recovery queue, a coordinator reminder prompted the missing
matching image-format extent check. Current primary-unit evidence shows required
8B fp16 width-packed image width 3584 exceeds the matching format/type/tiling/usage
limit of 2048. BLOCKER.md records the exact source derivation and private artifact
hashes. Earlier format-feature checks did not establish resource-extent legality.
The prepared parent/A/A queue was never launched. RESOURCE_LIMIT_BLOCKED is now
checked before adapter GPU/build entry and by both queues. The six-cell stage
remains incomplete; no historical successful run grants a limit exception.
