# NVIDIA WMMA matrix status

Updated: 2026-10-16 07:25 UTC

Stage: NATIVE TIMED MATRIX RUNNING under the owner's lean-finish decision
(2026-10-15 23:30 UTC). Job state: artifact jobs/native-matrix-r7/state.json (r6 stopped to reorder) (r1, r2 failed on a foreign client; r3 on a slow-load runner abort; r4 on a slow-load runner hang; r5 on four invalid attempts at c01 8b-8da4w r5). From r6 the replacement bound is 8 attempts per arm/repetition (was 4; c01 STAGE.json still records 4). It bounds retries only; validity criteria are unchanged.
Both arms: matrix-mm-r1 binary (commit dd57ce0e4), control 4070ti-fused1,
candidate adds only ET_VK_SARC_WMMA_PROFILE. Seven valid interleaved runs per
arm and cell, clock floor 2531 MHz, A/A s1-aa-mm-r1 reused (no new A/A).

Changes for the new protocol: a next-token difference on the timed, real or
unaligned prompt no longer stops a profile; it is recorded as
CORRECTNESS_PENDING (top-2 logits unavailable from llama_main) and the profile
is still timed. Timed-pair text differences are recorded, not aborted on.
Order: Orin-origin candidates first (c08, c09, c10, c11, c06), then c01, c02.

| Candidate | Linear kernel | Correctness | 1b-4w | 1b-8da4w | 3b-4w | 3b-8da4w | 8b-4w | 8b-8da4w | Geomean |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| c08 | sarc_dev_linear_q4gsw_coopmat_bx_t128x128k32g42s32f32c (4w, K>8192) | pdiff PASS; next token equal (36/36) | 1.000 | 1.000 | 1.000 | 0.992 | 0.917 | 1.000 | 0.9844 (8b-4w outside 8 % cell limit) |
| c09 | sarc_linear_q4gsw_coopmat_orin_t256x128k16g42s32bt (4w) | pdiff PASS; next token equal (36/36) | 1.036 | 1.000 | 1.028 | 0.992 | 1.046 | 1.000 | 1.0167 (84 valid, 1 invalid) |
| c10 | sarc_linear_dq8ca_coopmat_zpgtr_orin_bf_t128x128k64g24s32mk32ra (8da4w) | pdiff PASS; next token equal (36/36) | 1.000 | 1.040 | 1.000 | 1.058 | 1.003 | 1.038 | 1.0229 (84 valid, 0 invalid) |
| c11 | sarc_linear_q4gsw_coopmat_t128x128k16g22s32 (4w, K<=8192) | pdiff PASS; next token equal (36/36) | 1.018 | 1.000 | 1.021 | 1.016 | 1.032 | 1.000 | 1.0144 (84 valid, 2 invalid) |
| c06 | sarc_linear_dq8ca_coopmat_zpgtr_t128x128k64g44s32mk32ra tex3d (8da4w) | pdiff PASS; next token equal (36/36) | 1.000 | 1.000 | 1.000 | 1.008 | 1.003 | 1.000 | 1.0018 (84 valid, 0 invalid) |
| c01 | sarc_linear_q4gsw_coopmat_t256x128k16g42s32ga tex3d (4w, N>512) | pdiff PASS; next token equal (36/36) | 1.000 | 1.000 | 1.000 | 1.008 | 1.003 | 1.004 | 1.0024 (84 valid, 18 invalid; 8B rows mostly after slow loads) |
| c02 | sarc_linear_q4gsw_coopmat_t128x128k16g24s32ga tex3d (4w) | pdiff PASS; tokens pending | | | | | | | queued |
| c03, c04, c07 | buffer-IO variants | pdiff PASS (buffer microbench) | not timeable: the fixed models run texture3d IO, no dispatch | | | | | | n/a |
| c05 | buffer-weight variant | NO COVERAGE (0 dispatched cases) | not timeable, no dispatch | | | | | | n/a |

Ratios are candidate/control valid-run medians, 7 valid runs per arm, same session
and binary; c08 had 84/84 valid rows, 0 invalid. c08 dispatches only for K>8192
(8B 4w down projection); the other cells are unchanged within one timer step.

## Question for the coordinator

- 2026-10-16 02:30 UTC: the foreign ComfyUI (PID 336840) keeps 8086 MiB VRAM and about 12 GiB host RAM (3.7 GiB available, no swap). The 8B model no longer stays cached, and 8B runs now abort (rc -6/-11) or hang until the 1200 s timeout after slow loads. Valid 8B runs keep the same prefill rate. The queue replaces these runs (L9/D5).
- 2026-10-16 03:00 UTC: the slow loads also leave NVML's clock reading stale. 8B runs show a median clock of 270–600 MHz at an unchanged prefill rate (274 ms), so they fail the fixed 2531 MHz floor. c01 8b-8da4w r5 used all four attempts (r5 failed). From r6 every 8B process (both arms) waits until host MemAvailable >= model size + 1 GiB (memory-waits.jsonl). At 3.8 GiB available that wait lasts as long as ComfyUI holds its memory. c01 needs 8b-8da4w r5–r7 and c02 needs its 8B cells; the rest of the matrix is done. Can that workload release host memory before 2026-10-17 18:00 UTC?
- 2026-10-16 03:30 UTC, applying the coordinator answer (03:15): the memory wait and clock floor stay. From r7 the queue measures c02's 1B/3B cells first. The 8B remainder (c01 8b-8da4w r5–r7, c02 8B) then waits for memory until 2026-10-17 12:00 UTC; cells still missing then are recorded as NOT_MEASURED (foreign memory pressure). r6 was stopped by me while idle in the memory wait (no run in flight); the inner session needed a separate TERM because `timeout` runs in its own process group.
- 2026-10-16 07:25 UTC: the owner stopped zun-flux-pipeline.service (coordinator note); MemAvailable 14.1 GiB, no foreign GPU client. The r7 memory wait released by itself; c01 finished its 8b-8da4w r5–r7 and c02 8B cells are running. Validity rules unchanged; nothing restarted.

## Runner abort after slow load (c01, 2026-10-16 01:47 UTC)

At about 01:40 the foreign ComfyUI ran a generation. Since then it holds 8086 MiB VRAM and most host
RAM (3 GiB available, no swap), so the 8B model no longer fits in the page cache (cache_after
0.96). The c01 8b-4w candidate r4 a1 runner segfaulted (rc -11) after that slow load. By L9/D5
this is an invalid run that gets replaced, not a gate failure: the row is kept, the ABORTED marker
is archived in stage/s3-4070ti-wmma-c01/superseded/slow-load-abort/, and from r4 such aborts are
replaced automatically (slow-load-aborts.jsonl); from r5 also a runner timeout after a slow load
(c01 8b-8da4w control r5 a1 hung 1200 s; marker archived the same way). An abort without a slow load still stops the
queue. Four failed attempts at one position also stop it.

## Foreign GPU client (answered by the coordinator, 2026-10-16 00:20 UTC)

A ComfyUI server of another workload (`grand-omega/zun-flux-pipeline`, PID 335097, later
restarted as 336840) holds about 10–210 MiB on the GPU. It is never stopped. r1 stopped in
the c09 cooling check. r2 waited, then launched after the server restarted under a new PID;
observe.py refused that run (`8b-8da4w` candidate r3 a1, kept invalid, runner never started).
From native-matrix-r3: a run starts only after every foreign client has kept the same memory,
with 0 % device utilization, for 60 s (foreign-waits.jsonl). The run gets that client set; any
new foreign PID or memory change during the run makes the row invalid (kept, replaced).
Device utilization during our own run cannot be attributed per process, so it is not
checked for the foreign client. Each row records foreign_idle_client=<pid,MiB>. The c08 and
earlier c09 CSVs got the new column empty (originals in superseded/schema-upgrade).

## Earlier record (before the lean-finish decision)



Stage: ELEVEN GATES CAPTURED / SEVEN NATIVE SESSIONS PREPARED.
Running next: detached native-matrix-r1. Controller state is artifact
jobs/native-matrix-r1/state.json. No build or GPU job was active at preparation.
No owner question is pending. No candidate is yet accepted or timed; the finite
device matrix remains INCOMPLETE. Gate captures were committed as c77c47416.

The native queue uses only c01,c02,c06,c08,c09,c10,c11. Both arms use the same
retained matrix-mm-r1 runner/library and exact commit dd57ce0e4; the control
keeps the final tuned profile, while the candidate adds only its WMMA profile.
Attention dictionaries match the reviewed control and exact candidate gate.
Each native profile first obtains 36 fresh-process token runs: six cells ×
timed/real/unaligned prompts × two arms. The original unchanged prompt files
and D5 warming apply. A difference is preserved as pending D1/D3 evidence and
that profile is not timed; independent native profiles continue.

Only token-equal profiles run the six-cell interleaved session: control first
on odd repeats, candidate first on even, seven valid rows per arm/cell, clock
floor 2531 MHz, >=5 in-window samples, no thermal or foreign observations.
Invalid rows remain and have at most four attempts per logical arm/repeat.
Observer/model failures and timed output differences stop for diagnosis.
The first three token jobs and first three timed jobs record child cleanup.
Cooling/HOLD/lock/resume rules are copied from the reviewed A/A driver, with
no threshold recalibration. No new build or shader is involved.

Stage manifests record actual same-op production-shape microbench kernel names
for both schemes from the candidate gate, including inherited fallbacks.
Those names are not represented as an instrumented whole-model trace; the
retained runners have event tracing disabled. tokens.csv and runs.csv retain
identity/driver/commit/binary/profile/candidate, validity and raw evidence paths.
Per-cell medians, spreads and paired 95% intervals (seed 407014, 20,000 draws),
member-only numeric margins and six-cell geomean are saved per completed profile.
No helper makes a family acceptance or selection claim.

Projection: seven profiles × 120 processes (36 token + 84 timed), roughly
3–5 hours using the observed A/A pace, plus bounded replacements/cooling. Each
profile has a four-hour cap and the detached queue a 30-hour outer cap; this
is below the >48h reporting boundary. No parameter search or automatic retry
of a failed gate is queued. c03/c04/c07 alternative-IO correctness is retained;
c05 buffer-weight correctness remains independent work after this GPU queue.
No compilation will run alongside timing. Raw results remain under stage/s3-*;
results/native-sessions.json and STATUS update after each completed profile.

All eleven captures retained 407 observations, including 242 model runners rc0,
132 production-diff commands with all 528 cases PASSED, and the parent's unchanged
rc1 linear-benchmark dispatch-expectation behavior. No new normalized control
finding, default/tiled difference, thermal/foreign sample or observer abort.
The new assessment audit checks command return codes, all parsed cases and exact
shader coverage against raw candidate-assessment.json files. Its private evidence
is checks/candidate-gates-audit.json. Results/candidate-gates.json retains each
verdict and its exact stage path. Exact matching production-diff case counts are
c01=11, c02=12, c03=11, c04=12, c05=0, c06=12, c07=12, c08=1, c09=11,
c10=12, c11=11. The c05 zero count is an explicit coverage gap, not a pass.

Next: native c01,c02,c06,c08,c09,c10,c11 token checks and same-session six-cell
measurements at the reviewed seven repeats / 2531 MHz floor. Buffer-IO c03,c04,
c07 already have exact production-diff correctness dispatch; buffer-weight c05
still needs separate coverage. Fixed-model timing would not establish buffer
candidate performance, so it is not queued as such. All-six-cell token evidence
and numeric margins remain pending; no family-wide acceptance is inferred.

The fresh independent reviewer supported the feature/snapshot/calibration
checkpoint, checking the raw 84 rows and every referenced log/sample/record,
all reported medians/spreads/paired intervals, historical runs.csv medians,
all four golden checks, exact recursive exports (30 submodules per build),
compiled module audits and raw feature/snapshot outputs. Its reported exclusions
remain exclusions: no candidate evidence existed; selector binaries were not
recompiled; earlier full device/model/resource audits were not repeated; old
historical logs were unavailable; probes and CPU metadata were not re-executed.
The supplied review is evidence for this checkpoint, not family acceptance.
Completed preparation will not be repeated.

The next finite queue captures unmodified verify.sh gates for all eleven
committed 4070ti-wmma-c01 through c11 profiles on matrix-mm-r1, with unchanged
4070ti-fused1 attention controls. Each gate retains 37 observations, all command
return codes and dispatched linear names, a line-by-line rates-removed diff
against the reviewed control and a separate assessment. rc1 numerical failures,
next-token differences, unexpected dispatch and zero candidate coverage are
reported explicitly; no failed gate is retried or relabeled. Model/observer/
device failures stop the queue for diagnosis. Completed keys are saved atomically;
partial unassessed stages stop for diagnosis instead of being overwritten.
STATUS and a small candidate-gates.json update after each profile.

Seven native texture3d candidates have potential fixed-model dispatch. Four
buffer entries remain alternate correctness coverage. Existing microbench
buffer IO reaches c03/c04/c07 with texture weights. c05's buffer-weight shader
cannot be inferred from that: production shapes choose texture weights in
prepack_quantized_linear_weight, so c05 needs separate honest coverage. A gate
with no exact matching shader dispatch remains uncovered, even if its fallback
passes. Gate outputs record every actual kernel with shape/model/storage.

Projection: eleven fixed gates × roughly 4–8 minutes from the two completed
control captures, conservatively <=90 minutes plus staging/audit. Each gate has
a two-hour timeout; the detached queue has a 24-hour outer cap, below the 48h
reporting boundary. Estimated staging and sensor evidence fit available disk;
no rebuild, parameter search or new shader is included. HOLD is checked before
each gate; verify.sh holds the inherited lock. No timing, extra next-token
probe or acceptance occurs automatically after these gates. After inspecting
them, finish applicable native six-cell sessions (seven repeats / 2531 MHz),
all-six-cell timed/real/unaligned token checks, alternate correctness and margins.

## New measured evidence

All four probes (parent/matrix × llama/backend exact compiled libraries) observed
the actual vkCreateDevice chain: the memory-model extension enabled,
vulkanMemoryModel=1, DeviceScope=0, availability/visibility chains=0, driver rc=0.
They create/destroy the compiled Runtime with no model, shader or dispatch.
Source/library/helper/command hashes are retained. Probe instrumentation is absent
from the measured runners. Evidence: results/replacement-snapshots.json and
artifact checks/enabled-features/, stage/enabled-features-mm-r1/.

Replacement s0-parent-mm-verify and s0-topic-mm-control each have all 37 command
observations: 22 runners rc0, correctness rc0, twelve production-diff commands
rc0/ALL PASSED, four default/tiled SAME lines, two decode commands rc0. Both
linear benchmarks remain rc1: 24 cases per scheme, ok=true, twelve confirmed
and twelve unexpected_coopmat labels; benchmark correctness is SKIPPED because
verify.sh requests --skip-correctness there. This is preserved control behavior,
not a plain gate PASS. No foreign/thermal sample, slow-load possibility or runner
failure occurs across the 74 observations. Raw outputs are identical line by
line with rates removed, also matching the old parent observations. Empty diffs
and all logs stay in the replacement stages. Actor audit:
checks/snapshot-mm-audit.json; public summary results/replacement-snapshots.json.
Independent review must redo these checks from raw evidence.

## Calibration and remeasured historical control

A/A s1-aa-mm-r1 has 84/84 valid timed rows: seven per arm for all six cells,
zero invalid/replaced rows, 2048 prompt / 0 generated tokens, rc0, >=9 samples
inside each prefill window, no thermal/foreign observations and fully warmed
model pages. Actual odd/even arm order, binary/source/environment provenance
and generated-text equality match in every pair. The first three post-run child
checks find zero remaining descendants; no sampler process remains on recovery.

The pre-registered rule extended every cell from five to seven because the first
five include a >1% arm ratio and >3% spread. The lowest valid per-run median
clock is 2610 MHz, so the one device-wide floor is floor(0.97*2610)=2531 MHz.
No threshold changed. A/A geomean is 0.995829116934, within the 1%
limit; every cell is within 2%. All historical-baseline differences are within
3%, largest +2.44%, so none requires an out-of-band explanation. The table uses
valid-run medians and paired-bootstrap intervals, seed 407014 and 20,000 draws.

| Cell | Parent tok/s | Topic tok/s | Topic / parent | Paired 95% interval | Parent vs historical |
|---|---:|---:|---:|---:|---:|
| 1b-4w | 35310.30 | 35929.80 | 1.017544 | 1.000000–1.017544 | -1.72% |
| 1b-8da4w | 40156.90 | 39384.60 | 0.980768 | 0.961538–1.040000 | +1.96% |
| 3b-4w | 14222.20 | 14124.10 | 0.993102 | 0.993102–1.000000 | +1.39% |
| 3b-8da4w | 16650.40 | 16384.00 | 0.984000 | 0.976193–1.000000 | +2.44% |
| 8b-4w | 6420.06 | 6420.06 | 1.000000 | 0.996865–1.003145 | +0.31% |
| 8b-8da4w | 7501.83 | 7501.83 | 1.000000 | 1.000000–1.003676 | +0.00% |

The 1 ms runner timer step is about 1.7–2.0% in the 1B cells. Their median arm
differences are one timer step; 1B 8da4w's interval extends from about -3.85% to
+4.00%. This is noise/uncertainty, not a candidate gain or permission to widen
the limits. Seven repeats and the 2531 MHz floor apply to future candidate runs.

Actor recomputation reads every saved CSV row, runner JSON stats, sensor log,
record, binary, paired order and output hash, then recomputes the extension rule,
medians, spread, bootstrap intervals, floor, geomean and historical comparisons.
It matches the queue's results (geomean differs only by final floating-point
rounding). Source tool: tools/recompute_calibration.py. Raw audit:
checks/calibration-audit.json; public summary results/calibration.json. CSV:
stage/s1-aa-mm-r1/runs.csv; original calibration.json remains immutable there.
All raw observer records retain UNASSESSED for timing; per-row CSV decisions and
this raw-data audit supply the calibration validity, without relabeling records.

## Exact builds and approved source scope

parent-mm-r1: 5f02679b9a20d0b43ed44ea42cb1d779e1f269ac, original base plus
only the approved three-file feature fix. matrix-mm-r1:
dd57ce0e4fda4d7a5c72f9ce3c13b075428eec9e, the same runtime source bytes plus
eleven existing dev-only selectors and run-local evidence/tools. Both use exact
git-object exports of 30 pinned submodules and the pinned image/compiler.
All four unmodified golden checks PASS (53 shipped variants), and every one
of 2,091 SPIR-V modules per component/build matches the retained old parent.
All eleven shared-array audits match source. Exact binary hashes rechecked.
results/feature-builds.json; raw manifests, hashes and golden outputs remain
in build/parent-mm-r1/ and build/matrix-mm-r1/.

New parent CPU metadata execution passes all nine constant methods on all six
models, context/sequence length 2560. It links core runtime/mmap loader only;
ldd has no Vulkan dependency. Group-128 linear and packed 4-bit/group-32 embedding
facts remain from serialized-graph inspection. Evidence:
results/runtime-metadata-memory-model.json and metadata/parent-mm-r1/.

The 04:08 owner decision covers exactly Adapter.cpp, Device.cpp and Device.h
from Orin cbb7df9b6a7a605d28192fbc2a8b62c6f3611d9c. Supplied patch SHA256
5434b8e5c94ba8b18fb11e502577054bc3dea3af4e94263e6470113e316938a9.
Feature commit 8ebe66c2b and results/memory-model-fix.json record it. No release
shader, sarc/tools, golden, HOOKS, prompt or tolerance changed. Unmodified
check.sh --no-build still reports the two owner-authorized zone failures for
Device.cpp/Device.h; selector tests pass 1240 release / 1921 dev checks. That
check is NOT a plain PASS. Raw output: checks/check-memory-model-no-build.txt.

The old builds and snapshots remain preserved as historical observations.
Only superseded CMakeFiles were retired after configuration/link provenance and
removed-file hashes were archived; all exact binaries/libraries/SPIR-V/exports
and failed job records remain. About 30 GiB is free; no new build is needed.

## Reviewed inventory, controls and remaining stage

Inventory f393c3ff7ff4fc41e4651a5109a8509097c501f6; archive SHA256
65d1e3ca0eb7e9dffa8e9fbbb1a100bf2cf609868720b9053de67ff8bf5247d8;
NVIDIA manifest SHA256
ef5d13471421898c8515e9a8356caff64a20aa97432175cbcab7bf3a5cd748b1.
All 18 input hashes confirmed. Queried properties, compiled modules and the
whole-model matching-resource audit remain available; no aggregate peak-memory
or allocation-limit exception is inferred. The manifest has seven native
texture3d and four alternate-buffer candidates. All candidate entries remain
PENDING for correctness; fixed-model fallback must never count as candidate
coverage, and alternate buffer entries need honest separate correctness evidence.

Both arms retain 4070ti-fused1: k4070tiRefine1 fallback, 4070ti_nzf softmax,
fused3sb_d64_t32x32g11s32rko / fused3sb_d128_t16x64g11s32rko and existing K/V
copy. The full gate still checks decode; no attention/decode/fp-linear tuning.

Next after the prepared gate queue: inspect every assessment, then complete
applicable native six-cell sessions with same-
session controls, all-six-cell next-token checks on timed/real/unaligned prompts,
alternate-storage correctness, dispatch evidence, uncertainty and margins. A
D1/D3 difference needs its stated evidence; no candidate has been accepted.
This single member cannot establish family coverage, minimality or a split.

Reviewer: no builds or GPU jobs. Recompute from the raw stage logs/CSV, check the
exact measured-build goldens and exported sources, and verify the three approved
release edits. No new/changed shader exists for a synchronization diff audit.
No source-only review is requested and no existing owner blocker is re-reviewed.
HOLD remains untouched; all future jobs use the inherited lock and yield at
boundaries. No pushes, PRs, services, policy changes, new bodies or search.

2026-10-15T05:31:24.234845+00:00 Candidate 4070ti-wmma-c01: CAPTURE_COMPLETE; NEXT_TOKEN_AND_SESSION_EVIDENCE_PENDING. Coverage {"linear_benchmark_cases": 11, "production_diff_cases": 11}; assessment artifact stage/s2-gate-c01/candidate-assessment.json. No acceptance/timing.

2026-10-15T05:35:38.570071+00:00 Candidate 4070ti-wmma-c02: CAPTURE_COMPLETE; NEXT_TOKEN_AND_SESSION_EVIDENCE_PENDING. Coverage {"linear_benchmark_cases": 12, "production_diff_cases": 12}; assessment artifact stage/s2-gate-c02/candidate-assessment.json. No acceptance/timing.

2026-10-15T05:39:54.290903+00:00 Candidate 4070ti-wmma-c03: CAPTURE_COMPLETE; NEXT_TOKEN_AND_SESSION_EVIDENCE_PENDING. Coverage {"linear_benchmark_cases": 11, "production_diff_cases": 11}; assessment artifact stage/s2-gate-c03/candidate-assessment.json. No acceptance/timing.

2026-10-15T05:43:58.416637+00:00 Candidate 4070ti-wmma-c04: CAPTURE_COMPLETE; NEXT_TOKEN_AND_SESSION_EVIDENCE_PENDING. Coverage {"linear_benchmark_cases": 12, "production_diff_cases": 12}; assessment artifact stage/s2-gate-c04/candidate-assessment.json. No acceptance/timing.

2026-10-15T05:47:58.944596+00:00 Candidate 4070ti-wmma-c05: NO_PRODUCTION_DIFF_CANDIDATE_DISPATCH; ALTERNATE_COVERAGE_PENDING. Coverage {"linear_benchmark_cases": 0, "production_diff_cases": 0}; assessment artifact stage/s2-gate-c05/candidate-assessment.json. No acceptance/timing.

2026-10-15T05:52:08.066930+00:00 Candidate 4070ti-wmma-c06: CAPTURE_COMPLETE; NEXT_TOKEN_AND_SESSION_EVIDENCE_PENDING. Coverage {"linear_benchmark_cases": 12, "production_diff_cases": 12}; assessment artifact stage/s2-gate-c06/candidate-assessment.json. No acceptance/timing.

2026-10-15T05:56:18.584249+00:00 Candidate 4070ti-wmma-c07: CAPTURE_COMPLETE; NEXT_TOKEN_AND_SESSION_EVIDENCE_PENDING. Coverage {"linear_benchmark_cases": 12, "production_diff_cases": 12}; assessment artifact stage/s2-gate-c07/candidate-assessment.json. No acceptance/timing.

2026-10-15T06:00:28.905608+00:00 Candidate 4070ti-wmma-c08: CAPTURE_COMPLETE; NEXT_TOKEN_AND_SESSION_EVIDENCE_PENDING. Coverage {"linear_benchmark_cases": 1, "production_diff_cases": 1}; assessment artifact stage/s2-gate-c08/candidate-assessment.json. No acceptance/timing.

2026-10-15T06:04:38.254814+00:00 Candidate 4070ti-wmma-c09: CAPTURE_COMPLETE; NEXT_TOKEN_AND_SESSION_EVIDENCE_PENDING. Coverage {"linear_benchmark_cases": 11, "production_diff_cases": 11}; assessment artifact stage/s2-gate-c09/candidate-assessment.json. No acceptance/timing.

2026-10-15T06:09:20.534118+00:00 Candidate 4070ti-wmma-c10: CAPTURE_COMPLETE; NEXT_TOKEN_AND_SESSION_EVIDENCE_PENDING. Coverage {"linear_benchmark_cases": 12, "production_diff_cases": 12}; assessment artifact stage/s2-gate-c10/candidate-assessment.json. No acceptance/timing.

2026-10-15T06:13:44.047469+00:00 Candidate 4070ti-wmma-c11: CAPTURE_COMPLETE; NEXT_TOKEN_AND_SESSION_EVIDENCE_PENDING. Coverage {"linear_benchmark_cases": 11, "production_diff_cases": 11}; assessment artifact stage/s2-gate-c11/candidate-assessment.json. No acceptance/timing.

2026-10-15T23:57:53.952127+00:00 4070ti-wmma-c08: SESSION_COMPLETE; ACTOR_AUDIT_AND_REVIEW_PENDING; correctness=NEXT_TOKEN_EQUAL; acceptance=NONE; ONE_MEMBER_ONLY. Ratios {"1b-4w": 1.0, "1b-8da4w": 1.0, "3b-4w": 1.0, "3b-8da4w": 0.9920017437530653, "8b-4w": 0.917143412444397, "8b-8da4w": 1.0}; geomean=0.9843697995673315. Evidence artifact stage/s3-4070ti-wmma-c08.

2026-10-16T00:17:35.444930+00:00 4070ti-wmma-c09: SESSION_COMPLETE; ACTOR_AUDIT_AND_REVIEW_PENDING; correctness=NEXT_TOKEN_EQUAL; acceptance=NONE; ONE_MEMBER_ONLY. Ratios {"1b-4w": 1.0357147914347937, "1b-8da4w": 1.0, "3b-4w": 1.028166303092519, "3b-8da4w": 0.9919341277086434, "8b-4w": 1.0457515625, "8b-8da4w": 1.0}; geomean=1.0167226139885728. Evidence artifact stage/s3-4070ti-wmma-c09.

2026-10-16T00:37:55.470664+00:00 4070ti-wmma-c10: SESSION_COMPLETE; ACTOR_AUDIT_AND_REVIEW_PENDING; correctness=NEXT_TOKEN_EQUAL; acceptance=NONE; ONE_MEMBER_ONLY. Ratios {"1b-4w": 1.0, "1b-8da4w": 1.0400004062501587, "3b-4w": 1.0, "3b-8da4w": 1.05785, "8b-4w": 1.0031253624574064, "8b-8da4w": 1.0380227224557208}; geomean=1.0229080862476725. Evidence artifact stage/s3-4070ti-wmma-c10.

2026-10-16T01:01:20.250355+00:00 4070ti-wmma-c11: SESSION_COMPLETE; ACTOR_AUDIT_AND_REVIEW_PENDING; correctness=NEXT_TOKEN_EQUAL; acceptance=NONE; ONE_MEMBER_ONLY. Ratios {"1b-4w": 1.017544455866985, "1b-8da4w": 1.0, "3b-4w": 1.0209803669960222, "3b-8da4w": 1.0159990078134689, "8b-4w": 1.032362625894462, "8b-8da4w": 1.0}; geomean=1.0144159211102695. Evidence artifact stage/s3-4070ti-wmma-c11.

2026-10-16T01:22:48.859298+00:00 4070ti-wmma-c06: SESSION_COMPLETE; ACTOR_AUDIT_AND_REVIEW_PENDING; correctness=NEXT_TOKEN_EQUAL; acceptance=NONE; ONE_MEMBER_ONLY. Ratios {"1b-4w": 1.0, "1b-8da4w": 1.0, "3b-4w": 1.0, "3b-8da4w": 1.0079374922485427, "8b-4w": 1.0031253624574064, "8b-8da4w": 1.0}; geomean=1.0018394638639274. Evidence artifact stage/s3-4070ti-wmma-c06.

2026-10-16T07:19:08.056361+00:00 4070ti-wmma-c01: SESSION_COMPLETE; ACTOR_AUDIT_AND_REVIEW_PENDING; correctness=NEXT_TOKEN_EQUAL; acceptance=NONE; ONE_MEMBER_ONLY. Ratios {"1b-4w": 1.0, "1b-8da4w": 1.0, "3b-4w": 1.0, "3b-8da4w": 1.0079374922485427, "8b-4w": 1.0031253624574064, "8b-8da4w": 1.0036496595396702}; geomean=1.0024479347156887. Evidence artifact stage/s3-4070ti-wmma-c01.
