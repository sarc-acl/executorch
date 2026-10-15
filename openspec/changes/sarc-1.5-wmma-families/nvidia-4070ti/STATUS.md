# NVIDIA WMMA matrix status

Updated: 2026-10-15 01:13 UTC

Stage: PARENT SNAPSHOT COMPLETE / MATRIX SOURCE BUILD PREPARED.
Running: next launch is the single detached matrix-r1 build from the committed
branch head. It contains the eleven finite dev-only selectors, with no changed
shader or release code. Queue JSON is artifact jobs/matrix-r1/state.json.
No A/A or candidate GPU job is queued with this build.

Parent snapshot s0-parent-verify completed rc=0 with all 37 observations:
22 model runners returned 0; correctness returned 0; all twelve production-diff
commands returned 0 and ALL PASSED; next-token comparison lines are SAME; decode
commands returned 0. Both linear benchmark commands returned 1. Their 24 cases
per scheme all have ok=true, with twelve unexpected_coopmat dispatch labels
and twelve confirmed labels each. This is parent dispatch-expectation behavior,
not a plain verification pass or acceptance of unverified starting rows.
Raw reference: artifact stage/s0-parent-verify/verify.out and verify/ logs/JSON.
Source summaries: results/parent-snapshot.json and parent-verify-normalized.txt.

Observer audit: all 22 runner observations have zero foreign-client and zero
thermal samples, fully resident model pages after warming, and no runner crash.
Default timed-prompt prefill windows contain 11/9/27/23/58/50 sensor samples
(1B 4w/8da4w, 3B 4w/8da4w, 8B 4w/8da4w). No sampler process remained.
These are snapshot observations, not an interleaved timed session. No clock
floor, calibrated repeat count, baseline median, uncertainty or gain exists yet.
Raw observation-audit.json retains every per-invocation result. Original cache
fractions can slightly exceed one because fincore includes the final page;
future observer rows cap the displayed fraction at one and retain raw byte counts.

Whole-model resource audit completed after the coordinator's B580 reminder.
That finding is not used as a verdict for this device. All six original PTEs
contain RoPE prepack images of height 5120; the 8B images include width 3584.
The matching local 3D format queries support them. The audit checks ALL serialized
nonconstant tensors, constant-consuming op classes and source-derived generated
weight, RMSNorm, cache, attention, fused-copy and int8 buffer bounds against the
matching format/type/usage and buffer limits. No allocation-limit exception or
model re-export is used. This checks per-resource limits, not aggregate peak
memory. Evidence: artifact preflight/models-resources-raw.json and
whole-model-resources.json; small summary results/whole-model-resources.json.
The earlier projection-only image check was incomplete; parent execution is
not substituted for this matching-query evidence. No further GPU work preceded
this audit after the reminder.

Reviewed inventory: f393c3ff7ff4fc41e4651a5109a8509097c501f6.
Archive SHA256: 65d1e3ca0eb7e9dffa8e9fbbb1a100bf2cf609868720b9053de67ff8bf5247d8.
NVIDIA manifest SHA256: ef5d13471421898c8515e9a8356caff64a20aa97432175cbcab7bf3a5cd748b1.
The archive and all 18 per-file input hashes were confirmed before recovery work.
The immutable input and its source references remain the candidate inventory.
All eleven fixed compiled modules match source shared-array totals/natural
alignment and are byte-identical between parent llama and backend outputs.
Both unmodified golden checks pass for all 53 shipped variants. Matrix/type,
subgroup/workgroup, fp16/int8/storage features, image formats and production
resource prerequisites are recorded privately. Candidate correctness remains
pending, with seven native texture3d and four alternate buffer entries.

CPU-only runtime metadata already completed: all nine scalar methods on each of
six models match static evidence, including context/sequence length 2560.
Group-128 linear and packed 4-bit/group-32 embedding facts come from the actual
serialized graph; those fields have no runtime constant methods in these PTEs.
The owner resolved the initial metadata bootstrap question on 2026-10-15 00:32.
No unresolved owner question remains; no repeat preflight review is requested.

New selectors: impl/sarc_dev/NvidiaWmmaMatrix.cpp plus one appended 4070ti block
in Overrides.cpp. ET_VK_SARC_WMMA_PROFILE=4070ti-wmma-c01 through c11 requires
unverified mode and this device's pre-existing table service; each candidate
retains its recorded storage, tile, shape predicate and row-major layout. Only
the matching linear op changes. Before choices, attention controls and other
configurations stay inherited. No new shader body or family selection exists.
CPU tests of all eleven profiles passed exact kernel/storage/layout, unchanged
softmax control and decode exclusion. The original test_sarc_select is unchanged;
check.sh --no-build PASS (1240 release checks, 1921 dev checks). Evidence and the
CPU fixture binary are retained under artifact checks/.

The immutable build adapter now accepts an optional exact commit and a matrix
build tag. The same pinned image/export/manifest/binary/golden protocol applies.
Only one new intermediate build tree is created. Parent object directories were
retired after preserving CMake flags/link provenance; installed libraries/headers,
exact measured binaries, all SPIR-V modules, raw logs and source export remain.
Binary hashes rechecked after cleanup. Evidence: build/parent-r1/
retired-intermediates.json and retired-cmake-provenance.tar.gz. About 40 GiB is
free. Original failed parent queue remains failed; recovered build is reused.

Next after build: inspect exact binaries, golden and compiled-module identity;
then run interleaved control A/A on parent versus new build, both selecting the
unchanged final control profile. Fix the device-wide clock floor/repeat count
from that data and explain historical-reference differences before candidates.
Finite candidate gates, six-cell matrix sessions, alternate-storage coverage,
uncertainty, margins and independent matrix review remain incomplete. No family-
wide acceptance or measured candidate pass exists.

Scope remains finite linear matrix only. No attention/decode/fp tuning, search,
new bodies, release selection, promotion, push, PR, service or policy changes.
Private infrastructure/model paths and hardware policy stay in artifacts.

## Coordinator hold

The prior HOLD dry run remains valid; actual HOLD was not created or removed.
The next unit is one build, with the inherited lock, HOLD and foreign-client
checks, and a two-hour build timeout. No timing runs concurrently. No >48-hour
queue is launched. All later gates and resumable sessions check HOLD at unit
boundaries. Reviewer performs no build/GPU work.
