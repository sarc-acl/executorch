# sarc-1.5-m51-prefill-refine

Dev zone only. Branch `topic/m51-prefill-refine`, forked from `topic/780m-prefill-refine` at `f5f1bf10c` (the
parent of this campaign). Device: the Samsung Xclipse (M51) board, reached over adb from a build workstation.

**This directory holds tooling and number-free prose only.** By the device owner's rule, no figure of this device
(throughput, ratios, percentages, kernel times, clock values, register counts) and no host, board or driver
identifier is committed or pushed. Every table, the running `STATUS.md`, the thresholds and the raw data are kept
in the campaign's local artifact directory (`m51-LOCAL-ONLY`). Commit messages say what changed, never how much.

## Why

The xclipse rows of `impl/sarc/table_amd.cpp` (4w, 8da4w, attention QK^T and attention x V, all `kUnverified`)
already run the Llama prefill on this device with the SARC kernels. The device is in the Radeon 780M's position
(attention rows exist, RDNA-derived), so this change ports the 780M's second-layer results (owner decision N1)
and gates each one on this device: the fused attention kernel, the fp32 softmax without the zero tail, the linear
kernel chosen per layer shape, and the whole-texel 8da4w weight staging. No sampled parameter search.

## Thresholds

Fixed before the first measurement, in the local file `THRESHOLDS.md` of the artifact directory, sha256
`fcd8de6833afd9488a1e21ee95a58e9f16684b624be7b4ae2432f0ccc6fa564d`. They are the shared campaign rules (R4.4
baseline tolerance, R6 noise band, repeat rule and clock-floor rule, R8 kernel-screen margin, R11 stop rule, owner
decisions D1 and D3) applied to this device, plus the validity checks that pinned clocks allow (no active time at
any other GPU frequency, GPU cooling state zero, no GPU reset). Values derived from the A/A session go into a
separate local file, by the rule written in the frozen one.

## Method (adaptations to a board reached over adb)

- Builds on the workstation: `tools/build_tag.sh <tag> <commit>` exports exactly one commit and every submodule at
  its pinned commit from the git object stores (`tools/export_commit.sh`), then cross-compiles with
  `tools/build-native.sh` (NDK, Vulkan SDK glslc). The pinned build container is not available, so the shipped
  SPIR-V cannot match `sarc/golden/spirv.json`: the golden check is reported as pending, and the shipped variants
  of every candidate build are compared byte for byte with the parent build made by the same toolchain instead.
- The board has no bash, so `sarc/tools/verify.sh` runs unmodified on the workstation against a stage directory
  whose `llama_main` and `test_llama_microbench` are wrappers around `tools/adbshim.sh`. The shim runs the staged
  binaries on the board with the caller's `ET_VK_*` environment, maps the model and tokenizer paths and the
  `--json-out` file, and returns the device process's output and exit status. Every run is preceded by the
  device-state guard (driver identity, profiler configuration aside, clock pins read back) and logged.
- Timed sessions: `tools/e2e_m51.sh`, the 780M's `e2e5.sh` protocol on the board (fresh `llama_main` per run,
  `--warmup`, one new token, arms interleaved, cooling before each run), with a sampler on the board for the GPU
  clock, busy, temperature and thermal state, the per-frequency active time and the GPU reset count around each
  run, and the model load time of each run (whether the load was slow). `tools/summarize.py` summarises.
- Coordinator hold: every build, `verify.sh` call and timed session is one unit of `tools/hold.sh`.
- Board, serial and driver identity live in a local settings file read by `tools/dev.sh`, never in this tree.

### Tools taken from the 780M campaign and the device workspace

| tool | origin | what changed |
|---|---|---|
| `hold.sh` | 780M `tools/hold.sh` | artifact directory |
| `e2e_m51.sh` | 780M `tools/e2e5.sh` | runs on the board over adb; the sampler reads the board's GPU nodes; validity rules for pinned clocks; cooling also ends when the temperature stops falling; a second check prompt (unaligned real text) |
| `summarize.py` | 780M `tools/summarize.py` | column names; repeat count from the environment; load time; three next-token items |
| `stage.sh` | 780M `tools/stage.sh` | pushes the binaries and prompts to the board; adds the `verify.sh` wrappers and the flat model links |
| `build-native.sh` | workspace `tools/sarc-build-native.sh` | venv, glslc and NDK paths from the environment; lower default parallelism (another campaign measures on the workstation) |
| `sdpa_error_table.py` | 780M `tools/sdpa_error_table.py` | prints both arms' reference rms; the criterion requires identical inputs |
| `probe_compare.py` | 780M `tools/probe_compare.py` | unchanged |
| `probe_prompts.py` | 780M `tools/probe_prompts.py` | tokenizer read with tiktoken and the Llama 3 split pattern |
| `prof_decode.py` | 780M `tools/prof_decode.py` | K tile as a parameter |
| `build_probe.sh`, `probe_m51.sh` | 780M `tools/build_probe.sh`, `probe_run.sh` | native NDK build; runs on the board |
| `fscreen.sh`, `fscreen_summary.py` | new | ETDump screen of the fused attention variants, with the kernel-screen rule of the other screens |
| `gate_launcher.sh` | new | compiler launcher: no compile starts while a timed session runs |
| `build_tag.sh`, `export_commit.sh`, `adbshim.sh`, `dev.sh`, `push_models.sh`, `verify_m51.sh`, `verify_compare.py`, `mbstage.sh`, `spv_compare.py`, `trace_m51.sh`, `trace_families.py`, `lscreen.sh`, `lscreen_summary.py`, `phase_m51.sh`, `pdiff_error_table.py` | new | |
| `r1329.txt` | new | an unaligned real-text prompt (the first part of the kit's `prompt_real_2048.txt`) for the `r*.txt` item of `verify.sh` |

## Candidates (number-free; every figure is in the local `STATUS.md`)

Selected by `ET_VK_SARC_M51_PROFILE=<name>` on top of `ET_VK_SARC_UNVERIFIED=1`
(`impl/sarc_dev/Overrides.cpp`, m51 block). Without the variable every dispatch is the parent's: the shipped SPIR-V
of every build is byte-identical to the parent build's (`tools/spv_compare.py`), and the A/A session agreed within
the noise band.

1. **`c1`: fused attention.** The 780M's fused prefill attention kernel, ported as `sarc_dev_m51_sdpa_fused3`
   (`glsl/sarc_dev/`, node `impl/sarc_dev/m51/SdpaM51Fused.cpp`). Two findings on this device:
   - the kernel exchanges data between the invocations of its subgroup through shared memory with
     `memoryBarrierShared()` alone, which orders only an invocation's own accesses; the port uses
     `memoryBarrierShared()` followed by `barrier()` at every exchange (the workgroup is one subgroup);
   - the one-pass form (running row maximum) gave random wrong rows at S = 2048 on this driver with either barrier
     form, while the two-pass form passed every screen pass; the port uses the two-pass packed form.
   The node is installed into the override on the first selection, because the 780M's node registers into the
   same entry point from another file in an order the linker decides.
2. **Port item 2 (fp32 softmax without the zero tail) does not apply after `c1`:** the fused node serves every
   attention call of the timed prefill, so no softmax is dispatched there.
3. **`c2`: the linear kernel per shape for 8da4w, which is the 780M's texel-wise weight staging
   (`sarc_dev_linear_dq8ca_coopmat_zpg_bt`) on every 8da4w shape.** A kernel screen chose it on every shape (R8
   margin in every round); its production-diff errors equal the parent's on every shape.
4. **`c3`: `c2` plus the 4w pick for the shapes with K = 4096** (only the 8B model has them): the texel-wise B staging twin of
   the 4w tile `t128x128k32g42s32f32xp` (`sarc_dev_linear_q4gsw_coopmat_bx`, the xclipse drain flags). The other 4w tiles with the
   xclipse flags were not selected on any 1B or 3B shape. It was gated and timed in the 8B round, as part of `c5` (below).
5. **`c4`: `c2` plus the head_dim 64 fused attention variant that the ETDump screen of the fused variants selected** (the 16 x 64
   tile on a 64-wide subgroup; no head_dim 128 variant passed the screen, so the 780M's choice stays there). It acts on the 1B
   model only. Until `c4` existed as a profile it was selected through `ET_VK_SARC_M51_SDPA_FUSED` on top of `c2`, and the
   evidence of its gate was taken in that form.
6. Screened: texel-wise 4w weight staging (`sarc_dev_linear_q4gsw_coopmat_bx` on the xclipse flags), because
   the 4w kernel's phase timing shows about a third of each wave in shared-memory stores. It passed the screen on the
   K = 4096 shapes of the 8B model only (item 4).
7. **`c5`: `c4` plus the `c3` pick, one name for the final stack** (added in the 8B round; a profile-table entry in
   `impl/sarc_dev/Overrides.cpp` only). It is the stack that the 8B gate and the final sessions use. On the 1B and 3B models `c5`
   selects exactly what `c4` selects (K = 4096 does not occur there).

Test option added in the dev zone: `ET_VK_MB_SKIP_8B=1` leaves out the SDPA correctness cases named after the 8B head
configuration (no tolerance, tier or other case changes; the count of cases run shows it). `push_models.sh` takes
`PUSH_MODELS` to limit the model sizes pushed to the board.

Measurement aids added in the dev zone (never selected by default): 4w sweep tiles carrying the xclipse row's
drain flags, the phase-timing twin of the xclipse 4w row.

## Outcome and what stays open

- **R11 is met in part and the campaign ends by owner decision of 2026-10-08.** Three candidates were gated and timed
  (fused attention; whole-texel 8da4w staging; a fused-attention variant for one head dimension). The first two were
  faster outside the noise band; the third was inside it, so one sub-threshold candidate exists, not two. No further
  candidate passes a screen for the 1B and 3B models, and the owner closed the campaign on that basis. No new kernel
  was written for this. The campaign was reopened for the 8B model on 2026-10-08 (22:37 UTC, playbook decision N10): see the
  8B round below.
- **What the final profile is (`c8`):** the three candidates of round 1 (`c4`) plus the `bz` 4w kernel on every 4w shape (second
  round, below). `c5` (`c4` plus the 8B pick of `c3`) was the stack of the 8B round and is an intermediate stage: `c8` replaces
  its 4w pick. The final timed sessions of `c8` against the pristine parent, on the build of the committed head and with no local
  patch, were faster in every one of the six cells, the 8B cells included, outside the noise band; the figures are in the local
  `STATUS.md` only.
- **The 8B round (owner decision 2026-10-08, 22:37 UTC; playbook decision N10).** The 8B model is required for the campaign to
  be complete, and the earlier partial gates (1B and 3B) are now complemented for 8B. What was done, in the order of the decision:
  both 8B model files were pushed to the board and checked against the manifest's digests; the pristine parent's
  `verify.sh --models 8b` was stored as the snapshot (`s0-parent-verify-8b`); an A/A session of the 8B cells (pristine parent
  against the unmodified topic build, no profile) stayed inside the noise band, and the parent's 8B baseline agreed with the
  published 8B cells within the tolerance; the final stack was gated for 8B (below) and timed against the pristine parent; the
  4w pick `c3` was screened again on this board (it beat the incumbent by the margin in every round on the three K = 4096
  shapes and on no other shape), timed against `c4` in an interleaved A/B, adopted, gated as part of `c5` and timed against
  the pristine parent. **The setting for every 8B run, in both arms: `ET_VK_EXECUTE_NODE_THRESHOLD=32`** (a command-buffer
  submission every 32 graph nodes instead of 128; without it a 2048-token 8B prefill can trip the GPU's job watchdog, which
  crashed a board during an earlier campaign's 8B verification). The 1B and 3B runs never carry it. It is read by an opt-in
  block in an upstream file (the hook below).
- **Release-zone hook of the 8B round (owner decision 2026-10-08, 22:37 UTC: one additional hook beside those of owner decision
  D4), its own commit, with its line in `sarc/HOOKS`:** `backends/vulkan/runtime/graph/ComputeGraph.cpp`, in the constructor after
  the default thresholds are set. With the variable unset nothing changes.
  ```
  if (const char* thr = std::getenv("ET_VK_EXECUTE_NODE_THRESHOLD")) {
    const int n = std::atoi(thr);
    if (n > 0) {
      config_.execute_threshold_node_count = static_cast<size_t>(n);
      config_.execute_initial_threshold_node_count = static_cast<size_t>(n);
    }
  }
  ```
  (the commit also carries the owner's explanatory comment, unchanged). Shown: the hook commit on top of the parent changes no
  shipped or other SPIR-V (`tools/spv_compare.py`: every compiled shader identical); the dispatched kernels and their counts
  in an ETDump of every 1B and 3B cell are the same with the hook as without it; the 8B A/A session of the build with the hook
  against the parent with the hook agrees within the band. The 8B parent arm of every 8B comparison is the parent commit plus
  exactly this commit (a build tag `parent8b`); the candidate arms are builds of this branch (which contains it).
- **Not covered by the 8B round:** the 8B decode path (no item of the gate runs it for 8B), the pinned container's golden check
  (pending, as before), the roofs (not re-measured), and the group size and context length of the 8B files as read through the
  runtime (taken from the manifest of the model share). The 1B and 3B part of the final stack was re-checked on the build of
  the head (below).
- **Gate state of the final stack (profile `c4`, build of the head's sources, 1B and 3B only: PARTIAL):**
  recorded as `ACCEPTED (reference-error rule, owner decision 2026-10-04), PARTIAL: 1B and 3B end to end; 8B shapes by
  microbenchmark and SDPA cases only`. On the final board: the unmodified `verify.sh --models 1b,3b` of the pristine parent
  (`s0-parent-verify`) and of the final stack agree line by line except for the one line that names the dispatched 8da4w
  linear kernel (the intended change of candidate 2); the SDPA tiers `extended` and `full`, 12 passes each, include the 8B-head
  cases and have 0 mismatches and `pairing=ok`; the reference error of the fused attention is smaller than the parent's in every
  case, the production shapes and the 8B-head cases included; the 8da4w production-diff of the linear kernel is as exact as
  the parent's on every shape; the next token equals the parent's on the timed, real-text and unaligned prompts in all four
  cells; the real-text probe passes the gross-divergence check. The shipped SPIR-V of the final build equals the parent
  build's, and the golden check against `sarc/golden/spirv.json` stays pending (the pinned container is not available here;
  the native compiler gives the same set of differing variants for the parent build and for the final build).
  `sarc/tools/check.sh --no-build` passes.
- **Gate state of the final stack for 8B (profile `c5`, build of the head's sources, setting `ET_VK_EXECUTE_NODE_THRESHOLD=32`
  in both arms):** the unmodified `verify.sh --models 8b` of the pristine parent and of the final stack agree line by line
  except for the two lines that name the dispatched linear kernels (the intended changes of candidate 2 and of the `c3` pick);
  the 8B tiled and default prefill runs ran to the end; the production-diff of the 8B shapes (both schemes, both storages)
  passes and the final stack's errors are not larger than the parent's on every shape (the 4w kernel of the `c3` pick is
  bit-identical to `c4`'s output on the 32 real-text prompts of the probe); the SDPA tiers `extended` and `full`, 12 passes
  each, include the 8B-head cases and have 0 mismatches and `pairing=ok`; the attention reference error is smaller than the
  parent's in every case including the 8B heads; the real-text probe (32 prompts, 8B) passes the gross-divergence check. The
  next token of the three prompts equals the parent's in all 8B items but one: **the 8B 8da4w timed prompt differs
  (`ACCEPTED (reference-error rule, owner decision 2026-10-04), near-tie`)**. That prompt is the degenerate repetition of one
  word; the two top tokens are separated by a very small margin already in the parent (in both the parent's tiled and default
  arms), the final stack's tiled and default arms agree with each other and rank the other token first, and the real-text
  comparison, the reference error and the gross-divergence check hold; the logits of the four arms at that position are in the
  local evidence. The 1B and 3B side of the final stack was re-gated on the same build: `verify.sh --models 1b,3b` against its
  snapshot and the timed session (local figures).
- **Known property of the final stack at other prompt lengths (diagnosed from an ETDump of the unaligned prompt; the
  campaign's target is the 2048-token prefill, owner decision 2026-10-08, and the speed at other lengths is not a criterion;
  those prompts are in the gate for correctness only):** candidate 2 picks `zpg_bt`, a
  kernel of the 4h4w activation layout. That layout is fixed when the graph is built, with an aligned sequence length. At run
  time `q4gsw_coopmat_fits` requires the sequence length to be a multiple of 128; for a prompt that is not, nothing fits,
  and the release code keeps a coopmat kernel for any length only for the row-major layout (the parent's `zpgtr` row), so the
  4h4w layout falls to the release stock tiled linear kernel. The 8da4w cells of the final stack are therefore slower than
  the parent's for prompts whose length is not a multiple of 128 (seen in the gate's real-text and unaligned prompts, on both
  boards). The fused attention is not dispatched at such lengths either, so the 4w cells equal the parent's there. The timed
  2048-token prompt is aligned, so the headline is not affected. A shape predicate cannot cure it (the pick is made at build
  time). The final stack is kept unchanged by the owner's decision; this is recorded as a property, not as a finding to fix.
- **Evidence that exists** (figures local): timed sessions per candidate and for the final stack with all runs valid,
  next token equal on all items run, SDPA tiers 12 passes each with 0 mismatches on the cases that ran, reference error
  on identical inputs not larger than the parent's on the production shapes, production-diff of the 8da4w linear equal
  to the parent's, logits probes with the gross-divergence check passed, ETDump showing the dispatched kernels, and a
  read of the new shader (the fused attention kernel) for unsynchronised shared writes: every exchange between lanes of the
  workgroup, which is one subgroup, is separated by a shared-memory barrier, and no two lanes write one location.
- **Second round of linear kernels (owner decision 2026-10-08, 23:38 UTC).** In-kernel phase timing and the driver's
  pipeline statistics and ISA of the two linear kernels, on the shapes of all three models, located the cost: the 8da4w
  kernel spends about as much wave time staging the operands as multiplying and about a tenth in its per-quantization-group
  epilogue; the 4w kernels load their B operand from shared memory with hundreds of 16-bit gathers per loop. Both kernels
  already use double-buffered staging with one barrier per K step, so the order's "cut per-step synchronisation" had nothing
  to cut: twice the K per barrier changed the 8da4w kernel's time by less than the screen margin, and so did loading every
  fragment of a chunk before its MMAs (the compiler already schedules it so). Candidates, each default off, `m51` prefix, in
  the dev zone: (1) `zpgd`, 8da4w with the A operand loaded by `coopMatLoad` straight from the row-major activations: correct,
  slower than the incumbent on every shape (more vector registers, no occupancy gain); (2) smaller subgroup tiles of the
  8da4w kernel: slower on every shape; (3) `zpgf` (all fragments first): no effect; (4) **`bz`, the 4w texel-wise-staging kernel
  with the B operand stored N-major inside the shared pool** (the stock body refuses `B_COLMAJOR` together with the pool; this
  variant combines them, so the matrix unit's B loads are wide): correct (production-diff passes on both storages for all three
  models, with errors not larger than the parent's; its output is bit-identical to the previous stack's on the 32 real-text
  prompts of every cell), faster than the incumbent 4w kernel by the screen margin in every round on all twelve shapes of the
  three models, and faster than the K = 4096 pick of `c3` on those shapes; (5) `bw`, `bz` with packed fp16 dequantisation: about
  half the margin, so not selected. **Profile `c8` = `c4` plus `bz` on every 4w shape** (it replaces the `c3` pick): gated
  against `c5` (the unmodified `verify.sh` agrees with the parent snapshots except for the two intended kernel-name lines;
  SDPA tiers and reference error as before; next token equal in all items) and timed against `c5` (the 4w cells of all three
  models faster outside the noise band, the 8da4w cells, which it does not touch, unchanged). The final verification of `c8`
  ran on the build of the committed head (`fin2`, no local patch) after the board was restored by its owner (the board's driver had
  been replaced by another user while the queue was idle, and the verification waited for the restore; recorded in the local
  `STATUS.md`). PARTIAL as before, the 8B cells with the setting in both arms: the unmodified `verify.sh` (`--models 8b` and
  `--models 1b,3b`) agrees line by line with the parent snapshots except for the two lines that name the dispatched kernels (the
  intended change); the production-diff of both schemes on all three models, both storages, three passes, passes with errors not
  larger than the parent's on every shape; the SDPA tiers `extended` and `full`, 12 passes each, pass with no mismatch and
  `pairing=ok`, with the SDPA error not larger than the parent's; the real-text probe passes the gross-divergence check in every
  cell; next token on the three prompts is the parent's in all cells but the 8B 8da4w timed prompt, which stays the recorded
  near-tie (accepted under the reference-error rule, owner decision 2026-10-04: the same item as for `c4`, its path is the
  unchanged 8da4w kernel). The shipped SPIR-V of the head build equals both parent builds' (53 of 53 shipped variants
  byte-identical, none changed) and the golden check stays pending (native compiler; the same set of differing variants as the
  parent build). The final session against the pristine parent was valid in every run (8B cells and 1B/3B cells) and every cell
  was faster outside the noise band. Reference-error evidence for the whole stack: the `c4` and `c5` evidence stands for the
  8da4w path and the attention kernels, and the 4w `bz` kernel produces output bit-identical to `c5`'s on all 32 prompts of every
  cell (`c8` against `c5`) as well as errors not larger than the parent's in the production-diff.
- **Directions left for later work:** a new 4w or 8da4w GEMM kernel (the linear kernels are most of the prefill and the
  screens found no tile that beats the incumbents, apart from the 4w pick of `c3` on the 8B shapes), a fused SwiGLU (the elementwise operators are stock kernels), an 8B decode check, and a driver-side fix for the job watchdog that makes the 8B setting unnecessary.
- **Builds:** the golden check is pending; the compiler launches of a build wait while a timed session runs
  (`tools/gate_launcher.sh`; a stop signal to the build had no effect in the agent's environment). That gate was shown
  with a fake session on the build host only: the one build made with it (`f2`, the head that added it) was interrupted
  before it finished and was never used, so the gate has not been shown inside a completed real build. The final
  verification of the second round uses build `fin2` (commit of the last kernel change; later commits change this change
  directory only).
- The fused attention kernel's subgroup size inside the shader (`gl_SubgroupSize` read-back) is not verified on the device; the
  pipelines declare a required subgroup size and the correctness tiers pass.

## Status

Kept in the local `STATUS.md`.
