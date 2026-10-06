# Playbook: a GPU kernel-tuning campaign on a device you have not seen

What this is for: you are an agent on a machine that reaches only the git remote. You are to repeat, on new
GPUs, the campaign that raised 2048-token Llama prefill speed by +32 to +67 % on five devices. This document
is the ordered procedure. Follow it in order; where a step says "stop", stop and write the reason into
`STATUS.md`.

What it is not: `openspec/changes/sarc-1.5-e2e-benchmark/CONTRIBUTING-A-GPU.md` covers **how to benchmark a
device and deliver results** (builds, grid, CSV schema, deliverables). This package covers **how to run a
tuning campaign**: finding, gating and recording faster kernels. Read that guide for the benchmark protocol;
it is not repeated here.

| file | read it when |
|---|---|
| `PLAYBOOK.md` (this) | first; it is the order of work |
| `RULES.md` | before any work; it is part of every task |
| `LESSONS.md` | before touching a host; each lesson cost device time once |
| `OWNER-DECISIONS.md` | before judging any candidate; D1 to D9 in force, N1 to N9 the defaults for new devices |
| `COORDINATOR.md` | if you are the coordinating session |
| `results.md` | when you need a number to compare with |
| `templates/TASK.md`, `templates/PROMPT.txt` | when starting a device |
| `flow/gpu_campaign/__init__.py` | the hmz flow (actor and reviewer on the GPU host) |
| `tools/patrol.sh`, `tools/HOLD.md` | patrol script; the coordinator hold |

Placeholders used throughout: `<host>`, `<campaign-root>`, `<artifact-dir>` (= `<campaign-root>/.artifacts`),
`<tag>` (device tag: `7900xtx`, `rx7600`), `<change>` (= `openspec/changes/sarc-1.5-<tag>-prefill-refine`),
`<lock-uuid>`, `<parent>` (parent commit), `<model-dir>`. Never write real host names, user names, absolute
home paths, addresses, serial numbers or lock ids into a file that is committed: the repository is public.

Marking: **[owner]** a decision of the owner, **[measured]** a fact from the finished campaigns,
**[recommended]** advice of this package that nobody ordered and nobody measured.

## Part 0. The setting

- Devices finished: Radeon 780M, Arc B580, Arc Pro B70, RTX 4070 Ti SUPER, Jetson Orin Nano. Next: Radeon
  RX 7900 XTX and Radeon RX 7600. Both already have unverified kernel rows in the tree
  (`sarc/README.md`, "Devices").
- Roles: one actor and one reviewer per device (an hmz run of the flow in `flow/`), one coordinating Claude
  Code session for all devices, and the owner. Several devices run in parallel, one hmz workspace each.
  **[owner, N8]**
- Environment on your machine: Claude Code and hmz are available, Codex is not. The reviewer is Claude with a
  fresh context each round. **[owner, N6]**
- Expected gain on the two next devices: +20 to +30 % geometric mean, not +50 %: they already have attention
  kernels, like the 780M (+31.9 %). **[owner, N1]**

The method, in one paragraph: build the parent from an exported commit; re-measure the six cells against the
published table; run an A/A session to calibrate noise and validity thresholds before any candidate; locate
the time; take candidates from three sources (port a kernel proven on a sibling device, write a new kernel for
the located bottleneck, parameter search); pass every candidate through the unmodified gate and a timed
session with old and new builds interleaved; stop when two consecutive gated candidates gain under 2 %; a
reviewer with fresh context recomputes from raw files each round.

## Part 1. Before a campaign starts (coordinator)

### Step 1. Prepare the control machine

1. Check that Claude Code and hmz start, and record the hmz commit you run. This flow was last run
   with hmz commit `474add43cf42db8881741a29cb81ef3f9a8c1a87`; hmz is not pinned **[owner, N6]** and changed its
   API on 2026-10-05 (`AgentView.spawn()` no longer takes `env`; pass `env=` to `run()`). The flow file here
   already has the new form.
2. Install the flow: `cp -r flow/gpu_campaign ~/.hmz/flows/`. Check that it imports with hmz's own interpreter:
   `~/.local/share/uv/tools/hmz/bin/python -c "import sys; sys.path.insert(0, 'flow'); import gpu_campaign"`.
   If the import fails, hmz changed again: fix the flow before launching anything. After any change of the
   flow file, quit and reopen a running hmz window before `/resume`.
3. Fetch the branches you need:
   `git fetch origin topic/780m-prefill-refine topic/llamacpp-compare dev/1.5`.
4. Make one directory per device on the control machine (the hmz workspace) holding the filled-in task and the
   launch prompt. Keep these private: they contain host facts.

### Step 2. Collect the device and host facts

Fill sections 1 and 2 of `templates/TASK.md` from the host itself, read-only:

1. Device, driver and version, Vulkan device index (`vulkaninfo --summary`).
2. Cooperative-matrix shapes, subgroup size range, shared-memory limit. The 780M's attention kernels use
   subgroup 64 and 16 x 16 x 16 matrices **[measured]**; a device that does not expose those cannot run them
   unchanged. The RX 7600 under RADV (Mesa 26.2.3) exposes subgroup 32 to 64 and M = N = K = 16
   (`contrib/rx7600/NOTES.md`); for the RX 7900 XTX query the driver you will actually use.
3. Clock policy as found (frequency governor or performance level). Record it; do not change it (L18).
4. Sensors: where temperature, clock and power are read, and how fast they answer.
5. Other GPU users: display, services, CI runners, other agents. Stopping a service is the owner's act, not
   yours; write the restore command into the task file.
6. RAM against the model sizes (the six files total about 14 GB). If they do not fit in the page cache
   together, say so in the task file: D5 applies.
7. Free disk under `<campaign-root>`. **[recommended]** have 100 GB free: every build tag keeps its exported
   source tree and two builds, and the sizes the finished campaigns reached were not recorded.
8. The models **[owner, N7]**. The files on your machine are the same models but not byte-identical to the
   ones the published numbers were measured with. For each of the six `.pte` files:
   - size against the table in `../sarc-1.5-llamacpp-compare/kit/MODELS.md` (on branch `topic/llamacpp-compare`:
     `git show origin/topic/llamacpp-compare:openspec/changes/sarc-1.5-llamacpp-compare/kit/MODELS.md`);
   - context length: the constant method `get_max_context_len` must return 2560. One way, with the ExecuTorch
     Python runtime of a venv under `<artifact-dir>` (not run when this package was written; adapt if the API
     differs):
     ```python
     from executorch.runtime import Runtime
     prog = Runtime.get().load_program("<model>.pte")
     print(prog.load_method("get_max_context_len").execute([]))
     ```
   Write both results into the task file. A file with another context length or a size off by more than a
   fraction of a percent is a different export: report it to the owner before measuring anything.

### Step 3. Create the working copy, the task and the run

1. On the host: `git clone <public fork> <campaign-root>/executorch` (the directory must be named
   `executorch`), then
   `git switch -c topic/<tag>-prefill-refine origin/topic/780m-prefill-refine` for an AMD device. **Start from
   the branch of the same vendor's finished device, not from `dev/1.5`**: nothing is merged yet **[owner, N5]**.
   `<parent>` = the commit the new branch starts at (`git rev-parse HEAD`); write it into the task file.
2. Fill in `templates/TASK.md` completely. Section 4 for the two next devices, from `sarc/README.md`:

   | device | 4w row | 8da4w row | attention rows | state |
   |---|---|---|---|---|
   | RX 7900 XTX | `t256x128k32g24s32f32cbt` (fp32 accumulate) | 780M zpg `t128x64k32g42s32` | 780M QK^T and attention x V | unverified |
   | RX 7600 | same 4w tile | same | same | unverified |

   Unverified rows are inert unless `ET_VK_SARC_UNVERIFIED=1` is set: the parent arm of these devices needs it.
   Published numbers: `results.md` section 5.
3. Assemble the task file on the host, outside the working copy, device part last:
   `cat RULES.md LESSONS.md OWNER-DECISIONS.md <filled TASK.md> > <campaign-root>/CAMPAIGN.md`.
4. Fill in `templates/PROMPT.txt` (two placeholders).
5. Launch, from the device's workspace directory on the control machine:
   ```sh
   hmz exec -f @user/gpu_campaign \
       -a actor=<model> -a reviewer=<model> \
       -e box=ssh@[<user>@<host>]/<campaign-root>/executorch \
       -b duration=3d,cost=100 "$(cat PROMPT.txt)"
   ```
   Both models are passed with `-a`. Budgets are per run; a resumed run gets a fresh budget and fresh
   conversations, which is why everything the actor must know is in `CAMPAIGN.md` and `STATUS.md`.
6. Add the campaign to the table at the top of `tools/patrol.sh` and start patrolling (`COORDINATOR.md`).

## Part 2. The campaign (actor, on the GPU host)

### Step 4. Read, copy the tools, fix the thresholds

1. Read `CAMPAIGN.md` completely, then the files of `RULES.md` R2.
2. Create `<change>/`. Copy the sibling's `tools/` into it (for an AMD device:
   `openspec/changes/sarc-1.5-780m-prefill-refine/tools/`; list that directory first, its file names are the
   ones to use). Adapt only what is hard-coded for the sibling's host: paths, sensors, lock name, device index,
   the list of foreign GPU programs. Record every change in a table in `proposal.md`.
   What the tools of a finished campaign contain **[measured, B580 campaign]**: `host.sh` (constants, guard,
   `export_commit`, cooling), `gl.sh` (run one GPU job under the lock and the guard), `build-both.sh`,
   `stage.sh`, `parent_verify.sh`, `session.sh` and `e2e5.sh` (timed session), `gate.sh`, `gate_sdpa.sh`,
   `gate_check.py`, `trace.sh` and `trace_analysis.py` (ETDump), `prof.sh` and `prof_decode.py` (phase timing),
   `screen.sh`, `screen_sdpa.sh`, `screen_rows.py` (resumable kernel screens), `sdpa_ref.sh`, `sdpa_error.py`,
   `decide.py` (reference-error rule), `probe.sh`, `probe_analysis.py` (logits), `summarize.py`, `roof.sh`.
3. Put `TMPDIR`, the venv and every output under `<artifact-dir>` (L23).
4. Write the thresholds into `proposal.md` and commit before the first measurement (R6, D1 to D3):
   validity of a run; clock floor = 97 % of the lowest per-run median clock of the valid A/A runs; repeats 5
   (7 if any A/A cell is further than 1.0 % from 1 or any arm's spread exceeds 3.0 %); noise band +-2 %;
   baseline agreement 3 % per cell; screen margin 3 % in every round; reference-error limits (0.5 nat mean KL,
   one third of the prompts); done = two consecutive gated candidates under 2 %.
5. Add the coordinator hold to the queue runner you will use (`tools/HOLD.md`), test it once, record it in
   `STATUS.md`.

### Step 5. Build the parent and store the snapshot

1. Build `<parent>` from an exported commit, never from the working tree:
   `tools/build-both.sh parent` (it calls `sarc/tools/build.sh --llama` and `--traced` on an export made by
   `export_commit`, and records commit, tree, image id and glslc version beside the build).
2. Golden: `sarc/tools/spirv_golden.py <build>/backend/vulkan_compute_shaders sarc/golden/spirv.json` must report
   the shipped variants unchanged. If it does not, the build is not usable: find out why (toolchain, image)
   before anything else. The unverified AMD rows were last built with a native toolchain, not the container
   (`contrib/*/NOTES.md`): a golden difference on exactly those rows is a finding to report, not to repair by
   editing the golden.
3. Snapshot **[recommended; goes with owner decision N2]**: run the unmodified
   `sarc/tools/verify.sh --dir <stage> --lock <lock-uuid> --models 1b,3b,8b --schemes 4w,8da4w --pdiff` once on the
   parent build with the parent's environment and keep its output as `stage/s0-parent-verify/verify.out`
   (`tools/parent_verify.sh`). **This is not a verification pass.** Do not fix what it reports, do not stop for
   its failures: by N2 the existing unverified rows are not verified first. The snapshot exists so that a
   failure in the final verification can be told apart from one that was there before you started.

### Step 6. Re-measure the baseline

1. Stage parent and topic build (no profile) and run the first session with calibration:
   `tools/session.sh s1-aa --calibrate`. It is baseline and A/A in one: parent against the unmodified topic
   build, arms interleaved, 5 valid runs per cell.
2. Each run: fill the page cache (`cat <model>.pte > /dev/null`) before the first process of a cell, then
   `llama_main --prompt_file prompt_2048.txt --max_new_tokens 1 --temperature 0 --warmup` in a fresh process;
   read `prefill_token_per_sec` from the `PyTorchObserver` line and check `"prompt_tokens":2048`.
3. Compare the six parent medians with the published table (`results.md` section 5). **A cell more than 3 %
   away is explained before you continue [owner, N7].** Things that explained a difference before: a
   different driver or toolchain; a model file that is another export; a published median taken on a disturbed
   device (L21); a clock policy that differs (L18); the unverified rows not enabled.
4. A prompt whose token count is not a multiple of the tile runs the tiled kernels for every linear layer; a
   2047-token prompt measures stock. Check the dispatched kernel names in the snapshot.

### Step 7. Calibrate from the A/A session

1. The A/A geometric mean must be within 1 % of 1 and every cell within +-2 %. If not, nothing is optimised
   until the cause is found.
2. Store the clock floor, the idle temperature and, where the device has per-client engine accounting, the
   foreign-busy ceiling (max(5 %, 2 x the 90th percentile over the valid A/A runs)).
3. Decide 5 or 7 repeats by the rule fixed in step 4.4.
4. On a power-limited or integrated GPU, look at the clock per cell: if arms settle at different clocks
   reproducibly, the floor will reject whole arms (L17). Report such arms with their clock; do not lower the
   floor after the fact.
5. Check the sampler's process count after the first few runs (L39).

### Step 8. Locate the time

1. `tools/trace.sh` on both arms: warm ETDump, time per operator family (linear, QK^T, attention x V, softmax,
   quantize, other) for the six cells. Put the table into `STATUS.md`.
2. Phase timing inside the linear kernels (`tools/prof.sh`, the `sarc_dev_prof_*` shader-clock twins): fetch,
   stage, multiply, drain per wave.
3. Roofs: igpu-roofline `fast` plan on the current driver, then percent of roof per kernel. If the tool is not
   on the host, set it up under `<artifact-dir>` or report rates without percent-of-roof.
4. What the finished campaigns found **[measured]**: attention was about 30 to 40 % of the prefill where it
   was stock; in 8da4w the weight fetch (32 to 43 % of a wave) cost more than the multiply (22 to 35 %); on
   the 780M the softmax was the largest kernel that is not a matrix multiply.

### Step 9. Candidates

Default for a device that already has attention rows **[owner, N1]**: port the 780M's second-layer results, one
gated candidate each, in this order:

| # | candidate | gain where measured | where to take it from |
|---|---|---:|---|
| 1 | fused attention kernel (one kernel instead of QK^T, softmax, attention x V) | +13 % (780M) | the 780M change's dev-zone files and its `proposal.md`; entry point is a D4 hook |
| 2 | softmax that reduces in fp32 and writes no zero tail | +4 % (780M) | dev-zone softmax variant selected through `Override::softmax_variant` (D4.1) |
| 3 | linear kernel chosen per layer shape | +3 % (780M) | the 780M profile's shape predicates in `impl/sarc_dev/Overrides.cpp` |
| 4 | 8da4w whole-texel weight staging | +6 to +7 % (Intel) | the texel-wise `zpg` family in `glsl/sarc_dev/` |

Where these live on `topic/780m-prefill-refine` (names read from the branch on 2026-10-06; the branch's
`openspec/changes/sarc-1.5-780m-prefill-refine/{STATUS.md,proposal.md}` are the authority if they have moved):

- Fused attention kernel: `impl/sarc_dev/780m/Sdpa780mFused.cpp` and the shaders
  `glsl/sarc_dev/sarc_dev_780m_sdpa_fused{,2,3}.{glsl,yaml}` (three generations; the final configuration uses
  the last accepted one) with the helpers `sarc_dev_780m_sdpa_{kvt,vt}`. Switched by
  `ET_VK_SARC_780M_SDPA_FUSED`; its entry point into the upstream SDPA node is the hook D4.3.
- Softmax variants: `glsl/sarc_dev/sarc_sdpa_attn_weights_softmax_780m_{r1,r3,m1}.{glsl,yaml}`, selected through
  `Override::softmax_variant` (D4.1). `r3` is the accepted one.
- Profiles: `780m-refine1` to `780m-refine6` in `impl/sarc_dev/Overrides.cpp` (selected with
  `ET_VK_SARC_DEV_PROFILE`), and the later candidates `c7` to `c11` selected on top of `780m-refine3` with
  `ET_VK_SARC_780M_PROFILE`. The final configuration of that campaign is
  `ET_VK_SARC_DEV_PROFILE=780m-refine3 ET_VK_SARC_780M_PROFILE=c11`: `c10` carries the fused kernel, the `r3`
  softmax and the 4w kernel per shape, `c11` adds the 8da4w kernel per shape (+1.2 %, inside the noise band).
- The parameter space of that device's sweeps is described in `impl/sarc_dev/Space780m.inc`; it is not needed
  for a port.

Rules of the port:

1. Reach every kernel from the dev zone: your own file `impl/sarc_dev/<Device>Sdpa.cpp` /
   `<Device>Linear.cpp` registering `kUnverified` rows for your device string only, matching only while
   `ET_VK_SARC_DEV_PROFILE` names a `<tag>-*` profile; your profiles in an append-only block marked with your
   tag in `Overrides.cpp` (R3). Check whether the starting branch already carries the softmax hook:
   `git grep -n softmax_variant backends/vulkan/runtime/graph/ops/impl/sarc/`.
2. Candidate 0 is the sibling's final profile run unchanged on your device, if it runs at all. It tells you at
   once how much of the port is free.
3. Before a ported kernel is gated, check its static preconditions on your device: matrix shapes, subgroup
   size, shared-memory use against the limit (three 780M variants lost the Vulkan context by exceeding
   64 KiB), divisibility of the real shapes (`sarc/SWEEP-PARAMETERS.md`, "Hard preconditions").
4. Read each new or changed shader once for unsynchronised shared writes before gating it (L8, R7).
5. **No sampled parameter search by default [owner, N1].** No tile sweep of the linear kernels as a first move
   (L2). A screen of the handful of tiles that already exist in the dev zone is cheap and allowed; a screen
   selects only at 3 % in every round.
6. Anything projected to run longer than 48 hours: stop and report the count and the projection first (R8).

For a device that has no attention rows (not the case of the next two), the order is instead: attention
kernels first (L1), then the fp32 softmax, then 8da4w staging, 4w last.

### Step 10. Gate every candidate

Run `tools/gate.sh <session> [--sdpa]` on a staged candidate; the candidate's environment is the staged file
`cand/env` and nothing else. The gate is:

1. unmodified `verify.sh` with the candidate's environment on exactly the timed binaries, compared line by
   line with the snapshot;
2. attention changes: `test_llama_microbench --sdpa-correctness-only --sdpa-tier=extended` and `--sdpa-tier=full`,
   12 passes each, 0 mismatches, `pairing=ok`;
3. next token parent vs candidate, six cells, on the timed, the real-text and the unaligned prompt;
4. golden unchanged on the candidate build;
5. for an arithmetic change (every attention kernel that accumulates in fp32, every precision change): the
   reference-error evidence of D3, judged by `decide.py`: rms and maximum error against the fp32 CPU reference
   not larger than the parent's on the S = 2048 cases; the real-text comparison on at least 32 prompts; the
   gross-divergence limits. Recorded as `ACCEPTED (reference-error rule, owner decision 2026-10-04)`, never as a
   plain pass **[owner, N3]**.

If a gate fails: do not run it again. Find which run failed and what distinguishes it (L10). A runner abort
with rc 134 after the result line in a slow-load process is L9: the run is invalid and is replaced; any other
abort is a new problem and is reported. A failed candidate is recorded with its numbers and stays.

### Step 11. Timed session

Part of `gate.sh`, or alone with `tools/session.sh <session>`:

- parent build and candidate build interleaved in one session (parent first on odd repeats, candidate first
  on even ones), median of 5 valid runs per cell, clock and temperature sampled during every run;
- cool before the session and between runs; no build on the host during it; the device lock held throughout;
- a gain is a geometric mean over the six cells outside +-2 %. Inside the band it is noise, whatever its sign.

### Step 12. Record, commit, iterate, stop

1. After every candidate: rewrite `STATUS.md` (UTC time, what is running, the per-cell table against the
   parent, where the gain came from by the trace, next step, blocking items), commit, and move anything that
   landed in a temp directory.
2. A question only the owner can answer goes under "Decision needed from the owner" in `STATUS.md`, with the
   measured numbers and the options. Then continue with work that does not depend on the answer.
3. **Stop when two consecutive gated candidates each gain under 2 % geometric mean over their parent**
   **[owner, N3]**, or when the budget ends. Negative results are results: keep them with their numbers.

### Step 13. Final verification and closing

By N2 everything is verified together here, once, on the committed branch.

1. Commit the release-zone hooks the accepted candidates need, each as its own commit under the heading
   "Release-zone hook (owner decision 2026-10-05)" (D4). Show that with nothing selected nothing changed:
   `test_sarc_select` unchanged, golden unchanged, `verify.sh` with no environment identical to the snapshot.
2. Build the branch head from an exported commit, with no local patch.
3. Run the full gate of the final stack on that build, and the reference-error evidence for the stack as a
   whole, not only per candidate.
4. Run one final timed session of the final stack against the pristine parent.
5. If the final verification fails on an item that the snapshot `s0-parent-verify` already shows as failing
   with the same lines, it was there before the campaign: report it as a property of the starting rows. If the
   snapshot passed that item, a candidate broke it: find which, by re-running that item per candidate.
6. Update `proposal.md` (outcome, what changes, the hook diffs, thresholds, tool changes) and `STATUS.md`; run
   `sarc/tools/check.sh --no-build` and report its output unedited (it may name the hooks).
7. Commit and push your topic branch: `git push origin topic/<tag>-prefill-refine`. No force, no pull request.
8. Final report: per-cell tok/s against the parent and against the published table; where each gain came
   from, with the timing data; percent of the fresh roofs; negative results; what limits further progress.

The reviewer says "done" only after doing the checks of R12 itself on this final state. A campaign can be
reopened after it was pushed (L8): what was measured with a kernel that later changed is kept and labelled,
and is not evidence for the changed kernel.

## Part 3. After the tuning

### Step 14. The llama.cpp comparison on the device [owner, N9]

Run by the coordinator, not by the campaign, with the kit on branch `topic/llamacpp-compare`
(`openspec/changes/sarc-1.5-llamacpp-compare/kit/`; read its `README.md` and `../design.md`).

1. If the campaign is still running, take the device through the coordinator hold (`tools/HOLD.md`).
2. Add the device adapter `kit/hosts/<device>/host.sh`, written from the campaign's own session script: same
   lock, sensors, clock floor, foreign-workload list (functions `gpu_shared`, `guarded`, `gpu_others`,
   `drm_clients`; variables `LOCK`, `DEV_CLKMIN`, `DEV_BUSYMAX`; optionally `gtemp`, `dev_sampler`). Make the
   sampler kill its children (L39).
3. Build llama.cpp at the kit's pinned tag (`kit/VERSIONS.md`) outside the repository:
   `-DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DGGML_NATIVE=ON`, targets `llama-completion llama-bench`.
   Include the vendor's own llama.cpp backend where one exists (SYCL on Intel and CUDA on NVIDIA were
   measured). For an AMD card no vendor backend was measured on the 780M; **[recommended]** ask the owner
   which, if any, to add, and otherwise report Vulkan only and say so.
4. Arms: ExecuTorch `stock` (`kit/build-stock.sh`), `sarc` (the campaign's parent build), `tuned` (the accepted
   profile, with its commit), each in 4w and 8da4w; llama.cpp Q4_0 and Q4_K_M. Context `-c 2560`; pass
   `--override-kv tokenizer.ggml.add_bos_token=bool:false` so that the prompt is 2048 tokens.
   The GGUF files of the finished comparison are on storage you cannot reach: make your own from the same
   upstream model revisions (`kit/MODELS.md`, `kit/make-q4km.sh`), record sizes and bits per weight, and say in
   `ARMS.md` that they are not the same files.
5. **Screen llama.cpp's settings on this device**, on 1B, and record the screen: micro-batch 512, 1024, 2048
   with flash attention on and off. The best setting differed per device and backend (L42).
6. Write `arms.tsv` and `results/<device>/ARMS.md` and commit `ARMS.md` before the timed session. Then
   `kit/session.sh --tools <campaign tools> --stage <stage> --out <name>` and `kit/aggregate.py`.
7. **Compare warm against warm**: quote llama.cpp by `llama-bench`. Its fresh-process timer read 2.1 to 4.8
   times lower on SYCL (L43).
8. What to expect **[measured]**: tuned Vulkan kernels 1.16 to 1.80 times llama.cpp Vulkan on AMD and Intel,
   level on NVIDIA (0.93 to 1.21); vendor backends faster than tuned 4w (SYCL by 19 to 39 %, CUDA by 23 to 38 %
   on the RTX 4070 Ti SUPER, level on the Orin), tuned 8da4w within 6 % of SYCL; ExecuTorch's own upstream
   CUDA backend at 0.75 to 0.85 times stock Vulkan in 4w on the RTX 4070 Ti SUPER.

### Step 15. Deliver

Results of a public device go back to the public remote: the campaign's topic branch, and the comparison's
`results/<device>/` on `topic/llamacpp-compare`. Pushing is the owner's decision for the coordinator
(`COORDINATOR.md`); the campaign pushes its own topic branch as R11 says. Merging the topic branches into
`dev/1.5` is separate, later work **[owner, N5]**; do not start it.

## Part 4. Profiler tracing

Allowed on the new machines **[owner, N4]**; this reverses the rule of the finished campaigns. The record
**[measured]**: on the Radeon 780M (RADV, Mesa 25.2.7) two RGP captures of the three-kernel attention path
worked, and the third, of the fused attention kernel, hung the whole host with no kernel message and needed a
hand reboot. Candidate 1 of step 9 is that kernel, on the same vendor.

1. Save first: commit, update `STATUS.md` to say a capture is about to run and of which kernel.
2. Capture from a detached job with a timeout, one capture at a time.
3. Never during a timed session, a gate or a sweep.
4. After a capture, check that the device still answers before starting the next job.
5. A capture tool that was not tried on this device and driver is "untried", not "safe" (L41).

## Appendix. Quick reference

| what | command or file |
|---|---|
| build one exported commit | `tools/build-both.sh <build-tag> [commit]` -> `sarc/tools/build.sh --llama [--traced] <tree> <out>` |
| golden check | `sarc/tools/spirv_golden.py <build>/backend/vulkan_compute_shaders sarc/golden/spirv.json` |
| gate, unmodified | `sarc/tools/verify.sh --dir <stage> --lock <lock-uuid> --models 1b,3b,8b --schemes 4w,8da4w --pdiff` |
| attention tiers | `test_llama_microbench --sdpa-correctness-only --sdpa-tier=extended` / `--sdpa-tier=full`, 12 passes |
| select a profile | `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=<tag>-refineN` |
| zone and twin check | `sarc/tools/check.sh --no-build` |
| changed outside the dev zone | `git diff --name-status <parent> HEAD` |
| device lock | `flock` on `~/.cache/gpu-lab/lock-<lock-uuid>` |
| hold | `<artifact-dir>/HOLD` (coordinator), `<artifact-dir>/HELD` (queue) |
| exit codes of the campaign tools | 75 lock busy, 76 foreign GPU process, 70 device gone (Orin tools), 77 input missing (Orin tools) |
| status for the owner | `<change>/STATUS.md`, heading "Decision needed from the owner" |
