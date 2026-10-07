# Lessons of the first five campaigns

What this file is for: what happened in the campaigns on the Radeon 780M, Arc B580, Arc Pro B70,
RTX 4070 Ti SUPER and Jetson Orin Nano, what each event cost, and what to do about it. Read it before touching
the environment of a new device. Rules are in `RULES.md`; who decided what is in `OWNER-DECISIONS.md`. A lesson
here is a record of something that happened, with the cost where it was recorded.

The first three campaigns each spent about four hours before their first trustworthy end-to-end number, almost
all of it on sections C and D below. A campaign that forks from a sibling of the same vendor inherits the
adapted tools and should not pay that again.

## A. Where the gain was, and where it was not

- **L1. Attention kernels were the gain on devices that still ran the stock attention.** Cooperative-matrix
  QK^T and attention x V kernels, ported from the 780M's and adapted to each device's matrix shapes and subgroup
  size, gave +46 to +58 % geometric mean on the four devices without them. On the B580, QK^T became 5.8 to 7.9
  times faster and attention x V 6.0 to 8.7 times. Do: on a device without attention rows, do these first.
- **L2. Tile sweeps of the linear kernels did not pay on any device.** 4070 Ti: 27 tiles, best 0.995x. B580:
  17 4w tiles, none faster; a 1.2x reading after one round was a slow base run. B70: all slower except one
  8da4w tile. Cost: hours of device time per sweep. Do: do not start with a tile sweep of the linear kernels.
- **L3. In 8da4w the weight fetch cost more than the multiply.** Phase timing showed a wave fetching for 32 to
  43 % of its time and multiplying for 22 to 35 %. Reading whole texels of weights gained +6 to +7 % end to end
  on the Intel cards and also paid on the Orin. Do: look at phase timing before the tile shape.
- **L4. The stock softmax computes in fp16 and writes a full zero tail.** A softmax that reduces in fp32 and
  does not write the tail removed the last precision objection on the RTX 4070 Ti SUPER, and not writing the
  tail saves about 30 % of the softmax time. Its shader name is fixed in the release zone, so it needs the hook of `OWNER-DECISIONS.md`
  D4. On the Orin a softmax that reads its row once was slower. Do: port the fp32 no-tail softmax; do not assume
  a variant that won on one device wins on the next.
- **L5. On the device that already had the attention kernels (780M) the second layer was smaller.** One fused
  attention kernel +13 %, fp32 softmax +4 %, a linear kernel chosen per layer shape +3 %, an initial parameter
  refinement +8 %; +33.1 % in all. Do: on a device in that position expect +20 to +30 %, not +50 %.
- **L6. The sampled parameter search paid about 1 %.** On the B70 it ran 33 hours over four kernel families (two
  sampled at 2000 configurations, two enumerated): 4w tiles 2 to 6 % faster per shape gave +1.1 % end to end,
  8da4w nothing. The attention kernels' enumeration found kernels 3 to 22 % faster at kernel level that read
  +0.6 and +0.95 % end to end in two sessions, inside the noise band, and were not adopted. On the 780M the same
  enumeration found QK^T 9 to 30 % and attention x V 8 to 10 % faster, on a path its fused kernel no longer
  takes. A kernel-level gain in a kernel that is a few per cent of the prefill does not reach the end-to-end number. What the 780M's
  earlier search did yield was one derived quantity, the number of 16 x 16 matrix tiles one subgroup owns (8 or
  16 best; 64 and more 11 to 25 times slower), which explained as much variance as all yaml parameters together;
  five boolean options did not matter. Do: no sampled search by default (`OWNER-DECISIONS.md` N1); if one is
  ordered, follow D2.
- **L7. Two cards of one family, tuned separately, wanted the same kernels.** B580 and B70: every choice equal,
  same rankings, gains within a few points. Cost: one full campaign to learn it. Do: for a second card of a
  family already tuned, start from the first card's final profile as candidate 0.

## B. Correctness and review

- **L8. A reviewer reading the shader found a data race that every test had passed.** Each lane of a subgroup
  stored the subgroup's reduced value into one shared slot: equal values, but unordered non-atomic writes of one
  location. Cost: the campaign had already been closed and pushed; it was reopened, fixed (only the elected
  lane stores), rebuilt and re-gated (about 6.5 hours of device time were queued for it), and everything
  measured with the old kernel is kept but is not evidence for the fixed one. Do: read every new shader for shared writes before its
  gate (`RULES.md` R7), and make the reviewer do it again (R12).
- **L9. The runner aborts after printing its result, only after a slow model load.** `llama_main` ended with
  `corrupted double-linked list`, rc 134, in 5 of 60 processes whose model load was slow (file not in the page
  cache) and in 0 of 1261 normal ones, in the parent and the candidate alike. On a host whose RAM cannot hold
  the models, the first process after a model change is a slow load, so a gate that samples the first run of a
  cell fails at random. Cost: three gates. Do: read the model file into the page cache before each cell (D5);
  treat an aborted run as invalid and replace it; do not try to fix the runner inside a campaign.
- **L10. A blind retry of a failed gate is not a diagnosis.** Three gates were spent on L9 before the cause was
  looked for. One occurrence does not show "occasional crash"; the same position failing twice is a pattern.
  Do: after the first failed gate, find out which run failed and what distinguishes it, then decide.
- **L11. Next-token equality is a weak correctness measure.** Where the model is undecided, any change of
  summation order swaps the top two tokens; one position of 8B 8da4w does so on several devices. The stock
  attention kernels accumulate in fp16, so part of a candidate's distance from the parent is the parent's own
  rounding error. Cost: candidates rejected or held on
  one gate item until the owner ruled. Do: D1 and D3.
- **L12. A branch measured through an uncommitted local patch cannot be reproduced.** One campaign finished
  and pushed with its recommended configuration reachable only through a local hook patch. Cost: a follow-up
  task to commit the hook, rebuild, re-gate and re-time. Do: once D4 allows the hook, commit it and take the
  final numbers from the committed branch head.
- **L13. Screen scripts that are not resumable lose or duplicate work.** A copied screen script overwrote its
  logs on restart and re-ran everything; another skipped by a status marker written before the rows, so an
  interruption lost rows. A first fix inferred "complete" from whatever was saved and trusted a CSV record cut
  inside its last field. Cost: two review rounds and a 38-check regression test. Do: decide from saved row
  keys against a constant expected shape set; never trust an unterminated last record.
- **L14. A screen without cooling made a tile look 20 % faster.** The base run was hot. Do: cool before every
  screen run, and count a screen result only if it holds in every round.

## C. Measurement

- **L15. Compare against the parent's own gate output, not against an ideal.** The pristine parent already
  prints lines a naive checker calls failures, and which ones differs per device (`linear <scheme> rc=1`
  everywhere; on Intel also `correctness rc=1` and one `default vs tiled output DIFFER`). Do: store the parent's
  `verify.sh` output once and compare every candidate line by line with it.
- **L16. Clock and throttle rules are per device.** The rule "any throttle flag rejects a run" rejected every
  run on a card whose power-limit flag is set in every loaded run by design. Do: calibrate from the A/A session:
  reject on a thermal reason, or on a median clock below 97 % of the lowest per-run median seen there; one
  device-wide threshold.
- **L17. On a power-limited integrated GPU the clock a workload reaches depends on the workload.** Stock kernels
  and long runs settled 3 to 13 % lower than the tuned ones, reproducibly. A fixed floor calibrated on the
  campaign's own arms rejected whole arms of another workload. Do: report such arms with their clock instead of
  rejecting them, and say so next to the number.
- **L18. Two units of the same model, OS and driver disagreed by 20 % on one configuration, reproducibly.** One
  had its GPU frequency governor set to performance, the other scaled by load, and that configuration's load
  pattern let the scaling governor drop a step. The other 25 configurations agreed within 2.5 %. Do: record the
  governor of every device; do not assume a second unit is interchangeable; use it for rankings only (D7).
- **L19. Samplers and timers have a resolution.** A 1B prefill lasts 60 to 240 ms on the fast cards: a 0.1 s
  sampler sees one sample, and the runner's 1 ms timer quantises (ten identical readings; one step is up to
  1 %). Do: sample every 10 to 20 ms on fast devices; report the timer step; use ETDump dispatch time beside it.
- **L20. "Wait for idle + 5 C" never ends on a card whose idle temperature drifts.** Do: also end the wait when
  the temperature has stopped falling.
- **L21. A published baseline can itself be a disturbed measurement.** One cell re-measured 9.8 % above its
  published value; the old five runs were two fast and three slow on a desktop in use, and the published median
  sat in the slow group. Do: when a baseline disagrees, look at the old raw runs before suspecting the build.
- **L22. A GPU that drives a desktop is quiet only while nobody uses it.** Idle and locked: 0.00 % foreign
  engine time in 144 runs. In use: repeat spreads of 10 to 34 % in screens. Do: measure per-run foreign engine
  time where the driver exposes it, fix a ceiling from the A/A session, and use 7 repeats if the A/A asks.

## D. Environment and tools

- **L23. Outputs written under a temp directory are lost.** Do: builds, logs, stage directories, venvs and
  `TMPDIR` go under the artifact directory.
- **L24. The parent must be built from an export of its commit with pinned submodules.** The benchmark kit's
  tree script has a hard-coded repository path. Do: use the sibling's `export_commit` and check the golden.
- **L25. The GPU process guard produced three false aborts.** Causes: the tool's own children started with a
  cleaned environment, a child between `exit` and `wait` whose environment is no longer readable, and a build
  container whose command line contains the runner's name. Do: judge a process by the program it runs and by
  its parent chain; accept descendants of your own tagged processes; never build during a timed session; record
  an idle monitor that holds the device with zero engine time instead of aborting.
- **L26. A two-card guard treated the other card's own job as foreign and stopped a queue (rc 76).** Do: when a
  second device is added on a host, teach the guard about registered jobs before the first split run.
- **L27. `verify.sh` needs the device lock file to be writable.** Do: create
  `~/.cache/gpu-lab/lock-<lock-uuid>` and take the same lock for every GPU job of your own.
- **L28. igpu-roofline and the ETDump analysis need their own Python environments.** Neither was installed on
  the hosts. Do: create a checkout and a venv under the artifact directory; the roofline runners trip a naive
  guard (L25).
- **L29. A profiler capture hung a host.** On the Radeon 780M (RADV, Mesa 25.2.7) two RGP captures of the
  three-kernel attention path worked and the third, of the fused attention kernel, hung the whole host with no
  kernel message; it needed a hand reboot and every detached job on it was lost. Do: `RULES.md` R9.
- **L30. A device can disappear.** One card dropped off the bus twice while idle. Do: if the sensors or the
  vendor tool stop answering, stop, write a marker that blocks every later tool, record it in `STATUS.md`, and
  wait for a person. Never retry, reboot or reload a driver.
- **L31. On a device whose memory is shared and small, one process at a time.** 8 GB shared; the 8B model is
  4.4 GB. Do: check available memory before each 8B run, record swap counters, test the temperature probe on
  the device before the first session (an earlier start was aborted by a probe bug).

## E. Process

- **L32. Detached work survived the loss of the control session.** The control workstation lost power once and
  every host kept measuring. Do: every long job is a detached chain with a status file; after a restart, look
  before starting.
- **L33. An hmz run ends as `failed` after three consecutive model-call failures.** A few minutes of provider
  trouble (HTTP 403 on the model) ended two campaigns at once. Nothing was lost because the work ran detached;
  `/resume` continued. Do: check for ended runs on every patrol; the flow in this package now waits longer and
  gives up only after eight failures in a row.
- **L34. hmz is installed from git without a pin and updated itself mid-campaign.** It changed an API:
  `AgentView.spawn()` no longer takes `env`; `env=` goes to `run()`. The flow file had to be changed before
  the campaigns could be resumed. Do: record the hmz commit a flow was last run with; after changing a
  flow file, quit and reopen the hmz window before `/resume`.
- **L35. A resumed actor has no memory.** What the owner decided in conversation is gone unless it is in the
  task file on the host. Do: append every owner decision to the task file, dated.
- **L36. Habits that paid off.** `STATUS.md` rewritten after every candidate; a commit at every consistent
  state; a failed gate recorded as failed with its numbers and the question put under "Decision needed from the
  owner"; nobody edits the gate; negative results kept with their numbers.

## F. Coordinator mistakes (all made once; do not repeat them)

- **L37. Killing "all hmz processes of a workspace" stopped every campaign.** The first-opened window owns a
  shared hmz process. Do: kill only that workspace's own TUI and daemon pair.
- **L38. A background command that was to remove a `HOLD` file silently failed.** It used bash syntax through a
  fish login shell. Cost: a campaign stayed held. Do: always `ssh <host> 'bash -s' <<'EOF'`, and verify that the
  `HOLD` is gone.
- **L39. A sampler built as a shell pipeline left its poller running after every run.** 160 had piled up
  before it was noticed; run-to-run spread had reached 40 to 66 % and a whole session was discarded. Do: make
  the sampler kill its children, and check the process count after the first few runs, not after the session.
- **L40. An ssh that starts a remote background job does not return** unless stdin and stdout are detached.
  Do: `ssh -n -f <host> 'nohup <command> > <log> 2>&1 < /dev/null'`.
- **L41. A driver-level capture was recommended as "safe" before it had been tried on that device** (L29).
  Do: say "untried on this device" when that is the case.

## G. From the llama.cpp comparison

- **L42. llama.cpp's best settings differ per device and backend.** Flash attention off was faster on Vulkan on
  the two Intel cards; on was faster on SYCL and on Vulkan on the two NVIDIA devices; on the 780M the tiers
  were within 2 %. The best micro-batch was 1024 or 2048. Do:
  screen the settings on each device and record the screen.
- **L43. llama.cpp's fresh-process timer can read several times lower than its warm timer.** On SYCL 2.1 to 4.8
  times: first-use cost on the CPU side, with the GPU clock at idle for most of the window. Do: compare warm
  against warm.
- **L44. The comparison protocol is the campaign's protocol.** The same lock, guard, sampler and validity rules
  were reused through one adapter file per device. Do: add `kit/hosts/<device>/host.sh`, nothing more.
