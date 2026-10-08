# sarc-1.5-4070ti-fused-port: the fused attention kernel on the RTX 4070 Ti SUPER

Second campaign on this card. The first (`sarc-1.5-4070ti-prefill-refine`) ended with the cooperative-matrix
attention kernels and the fp32 no-tail softmax (`4070ti-refine1`), +46.35 % geomean over `dev/1.5` from the
committed branch. This one ports one more thing on top: the fused attention kernel of the Radeon 780M
(`fused3`, in the RX 7600's `fused3sb` form with subgroup barriers), one-pass variants. It is a port: no sampled
search, no enumeration, no tile sweep; two candidates at most.

- Device: NVIDIA GeForce RTX 4070 Ti SUPER, driver 615.71.09 (as found 2026-10-08), the only GPU of a KVM guest;
  subgroup size 32 (minimum = maximum), `computeFullSubgroups` and `subgroupSizeControl` available,
  `maxComputeSharedMemorySize` 49152 bytes, at most 1024 invocations per workgroup. Clock policy as found:
  P0, 3120 MHz maximum, 285 W limit; not changed.
- Parent of every comparison: commit `6050b1287` with
  `ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=4070ti-refine1`. Pristine arm of the closing session: the same
  build with no environment.
- Candidate profile names: `4070ti-fusedN`, selected by `ET_VK_SARC_DEV_PROFILE` alone.

## Thresholds (fixed before the first measurement; `tools/thresholds.txt`)

| what | value |
|---|---|
| valid timed run | rc 0, 2048 prompt tokens, 0 generated tokens, no GPU process of another owner before, during or after, at least 2 clock samples in the prefill window (sampler 20 ms, a 1B prefill is 60 to 70 ms), median clock at or above the floor |
| clock floor | floor(0.97 x the lowest per-run median clock of the usable A/A runs), one device-wide value; a run whose median is below 1500 MHz (still ramping from the 210 MHz idle clock) is listed and not used for the floor |
| repeats | median of 5 valid runs per arm and cell; 7 if any A/A cell is further than 1.0 % from 1 or any arm's repeat spread exceeds 3.0 % |
| noise band | a difference inside +-2 % is noise |
| baseline | each parent cell within 3 % of the committed-build numbers of the first campaign (1B 29681 / 33032, 3B 12881 / 14949, 8B 5988 / 6954 tok/s, 4w / 8da4w), else explained first |
| A/A | geomean within 1 % of 1 and every cell within +-2 %, else nothing is optimised |
| kernel screen | a kernel replaces the incumbent only when at least 3 % faster in every round |
| reference-error rule (D3) | per production-shape case the candidate's rms and maximum error against the fp32 reference not larger than the parent's; 0 mismatches in every tier; no cell with mean KL above 0.5 nat or top-1 differing on more than one third of at least 32 real-text prompts |
| stop | two consecutive gated candidates under 2 % geomean, or the port list of the task exhausted (task section 6.4), or candidate 1 not correct within 24 hours of device time |

The timer of the runner has 1 ms steps: one step is 1.4 to 1.6 % of a 1B prefill at the parent's rate. 1B cells
are reported with the ETDump dispatch total beside tok/s.

## Tools

Copied from `sarc-1.5-4070ti-prefill-refine/tools/` (same host, same card). What changed:

| file | change | why |
|---|---|---|
| `common.sh` | campaign root, artifact directory (`<campaign-root>/.artifacts`), change directory, process tag `4070ti-fused-port`; `PARENT_COMMIT=6050b1287` and `PARENT_ENV`; `TMPDIR` under the artifact directory; `hold_point` | paths of this campaign; the parent carries a profile; L23; coordinator hold (D6) |
| `gatelib.sh`, `e2e5.sh`, `trace.sh`, `build-both.sh` | `hold_point` before every gate step, every cell of a timed session, every traced run, every build | D6: the smallest unit is one cell, one `verify.sh`, one SDPA pass, one traced run or one build |
| `parent_verify.sh` | takes a session name and an environment (default: the parent's) | two snapshots: `s0-parent-verify` (parent with its profile) and `s0-pristine-verify` (no environment, reference of the hook control) |
| `hook_control.sh` | takes the reference snapshot; the staged environment must equal the reference's | the hook is shown inert with nothing selected and under the parent's profile |
| `calibrate_clock.py` | the floor is taken from the lowest per-run median (rule R6), ramp runs listed; constants read from `thresholds.txt` | the shared rule replaces the first campaign's per-cell-median rule |
| `collect.sh` | paths | |
| `thresholds.txt` | new | task section 6.1 |

Not carried over: the generators (`gen_4070ti_*.py`, `devzone.py`), the linear screens and phase-timing tools
(`screen.sh`, `prof.sh`, ...), the runner probes (`exit_probe.sh`, `first_use_probe.sh`), the local hook patches.
This campaign generates nothing and sweeps nothing. `gate_check.py`, `runrow.py`, `nexttoken.py`, `summarize.py`,
`llama_main_rc.sh`, `warm_file.py`, `stage.sh`, `gate*.sh`, `session.sh`, `gl.sh`, `mktree.sh` are unchanged
(`tools/test_gate_check.py`: 40 tests pass).

Builds: `tools/build-both.sh <tag> <commit>` exports the commit and its pinned submodules from the git object
stores, runs the tree's own `sarc/tools/build.sh --llama` (and `--traced`) in `localhost/et-vk-build:rocky10`
through the docker shim, and fails unless `sarc/tools/spirv_golden.py` reports the shipped variants unchanged.
