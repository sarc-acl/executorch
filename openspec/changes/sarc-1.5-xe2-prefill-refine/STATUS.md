# sarc-1.5-xe2-prefill-refine: status

**2026-10-04 — BLOCKED before the first build. Nothing has been built or measured. No candidate exists.**

Branch `topic/xe2-prefill-refine`, parent `6a7cc8cc6` (head of `topic/780m-prefill-refine`).

## Running now

Nothing. No GPU job was started on either card.

## Blocking

The agent session on `fedora-gpu-eval` runs every command inside a write fence (`humanize ... internal fence`,
no-new-privileges). Its policy allows reads everywhere but writes only to the working copy
(`~/hmz-sarc-xe2/executorch`), `/dev/dri` and a private `/tmp`. That fence was not worked around. Inside it:

| needed by the campaign | result |
|---|---|
| artifact directory `~/hmz-sarc-xe2/.artifacts/` | `Permission denied` on write |
| `podman` (for `sarc/tools/build.sh`) | `acquiring runtime init lock: open /run/user/1000/libpod/tmp/alive.lck: permission denied`; user namespaces cannot be created either (`/proc/self/uid_map: Permission denied`) |
| gpu-lab lock `~/.cache/gpu-lab/lock-868023e2-0000-0000-0100-000000000000` | cannot be opened for append, which is how `verify.sh`, `gl.sh` and `e2e5.sh` take it (`exec 9>>`) |
| pushing the branch | not attempted (the campaign pushes at the end) |

Independent of the fence, three things are missing on this host:

- The build image `localhost/et-vk-build:rocky10` is not in the podman store (`/srv/cache/containers/storage`
  holds only `open-webui` and `vllm-openai-xpu`). It has to be built from `tools/Containerfile` (pulls
  `rockylinux:10`, `dnf`, `pip install torch==2.13.0`), which needs a working podman and network.
  The host toolchain is not a substitute: glslc is shaderc v2026.1 (the pinned one is v2023.8), GCC 16.2.1, no
  `torch` for the codegen.
- igpu-roofline is not checked out on this host; only result caches exist in `~/.cache/igpu-roofline/`
  (`fleet-fast-20260926` and others). Without the tool the roofs cannot be re-measured and rates would be
  reported without percent-of-roof.
- `tools/trace.sh` needs a Python with the `executorch` devtools for `trace_analysis.py`; the 780M host used
  `~/sarc-acl/dev/executorch/.venv`, which does not exist here (`XE2_PYTHON` selects the interpreter).

The reviewer confirmed the same denials independently (2026-10-04) and ruled that the fence is not to be
bypassed and outputs are not to be moved into the checkout. What unblocks the campaign: an agent environment
with write access to `~/hmz-sarc-xe2/.artifacts`, `~/.cache/gpu-lab`, `/run/user/1000` and
`/srv/cache/containers/storage`, without no-new-privileges (rootless podman needs `newuidmap`).

## Established so far (read-only checks)

- Card `b70-0` = guest PCI `0000:01:00.0` = Vulkan device 0, deviceUUID `868023e2-0000-0000-0100-000000000000`
  (matches the lock UUID): **`ETVK_DEVICE_INDEX=0`**. Device 1 is the second B70 (`02:00.0`), device 2 llvmpipe.
  Driver ANV, Mesa 26.2.3, `Intel(R) Graphics (BMG G31)`.
- Sensors of `b70-0`: hwmon `xe` under `0000:01:00.0` (`temp2_input` = package, `energy1_input` = card energy,
  no instantaneous power, no busy percent); clock `tile0/gt0/freq0/act_freq` (0 when idle; min 1200, max and
  RP0 2800, RPe 400 MHz); throttle flags `freq0/throttle/status`. Idle package temperature at the time of the
  check: 59 C.
- No GPU process was running; `llama-server`, `comfyui`, `vllm`, `ollama` units inactive. `open-webui` runs in
  a container (no GPU).
- Models and tokenizers for 1B/3B/8B are present under `/mnt/linux-share/models`.

## Done in the tree

`tools/`: the 780M campaign tools copied and adapted for this host. **None of them has run a build or a GPU
job**; what could be exercised inside the fence is listed at the end of this section.

| tool | what it does on this host |
|---|---|
| `host.sh` | paths, lock UUID, `ETVK_DEVICE_INDEX=0`, parent commit, sensors, `cool_start`, and the foreign-GPU-process guard (`gpu_others`, `guarded`): a job is not started while a known GPU workload that the campaign did not start is running, and fails with 76 if one appears during it |
| `build-both.sh <tag> [commit]` | builds an export of one exact commit (`git archive` + the pinned submodule trees), never the live tree; `parent` = `6a7cc8cc6`; creates the output directories; records commit, tree, image id and glslc; fails on a failed build or on a shipped-SPIR-V mismatch against `sarc/golden/spirv.json`; a tag is built once |
| `stage.sh` | stages only `BUILD_BOTH_OK` builds; adds the unaligned prompt `r1304.txt` that `verify.sh` looks for (from `~/.cache/et-e2e/sarc15-r4/`, sha256 `881de104...`, identical in the four earlier B70 studies on this host) |
| `e2e5.sh` | samples act_freq, throttle status, card energy, package temperature. A clock threshold is required: `--calibrate` (baseline / A-A session only) stores the idle temperature and 97 % of the lowest per-run median clock in the artifact directory; every other session refuses to start without it. A throttled run is invalid. A foreign GPU process aborts the session. Next tokens are compared only between completed runs that printed text (otherwise `INVALID`). Exit status and `done.txt` are `E2E5_OK` only with 5 valid runs per build in every cell and all comparisons `SAME` |
| `parent_verify.sh` | parent control `s0-parent-verify`: unmodified `verify.sh` on the pristine parent build and one pass of each SDPA tier |
| `sdpa_passes.sh` | N passes each of `--sdpa-tier=all`, `extended`, `full`; every log and return status kept |
| `gate.sh <session> "<env>" [--sdpa]`, `gate_sdpa.sh` | SDPA passes (12 per tier with `--sdpa`), guarded `verify.sh`, timing session, traces; every step runs even after a failure and nothing is deleted; `gate.done` is `GATE_PASS` / `GATE_FAIL` / `GATE_ABORTED` from `gate_check.py`, never unconditional |
| `gate_check.py` | the acceptance decision, one PASS/FAIL line per requirement: `verify.sh` results and line-by-line equality with the parent control, six complete timing cells with a clock threshold in force, next token `SAME` in six cells on both prompts, traces for six cells x two arms, and for SDPA candidates 12 passes x 3 tiers with the expected case counts (4 / 8 / 4), 0 mismatches and `pairing=ok` |
| `trace.sh`, `trace_analysis.py` | guarded warm ETDump runs under `raw/xe2`; the analyzer is a campaign-local copy of the kit's (the kit file is unchanged) that reads `xe2` and exits non-zero on a missing, empty or single-execution trace; `trace.ok` only when every run and the analysis succeeded |
| `roof_util.py` | no built-in roofs: percent-of-roof only with `--roof` and `--source` naming a fresh igpu-roofline run, otherwise rates only |
| `session.sh`, `screen.sh`, `gl.sh`, `collect.sh`, `summarize.py`, `screen_summary.py`, `prof_decode.py`, `Containerfile` | as on the 780M with this host's paths; `gl.sh` uses the guard |

Exercised inside the fence (no GPU): shell syntax and Python compilation of every tool; `gate_check.py` on the
780M session `s8-r3final` evidence (passes what that session contains) and on copies with an injected
mismatch, a `pairing` failure, an `INVALID` next token and a failed production-diff case (each reported as
FAIL); `roof_util.py` on a 780M `gemm.csv`; the process guard with dummy processes (own child accepted, own
launcher shells ignored, a foreign process before or during a job gives 76). Not exercised: `build-both.sh`,
`stage.sh`, `e2e5.sh`, `trace.sh`, the gate scripts end to end, and whether `act_freq` is readable under load
in this VM (the calibration session fails with `clock_not_readable` if it is not).

The shader generators (`gen_*.py`, `add_batch*.py`) were not copied yet; they come with the first Xe2 kernels.
`proposal.md` is not written: there is no result to propose.

## Per-cell numbers

None measured. Expected parent (from `sarc-1.5-e2e-benchmark/results/cells.csv`, B70 SARC arm, tok/s):
4w 11702.9 / 4864.61 / 2438.10 and 8da4w 12412.1 / 5251.28 / 2737.97 for 1B / 3B / 8B.

## Next step (once unblocked)

1. Build the image (`podman build -t localhost/et-vk-build:rocky10 -f tools/Containerfile tools`), then
   `tools/build-both.sh parent` (`6a7cc8cc6`, includes the shipped-SPIR-V comparison) and `build-both.sh topic0`.
2. `tools/parent_verify.sh`; then the baseline + A/A session (`stage.sh s1-aa parent "" topic0 ""`,
   `session.sh s1-aa --calibrate`), which must agree with the expected numbers below within a few percent.
3. Per-op ETDump breakdown and phase timing of the two Intel linear kernels.
4. Xe2 SDPA kernels (QK^T, attn*V), then 8da4w linear, then 4w linear; sweeps by resumable script, pruned
   statically; every candidate through `gate.sh` / `gate_sdpa.sh`; stop after two consecutive gated candidates
   under 2 % geomean.

## Awaiting B580 confirmation

Nothing yet.
