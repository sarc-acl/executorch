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
| pushing the branch, memory directory | not attempted / not writable |

Independent of the fence, two things are missing on this host:

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

What unblocks the campaign (owner's decision, any one of these):

1. run the agent with write access to `~/hmz-sarc-xe2/.artifacts`, `~/.cache/gpu-lab`, `/run/user/1000` and
   `/srv/cache/containers/storage`, without no-new-privileges (rootless podman needs `newuidmap`); or
2. provide the image and prebuilt trees some other way and state that builds and artifacts may live in an
   untracked directory inside the working copy.

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

`tools/`: the 780M campaign tools copied and adapted for this host (not yet run, because nothing can be built):
`host.sh` (new: paths, lock UUID, device index, sensors, GPU process check, cool-down), `build-both.sh`,
`stage.sh`, `gl.sh`, `session.sh`, `e2e5.sh` (samples act_freq, throttle status, card energy, package
temperature; a throttled run is invalid; `--clkmin` defaults to 0 until the baseline session shows the normal
clock under load), `gate.sh`, `gate_sdpa.sh`, `screen.sh`, `trace.sh`, `collect.sh`, `summarize.py`,
`screen_summary.py`, `prof_decode.py`, `Containerfile`. `roof_util.py` still holds the 780M roofs and says so.
The shader generators (`gen_*.py`, `add_batch*.py`) were not copied yet; they come with the first Xe2 kernels.

## Per-cell numbers

None measured. Expected parent (from `sarc-1.5-e2e-benchmark/results/cells.csv`, B70 SARC arm, tok/s):
4w 11702.9 / 4864.61 / 2438.10 and 8da4w 12412.1 / 5251.28 / 2737.97 for 1B / 3B / 8B.

## Next step (once unblocked)

1. Build the image, then `tools/build-both.sh parent` at `6a7cc8cc6`; check the shipped SPIR-V against the golden.
2. Baseline for the six cells, then the A/A session; set `--clkmin` and the idle temperature from them.
3. Per-op ETDump breakdown and phase timing of the two Intel linear kernels, then the Xe2 SDPA kernels.

## Awaiting B580 confirmation

Nothing yet.
