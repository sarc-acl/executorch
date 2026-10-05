# Fedora → Jetson ExecuTorch cross-build

Builds Linux AArch64 programs inside an x86_64 Ubuntu 24.04 Podman container.
No QEMU, GPU passthrough, CUDA toolkit or on-device compilation is used. The image
is project-owned; `jp7.2.1` names the tested target, not an NVIDIA image tag.

Target: `duck-naughty`, Orin Nano, L4T 39.2.1, NVIDIA Vulkan 595.78.
The native driver remains on the device. Ubuntu AArch64 target libraries supply
build-time ABI dependencies; device execution is the compatibility check.

## Build

Run from the igpu-roofline repository root. A full ExecuTorch checkout with its
submodules must be at the fixed revision below. Dirty tracked files are excluded
from the snapshot, and the original checkout is never modified.

```sh
python3 tools/jetson-cross/prepare.py \
  /home/doremy/Desktop/sarc-acl/yanwen/release14-quant-shaders-4070ti/executorch \
  out/jetson-cross
podman build -t localhost/et-jetson-cross:jp7.2.1 \
  -f tools/jetson-cross/Dockerfile tools/jetson-cross
tools/jetson-cross/container.sh smoke
tools/jetson-cross/container.sh
```

The scripts pin ExecuTorch `0270403ba53ea4b3c4671b8037d468bdeb0f8d8b`
and Vulkan SDK 1.4.341.1. It applies `patches/restore-sdpa-decode.patch`:
four SDPA `_coop` shader files restored verbatim from `8ff908a36^`. The branch's
cleanup removed these files while the single-token dispatch still referenced
them; without the patch the model generated its first token, then aborted on
`sdpa_compute_attn_weights_coop_buffer_buffer_half`. This is an existing decode
packaging regression, not an ARM64 compiler or Jetson driver workaround.

`patches/enable-llama-etdump.patch` also enables and links ETDump in the separately
configured runner; the core tracing option alone does not propagate to it.

Package versions and image ID for the measured build
are retained in `out/jetson-cross/tool-versions.txt` and `image-id.txt`.
Ubuntu repositories can update: reuse the recorded image ID for an exact rebuild.
The build uses eight jobs and disables XNNPACK/CUDA; portable CPU kernels remain
available for operations outside the Vulkan delegate.

`JETSON_CROSS_WORK` overrides the container workspace. The default is
`out/jetson-cross`, which is ignored by Git. SELinux labels are applied only to
this dedicated workspace and this recipe directory, not the original checkout.

`flatc` and `flatcc` are built as host programs by ExecuTorch's ExternalProject
rules. Python and glslc are also host executables; the runner and backend are
AArch64 binaries. Do not export target `CC`/`CXX` globally, which would contaminate
host tool builds. The cross compiler's own target root is used; CMake searches
only target roots for libraries and packages.

## Deploy and validate

Use a fresh remote campaign directory when rerunning measured tests. The supplied
device script targets `~/.cache/et-jetson-cross-0270403ba` and the recorded Orin
UUID. Adapt these explicit values together if targeting another device.

Copy `bundle/` contents, `device-test.sh`, `prompt_256.txt`, both existing 1B Vulkan
PTEs and `tokenizer.model` into that directory. The existing model source is
`/mnt/linux-share/models/llama-3.2-1b/{exported,original}`.

First execute `vulkan-smoke` and `ldd` on the device. Then:

```sh
python3 tools/jetson-cross/validate.py
```

The controller preserves logs and return codes, refuses to overwrite failed
attempts, and bounds the campaign to ten minutes plus timeout termination grace.
The device script acquires gpu-lab's native advisory lock and checks for known
concurrent workload processes. Check gpu-lab availability before starting as
well: unrelated applications need not honor this lock.

Validation includes default/tiled generation for both schemes, small correctness,
sampled 1B production-shape reference, then explicit WMMA prefill plus decode.
The WMMA prompt contains 256 repetitions of ` the`; this is a dispatch exercise,
not a language-quality test. Default generation uses a normal short prompt.
Sampled reference is not full-output validation. Successful process exit alone
is not numerical validation: inspect the check summaries and actual kernel names.

Collect generated `*.etdp` files for actual shader dispatch evidence. Performance
from these cold smoke runs is diagnostic, not a confirmed benchmark. No clocks,
power settings, system libraries or services are changed.

For a separate trace campaign without repeating numerical checks:

```sh
python3 tools/jetson-cross/validate.py --campaign validation-trace --modes tiled wmma
# After copying the device's *.etdp into out/jetson-cross/traces/:
python3 tools/jetson-cross/inspect-traces.py
```

ETDump is size-prefixed; the inspector passes `--size-prefixed` to host flatc.
It writes `dispatch-summary.json` with actual kernel-event counts.

[Verified device results](../../docs/history/JETSON-CROSS-BUILD-2026-09-26.md)
record both quantization schemes, correctness scope and the two source fixes.
