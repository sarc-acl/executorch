# Jira draft: use-after-free regression from GFXSW-74885

**Type**: Bug (regression) | **Component**: SUMD / PAL (resource_tracking) | **Priority**: suggest High (SIGSEGV, no workaround short of reverting)

## Summary

On M5 EVT1, this issue caused our WMMA (cooperative-matrix) correctness test to fail with a
SIGSEGV. Bisected to a single driver commit; root cause is a use-after-free in
`ProcessBoundResourcesForCompute`, a regression from GFXSW-74885.

- **Known-good driver commit**: `f14c51b6f850dbe6d1becfccef8e264e435c373b` (short `f14c51b6f8`,
  2026-06-15) — on-device driver `.so` md5 `c9861e9906d03fa2c7d48b804e1a1c80`
- **Broken driver commit (first-bad)**: `805609f0dabbbbe4f1b1687adf1d35a0b1e8a6f9` (short
  `805609f0da`, 2026-06-22) — reproduced through tip `7bb715f7cc3feda5460a35f74d9618e3855acfe6`
  (2026-07-21), on-device driver `.so` md5 `1ebb7318b5dd8cd3fb2449d7b0b8b6ce`; still present on
  current `origin/main` tip `5ce9e559e9c27b8ebcca1350dd38b0ad2281cf9b`

## Environment

- Device: M5 EVT1 (Exynos 2500 / Xclipse 970), serial `0000088f8e579c33`, host `sj1-dmckee-d01`
- Driver: SUMD `main`; see commit hashes above
- Last known-good's immediate successor in the bisect: `c259018f96` (parent of the culprit)
- Workload: ExecuTorch Vulkan backend, `test_coopmat_linear_bench_origcm`
  (`COOPMAT_BENCH_CORRECTNESS_ONLY=1`), quantized-linear coopmat vs. tiled correctness matrix

## Description

`COOPMAT_BENCH_CORRECTNESS_ONLY=1 test_coopmat_linear_bench_origcm` SIGSEGVs inside
`vulkan.samsung.so` partway through its 16-case correctness matrix: the first (tiled,
`Texture3D`-backed) test case passes, and the process crashes immediately on the very next
(`coopmat`, buffer-backed) dispatch. Confirmed via `/data/tombstones/`: `SIGSEGV`/`SEGV_MAPERR`
(read fault), fault addresses that look like stale/freed memory (e.g.
`0x0000000000000000`, `0x00000003001040d8`), not a well-formed out-of-bounds access. Call stack
bottoms out in `ComputeGraph::execute()` → `DispatchNode::encode()` →
`register_shader_dispatch()` → `CommandBuffer::dispatch()` → driver.

## Root cause

Bisected the 303 commits between the two driver builds above to a single first-bad commit,
`805609f0da` ("spal: Skip GL2 WB+Inv WA on clean global barriers", GFXSW-74885). That commit's own
title mechanism (the GL2 flush-skip decision) was ruled out directly: forcing
`Feature::Performance::Gl2FlushOpt`'s device default off (fully reverting that decision to its
pre-commit "always flush" behavior) does not fix the crash.

Diff-bisecting the commit's 19 changed files further isolated the actual cause to one ~9-line
addition in `ResourceTrackerBuilder::ProcessBoundResourcesForCompute`
(`drivers/xgl/icd/api/resource_tracking/ResourceTrackerBuilder.cpp`):

```cpp
if (foundResource || addResource)
{
    ...
    if ((computeResource.descTypeBits == DescriptorTypeBits::DescriptorTypeStorageImage) &&
        ((accessType == ResourceAccessType::Write) || (accessType == ResourceAccessType::ReadWrite)))
    {
        auto* pImage = vk::Image::ObjectFromHandle(reinterpret_cast<VkImage>(handle));   // <-- new
        if ((pImage != nullptr) && pImage->PalImage(DefaultDeviceIndex)->HasMisalignedMetadata())
        {
            m_hasPendingMisalignedMetadataWrite = true;
        }
    }
}
```

A few lines above this, pre-existing code performs the identical `vk::Image::ObjectFromHandle()`
cast, but only under `if (addResource)` -- i.e. only for a resource being tracked for the first
time this pass, when its descriptor is guaranteed freshly bound and the backing `VkImage` is known
live. The new code's broader `foundResource || addResource` condition has no such guarantee:
`foundResource` means the resource was already in the dirty-resource list from *earlier* in the
command buffer -- for this workload, the preceding tiled test case's `Texture3D` output. If that
entry is not evicted from the tracker's dirty-resource list when the underlying image is destroyed
between test cases, this code re-dereferences a now-dangling handle -- a use-after-free.

This explains every observed symptom: the crash always occurs immediately after the tiled case
(the earliest point a stale `StorageImage` entry can exist), is unaffected by `Gl2FlushOpt`'s
state (this code runs unconditionally, regardless of that feature), and the tombstone fault
addresses look like reused/freed memory rather than a clean out-of-bounds access.

## Steps to reproduce

1. Flash SUMD `805609f0da` or later (through at least `5ce9e559e9`) to M5 EVT1.
2. `cd /data/local/tmp/llama_vk && COOPMAT_BENCH_CORRECTNESS_ONLY=1 ./test_coopmat_linear_bench_origcm`
3. Observe: first (tiled) case PASSES, process then crashes (`SIGSEGV`) on the second (coopmat)
   case. Exit code 139.

## Evidence

- Full bisect trace, tombstones, and the `Gl2FlushOpt`-disable negative-control test:
  `specs/033-sumd-coopmat-segfault-bisect/results/bisect-report.md` and
  `results/gl2flushopt-disable-test.md` (ExecuTorch workspace, not attachable here directly --
  paste relevant sections into this ticket, or link if the tracker can reach that repo).

## Suggested fix (candidate patch attached/available)

Cache the `HasMisalignedMetadata()` result on the resource's dirty-resource entry
(`CmdBufferData::DirtyResource`) at add-time, when the `VkImage` is provably live, instead of
re-deriving it from the raw handle later. The `foundResource` path then reads the cached bit and
never touches the raw handle again. Preserves the original commit's tracking behavior (unlike
simply removing the check, which would silently stop tracking these writes). Built and validated
on `origin/main` tip and on `7bb715f7cc`: `test_coopmat_linear_bench_origcm` correctness matrix
16/16 PASSED on M5 EVT1 with the fix, reproduced crashing (exit 139) without it, across multiple
builds. Patch available on request / at
`specs/033-sumd-coopmat-segfault-bisect/results/candidate-fix.patch`.

Not yet done: confirming the use-after-free mechanism with a debugger/ASAN pointer trace -- the
above is inferred from the code and the diff-bisect result, not proven at the pointer level.

## Suggested title

`spal/PAL: use-after-free in ProcessBoundResourcesForCompute's misaligned-metadata check (regression from GFXSW-74885 / 805609f0da)`
