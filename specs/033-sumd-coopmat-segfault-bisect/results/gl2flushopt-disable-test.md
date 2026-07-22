# Follow-up: Does disabling `Gl2FlushOpt` fix the crash?

**Date**: 2026-07-21 | **Device**: M5 EVT1 (`0000088f8e579c33` / `sj1-dmckee-d01`) | **Status**: Hypothesis refuted

`bisect-report.md` names `805609f0da` ("spal: Skip GL2 WB+Inv WA on clean global barriers",
`Feature::Performance::Gl2FlushOpt`) as the first-bad commit and proposes, as a plausible
mechanism, that the feature's tracker-gated cache flush is being incorrectly skipped for the
coopmat dispatch. This note tests that mechanism directly: does turning `Gl2FlushOpt` off make the
crash go away?

## Runtime override knob: does not exist for this feature

`drivers/pal_common/src/util/palFeatureUtils.cpp` implements a real, generic override mechanism —
Android property (`debug.mariner.Feature::<Name>`), an app-scoped or global
`/data/vendor/gpu/[<appName>/]featureOverride.cfg` file, or the `VK_SECX_feature_control`
extension. Confirmed live on-device (logcat shows the driver finding and loading our app-scoped
cfg file at `/data/vendor/gpu/test_coopmat_linear_bench_origcm/featureOverride.cfg`):

```
Feature::Performance::Gl2FlushOpt = false
```

But it's rejected: `Error: Feature: Override: Failed due to unknown feature name in override:
'Feature::Performance::Gl2FlushOpt = false' -- Source: Config`. Checking the generated
`Feature::SetterMap` in the actual `7bb715f7cc` build (`generated/feature/g_feature.cpp`) confirms
why: `Gl2FlushOpt` has no entry there at all (only 3 of the many `Performance::*` features do). Its
generated class (`generated/feature/performance/g_performance.h`) declares
`ValidSourceMask() { return DeviceDefault | IfhDeviceDefault; }` — this feature can *only* be set
by hardware-generation-based device defaults, never by a `.cfg`/property/env override, regardless
of name/syntax correctness.

## Build-time toggle: tested, does not fix the crash

Since there's no runtime knob, tested via the device-default resolution instead — the only lever
that actually controls this feature:

1. In a local build worktree (`/local/yanwen.xu/sumd/7bb715f7cc`, not shared/upstream), flipped
   `drivers/features/mgfx/performanceFeatures.json`'s `Gl2FlushOpt.DeviceDefaults[0].Enable` from
   `true` back to `false` (this entry had flipped true→false→true itself somewhere between the
   introducing commit and this tip — see note below).
2. Rebuilt (`uv run scripts/run.py --os android --build --build-type release`, ~2 min incremental).
3. Verified the regenerated `g_performance.cpp` reflects the change:
   `Gl2FlushOpt::SetDeviceDefault()` now calls `Set(sourceLevel, false)` (was `true`) whenever
   `genID >= MGFX4_GRBM_CHIP_REVISON_GEN_ID` — an unconditional numeric generation-ID threshold
   check, not a string match, so this does apply to M5's chip.
4. Flashed to M5 EVT1 (backed up the board's pre-existing driver first — see below — and verified
   the new build's hash post-flash: `10cd78c7808dca68c0b394565fdd5285`).
5. Re-ran `COOPMAT_BENCH_CORRECTNESS_ONLY=1 ./test_coopmat_linear_bench_origcm`.

**Result: still crashes.** Identical signature to the unmodified `7bb715f7cc` build and to every
`bad` row in `bisect-report.md`'s trace: the tiled test case (`linear_q4gsw_tiled_texture3d_texture2d_half`)
passes, then `EXITCODE:139` on the very next (coopmat) case.

| build | `Gl2FlushOpt` device default | verdict |
|---|---|---|
| `7bb715f7cc` unmodified | `true` (enabled) | bad (exit 139) — re-confirmed this session before testing the fix |
| `7bb715f7cc` + `Gl2FlushOpt` forced off | `false` (disabled) | **still bad (exit 139)** |

## Conclusion

The `bisect-report.md` mechanism — "the tracker's `HasMisalignedMetadataWrite` heuristic skips a
needed flush" — is **refuted** as the cause, at least as the sole/dominant cause. Disabling the
entire feature (not just working around the heuristic) doesn't change the outcome. The actual
regression must be elsewhere in `805609f0da`'s diff, which touched more than the flush-skip logic:
`pm4CmdBuffer.h`/`.cpp` (struct member layout in the vicinity of the new
`MustWaRefreshTccOnGlobalBarrier()`), `hwImage.cpp`/`.h`, `gfxImage.h`, `image.cpp`/`.h`,
`mgfxReleaseAcquireEvent.cpp`, `ResourceTrackerBuilder.h` — none of which have been examined yet.
Next step: bisect *within* this one commit (selectively revert individual files/hunks of
`805609f0da` on top of `c259018f96`, rebuild, retest) to isolate which specific change actually
causes the crash, rather than assuming it's the named feature the commit's title advertises.

## Aside: `Gl2FlushOpt` device default drift

At the introducing commit `805609f0da` itself, `DeviceDefaults[0].Enable` was `false` (confirmed by
reading that commit's own diff). At tip `7bb715f7cc`, it's `true` — some later commit in the range
flipped it. This doesn't affect the bisect's validity (the *code path* — gating the flush behind
the tracker at all — was introduced at `805609f0da` regardless of the device-default value at that
exact point), but it means "is `Gl2FlushOpt` active on M5" isn't a constant across the bisected
range; not investigated further since it's now moot given the conclusion above.

## 2026-07-21 follow-up #2: diff-bisect within `805609f0da` — found it

Since the `Gl2FlushOpt` flush-*decision* logic is proven not to matter, split the rest of the
commit's diff and tested each piece on M5 EVT1 (same board, same backup/restore discipline as
above; all builds from `/local/yanwen.xu/sumd/c259018f96` + a partial cherry-pick of
`805609f0da`'s diff, `uv run scripts/run.py --os android --build --build-type release`):

| build (on top of `c259018f96`) | result |
|---|---|
| + all of `805609f0da` except `mgfxReleaseAcquireBarrier/Event/Util.cpp`, `pm4CmdBuffer.cpp/.h`, `performanceFeatures.json` (i.e. everything except the flush-skip decision) | **bad**, exit 139 |
| + only the storage/plumbing additions (`Nodes.h/.cpp`, `ResourceTracker.h/.cpp`, `palImage.h`, `gfxImage.h`, `hwImage.cpp/.h`, `image.cpp/.h`, `decorators.h`) — no `ResourceTrackerBuilder.cpp/.h` call sites at all | **good**, all 16 PASSED |
| + everything above *plus* `ResourceTrackerBuilder.cpp/.h`'s call sites, minus just the 9-line `ProcessBoundResourcesForCompute` hunk (see below) | **good**, all 16 PASSED |
| + the full `ResourceTrackerBuilder.cpp/.h` diff including that hunk | **bad**, exit 139 |

This isolates the crash to one specific ~9-line addition in
`ResourceTrackerBuilder::ProcessBoundResourcesForCompute` (`drivers/xgl/icd/api/resource_tracking/ResourceTrackerBuilder.cpp`):

```cpp
// Add the resource as an input/output to the compute node.
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
    ...
}
```

**Likely mechanism**: the pre-existing code a few lines above this (also calling
`vk::Image::ObjectFromHandle(reinterpret_cast<VkImage>(handle))`) only runs `if (addResource)` —
i.e. only for a resource being tracked for the *first* time this pass, when it's guaranteed to
still be a live, just-bound descriptor. The new hunk runs on the broader `foundResource ||
addResource` condition — `foundResource` means "already in the dirty-resource list from earlier,"
which for this bench means: the tiled test case's `Texture3D` output image, tracked once during
test case 1. If that entry isn't evicted from the tracker's dirty-resource list when test case 1's
tensors are torn down before test case 2 runs, this new code re-derefs
`vk::Image::ObjectFromHandle()` on what is now a dangling handle the moment test case 2's global
barrier processing walks it — a **use-after-free**, not a cache-coherency issue. This fits every
observed fact: crash always immediately after the tiled case (the first point a stale `StorageImage`
entry could exist), unaffected by `Gl2FlushOpt` (this code runs regardless of that flag), and
tombstone fault addresses that look like freed/reused memory rather than a clean OOB access.
This mechanism is inferred from the code, not confirmed with a debugger/ASAN — that would be the
next step for full certainty.

**Proper fix, in order of preference**:
1. Don't re-derive `pImage` from the raw handle in this new block at all. Cache the
   `HasMisalignedMetadata()` bit (or the `vk::Image*`) once, at the point the resource is added to
   the tracker (`if (addResource)`, a few lines up, where the object is provably still live), store
   it on the dirty-resource entry, and just read the cached bit here. Never dereferences a
   potentially-stale handle.
2. If (1) is impractical, gate the whole new block on `addResource` only (matching the adjacent
   pre-existing pattern) instead of `foundResource || addResource`. Simpler, but silently stops
   tracking misaligned-metadata writes to a resource on its 2nd+ use within a command buffer —
   probably an acceptable trade for correctness but changes the feature's coverage.
3. Root-cause option (bigger, not scoped to this bug): make the resource tracker's dirty-resource
   list evict entries when the underlying `VkImage`/`VkBuffer` is destroyed, so *no* code path can
   ever dereference a dead handle this way again. Worth raising with the tracker's owners
   regardless of which of (1)/(2) ships as the immediate fix.

Not yet done: confirming the use-after-free mechanism with an instrumented build (e.g. logging the
pointer value and comparing to a `vkDestroyImage` call, or an ASAN build) before this goes into a
bug report as fact rather than a well-supported hypothesis.

## 2026-07-21 follow-up #3: candidate fix implemented and validated on-device

Implemented fix option (1) from the follow-up #2 analysis: cache `HasMisalignedMetadata()` on the
`CmdBufferData::DirtyResource` entry at the point it's added (`if (addResource)`, where `pImage` is
provably still live), and read that cached bit in the `foundResource || addResource` block instead
of re-deriving `pImage` from the raw handle. This preserves the original commit's intended
tracking (unlike simply deleting the 9-line hunk, which would silently stop tracking
misaligned-metadata writes on a resource's 2nd+ use in a command buffer — a correctness regression
of its own, just not a crashing one).

Patch: [`candidate-fix.patch`](candidate-fix.patch) (`git diff` from tip `7bb715f7cc`, 2 files, ~35
lines: `CmdBufferData.h` adds one `bool` field to `DirtyResource`; `ResourceTrackerBuilder.cpp`
moves the `HasMisalignedMetadata()` query to add-time and caches it, and replaces the crash site's
re-derivation with a cache read).

Built on top of unmodified tip `7bb715f7cc` (not the `c259018f96`-based diff-bisect scratch builds
used above), flashed to M5 EVT1 (hash `01f2546479b0748d0f3ae02aeb92c160`, confirmed post-flash),
clocks re-pinned (509/2730/663 MHz, confirmed). Ran
`COOPMAT_BENCH_CORRECTNESS_ONLY=1 ./test_coopmat_linear_bench_origcm` twice:

**Result: all 16 test cases PASSED, exit 0, both runs.** No regression in the printed GFLOP/s
figures for the passing tiled/coopmat cases vs. the unmodified-build numbers earlier in this doc.

This is a locally-built, locally-validated candidate patch against Samsung's proprietary SUMD
driver source — not something we can or should ship ourselves. Next step is handing this (bisect
report + root-cause analysis + this patch) to the driver team (`GFXSW-74885` / Aaron Zhong) for
their own review; they may prefer option (2) or (3) from the follow-up #2 analysis instead. The
UAF mechanism this patch assumes is still inferred from code + the diff-bisect result, not
confirmed with a debugger/ASAN pointer trace.

## Device state

- Board had drifted to an unrecognized driver (`8a8ed6308b260a2e0091249b31984e45`, not in any prior
  backup log) before this session touched it — backed up to
  `/sarc-c/gpusw/users/yanwen.xu/vulkan.samsung.so.8a8ed6308b260a2e0091249b31984e45-backup-20260721`
  before flashing anything.
- Restored to that exact driver at the end of this session (hash-verified).
- The `7bb715f7cc`-with-`Gl2FlushOpt`-off build is staged at
  `/sarc-c/gpusw/users/yanwen.xu/sumd-deploy/7bb715f7cc-gl2flushopt-off/vulkan.samsung.so` if needed
  again; the local edit in `/local/yanwen.xu/sumd/7bb715f7cc` (JSON change only) is uncommitted and
  can be reverted with `git checkout -- drivers/features/mgfx/performanceFeatures.json`.
