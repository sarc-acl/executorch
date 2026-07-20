# Research: M41 Release/1.3 Baseline Clock & Quant-Mode Study

## Decision 1: Reuse existing PTEs, no fresh export

**Decision**: Use the already-exported `4w`/`8da4w` texture-storage PTEs already staged on NFS
(`/sarc-c/gpusw/users/yanwen.xu/android-run/models/`) rather than exporting anything new.

**Rationale**: Per constitution's Default Scope, every PTE for this workload
(2048-prefill/1024-decode, `ctx3072`) already exists for all three target models at both `4w` and
`8da4w`, texture storage — the exact set this feature needs. Re-exporting would duplicate work and
risk introducing a subtly different config (e.g. a different `group_size`) than the one already
used for the collected 4w-floating data, breaking cross-table comparability (spec FR-008).

**Alternatives considered**: Fresh export per model/scheme — rejected, unnecessary and risks
methodology drift from the already-collected 4w-floating baseline.

**Status as of plan time**: All six PTEs (3 models × 2 schemes) are now pushed to the M41 device's
working directory (`/data/local/tmp/llama_vk/`), ~21.7GB total staged. See Decision 4 for the
device-headroom implication.

## Decision 2: `llama_main_rel1.3` runner, no rebuild

**Decision**: Continue using `llama_main_rel1.3` (already built, already on-device) for every run
in this feature — no rebuild, no swap to a different runner variant.

**Rationale**: This binary is what produced the already-collected 4w-floating dataset (spec FR-002)
— using anything else (e.g. a `dev`-branch runner with the WMMA coopmat port, or the
`release13-node-threshold` branch's runner) would compare apples to oranges across this feature's
four tables. Per workspace-root `CLAUDE.md`'s worktree table, `release13-node-threshold` is a
*separate* feature branch off `release/1.3` — its `ET_VK_EXECUTE_NODE_THRESHOLD` addition is not
assumed present in `llama_main_rel1.3` without checking (see Decision 3).

**Alternatives considered**: Rebuild `release-1.3/executorch` with the node-threshold patch
cherry-picked in, to get a workaround for the already-observed 8B pinned watchdog crash —
rejected for this feature. That would no longer be "the same release/1.3 vanilla baseline" the
spec's own Assumptions define; it belongs to a follow-up feature if the crash rate makes the
4w-pinned/8da4w-pinned 8B cells unusable as-is.

## Decision 3: Test, don't assume, whether `ET_VK_EXECUTE_NODE_THRESHOLD` has any effect on this binary

**Decision**: Before the pinned sweep, try setting `ET_VK_EXECUTE_NODE_THRESHOLD=16` on one throwaway
8B pinned run (not counted as one of the 3 reported reps) and check whether it changes behavior at
all (env var recognized vs. silently ignored). Document the finding either way; do not blanket-apply
it to the real reps unless it's confirmed both present and needed.

**Rationale**: `.specify/memory/gotchas.md` G12 found, on the *primary* M5 EVT1 target, that this
threshold is a per-config decision, not a blanket setting — required for 8B T-tiled (crashes
without it) but actively harmful for 3B T-tiled (~11% slower with it, no benefit). If the flag
turns out to exist and work on `llama_main_rel1.3`, applying it indiscriminately to all 27 pinned
reps in this feature would risk the same ~11% tax on cells (1B, 3B T-tiled) that don't need it,
per G12's own finding. An unset/unrecognized env var is harmless to test (the C++ `getenv` call
either finds it or doesn't — no risk to try).

**Alternatives considered**: Assume it's absent (per Decision 2's branch-provenance argument) and
skip testing — rejected: G12's finding was surprising and per-config, and this is a one-throwaway-run
check, cheap enough to just confirm rather than assume.

## Decision 4: Distinguish GPU-watchdog crashes from host-OOM kills via `dmesg`/`/proc/meminfo`

**Decision**: For every `VK_ERROR_DEVICE_LOST` crash encountered in this feature's sweep, before
recording a "watchdog"/"thermal" cause in the results table, check `adb shell dmesg | tail` for an
Android OOM-kill signature and `adb shell cat /proc/meminfo` for `MemAvailable` at the time. Record
whichever cause the evidence actually supports.

**Rationale**: `.specify/memory/gotchas.md` G11 documents a case where an apparently-identical
`vkQueueWaitIdle(...) returned -4` crash was root-caused to a host-side Android OOM kill (the
on-device working directory had accumulated too much staged data, leaving too little `MemAvailable`
headroom for a large model's PTE + warmup's doubled peak memory) — not a genuine GPU/driver
watchdog defect. This feature's own on-device working directory is now at ~21.7GB (Decision 1) on
a device whose total RAM is unconfirmed as of plan time (M41's hardware spec table in
`instruction-for-ai/hardware/other-devices.md` lists it as `_TBD_`) — exactly the risk profile G11
warns about. This session's earlier crashes (pinned 8B watchdog crash, floating 8B rep-2 crash)
were both attributed to GPU-watchdog/thermal causes based on inference from known M5 EVT1 patterns,
without checking `dmesg`/`meminfo` on M41 itself — this decision closes that verification gap before
any more crashes get the same unchecked attribution.

**Alternatives considered**: Keep attributing crashes to watchdog/thermal by pattern-matching to
the M5 EVT1 precedent — rejected, since G11 is direct evidence that pattern-matching this specific
symptom to "GPU watchdog" without checking `dmesg` has produced a wrong root cause before, on this
same workstream, for what looked like an identical crash signature.

**Follow-up**: If any already-recorded crash this session (pinned 8B, floating 8B rep 2) can still
be checked via `dmesg` (logs may have rotated since), retroactively verify or correct its
attribution when writing the final report.

## Decision 5: 3 reps per cell, matching the already-collected dataset

**Decision**: 3 reps per (model, quant-mode, clock-mode) cell, same as the existing 4w-floating
data.

**Rationale**: Matches spec Assumptions (this is also the minimum needed to compute a CoV per
FR-011); keeps all four tables' sampling directly comparable.

**Alternatives considered**: More reps for higher-confidence CoV — rejected for this feature given
the end-of-day deadline (spec SC-007) and the workspace's established 3-rep convention elsewhere.
