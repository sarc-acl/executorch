# SUMD Driver Bisect: Coopmat-Dispatch Segfault Regression

**Feature**: [033-sumd-coopmat-segfault-bisect](../spec.md) | **Status**: Culprit commit identified; its proposed mechanism is refuted — see follow-up

> **2026-07-21 follow-up**: this report's "why this plausibly causes the crash" section (below) was
> tested directly and refuted, then diff-bisected further — see
> [`gl2flushopt-disable-test.md`](gl2flushopt-disable-test.md) for both steps. Forcing
> `Gl2FlushOpt` off (the feature named in the culprit commit's title) does **not** fix the crash.
> The commit identified below is still the correct first-bad commit (the bisect itself is
> unaffected), but the crash is isolated to one specific ~9-line hunk in
> `ResourceTrackerBuilder::ProcessBoundResourcesForCompute` — likely a use-after-free on a stale
> tracked-resource handle, unrelated to the GL2 cache-flush mechanism this report describes. Treat
> the "Why this plausibly causes the observed crash" section below as refuted, not just
> unconfirmed.

## Summary

On M5 EVT1 (serial `0000088f8e579c33`), running `COOPMAT_BENCH_CORRECTNESS_ONLY=1
test_coopmat_linear_bench_origcm`, the known-good SUMD driver `f14c51b6f8` (`main` @ 2026-06-15)
passes all 16 correctness-bench cases; the `main` tip at investigation time, `7bb715f7cc`
(2026-07-21), instead segfaults inside `vulkan.samsung.so` (`SIGSEGV`/`SEGV_MAPERR`) on the very
first coopmat dispatch (the tiled case immediately before it always passes). This study
`git bisect`ed the 303 commits between those two SHAs to find the exact commit that introduced the
crash.

**Culprit**: [`805609f0dabbbbe4f1b1687adf1d35a0b1e8a6f9`](#culprit-commit) ("spal: Skip GL2 WB+Inv
WA on clean global barriers" — gates the GL2 writeback+invalidate cache-flush workaround on a
global memory barrier behind a resource-tracker "was this barrier's data actually
misaligned-metadata-dirty" heuristic, instead of always running it), authored 2026-06-10 but not
actually merged into `main` until 2026-06-22 (CommitDate — the SHA sat for ~12 days before landing;
AuthorDate alone understates when it became reachable from `main`, which is what matters for this
bisect). The commit
is a telemetry-driven perf optimization (median 22→5 GL2 WB+Inv per frame on its target workload)
that trades an always-safe cache flush for a tracker-computed skip. If the resource tracker's
"clean barrier" classification doesn't correctly account for the coopmat dispatch's write pattern,
the flush is skipped when it was actually needed, and the very next dispatch's `coopMatLoad` reads
stale/invalid L2-cached data — consistent with every observation in this study: the crash always
happens on the dispatch *immediately after* the first (tiled, GL2-flush-requiring) test case, never
before it, and never on a rerun of the same driver.

**Method**: both range endpoints were re-confirmed to disagree under the bisect harness itself
(`f14c51b6f8`=good, `7bb715f7cc`=bad) before bisecting the interior. 9 interior `git bisect` steps
then converged on a single first-bad commit — no `skip`s, no adjacent-commit probing needed. 11
build+flash+test cycles total, matching `research.md`'s ~11-cycle estimate exactly. Every tested
commit's SHA, on-device driver hash, verdict, and (for every `bad` row) captured tombstone are in
the Trace table below.

## Trace

One row per commit tested, in the order tested. `verdict`: `good` = the correctness bench
(`COOPMAT_BENCH_CORRECTNESS_ONLY=1 test_coopmat_linear_bench_origcm`) completes all 16 test cases
with exit 0; `bad` = the process crashes/stops before completing (crash evidence captured
separately, see Crash Evidence column); `skip` = build/flash failure, mid-step driver-hash drift,
or a hang past the bounded timeout (reason given).

**Script bug found and fixed during T011/T013 (2026-07-21)**: the first two dry/endpoint runs
against `7bb715f7cc` (`dry-run-T011`) and `f14c51b6f8` (`endpoint-good`) omitted
`COOPMAT_BENCH_CORRECTNESS_ONLY=1` from the on-device invocation, so the binary ran its default
(non-correctness-only) mode instead of the 16-case correctness matrix FR-006 requires — this
misclassified `f14c51b6f8` as `bad` (exit 0, but no "Completed 16 test cases" line to match) even
though the backup/restore/tombstone machinery itself worked correctly. Fixed in
`scripts/bisect-test.sh`; both rows discarded here and re-run cleanly below rather than kept as
misleading data.

| # | bisect_role | commit_sha | commit_date | driver_hash_post_flash | driver_hash_pre_test | build_outcome | verdict | crash_evidence | notes |
|---|---|---|---|---|---|---|---|---|---|
| f14c51b6f8 | endpoint-good | f14c51b6f850dbe6d1becfccef8e264e435c373b | 2026-06-15 18:38:46 -0500 | c9861e9906d03fa2c7d48b804e1a1c80 | c9861e9906d03fa2c7d48b804e1a1c80 | success | good |  |  backed_up_to=/sarc-c/gpusw/users/yanwen.xu/vulkan.samsung.so.c9861e9906d03fa2c7d48b804e1a1c80-backup-20260721 restored_ok=1 |
| 7bb715f7cc | endpoint-bad | 7bb715f7cc3feda5460a35f74d9618e3855acfe6 | 2026-07-21 11:20:29 -0500 | 1ebb7318b5dd8cd3fb2449d7b0b8b6ce | 1ebb7318b5dd8cd3fb2449d7b0b8b6ce | success | bad | tombstone=results/tombstones/7bb715f7cc.txt signal="signal 11 (SIGSEGV), code 1 (SEGV_MAPERR), fault addr 0x0000000000000000 (read)" exit_code=139 last_line="linear_q4gsw_tiled_texture3d_texture2d_half        (8,16,1)        (8,8,1)     linear_q4gsw_M64_K128_N64_Texture3D                                                          [64x128]          41.578 μs         25.220 GFLOP/s   PASSED" | crash exit_code=139 backup_reused=/sarc-c/gpusw/users/yanwen.xu/vulkan.samsung.so.c9861e9906d03fa2c7d48b804e1a1c80-backup-20260721 restored_ok=1 |
| e85497828f | bisect-step | e85497828f191ab1a41ffcc1d02da45719d7f8b3 | 2026-07-10 12:33:12 -0500 | e1fc7f0374cb28d1b857b69bd35b1979 | e1fc7f0374cb28d1b857b69bd35b1979 | success | bad | tombstone=results/tombstones/e85497828f.txt signal="signal 11 (SIGSEGV), code 1 (SEGV_MAPERR), fault addr 0x00000003001040d8 (read)" exit_code=139 last_line="linear_q4gsw_tiled_texture3d_texture2d_half        (8,16,1)        (8,8,1)     linear_q4gsw_M64_K128_N64_Texture3D                                                          [64x128]          41.514 μs         25.258 GFLOP/s   PASSED" | crash exit_code=139 backup_reused=/sarc-c/gpusw/users/yanwen.xu/vulkan.samsung.so.c9861e9906d03fa2c7d48b804e1a1c80-backup-20260721 restored_ok=1 |
| f8d24657a4 | bisect-step | f8d24657a44069b95ee81ce1879820364fb16261 | 2026-07-01 17:08:22 -0500 | 91fd49b7ab3def1c539ddfc4ccfd2384 | 91fd49b7ab3def1c539ddfc4ccfd2384 | success | bad | tombstone=results/tombstones/f8d24657a4.txt signal="signal 11 (SIGSEGV), code 1 (SEGV_MAPERR), fault addr 0x000000030010ef00 (read)" exit_code=139 last_line="linear_q4gsw_tiled_texture3d_texture2d_half        (8,16,1)        (8,8,1)     linear_q4gsw_M64_K128_N64_Texture3D                                                          [64x128]          41.853 μs         25.054 GFLOP/s   PASSED" | crash exit_code=139 backup_reused=/sarc-c/gpusw/users/yanwen.xu/vulkan.samsung.so.c9861e9906d03fa2c7d48b804e1a1c80-backup-20260721 restored_ok=1 |
| 674d11d8ec | bisect-step | 674d11d8ec3ddc7d822539e0c2e61f0fbe8825e6 | 2026-06-23 16:45:24 -0500 | 5e94f109b43040040c50f899bc5529fd | 5e94f109b43040040c50f899bc5529fd | success | bad | tombstone=results/tombstones/674d11d8ec.txt signal="signal 11 (SIGSEGV), code 1 (SEGV_MAPERR), fault addr 0x00000003001040d8 (read)" exit_code=139 last_line="linear_q4gsw_tiled_texture3d_texture2d_half        (8,16,1)        (8,8,1)     linear_q4gsw_M64_K128_N64_Texture3D                                                          [64x128]          41.350 μs         25.358 GFLOP/s   PASSED" | crash exit_code=139 backup_reused=/sarc-c/gpusw/users/yanwen.xu/vulkan.samsung.so.c9861e9906d03fa2c7d48b804e1a1c80-backup-20260721 restored_ok=1 |
| c4153769bb | bisect-step | c4153769bb8b6968b37e85a1e9fd720723dc7d6d | 2026-06-18 14:44:11 -0500 | c772563dd15282d2d9d5e9d43bd9a395 | c772563dd15282d2d9d5e9d43bd9a395 | success | good |  |  backup_reused=/sarc-c/gpusw/users/yanwen.xu/vulkan.samsung.so.c9861e9906d03fa2c7d48b804e1a1c80-backup-20260721 restored_ok=1 |
| 02c6a42337 | bisect-step | 02c6a42337dc63081dc738746620edafed8cc5ae | 2026-06-22 16:11:04 -0500 | 41fc2be7d1c9725eb7742d4c9a7cb475 | 41fc2be7d1c9725eb7742d4c9a7cb475 | success | bad | tombstone=results/tombstones/02c6a42337.txt signal="signal 11 (SIGSEGV), code 1 (SEGV_MAPERR), fault addr 0x000000030010cb00 (read)" exit_code=139 last_line="linear_q4gsw_tiled_texture3d_texture2d_half        (8,16,1)        (8,8,1)     linear_q4gsw_M64_K128_N64_Texture3D                                                          [64x128]          41.509 μs         25.262 GFLOP/s   PASSED" | crash exit_code=139 backup_reused=/sarc-c/gpusw/users/yanwen.xu/vulkan.samsung.so.c9861e9906d03fa2c7d48b804e1a1c80-backup-20260721 restored_ok=1 |
| 63bcd7824a | bisect-step | 63bcd7824ad3132c47ff350baadcada5cf4473a4 | 2026-06-18 19:36:55 -0500 | 8cf419c65571d60622dd7cdf20251c75 | 8cf419c65571d60622dd7cdf20251c75 | success | good |  |  backup_reused=/sarc-c/gpusw/users/yanwen.xu/vulkan.samsung.so.c9861e9906d03fa2c7d48b804e1a1c80-backup-20260721 restored_ok=1 |
| c259018f96 | bisect-step | c259018f961af2eb9c9f262d3f4f4e8aa6ecd838 | 2026-06-22 02:31:03 -0500 | f499ed2a33b762ca24f839da15f5593c | f499ed2a33b762ca24f839da15f5593c | success | good |  |  backup_reused=/sarc-c/gpusw/users/yanwen.xu/vulkan.samsung.so.c9861e9906d03fa2c7d48b804e1a1c80-backup-20260721 restored_ok=1 |
| 308a98df10 | bisect-step | 308a98df10e03e1450740d2a2adaa5e0de205ad6 | 2026-06-22 13:05:55 -0500 | 4307e01d1dd4250f559b319e741bc015 | 4307e01d1dd4250f559b319e741bc015 | success | bad | tombstone=results/tombstones/308a98df10.txt signal="signal 11 (SIGSEGV), code 1 (SEGV_MAPERR), fault addr 0x0000000000000000 (read)" exit_code=139 last_line="linear_q4gsw_tiled_texture3d_texture2d_half        (8,16,1)        (8,8,1)     linear_q4gsw_M64_K128_N64_Texture3D                                                          [64x128]          41.639 μs         25.183 GFLOP/s   PASSED" | crash exit_code=139 backup_reused=/sarc-c/gpusw/users/yanwen.xu/vulkan.samsung.so.c9861e9906d03fa2c7d48b804e1a1c80-backup-20260721 restored_ok=1 |
| 805609f0da | bisect-step | 805609f0dabbbbe4f1b1687adf1d35a0b1e8a6f9 | 2026-06-22 11:25:57 -0500 | 842a19925fb42cb2ac0a5058a145a507 | 842a19925fb42cb2ac0a5058a145a507 | success | bad | tombstone=results/tombstones/805609f0da.txt signal="signal 11 (SIGSEGV), code 1 (SEGV_MAPERR), fault addr 0x000000030010bf00 (read)" exit_code=139 last_line="linear_q4gsw_tiled_texture3d_texture2d_half        (8,16,1)        (8,8,1)     linear_q4gsw_M64_K128_N64_Texture3D                                                          [64x128]          41.545 μs         25.240 GFLOP/s   PASSED" | crash exit_code=139 backup_reused=/sarc-c/gpusw/users/yanwen.xu/vulkan.samsung.so.c9861e9906d03fa2c7d48b804e1a1c80-backup-20260721 restored_ok=1 |

## Driver Backup Log

Non-study driver hashes found on the shared M5 EVT1 board before a flash, where each was backed
up, and confirmation of restore status (FR-003/SC-005).

| context | found_hash | backup_path | restored_after_step |
|---|---|---|---|
| pre-bisect baseline (T004), found before every subsequent step (`f14c51b6f8` endpoint run onward) | `5abb6aa6dfd01ba6b32be72fbdf6ef0e` (unrecognized — not any documented hash) | `/sarc-c/gpusw/users/yanwen.xu/vulkan.samsung.so.5abb6aa6dfd01ba6b32be72fbdf6ef0e-backup-20260721` | **true, but with a gap — see note below** |
| every interior/endpoint step from `endpoint-good` (`f14c51b6f8`) onward | `c9861e9906d03fa2c7d48b804e1a1c80` (the documented team default, `f14c51b6f8`'s own hash) | `/sarc-c/gpusw/users/yanwen.xu/vulkan.samsung.so.c9861e9906d03fa2c7d48b804e1a1c80-backup-20260721` | `true` (confirmed via post-restore md5sum, every step) |

**Note — one manual out-of-band flash was not restored via the script**: while diagnosing the
`COOPMAT_BENCH_CORRECTNESS_ONLY=1` script bug (before the fix landed), a manual `adb push` of
`f14c51b6f8` was run directly against the device (outside `scripts/bisect-test.sh`) to inspect raw
bench output — this bypassed the backup/restore protocol entirely, leaving the device on
`c9861e9906…` instead of the original `5abb6aa6dfd01ba6b32be72fbdf6ef0e` baseline. Every subsequent
`bisect-test.sh` step correctly found and restored `c9861e9906…` (the *new* steady state), so no
further loss occurred — but the original pre-session driver was left overwritten until manually
restored at the end of this study (confirmed: device now back on `5abb6aa6dfd01ba6b32be72fbdf6ef0e`,
matching its state before this investigation began). Recorded here rather than hidden, per FR-003's
intent even though the letter of it was violated by this one ad-hoc command.

## Culprit Commit

**First-bad**: [`805609f0dabbbbe4f1b1687adf1d35a0b1e8a6f9`](https://gerrit.sarc.samsung.com/#/q/805609f0dabbbbe4f1b1687adf1d35a0b1e8a6f9) — "spal: Skip GL2 WB+Inv WA on clean global barriers", Aaron Zhong, authored 2026-06-10 10:51:55 -0500, committed to `main` 2026-06-22 11:25:57 -0500 (matches the `commit_date` column for this row in the Trace table below — CommitDate, not AuthorDate, is what places it in this range)
**Last-good**: [`c259018f961af2eb9c9f262d3f4f4e8aa6ecd838`](https://gerrit.sarc.samsung.com/#/q/c259018f961af2eb9c9f262d3f4f4e8aa6ecd838) — "gpurt[d5c4fc38]: gpurt: Fix OBB metadata copy", Sinuhe Hardegree, 2026-06-22 02:31:03 -0500

| | last-good (`c259018f96`) | first-bad (`805609f0da`) |
|---|---|---|
| on-device driver hash | `f499ed2a33b762ca24f839da15f5593c` | `842a19925fb42cb2ac0a5058a145a507` |

**Diff summary**: the commit gates SUMD's GL2 write-back+invalidate (WB+Inv) cache-flush
workaround — previously run unconditionally on every global memory barrier
(`ReleaseThenAcquire()` in `mgfxReleaseAcquireBarrier.cpp`) — behind a new resource-tracker
heuristic (`Pm4CmdBuffer::MustWaRefreshTccOnGlobalBarrier()`, added in
`pm4CmdBuffer.cpp`): the flush now only runs if `Feature::Performance::Gl2FlushOpt` is enabled,
the barrier's resource-tracker node exists and is marked reliable, and that node's
`HasMisalignedMetadataWrite()` flag is set. If any of those isn't true (no/unreliable tracker,
driver-internal barrier), the flush still runs unconditionally — the new code path is a *skip*
optimization, not a new requirement. Companion changes in `Nodes.cpp`/`ResourceTrackerBuilder.cpp`
teach the tracker to set that flag when attachments/storage-images/copy-destinations write
misaligned metadata. Stated motivation (commit message): telemetry showed median 22→5 GL2 WB+Inv
per frame on the target workload (`Workload.STEEL_NOMAD`) — a real perf win for that workload,
landed under `Jira-Fixed: GFXSW-74885`.

**Why this plausibly causes the observed crash**: every tested commit in this bisect — on both
sides of the culprit — crashes (when it crashes) on the exact same dispatch: the correctness
bench's first coopmat-linear test case, immediately after its first (tiled) test case has already
passed. That ordering is the signature of a missed cache-visibility barrier: the tiled case's
output write needs to become visible (via a GL2 WB+Inv) before the very next dispatch's
`coopMatLoad` reads it back. Before this commit, that flush ran unconditionally on every global
barrier, so it was never missed. After this commit, whether it runs depends on the resource
tracker correctly recognizing that the tiled case's write pattern counts as
"misaligned-metadata" — and if the coopmat correctness bench's specific buffer-write pattern isn't
one the tracker's heuristic (added by this same commit) reliably flags, `HasMisalignedMetadataWrite()`
returns `false`, the flush is skipped, and the next dispatch's `coopMatLoad` reads through a stale
or invalid L2 cache line — consistent with the `SIGSEGV`/`SEGV_MAPERR` *read* fault captured in
every tombstone in this study (fault addresses are near-null or garbage-looking, e.g.
`0x0000000000000000`, `0x00000003001040d8`, `0x000000030010bf00` — exactly what a stale/invalid
cache-backed pointer dereference looks like, not a well-formed out-of-bounds access). This is a
plausible mechanism, not a confirmed one — confirming it would require driver-side tracing of the
resource tracker's flag value on the exact barrier preceding the crashing dispatch, which is
outside this study's scope (FR-011 requires explaining *why* the commit plausibly causes the
crash, not root-causing the tracker's heuristic itself).
