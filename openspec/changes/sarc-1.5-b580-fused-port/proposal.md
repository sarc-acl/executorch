# sarc-1.5-b580-fused-port: the fused attention kernel on the Arc B580

Device tag `b580`. Branch `topic/b580-fused-port`, forked from the first B580 campaign's pushed head
(`topic/b580-prefill-refine` at `51d9d757f`). **Parent of every comparison: `51d9d757f` with
`ET_VK_SARC_UNVERIFIED=1 ET_VK_SARC_DEV_PROFILE=b580-refine3`**, the tuned state, not the pristine one. Host: the
owner's desktop `fedora` (Fedora 44, Mesa ANV; the version found is in `STATUS.md`); the B580 is PCI
`0000:03:00.0`, Vulkan device 0, and drives the display. Artifacts:
`/mnt/linux-share/hmz-campaigns/b580-fused/.artifacts/`.

## Why

The first campaign on this card (`sarc-1.5-b580-prefill-refine`, +57.65 %) left attention as three kernels: QK^T,
the release fp16 softmax and attention x V, together 8.7 % (8B 4w) to 20.6 % (1B 4w) of the prefill. On the
Radeon 780M and the RX 7600 one fused kernel replaced the three and gained +13 % and +18 %. This campaign ports
that one kernel to the B580: a port, no sampled search, no enumeration, no tile sweep, two candidates at most.
Expected ceiling from the first campaign's trace: about +6 to +7 % on 8B, +15 % on 1B, +5 to +8 % geomean.

## Thresholds, fixed before any measurement

`tools/thresholds.txt` (rules committed with this file, before the first measurement; calibrated values appended
after the A/A session `s1-aa` and before candidate 1).

## Order of work

1. Parent build (`51d9d757f`), parent snapshots `s0-parent-verify` (parent environment) and `s0-parent-noenv`
   (no environment, for the hook conditions), baseline and A/A.
2. Hook commit: cherry-pick of `1c8861aa7e` (fused attention entry point, owner decision D4.3), with the D4
   evidence that nothing changes while nothing is selected.
3. Candidate 1 `b580-fused1`: the fused attention kernel (`fused3sb` semantics, Intel 8 x 16 x 16 matrix shape).
4. Candidate 2: the one-pass form if candidate 1 is two-pass, else the fp32 no-tail softmax for the calls the
   fused kernel does not take.
5. Final verification on the committed head, one timed session against the parent and one against the pristine
   `6a7cc8cc6`.

## Tools: what was changed from the first B580 campaign's `tools/`

Copied from `sarc-1.5-b580-prefill-refine/tools/`; the originals are not edited. Paths follow from the location of
the copy (`host.sh` derives the artifact directory from it).

| file | change | reason |
|---|---|---|
| `host.sh` | `PARENT_COMMIT=51d9d757f`, `PARENT_ENV` (the `b580-refine3` profile), `PRISTINE_COMMIT=6a7cc8cc6`, `hold_wait` | the parent of this campaign is the tuned head; the closing session also needs the pristine parent; owner decision D6 |
| `hold.sh` (new, from the RX 7600 campaign's `tools/hold.sh`) | artifact directory of this campaign | every detached unit waits while `.artifacts/HOLD` exists |
| `gl.sh`, `session.sh`, `build-both.sh` | call `hold_wait` before taking a lock | as above |
| `parent_verify.sh` | takes `<name> <build tag> "<env>"` instead of the fixed pristine parent without environment | two snapshots are needed: the parent with its profile (compared with every candidate) and without (hook conditions) |
| `collect.sh` | names of this change and artifact directory in the header | |
| `thresholds.txt` (new) | the thresholds | task section 6.1 |
