# Research: SUMD Driver Bisect for the 8da4w-Slower-Than-4w Regression

**Feature**: `032-sumd-driver-bisect` | **Date**: 2026-07-16

No `[NEEDS CLARIFICATION]` markers remained in the spec after `/speckit-clarify` — this document
resolves the remaining *technical* unknowns needed to execute the bisect, not spec-level ambiguity.

## 1. Range endpoint resolution

**Decision**: Use `git log -1 --before="<date> 23:59:59" main` (equivalently `git rev-list -n1
--before=...`) to pick the nearest commit on-or-before each calendar boundary — a single
reproducible git primitive rather than a hand-rolled "closest by absolute distance" rule.

**Resolved endpoints** (computed 2026-07-16, subject to `origin` not having force-moved `main`
since):
- Nov 2024 boundary → `898709039d173379d987ff4c9289cc5be7ee09ef` (2024-11-01 16:18:45, "scripts:
  Disable tests in --test")
- Mar 2026 boundary → `ec3958eae55ec3826d829d2a1149ddb4765b8af4` (2026-03-31 22:00:45,
  "xgl,unittests: just addressed minor nits")

**Rationale**: `--before` is deterministic and re-runnable; "closest by absolute distance" would
occasionally pick a commit *after* the boundary instead, which doesn't matter for bracketing a
regression but adds needless custom logic for no benefit here.

**Alternatives considered**: Picking the nearest commit *after* each date — rejected, no
material difference in outcome (FR-002 re-verifies both endpoints empirically regardless of which
side of the date line they fall on), and `--before` is the more common/idiomatic git idiom for
this kind of "commit as of date X" query.

## 2. Range size and bisect step budget

**Finding**: `898709039d1..ec3958eae55` on SUMD `main` spans **3,055 commits**. `git bisect`
converges in ⌈log2(N)⌉ steps for a monotonic range, so **~12 interior steps** are expected, plus
the 2 endpoint measurements from User Story 1 — call it **~14 build+flash+measure cycles**
nominal, more if any commits are `skip`ped (per spec Edge Cases/FR-009, skips can add extra
adjacent-commit probes to keep `git bisect` converging).

**Implication for planning**: each cycle is a full SUMD Android release build (already observed
to take multiple minutes on this NFS workstation, per `sumd/CLAUDE.md`) + NFS stage + adb flash +
two runner invocations (4w, 8da4w). This is a multi-hour investigation, not a quick check — worth
setting that expectation up front rather than discovering it mid-bisect.

## 3. Driving `git bisect` against a hardware-in-the-loop test

**Decision**: Drive `git bisect` interactively (`git bisect start/good/bad`, one step at a time,
each verdict computed by the agent running the build/flash/measure procedure) rather than
`git bisect run <script>`.

**Rationale**: `git bisect run` is designed for exactly this kind of scripted test, and its exit
code contract (`0`=good, `1`-`124` except `125`=bad, `125`=skip) already matches this spec's
predicate cleanly. However, a fully unattended `run` would hide intermediate progress for a
multi-hour, hardware-dependent loop where a shared device (M41) can go unresponsive mid-run
(spec Edge Cases) — better to drive it step-by-step so a stuck/unresponsive device doesn't strand
an unattended script mid-build. The step-by-step loop still uses the *same* underlying test
procedure/script (see `contracts/bisect-test-script.md`) — it differs only in who calls `git
bisect good`/`bad`/`skip` after each result, not in how the verdict itself is computed.

**Alternatives considered**: `git bisect run scripts/bisect-test.sh` fully unattended — rejected
for this pass given the shared-device liveness risk; may be revisited once the procedure is
proven reliable on a few manual steps.

## 4. Clock pinning to M41's own maximum (non-default target)

**Finding**: This workspace's usual pin target is the workspace-wide default 509/2730/663 MHz
(GPU/MIF/INT) — but this spec explicitly requires M41's own maximum instead: `23400000.sgpu` =
980 MHz, `17000010.devfreq_mif` = 5333 MHz, `17000020.devfreq_int` = 800 MHz.

**Decision**: `/sarc-c/gpusw/users/yanwen.xu/android-run/pin_freqs.sh` (read directly, 2026-07-16 —
it's workspace tooling, not SUMD driver source, so Rule 0 doesn't apply) already takes
`S`/`GPUFREQ`/`MIFFREQ`/`INTFREQ` as env-var overrides (defaults are the M5 EVT1 serial and the
workspace-default clocks). Reuse it as-is:

```bash
S=00000a34cdd4abd3 GPUFREQ=980000 MIFFREQ=5333000 INTFREQ=800000 \
  /sarc-c/gpusw/users/yanwen.xu/android-run/pin_freqs.sh
```

It writes `/sys/kernel/gpu/{min,max}_freq`, `/sys/class/devfreq/23400000.sgpu/{min,max}_freq`, and
`.../17000010.devfreq_mif/scaling_devfreq_{min,max}` / `.../17000020.devfreq_int/scaling_devfreq_{min,max}`,
then itself prints a readback of `gpu/{min,max}_freq` and both devfreq nodes' `cur_freq` — this
script's own printed readback *is* the FR-006 verification, no separate step needed; the bisect
script just has to check the printed values match the requested ones rather than trusting the
writes blindly.

**Rationale**: Removes DVFS/thermal variance as a confound while isolating the driver commit
(spec Assumptions) — a deliberate, spec-directed deviation from the workspace default, not an
oversight. Reusing the existing script (rather than hand-rolling sysfs writes) avoids duplicating
already-validated pin logic and its readback command.

## 5. Build/deploy mechanics (already documented, reused as-is)

**Decision**: Follow `/local/yanwen.xu/sumd/CLAUDE.md` verbatim for every bisect step: `uv run
scripts/run.py --os android --build --build-type release` (with `vulkan-sdk` stripped from
`LD_LIBRARY_PATH` to avoid the documented GpuRt "Too many users" build failure), stage the
resulting `.so` to a per-worktree NFS subdir with a `cmp` verify, then the standard adb push
sequence to M41.

**Rationale**: This procedure is already validated and documented; re-deriving it per bisect step
would be pure risk with no upside. Building happens inside a fresh `git worktree add
../<short-sha> <sha>` per candidate commit (that repo's own documented pattern for testing a
specific SHA without disturbing `main`), consistent with the existing SHA-named worktrees
(`be1273bcbb/`, `c0d117aaf2/`, `f14c51b6f8/`) already in that directory.

## 6. Driver identity capture

**Decision**: After flashing, capture `adb shell md5sum /vendor/lib64/hw/vulkan.samsung.so` and
record that hash alongside the commit SHA for every step (FR-005), not just at the two endpoints.

**Correction (found while authoring the bisect-test script, 2026-07-16)**: the original decision
here was `logcat -d | grep SUMD`, based on a stale reading of the `flash-sumd-driver` memory
note. `.shared-context/instruction-for-ai/access-and-run/README.md` §6 explicitly documents this
as **unreliable**: `logcat`'s `SUMD` lines include `LogBuildDetails`, which dumps a chunk of the
driver's own git *ancestry* — every commit hash that happened to land in that build's history, not
just the one it's built from. A driver built well after some reference commit will still print
that commit's hash in the list, even though it's a different, unidentified build (verified
2026-07-08 in that doc via a documented false positive). `logcat | grep SUMD` is fine for
confirming *some* driver reload happened after a flash, but not for confirming *which* build is
active — which is exactly what this bisect needs at every single step. `md5sum` is the
documented, reliable identity check and is used instead.

## 7. Measurement procedure (release/1.3 vanilla, 1B, 2048 prefill)

**Decision**: Reuse the exact `llama_main_rel1.3` runner and existing NFS/`.pte_out`-staged 1B
4w/8da4w PTEs (ctx supporting 2048-token prefill) from prior specs (`029`, `030`) — no new export.
One rep per quant mode per commit, prefill tok/s only feeds the predicate (FR-007/FR-008); decode
tok/s may be recorded if trivially available but isn't required.
