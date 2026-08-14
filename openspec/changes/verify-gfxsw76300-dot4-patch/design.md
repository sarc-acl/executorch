## Context

See proposal.md - Why/What Changes. This is a two-repo verification: the patch lives in the SUMD driver checkout (`/local/yanwen.xu/sumd/main` or equivalent), the benchmark harness and PTEs live in this ExecuTorch workspace. Flashing an experimental driver build carries real device risk — GFXSW-76434 was a UAF that SIGSEGV'd M51 during an earlier verification pass on this same ticket, so a bad build is not just "results are wrong," it can crash the board.

**Device access (per yanwen, 2026-08-12 — supersedes any other serial recorded elsewhere for M41):**
- M41: `ssh xgpusw-debug07`, then `export ANDROID_SERIAL=00000bb7cc34abd3`
- M51: `ssh yanwen.xu@sj1-dmckee-d01`

**Crash mitigation:** if the patched build reproduces the GFXSW-76434 UAF (SIGSEGV), pull in its fix before re-testing:
```
git fetch https://yanwen.xu@gerrit.sarc.samsung.com/a/sumd refs/changes/92/65292/9 && git checkout -b change-65292 FETCH_HEAD
```
then rebase/cherry-pick Jayati's dot4 patch on top of `change-65292` and rebuild.

## Goals / Non-Goals

**Goals:**
- Decide which SUMD base commit to apply the patch to, and in what order to touch M41 vs M51.
- Have a concrete rollback path if the patched build misbehaves on-device.

**Non-Goals:**
- Reviewing or improving Jayati's patch itself (that's the SUMD team's call) — we're only measuring its effect on our workload.
- Re-running the full 15-commit bisect — we only need last-good, first-bad, and first-bad+patch for comparison.

## Decisions

- **Base commit for the patch: first-bad (`69e887275e`), not SUMD `main` tip.** Applying on top of the exact commit already characterized in the bisect isolates the patch's effect from unrelated drift on `main` since 2025-08-18. If it doesn't cleanly apply there, next choice is `main` tip with a note that the comparison is no longer apples-to-apples.
- **Device order: M41 first, M51 only if M41 looks clean.** M41 is the existing bisect device with a known-safe recovery path (re-flash last-good md5 `b460447da2...`); M51 is scarcer and the board that already crashed once on this ticket (GFXSW-76434). Don't put M51 at risk for a patch that hasn't been sanity-checked anywhere yet.
- **Reuse the existing bisect harness (`scripts/bisect-test.sh`) rather than a new script.** It already encodes the correct clocks, PTE paths, and rep count for this exact comparison; a new script risks silently drifting from the original bisect's methodology and invalidating the throughput comparison.
- **Confirm via disassembly, not just throughput.** Per the bisect's own comment thread, `v_dot4_i32_i8` instruction count is the direct causal signal; throughput alone could recover for an unrelated reason (clock drift, thermal, noise) and give a false pass.

## Risks / Trade-offs

- [Patched build crashes or hangs the device] → Keep the last-good driver md5 on hand for immediate re-flash; treat M41 as expendable for this test, M51 as not-first. If the crash matches the known GFXSW-76434 UAF signature, apply that fix (`refs/changes/92/65292/9`, see Context) rather than treating it as a new bug.
- [Patch doesn't apply cleanly to `69e887275e`] → Fall back to `main` tip, note the base-commit change when reporting results so Jayati/Pavan know the comparison isn't identical to the original bisect.
- [Throughput recovers but disassembly still shows emulation, or vice versa] → Report both signals independently rather than collapsing to a single pass/fail; a mismatch is itself useful information for the SUMD team.

## Open Questions

- Whether `Feature::M5::CompilerTarget::GetData().evt` actually reports a non-zero value on M41 hardware (vs. M51 evt0/evt1) — this determines whether the patch's `evt == 0` branch takes the same path on M41 as on M51 evt0. Not needed before running the test (the benchmark will answer it empirically), but worth asking Jayati if the result is surprising.
