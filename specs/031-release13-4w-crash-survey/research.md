# Phase 0 Research: Release/1.3 Vanilla 4w Crash Survey on M5 EVT1 (Floating Clocks)

No `[NEEDS CLARIFICATION]` markers were left open by the spec or the plan's Technical Context —
this document records the small set of methodology decisions the spec deliberately left flexible
("mean or median, consistent with this workstream's existing convention", exact run order, exact
crash-recovery mechanics) so `/speckit-tasks` has a single unambiguous procedure to schedule.

## Decision: summary statistic = median (not mean), CoV = stddev/mean over completed reps

**Rationale**: `specs/029-release-version-4w-baseline` — the most directly comparable prior
spec (same workload, same device, same runner family) — used the median of 3 reps as its headline
number. Reusing the same convention keeps this survey's numbers comparable to it. CoV is computed
as sample standard deviation ÷ sample mean of the *completed* reps only (excluding crashed
attempts, per spec FR-005/Edge Cases) — the standard definition, expressed as a percentage.

**Alternatives considered**: Mean-only (rejected — `specs/029` already established median as this
family's convention; switching would make the two specs harder to compare side by side). Reporting
only a single blended number with no per-rep table (rejected outright — Principle VII explicitly
requires per-rep visibility for floating-clock results, and the spec's FR-006 already mandates it).

## Decision: run order = 3B → 1B → 8B (already-observed-safest first, then smallest-untested, then largest/highest-risk last)

**Rationale**: Prior single-run evidence this session: 3B completed cleanly once, 1B crashed once
(second heavy run in its boot session), 8B crashed once (first heavy run in its boot session).
There is no clean deterministic ordering theory yet (the spec's own Edge Cases acknowledge this),
so the practical goal is simply to front-load the model most likely to yield 3 clean reps quickly
(3B) before spending crash-recovery time-budget on 1B and 8B, and to attempt 8B last since a
crash there is the most expensive to recover from in wall-clock terms (largest `.pte`, longest
per-rep runtime) if it turns out to crash repeatedly.

**Alternatives considered**: Fixed size order (1B→3B→8B) — rejected, no evidence it's safer, and
it front-loads two already-observed-crashing models before the one already-observed-safe model.
Interleaving reps across models (e.g. 1B rep1, 3B rep1, 8B rep1, 1B rep2, ...) — rejected as
unnecessary complexity; this is not a thermal-interleaving A/B (Principle VII's interleaving
guidance targets floating-clock A/B *comparisons* between two configs, not a per-model crash
census), and it would multiply the number of driver/clock re-verification checkpoints needed.

## Decision: crash recovery = plain `fastboot reboot` only, escalate on failure

**Rationale**: Already validated twice this session (8B crash, 1B crash) — `fastboot devices`
sees the board immediately after it drops to `S5E9975_LK_Bootloader`, and a plain `fastboot -s
<serial> reboot` (no flashing, no wipe) reliably returns it to a fully booted, `adb`-reachable
Android state within ~10-30s. This is the only recovery mechanism in scope per the spec's Edge
Cases — if it ever fails to bring the device back, the survey stops and escalates rather than
attempting anything more invasive (`fastboot flash`, factory reset, etc.) unattended.

**Alternatives considered**: Physical power-button reboot — rejected as the default (works, but
requires someone physically at the board; `fastboot reboot` is remote and has a 100% success rate
so far this session). Waiting for auto-recovery from bootloader with no intervention — rejected,
already observed the device sits in bootloader indefinitely (4.5+ minutes, no auto-continue)
without an explicit reboot command.

## Decision: verification checkpoints = driver hash + clock floating-range, at model start and after every crash recovery

**Rationale**: Directly required by spec FR-003/FR-004/FR-007 and constitution Principle VIII.
The exact checks (`adb shell md5sum /vendor/lib64/hw/vulkan.samsung.so` compared against the
documented default `c9861e9906d03fa2c7d48b804e1a1c80`, and `cat
/sys/class/devfreq/23400000.sgpu/{min_freq,max_freq}` compared against the HW range `255000`/
`980000`) are exactly the ones already used successfully this session for the 8B/3B/1B single-rep
attempts — no new tooling needed, per constitution Principle X (reuse the documented procedure,
don't invent a new one).

**Alternatives considered**: `logcat | grep SUMD` for driver identity — explicitly rejected by
`access-and-run/README.md` §6 as unreliable (false-positive risk); md5sum is authoritative.

## Decision: no ETDump / kernel-dispatch confirmation run

**Rationale**: Principle IV's "separate ETDump-confirmation run" requirement exists to confirm a
WMMA/coopmat kernel actually dispatched. Vanilla `release/1.3` has no coopmat dispatch path at
all (confirmed by source grep during this session: `execute_threshold_node_count` defaults to 128
with no env-var override anywhere in this worktree) — there is no kernel-selection claim to
verify. This requirement is N/A for this feature (already recorded in the plan's Constitution
Check), not skipped by oversight.

---

# Extension decisions (2026-07-14, continued): 4w pinned gap-fill + 8da4w matrix + threshold policy

## Decision: default fallback threshold changed from 32 to 64, with per-cell empirical fallback further to 32

**Rationale**: Per explicit user direction mid-extension ("actually we should retry all threshold
to 64" / "for all the previous 32 entry, recollect on 64 threshold"). The two cells originally
collected at `32` (8B floating — 1 rep; 8B pinned — 2 reps) were re-collected fresh at `64`. `64`
turned out sufficient for 8B floating (3/3 clean) but **insufficient for 8B pinned** (crashed
2/2, confirmed reproducible before falling back). The fallback to `32` for 8B pinned specifically
is empirically justified, not a reversion to the old default — `64` was given a genuine, fair
try on that exact cell first.

**Alternatives considered**: Keep `32` as the universal fallback (rejected — explicit user
direction). Try an intermediate value (e.g. `48`) for 8B pinned before falling back to `32`
(rejected as unnecessary — `32` was already known-good from the pre-extension data, and probing
for the exact boundary value wasn't requested and would cost additional crash-recovery cycles for
no requested benefit).

## Decision: pinned clocks are empirically riskier than floating for this crash mode, on both quant schemes

**Rationale**: Counter to a naive assumption that pinning to a lower, more predictable clock
would be "safer," this session found the opposite for the sgpu-watchdog crash mode: 3B, which
never crashes on vanilla floating (beyond one transient rep-1 crash), crashes **reliably** on
vanilla pinned (3/3 for `4w`, 1/1 not-retried for `8da4w` — both immediately confirmed and both
fixed by `threshold=64`). The mechanism is straightforward once observed: pinned 509MHz is
*slower* than floating's typical operating point (up to 980MHz), so the same 128-node command
buffer takes longer to execute at the pinned clock, pushing it closer to the ~2.56s watchdog
timeout that a faster floating clock would clear comfortably. This is a genuine, reproducible
finding (not a one-off), consistent across both `4w` and `8da4w` for 3B, and consistent with why
8B — which already crashes at floating — needs the *smallest* threshold (`32`) specifically when
pinned (slowest clock × largest per-node compute × largest attempted chunk size = the most
watchdog-exposed combination in the whole matrix).

**Alternatives considered**: Assume pinned is always safer (the naive prior) — directly refuted
by 3B's vanilla-pinned crash data; discarded once contradicted by the first crash.

## Decision: extend the 4w methodology to 8da4w unchanged, model order 1B → 3B → 8B (not 3B → 1B → 8B)

**Rationale**: The 4w survey's original run-order rationale (front-load the already-observed-
safest model, save the most expensive/highest-risk model for last) still applies conceptually,
but for 8da4w there was no prior single-run evidence to rank 1B vs 3B by risk — so plain size
order (1B → 3B → 8B) was used instead, still saving 8B for last. This turned out to reveal that
1B itself is not immune to transient crashes on `8da4w` floating (2 transient crashes across its
3 reps) — evidence that would have been missed if 8B's known-highest-risk status had caused 1B to
be skipped or under-tested.

**Alternatives considered**: Reuse the exact 3B → 1B → 8B order from the 4w survey (rejected —
that order was justified by 4w-specific single-run evidence that doesn't transfer to a different
quant scheme's compute profile; applying it here would have been assuming rather than verifying,
against Principle VI).

## Decision: for 8B pinned 8da4w, skip the threshold=64 attempt entirely and go straight to 32

**Rationale**: By the time this cell was reached, `64` had already been empirically confirmed
insufficient for 8B-pinned on `4w` (2/2 crashed), and the underlying mechanism (slowest clock +
largest model + node-count-independent per-node compute overhead) is not quant-scheme-specific.
Spending a crash-recovery cycle to re-confirm `64`'s insufficiency on `8da4w`-pinned-8B specifically
would have cost ~1-2 minutes of crash+reboot for a near-certain outcome, with no plan to report
"64 also crashed here" as new information beyond what was already established. This is a
pragmatic time/device-wear trade-off, explicitly noted in `report.md` rather than silently
presented as if `64` had been tried and failed on this exact cell too.

**Alternatives considered**: Test `64` on this cell anyway for full per-cell empirical rigor
(rejected — the marginal evidence value was judged not worth another crash-recovery cycle given
the mechanism was already well-established across 3 other cells; this is the one cell in the
whole extension where a threshold value was chosen by inference rather than direct test, and it
is called out as such rather than presented as equally rigorous).
