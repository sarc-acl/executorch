# Research: M5 EVT1 Floating-Clock Speedup Table

## Decision 1: Reuse every PTE and dispatch-confirmation result from `specs/015`/`018` -- no new export

**Decision**: All 12 configs' PTEs (6 T-tiled baselines from `specs/018`,
6 full-stack optimal from `specs/015`) are reused as-is from `.pte_out`.
Dispatch status (`tiled_confirmed` / `confirmed` coopmat) is likewise
reused, not re-derived.

**Rationale**: Clock frequency affects execution *speed*, not *which
shader/kernel dispatches* -- the eligibility gate (`can_use_q4gsw_coopmat`)
and the storage-type-driven tiled/coopmat split are both compile-time/
export-time decisions, unrelated to runtime clock state. Re-exporting or
re-running dispatch confirmation under floating clocks would burn device
time to re-derive a fact that cannot change with clock state.

**Alternatives considered**: Re-confirming dispatch under floating clocks
"just in case" -- rejected as unnecessary given the above; if a future
session finds evidence clock state somehow affects dispatch (it
structurally shouldn't), that would be a new, separate finding worth its
own investigation, not a reason to duplicate this feature's own work
preemptively.

## Decision 2: Unpin via the documented method -- write hardware min to `min_freq`, hardware max to `max_freq`

**Decision**: Per `.shared-context/instruction-for-ai/commands.md` §5,
floating means writing the device's own hardware-reported minimum
frequency to the `min_freq` sysfs node and the hardware-reported maximum
to `max_freq`, for all three domains (GPU, MIF, INT) -- freeing the DVFS
governor to move anywhere in that full range, rather than any specific
"floating" frequency value.

**Rationale**: This is the only documented, established floating
procedure in this workspace; inventing a different one (e.g., picking an
arbitrary mid-range frequency and calling it "floating") would not match
what "floating" means anywhere else in this workstream's prior reports,
making this feature's numbers incomparable to any future floating
measurement that follows the standard procedure.

**Alternatives considered**: Rebooting the device to clear any pin state
instead of explicitly writing hardware min/max -- rejected, `pin_freqs.sh`
itself is documented as "not persistent across reboots" but nothing
guarantees a reboot leaves the governor in a fully-open state either; the
explicit write is the only way to know the exact state achieved (and to
verify it via FR-004's sysfs readback).

## Decision 3: Verify genuinely-floating state via sysfs readback before trusting any capture

**Decision**: After writing hardware min/max, read back
`/sys/kernel/gpu/{min,max}_freq` (and the `devfreq` equivalents for
MIF/INT) and confirm they reflect the hardware's full range, not the
previously-pinned 509/2730/663 MHz triple, before running any timed
measurement.

**Rationale**: This workspace has already hit the mirror-image failure
mode once (Q10: a ~980MHz DVFS-boost number was mistaken for something
it wasn't, only caught by a GFLOP/s cross-check) -- the analogous risk
here is a "floating" run that's actually still capped low (e.g., a
leftover pin write didn't fully clear), which would silently produce
numbers *lower* than genuine floating throughput and understate the
floating speedup. A cheap sysfs readback closes this gap up front rather
than requiring a retroactive cross-check.

**Alternatives considered**: Skipping the readback and relying on the
observed tok/s being "higher than pinned" as implicit proof of floating
-- rejected; that reasoning is circular for exactly the tiled-baseline
configs this feature cares most about (a modestly-higher-than-pinned
number could just as easily be a partially-open, not fully-floating,
state).

## Decision 4: Report per-rep values; do not collapse into a single mean when variation is meaningful

**Decision**: For every one of the 36 timed runs (12 configs x 3 reps),
publish all 3 per-rep tok/s values in the results file. A mean is
reported alongside for convenience, but is never the *only* number shown,
and is explicitly flagged as a "cold-start+steady-state blend" if the
3 reps show more than a few percent spread (consistent with Principle
VII's own -19%/-27% precedent for tiled configs specifically).

**Rationale**: Constitution Principle VII states this requirement
directly: "report per-rep numbers (or note explicitly that a mean mixes
cold-start peak with throttled steady state) rather than a single
blended average, especially when comparing a tiled baseline to a coopmat
config" -- which is exactly this feature's comparison. This is not a new
methodological invention, it is applying an existing, already-ratified
rule to the first feature that actually needs it.

**Alternatives considered**: Reporting only a mean with a footnote caveat
-- rejected; a footnote is easy to skip past, while the per-rep values
being visible in the table itself (Decision 5) makes the throttle
behavior (or its absence) impossible to miss.

## Decision 5: Speedup ratio uses matched cold-start-vs-cold-start (rep 1 of each config), not blended means

**Decision**: The consolidated floating speedup table's ratio column
divides each config's *first-rep* (cold-start) value against its
T-tiled-baseline counterpart's first-rep value, with the full per-rep
data available in the per-model results files for readers who want the
steady-state comparison instead.

**Rationale**: Per spec.md's Edge Cases and User Story 4, mixing a tiled
config's throttled steady-state value against a coopmat config's
(flatter) cold-start value in the same ratio would either overstate or
understate the real comparison depending on which direction the mismatch
runs. Rep 1 (cold start) is the one point every config has in common
before any throttle has had a chance to develop, making it the only
directly comparable value across configs without needing to define and
justify a "steady state" window (e.g., "reps 2-3 averaged") that would
itself need its own justification for a 3-rep capture.

**Alternatives considered**: Using the steady-state (later-rep) values
instead -- rejected as the primary choice (though reported alongside)
since 3 reps is a short capture and "steady state" is less well-defined
than "the first rep," but flagged as worth comparing directly in the
report's caveat paragraph, per FR-007.
