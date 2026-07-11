# Phase 0 Research: 8da4w Tile/Subgroup Sweep Ranked by End-to-End Throughput

No `NEEDS CLARIFICATION` items remain from the plan's Technical Context — this feature
reuses `025`'s and `026`'s already-built shader variants, correctness data, and
microbenchmark scores directly, and this same session's own `026` Tier-2 validation
already resolved the one genuinely open methodology question (how to avoid a
shape-mismatched e2e comparison) by trial and correction.

## Decision 1: Combine `025`'s and `026`'s microbenchmark data into one pre-filter, don't re-derive scores

**Decision**: Build the top-8 shortlist (Clarifications) directly from `025`'s
`round2_results.json`/`round3_results.json` (25 subgroup=64 candidates) and `026`'s
`correctness_matrix.json`/`round3_results.json` (5 subgroup=32 candidates, 2 of which are
`all_shapes_pass: true`), ranked together by measured GFLOP/s — not by re-running any
microbenchmark.

**Rationale**: Both files already exist, are correctness-gated, and are on the same
device/driver/clock-pin state this feature reuses. Re-measuring them would spend device
time on information this workstream already has. The only genuinely new work this feature
does is take *already-known-good* candidates to e2e — that's what "(smartly)" means in
context (Clarifications, `spec.md` Context section).

**Alternatives considered**:
- *Re-run the microbenchmark to get one canonical combined ranking*: rejected — `025` and
  `026` used the same harness, same device family, same driver hash
  (`c9861e9906d03fa2c7d48b804e1a1c80`); no evidence the numbers are stale or incomparable.
- *Only use `026`'s 2 correctness-passing candidates, skip `025` entirely*: rejected — `025`
  has 25 correctness-passing candidates with real microbenchmark scores; excluding them
  would shrink the shortlist to fewer than the 8 candidates Clarifications settled on, and
  would bias the shortlist toward `026`'s narrower 5-shape probe instead of `025`'s broader
  25-candidate search.

## Decision 2: Shape-matched model is a per-candidate lookup, not a single project-wide choice

**Decision**: Every candidate's e2e measurement uses the Llama model size (1B/3B/8B) whose
per-layer shapes match the specific representative shape the candidate's microbenchmark
score is keyed to — not a single model chosen once for the whole feature. In practice, both
`025`'s and `026`'s existing scores are keyed to the 8B-shaped representative shapes
(K=4096/14336, this workstream's standing `wq`+`w1_gate` convention measured at those
dimensions specifically in the `test_coopmat_linear_bench` default `M=2048` sweep), so all
8 shortlisted candidates in this feature's initial pass resolve to the same model (Llama
3.1 8B) — but this is a resolved lookup per candidate, not an assumption baked into the
tooling, so a future candidate keyed to a different shape family doesn't silently reuse the
wrong PTE.

**Rationale**: This is directly, freshly learned from this session's own mistake: the
first `026` Tier-2 check used the 1B model against an 8B-shaped-microbenchmark winner and
got a wrong (more pessimistic) answer that had to be re-run and corrected. Baking the
lookup into the tooling as an explicit per-candidate field (spec FR-003;
`data-model.md`'s `Candidate.shape_family` → `model_used` mapping) rather than a global
constant is exactly the fix that prevents this specific class of error recurring in a
scripted, unattended sweep — a global constant would silently paper over a future
candidate that happens to be keyed to a different shape family.

**Alternatives considered**:
- *Just always use 8B, since that's what happens to be needed today*: rejected — this
  reintroduces the exact silent-assumption failure mode this feature exists partly to
  correct; a hardcoded model choice is one dropped assumption away from repeating this
  session's mistake the next time this tooling is reused.

## Decision 3: Adaptive screen→confirm bar implementation

**Decision**: `run_e2e_screen.py` runs exactly 1 prefill measurement per candidate and
computes `(screen_tok_s - baseline_tok_s) / baseline_tok_s`. If that ratio is `>= -0.10`
(within 10% of baseline, or ahead of it — Clarifications), the candidate is queued for
`run_e2e_confirm.py`, which runs 3 more measurements (4 total data points for that
candidate: the original screen + 3 confirm runs, or a clean 3 fresh runs — see Decision 4)
and computes a mean/stddev to compare against the baseline's own multi-run range (not a
single baseline number), per spec FR-005/FR-011.

**Rationale**: Directly implements the user's own stated design in this session ("if the
candidate is too far away from the current winner, no need to do 3 runs, we do 3 runs only
for the close results"). The 10% threshold is this feature's one informed-default choice
(not asked as a separate clarification question, per the clarify session's own reasoning) —
chosen because it's comfortably wider than the noise band `026`'s Tier-2 check actually
observed (non-overlapping 3-run distributions with ~1-2% spread at both 1B and 8B), so a
candidate within 10% of baseline is plausibly a real contender worth the extra 3 runs,
while a candidate 20-30%+ behind (like most of `025`'s/`026`'s non-winning candidates) is
not going to close that gap from run-to-run noise alone.

**Alternatives considered**:
- *Flat 3-run-for-everyone*: rejected per the user's explicit instruction and the
  Clarifications session — wastes device time on candidates with no plausible path to
  winning.
- *Skip the screening run entirely, use existing microbenchmark rank to decide who gets 3
  runs*: rejected — this is precisely what `026`'s single post-hoc Tier-2 check did
  (trust microbenchmark rank, validate the presumed winner directly) and it produced a
  wrong answer once already; a cheap real e2e screening run per candidate is only
  marginally more expensive than trusting the microbenchmark rank, and actually observes
  the metric this feature cares about before committing to 3x the device time.

## Decision 4: Whether the screening run counts toward the 3-run confirmation

**Decision**: `run_e2e_confirm.py` takes 3 *fresh* runs (not reusing the screening run's
result as one of the 3) — 4 total data points per confirmed candidate, but only 3 are used
for the reported mean/stddev.

**Rationale**: The screening run's purpose is triage, not measurement — model load timing,
thermal state, and OS scheduling jitter immediately after a cold model load (the screening
run) are not necessarily representative of the model's warmed-up steady state the 3
confirmation runs capture back-to-back, matching `025`'s/`026`'s existing 3-run convention
which always uses a clean run set. This avoids a subtle inconsistency where confirmed
candidates have differently-composed data (screen+2 fresh vs. 3 fresh) depending on
whether they screened above or below the 10% line.

**Alternatives considered**:
- *Reuse the screening run as run 1 of 3, only measure 2 more*: rejected — saves one run
  per confirmed candidate (at most 8 runs total across the whole shortlist) at the cost of
  a subtle methodology inconsistency; not worth it given this feature's overall run count
  is already small (spec SC-004).

## Decision 5: Execution worktree — reuse `dbuf-int8-sweep`, same as `026`

**Decision**: Execute in the same `dbuf-int8-sweep` worktree (`023-8da4w-int8-dbuf-sweep-impl`
branch) `026` used, not a new worktree.

**Rationale**: This worktree already has every dependency User Story 1 needs: the built
`llama_main` runner, the `linear_dq8ca_q4gsw_coopmat_tsweep` shader catalog covering all 8
shortlisted candidates (assuming the top 8 by combined rank are a subset of `025`'s 25 +
`026`'s 5 — verified in Phase 1's `data-model.md` derivation), and a warm, installed
`cmake-out-android-vk` build tree. Branching fresh would only add bootstrap cost for zero
benefit, following `026`'s own Decision 5 reasoning exactly (and even more strongly here,
since this feature — unlike `026` — needs no new shader source at all for its MVP).

**Alternatives considered**: None seriously — `026`'s own research.md already settled this
question with directly-applicable reasoning; re-deriving it here would be redundant.
