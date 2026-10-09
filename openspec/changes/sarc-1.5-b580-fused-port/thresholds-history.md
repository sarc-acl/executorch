# History of `tools/thresholds.txt` and of the measurement protocol (not operative)

What this file is for: `tools/thresholds.txt` holds the operative thresholds. This file records what was added
to it or to the measuring tools after candidate 1 had been measured, why that was wrong, and what was done
about it. Nothing in this file is a rule.

## What `thresholds.txt` contains, and when each block was written

| block | written | relation to the data |
|---|---|---|
| the rules (lines 1 to 26) | commit `7112e8930`, before the first measurement | required by the task (section 6.1) |
| 2026-10-08 21:02 UTC: `s1-aa` is not the calibration session; `idle_wait`; reading of the screen rule's incumbents | before any screen timing was read and before any candidate was timed | |
| 2026-10-08 21:40 UTC: calibrated values from `s1-aa2` | before candidate 1 was timed | required by the task |
| 2026-10-09 00:32 UTC: `idle_wait` lifted | before candidate 1 was timed (`s2-c1` began 00:34 UTC) | owner decision 2026-10-09 00:22 UTC |

The file is back to exactly this content (it is byte-identical to its state before commit `e1e450530`).

## Removed: the block of commit `e1e450530` (2026-10-09 03:17 UTC), added after candidate 1 was measured

The actor appended three entries after session `s3-c2` (candidate 2's first gate, no valid timed run). The
reviewer of 2026-10-09 ruled all three unauthorized: the rule of the file is that nobody changes it after a
candidate's numbers exist, and no owner decision covered them. They were removed in commit `917b471af`.

| entry | what it said | where it had effect | status now |
|---|---|---|---|
| `valid_run` "clarified" | a foreign engine share outside 0 to 100 % makes a run invalid (`busy_unreadable`) | `e2e5.sh`, from `e1e450530` to `917b471af`. No run was rejected by it: sessions `s3b-c2`, `s4-final`, `s5-pristine` have no share outside 0 to 100 % | removed from `e2e5.sh`, which again tests only `fbusy_pct > BUSYMAX`, as calibrated. The one such row that exists (`s3-c2`) is handled by a keyed adjudication, below. Making it a rule of the runner needs an owner decision (asked in `STATUS.md`) |
| `busy_wait` | a timed session starts only when the desktop's share over 5 s is at or below `BUSYMAX`, checked once a minute for at most 3 hours | `host.sh`, `session.sh`, `chain11.sh`, `chain12.sh`, from `e1e450530` to `917b471af` | removed. **It contradicted the owner decision of 2026-10-09 00:22 UTC ("Start every timed run, trace run and screen run at once, whatever the desktop reports").** The actor had called it a different start condition; that does not authorize it |
| `final_stack` | candidate 2 enters the final stack only with a passed gate and a geomean gain above +2 % | the choice of `b580-fused1` as final stack | removed. The choice does not need it: by the `noise_band` rule of the original file a difference inside +-2 % is not a gain, and candidate 2 reads inside the band |

What the wait did, from `.artifacts/logs/busy_wait.log` (kept; copied to `results/b580/`):

| session | first reading | started | delay caused by the wait |
|---|---|---|---|
| `s3b-c2`, before the gate | 05:03:21 UTC, 23.5 % | 05:04:26, 4.2 % | about one minute |
| `s3b-c2`, before the timed session | 06:03:34, 5.4 % | 06:04:39, 2.4 % | about one minute |
| `s4-final`, before the gate and before the timed session | 07:27:50, 0.3 %; 08:08:59, 0.4 % | at once | none beyond the 5 s reading |
| `s5-pristine` | 08:25:18, 0.4 % | at once | none beyond the 5 s reading |

So the timed session of `s3b-c2` did start later than the authorized protocol would have started it, and the
two closing sessions started 5 s later. All three stay where they are as measured; none is the evidence of the
close any more. They are repeated without the wait by `tools/chain13.sh` (`s3c-c2`, `s6-final`, `s7-pristine`).

## Kept, with its own decision: `--extra 12`

`session.sh` passes `--extra 12` (up to 12 replacement pairs a cell instead of 3) since commit `9dd8b782a`,
written at 00:27 UTC with the owner decision of 00:22 UTC and before candidate 1 was timed. It replaces
rejected runs, which that decision keeps; it changes no validity rule and no threshold. It is listed here so
that the list of protocol changes is complete; whether it needs its own ruling is for the owner to say.

## Erratum in the calibration block (the file is not edited)

`thresholds.txt` line 51 says "the 90th percentile of the per-run foreign engine share is 1.03 %". With the
index `e2e5.sh` uses (the sorted 60 valid timed shares of `s1-aa2`, element `int(0.9 * 60)` = 54, counted from
0) the value is **0.94 %**; 1.03 % is the next element. `BUSYMAX` = max(5, 2 x 0.94) = 5.0 % either way, and the
file `busymax_pct` written by `e2e5.sh` holds 5.0. The line stays as written because the file is not changed
after a candidate's numbers exist; this erratum is the correction.

Recomputed again at closing (2026-10-09, second review round) from `stage/s1-aa2/raw/runs.csv`: 60 valid timed
rows; the ten largest shares, sorted, are 0.75, 0.78, 0.85, 0.92, **0.94**, 1.03, 1.31, 1.31, 1.63, 1.74 %
(indices 50 to 59), so `sorted_shares[int(0.9 * 60)]` = index 54 = 0.94 %. `BUSYMAX` stays max(5, 2 x 0.94) =
5.0 %; no run's validity depends on the misquoted 1.03 %, and `thresholds.txt` and its effective values are
unchanged.

## Adjudication of the invalid run of `s3-c2`

`stage/s3-c2/raw/runs.csv` (and its byte-identical copy `results/b580/sessions/s3-c2/runs.csv`) is kept as
written: 252 rows, 228 timed runs, 227 with `valid=0` and reason `foreign_busy` (shares 32.87 to 45.15 %), and
one, `logs/prefill-3b-8da4w-parent-r12.log` (rep 12, slot 2), stored with `valid=1` and an empty reason although
its share reads -5274.35 %, which is not a share. `tools/adjudication.csv` holds the keyed ruling for that row
(session, log, rep, slot -> not valid, `busy_unreadable`). `tools/adjudicate.py` applies it and is what
`summarize.py` and `gate_check.py` now count with: an adjudicated row takes the adjudicated validity (an
adjudication can only invalidate), and any other timed row stored valid without a share between 0 and 100 % is
not counted and fails the analysis until a person adds a keyed row. Recomputed for `s3-c2`: 0 countable runs in
all six cells (stored: 1 in 3B 8da4w parent). Its gate stays `GATE_FAIL`; `stage/s3-c2/gate.txt` is not
rewritten (the recheck has 7 FAIL lines as before, the 3B 8da4w line now "parent 0 cand 0").

## Owner decision, 2026-10-09 (09:55 UTC): items 2 and 3 of "Decision needed from the owner"

Appended to the task file by the coordinator under the owner's standing authorisation.

- **Item 2.** A foreign engine share outside 0 to 100 % is not a reading: the runner rejects the run with reason
  `busy_unreadable` and replaces it inside the session like any other rejected run ("a validity predicate made
  complete, not a threshold change"). Applied to `tools/e2e5.sh` at 10:08 UTC, while chain 13 was in the
  correctness part of the gate of `s6-final` and before that session's timed runs began; `s7-pristine` and
  `s3c-c2` ran before it with the calibrated runner and have no share outside 0 to 100 % (lowest 0.3 %), so the
  predicate would not have changed them. `thresholds.txt` is not edited. The keyed adjudication and
  `tools/adjudicate.py` stay: they cover the row of `s3-c2`, written before the predicate.
- **Item 3.** No wait before a session. A session of chain 13 that ends with a cell short of its valid runs,
  every rejection in that cell being the foreign share, is repeated once, started at least 30 minutes after the
  first ended; if the repeat ends the same way it is reported and not repeated again. Nothing is kept from a
  session that did not complete.
- **Item 1 (F1)** is with the owner.

## Closing, 2026-10-09 (after the owner decision of 15:25 UTC): a gate requirement added, no threshold changed

`tools/gate_check.py` additionally requires, in every pass of the SDPA tiers, one `[sdpa-error]` record per case
with finite `rms_err`, `max_abs_err` and `ref_rms` (`STATUS.md`, "Finding F2": the test's mismatch count is false
for a NaN). It was added after every gate had run and can only turn a pass into a fail; the five recorded gates
decided again with it give the same verdicts (`results/b580/gate-recheck/`), and each session's `gate.txt` is
kept as written. `thresholds.txt` is not edited.
