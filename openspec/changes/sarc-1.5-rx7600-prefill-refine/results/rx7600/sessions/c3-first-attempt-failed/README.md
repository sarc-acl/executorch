# Candidate 3, first attempt: FAILED, superseded (2026-10-07 04:37 UTC)

Stage `c3-linear-per-shape`: every run has rc 127 (`timeout: failed to run command .../cand/llama_main: No such file
or directory`, 16 of 16 rows of `runs.csv`); `cand/` and `parent/` held only an `env` file, no binaries were staged.
The candidate env was `ET_VK_SARC_RX7600_PROFILE=rx7600-refine1` on top of candidate 1, but the binary would have been the
parent build, which does not contain the `rx7600` block of `Overrides.cpp` (committed later, never built), and
`rx7600-refine1` has no linear picks (only the softmax `780m_r3`, as in candidate 1). So even with binaries the
profile would have changed nothing. No tok/s of this attempt exists or counts. Raw files: `.artifacts/superseded/
c3-linear-per-shape-failed-no-binaries-wrong-env-20261007062118/`. Candidate 3 is redone from a real build of an
exported commit that contains the profile, after the complete linear screens.
