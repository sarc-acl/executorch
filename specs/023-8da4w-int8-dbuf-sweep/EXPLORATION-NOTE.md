# Exploration note

This feature's actual sweep run and results were done as pure exploration in a
separate worktree/branch, deliberately kept out of `quant-perf-optimization`
(active dev) history:

- Worktree: `/local/yanwen.xu/workspace/dbuf-int8-sweep/executorch`
- Branch: `023-8da4w-int8-dbuf-sweep-impl` (branched from `quant-perf-optimization` @ `0da7f5dad`)
- Result commits: `cb664bacf` (sweep run — dbuf2 wins, dbuf3 hypothesis refuted), `09efbb9e6` (all 33 tasks marked complete)
- Report: `specs/023-8da4w-int8-dbuf-sweep/results/m5-dq8ca-dbuf-sweep-report.md` (in that worktree)

**Headline result:** dbuf2 is the fastest double-buffer variant for the int8
`dq8ca_q4gsw` coopmat shader — wins 6/6 tested shapes, +18.15% over shipped
dbuf4, +7.44% over dbuf3.

⚠️ That branch is **local-only, not pushed to any remote** — no backup exists
beyond this machine's disk. If this result should be preserved long-term or
acted on, promote it deliberately (push the branch, or cherry-pick into
active dev) rather than assuming it will persist.
